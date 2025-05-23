from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
import spacy
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from main import predict, suggest_roles, analyze_skill_gaps
import sqlite3
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'jndjsahdjxasudhas-09vzx2223'
app.config['SESSION_TYPE'] = 'filesystem'
socketio = SocketIO(app)
DATABASE = "new.db"

# Initialize database
with sqlite3.connect(DATABASE) as conn:
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS register (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT, user_email TEXT, password TEXT
        )
    ''')
    conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        user_name = request.form['user_name']
        user_email = request.form['user_email']
        password = request.form['password']
        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO register (user_name, user_email, password) VALUES (?, ?, ?)",
                    (user_name, user_email, password)
                )
                conn.commit()
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_email = request.form['user_email']
        password = request.form['password']
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM register WHERE user_email=? AND password=?",
                (user_email, password)
            )
            user = cursor.fetchone()
        if user:
            session['user_email'] = user_email
            session['user_name'] = user[1]  # user_name is the second column
            return render_template('upload.html', name=user[1], email=user_email)
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html')

nlp = spacy.load('en_core_web_sm')
result = []
with open('linkedin skill', encoding='utf-8') as f:
    external_source = list(f)

for element in external_source:
    result.append(element.strip().lower())

def extract_skill_1(resume_text):
    nlp_text = nlp(resume_text)
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skills = result
    skillset = []
    for i in tokens:
        if i.lower() in skills:
            skillset.append(i)
    for i in nlp_text.noun_chunks:
        i = i.text.lower().strip()
        if i in skills:
            skillset.append(i)
    return [word.capitalize() for word in set([word.lower() for word in skillset])]

STOPWORDS = set(stopwords.words('english'))
EDUCATION = ['CSE', 'EEE', 'ECE', 'IT', 'MCA']

def extract_education(resume_text):
    nlp_text = nlp(resume_text)
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]
    edu = {}
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in STOPWORDS:
                edu[tex] = text + (nlp_text[index + 1] if index + 1 < len(nlp_text) else '')
    education = []
    for key in edu.keys():
        year = re.search(r'(((20|19)(\d{2})))', edu[key])
        if year:
            education.append((key, ''.join(year[0])))
        else:
            education.append(key)
    return education

def extract_marks(resume_text):
    nlp_text = nlp(resume_text)
    mark_patterns = [
        [{'POS': 'NUM'}, {'ORTH': '%'}],
        [{'LOWER': 'grade'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'NUM'}],
        [{'LOWER': 'cgpa'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'NUM'}],
        [{'LOWER': 'marks'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'NUM'}],
        [{'LOWER': 'percentage'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'NUM'}],
        [{'LOWER': 'percent'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'NUM'}],
        [{'POS': 'NUM'}, {'ORTH': '%'}]
    ]
    matcher = spacy.matcher.Matcher(nlp.vocab)
    matcher.add('MARK_PATTERN', mark_patterns)
    matches = matcher(nlp_text)
    extracted_marks = []
    for match_id, start, end in matches:
        mark_span = nlp_text[start:end]
        extracted_marks.append(mark_span.text)
    return extracted_marks

def extract_skill(resume_text):
    nlp_text = nlp(resume_text)
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skills = [
        'python', 'machine learning', 'css', 'c++', 'data science',
        'php', 'mysql', 'html', 'sql', 'tensorflow', 'deep learning',
        'pandas', 'opencv', 'typescript', 'c#', 'data factory', 'ci/cd'
    ]
    skillset = []
    for i in tokens:
        if i.lower() in skills:
            skillset.append(i)
    for i in nlp_text.noun_chunks:
        i = i.text.lower().strip()
        if i in skills:
            skillset.append(i)
    return [word.capitalize() for word in set([word.lower() for word in skillset])]

@app.route('/back')
def back():
    if 'user_email' not in session:
        flash('Please log in to upload a resume.', 'error')
        return redirect(url_for('index'))
    return render_template('upload.html', name=session.get('user_name'), email=session.get('user_email'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_email' not in session:
        flash('Please log in to upload a resume.', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        start_time = time.time()
        logger.info("Starting file upload processing...")

        if 'resume_file' not in request.files:
            flash('No file part in the request.', 'error')
            return redirect(url_for('upload'))

        i_f = request.files['resume_file']
        if i_f.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))

        # Validate file type
        if not i_f.filename.lower().endswith('.pdf'):
            flash('Only PDF files are allowed.', 'error')
            return redirect(url_for('upload'))

        try:
            # Emit progress updates
            socketio.emit('progress', {'progress': 10, 'message': 'Extracting text from PDF...'})
            logger.info("Extracting text from PDF...")
            resMgr = PDFResourceManager()
            retData = io.StringIO()
            TxtConverter = TextConverter(resMgr, retData, laparams=LAParams())
            interpreter = PDFPageInterpreter(resMgr, TxtConverter)
            for page in PDFPage.get_pages(i_f):
                interpreter.process_page(page)
            txt = retData.getvalue()
            retData.close()
            TxtConverter.close()
            logger.info(f"Text extraction completed in {time.time() - start_time:.2f} seconds")

            socketio.emit('progress', {'progress': 30, 'message': 'Extracting resume details...'})
            logger.info("Extracting resume details...")
            def extractResume(resume_text):
                skill = extract_skill(resume_text)
                skill_from_external = extract_skill_1(resume_text)
                mark = extract_marks(resume_text)
                degree = extract_education(resume_text)
                return skill, skill_from_external, mark, degree

            skill, skill1, mark, degree = extractResume(txt)
            totallskill = skill + skill1
            low = [i.lower() for i in totallskill]
            num_skills = len(totallskill)  # New feature: number of skills
            logger.info(f"Resume extraction completed in {time.time() - start_time:.2f} seconds")

            socketio.emit('progress', {'progress': 50, 'message': 'Processing academic performance...'})
            logger.info("Processing marks...")
            percentages = []
            cgpa_values = []
            grades = []
            other_values = []

            for a in mark:
                item = a.lower()
                if '%' in item:
                    percentage_value = float(item.strip('%')) / 10
                    percentages.append(percentage_value)
                elif 'cgpa' in item:
                    cgpa_value = ''.join(filter(str.isdigit, item))
                    cgpa_values.append(int(cgpa_value) if cgpa_value else 0)
                elif 'grade' in item:
                    grade_value = float(''.join(filter(str.isdigit, item)))
                    grades.append(grade_value)
                else:
                    other_values.append(item)

            if len(cgpa_values) > 0:
                l = max(cgpa_values)
            elif len(percentages) > 0:
                l = max(percentages)
            elif len(grades) > 0:
                l = max(grades)
            elif len(other_values) > 0 and any(x.replace('.', '', 1).isdigit() for x in other_values):
                l = max([float(x) for x in other_values if x.replace('.', '', 1).isdigit()])
            else:
                l = None

            if l is None:
                marks_message = "Marks information was not found in the resume."
            else:
                marks_message = f"Extracted marks: {l}"
            logger.info(f"Marks processing completed in {time.time() - start_time:.2f} seconds")

            socketio.emit('progress', {'progress': 70, 'message': 'Mapping skills...'})
            logger.info("Mapping skills...")
            value = {
                'java': 0, 'javascript': 1, 'php': 2, 'python': 3, 'ruby': 4,
                'sql': 5, 'c': 6, 'css': 7, 'html': 8, 'bootstrap': 9
            }
            skill_indices = [j for i, j in value.items() if i in low]

            if not skill_indices:
                flash('Could not find any relevant skills in the resume.', 'error')
                return redirect(url_for('upload'))
            logger.info(f"Skill mapping completed in {time.time() - start_time:.2f} seconds")

            socketio.emit('progress', {'progress': 80, 'message': 'Predicting companies...'})
            logger.info("Predicting companies...")
            final = set()
            for skill_idx in skill_indices:
                companies = predict(l if l else 0, skill_idx, num_skills)
                final.update(companies)
            companies = list(final)
            logger.info(f"Company prediction completed in {time.time() - start_time:.2f} seconds")

            socketio.emit('progress', {'progress': 90, 'message': 'Suggesting job roles...'})
            logger.info("Suggesting job roles...")
            job = set()
            for skill in low:
                role = suggest_roles(skill)
                if role != 'Unknown Role':
                    job.add(role)
            job = list(job)
            logger.info(f"Job role suggestion completed in {time.time() - start_time:.2f} seconds")

            socketio.emit('progress', {'progress': 95, 'message': 'Analyzing skill gaps...'})
            logger.info("Analyzing skill gaps...")
            skill_gaps = analyze_skill_gaps(totallskill, job)
            logger.info(f"Skill gap analysis completed in {time.time() - start_time:.2f} seconds")

            socketio.emit('progress', {'progress': 100, 'message': 'Processing complete!'})
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            return render_template('result.html', companies=companies, job=job, marks=marks_message, skill_gaps=skill_gaps)

        except Exception as e:
            logger.error(f"Error processing the resume: {str(e)}")
            socketio.emit('progress', {'progress': 0, 'message': f'Error: {str(e)}'})
            flash(f'Error processing the resume: {str(e)}', 'error')
            return redirect(url_for('upload'))

    return render_template('upload.html', name=session.get('user_name'), email=session.get('user_email'))

if __name__ == '__main__':
    socketio.run(app, debug=True)