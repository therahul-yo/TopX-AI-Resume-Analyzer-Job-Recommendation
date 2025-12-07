from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
import re
import sqlite3
import logging
import time

from main import predict, suggest_roles, analyze_skill_gaps

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

# Load external skills database
LINKEDIN_SKILLS = set()
try:
    with open('linkedin skill', encoding='utf-8') as f:
        for line in f:
            LINKEDIN_SKILLS.add(line.strip().lower())
except:
    pass

# Common programming skills list
COMMON_SKILLS = {
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
    'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
    'html', 'css', 'sql', 'mysql', 'postgresql', 'mongodb', 'redis',
    'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'spring',
    'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'scikit-learn',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
    'linux', 'bash', 'powershell', 'selenium', 'junit', 'pytest',
    'machine learning', 'deep learning', 'data science', 'artificial intelligence',
    'rest api', 'graphql', 'microservices', 'devops', 'ci/cd',
    'agile', 'scrum', 'jira', 'confluence', 'figma', 'adobe xd',
    'power bi', 'tableau', 'excel', 'spark', 'hadoop', 'kafka',
    'elasticsearch', 'rabbitmq', 'nginx', 'apache', 'terraform',
    'opencv', 'nlp', 'computer vision', 'blockchain', 'solidity',
    'unity', 'unreal engine', 'flutter', 'react native', 'android', 'ios',
    'laravel', 'rails', 'express', 'fastapi', 'spring boot',
    'hibernate', 'maven', 'gradle', 'webpack', 'vite',
    'sass', 'less', 'bootstrap', 'tailwind', 'material ui'
}

# Education keywords
EDUCATION = ['CSE', 'EEE', 'ECE', 'IT', 'MCA', 'BCA', 'BTECH', 'MTECH', 'BSC', 'MSC', 'MBA', 'BE', 'ME']

def extract_skills(resume_text):
    """Extract skills using keyword matching (works with Python 3.14)"""
    text_lower = resume_text.lower()
    
    # Clean the text - replace common separators with spaces
    text_clean = re.sub(r'[,;:\|\•\-–—]', ' ', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_clean)
    
    found_skills = set()
    
    # Check common skills with whole word matching
    for skill in COMMON_SKILLS:
        # For multi-word skills
        if ' ' in skill:
            if skill in text_clean:
                found_skills.add(skill.title())
        else:
            # For single word skills, use word boundary matching
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_clean):
                found_skills.add(skill.title())
    
    # Filter out any garbage (must be at least 2 chars and not just numbers)
    valid_skills = []
    for skill in found_skills:
        if len(skill) >= 2 and not skill.isdigit():
            valid_skills.append(skill)
    
    return valid_skills[:20]  # Limit to top 20 skills

def extract_marks(resume_text):
    """Extract marks/percentages from resume using regex"""
    patterns = [
        r'(\d{1,2}(?:\.\d+)?)\s*%',  # Percentages like 85% or 85.5%
        r'cgpa[:\s]*(\d+(?:\.\d+)?)',  # CGPA patterns
        r'gpa[:\s]*(\d+(?:\.\d+)?)',   # GPA patterns
        r'(\d{1,2}(?:\.\d+)?)\s*cgpa',
        r'percentage[:\s]*(\d{1,2}(?:\.\d+)?)',
    ]
    
    marks = []
    for pattern in patterns:
        matches = re.findall(pattern, resume_text.lower())
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            try:
                val = float(match)
                if 0 < val <= 10:  # GPA/CGPA
                    marks.append(f"CGPA: {val}")
                elif 30 <= val <= 100:  # Percentage
                    marks.append(f"{val}%")
            except:
                pass
    
    return marks[:5]  # Limit to 5 marks

def extract_education(resume_text):
    """Extract education from resume"""
    education = []
    text_upper = resume_text.upper()
    
    for edu in EDUCATION:
        if edu in text_upper:
            education.append(edu)
    
    return list(set(education))

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
            session['user_name'] = user[1]
            return render_template('upload.html', name=user[1], email=user_email)
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html')

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

        if not i_f.filename.lower().endswith('.pdf'):
            flash('Only PDF files are allowed.', 'error')
            return redirect(url_for('upload'))

        try:
            # Step 1: Extract text from PDF
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

            # Step 2: Extract skills
            socketio.emit('progress', {'progress': 30, 'message': 'Analyzing skills...'})
            logger.info("Extracting skills...")
            all_skills = extract_skills(txt)
            
            # Step 3: Extract marks/education
            socketio.emit('progress', {'progress': 50, 'message': 'Processing academic info...'})
            logger.info("Extracting marks and education...")
            marks = extract_marks(txt)
            education = extract_education(txt)
            
            # Format marks message
            if marks:
                marks_message = "Detected: " + ", ".join(marks[:5])
            else:
                marks_message = "No specific marks/grades detected"
            
            # Step 4: Predict companies
            socketio.emit('progress', {'progress': 70, 'message': 'Matching companies...'})
            logger.info("Predicting companies...")
            
            # Get first skill for prediction
            first_skill = all_skills[0].lower() if all_skills else 'python'
            
            # Simple skill encoding
            skill_mapping = {
                'python': 0, 'java': 1, 'javascript': 2, 'sql': 3,
                'php': 4, 'css': 5, 'html': 6, 'c++': 7, 'ruby': 0
            }
            skill_encoded = skill_mapping.get(first_skill, 0)
            
            mark_value = 7.0
            num_skills = len(all_skills)
            companies = predict(mark_value, skill_encoded, num_skills)
            
            # Step 5: Suggest job roles
            socketio.emit('progress', {'progress': 85, 'message': 'Finding job matches...'})
            logger.info("Suggesting job roles...")
            
            job_roles = []
            for skill in all_skills[:5]:
                role = suggest_roles(skill)
                if role != 'Unknown Role':
                    job_roles.append(role)
            
            # Remove duplicates
            unique_roles = []
            for role in job_roles:
                if role not in unique_roles:
                    unique_roles.append(role)
            
            # Step 6: Analyze skill gaps
            socketio.emit('progress', {'progress': 95, 'message': 'Analyzing skill gaps...'})
            logger.info("Analyzing skill gaps...")
            skill_gaps = analyze_skill_gaps(all_skills, unique_roles)
            
            # Complete
            socketio.emit('progress', {'progress': 100, 'message': 'Analysis complete!'})
            logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
            
            return render_template(
                'result.html',
                companies=companies,
                job=unique_roles,
                marks=marks_message,
                skill_gaps=skill_gaps,
                skills=all_skills
            )

        except Exception as e:
            logger.error(f"Error processing the resume: {str(e)}")
            socketio.emit('progress', {'progress': 0, 'message': f'Error: {str(e)}'})
            flash(f'Error processing the resume: {str(e)}', 'error')
            return redirect(url_for('upload'))

    return render_template('upload.html', name=session.get('user_name'), email=session.get('user_email'))

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)
