from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import io
import re
import os
import sqlite3
import logging
import time

from main import predict, suggest_roles, analyze_skill_gaps, calculate_resume_score

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-fallback-change-in-production')
app.config['SESSION_TYPE'] = 'filesystem'
socketio = SocketIO(app)
DATABASE = "new.db"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

with sqlite3.connect(DATABASE) as conn:
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS register (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            user_email TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()

LINKEDIN_SKILLS = set()
try:
    with open('linkedin skill', encoding='utf-8') as f:
        for line in f:
            LINKEDIN_SKILLS.add(line.strip().lower())
except Exception:
    pass

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
    'sass', 'less', 'bootstrap', 'tailwind', 'material ui',
    'nextjs', 'svelte', 'firebase', 'supabase', 'prisma',
    'jwt', 'oauth', 'websocket', 'grpc', 'redis',
}

SKILL_ALIASES = {
    'react.js': 'react', 'reactjs': 'react',
    'node.js': 'nodejs', 'node js': 'nodejs',
    'vue.js': 'vue', 'vuejs': 'vue',
    'next.js': 'nextjs', 'nuxt.js': 'nuxt',
    'express.js': 'express',
    'mongo': 'mongodb', 'mongo db': 'mongodb',
    'postgres': 'postgresql', 'psql': 'postgresql',
    'scikit learn': 'scikit-learn', 'sklearn': 'scikit-learn',
    'natural language processing': 'nlp',
    'tailwind css': 'tailwind', 'tailwindcss': 'tailwind',
    'material-ui': 'material ui', 'mui': 'material ui',
    'google cloud': 'gcp', 'google cloud platform': 'gcp',
    'amazon web services': 'aws',
    'microsoft azure': 'azure',
    'spring boot': 'spring boot',
    'react native': 'react native',
    'ci cd': 'ci/cd', 'cicd': 'ci/cd',
    'rest': 'rest api', 'restful': 'rest api',
    'artificial intelligence': 'artificial intelligence',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'js': 'javascript',
    'ts': 'typescript',
    'k8s': 'kubernetes',
}

SKILL_ENCODING = {
    'python': 0, 'java': 1, 'javascript': 2, 'sql': 3,
    'php': 4, 'css': 5, 'html': 6, 'c++': 7, 'ruby': 8,
    'typescript': 2, 'react': 2, 'nodejs': 2, 'angular': 2, 'vue': 2,
    'mysql': 3, 'postgresql': 3, 'mongodb': 3,
    'django': 0, 'flask': 0, 'fastapi': 0,
    'machine learning': 0, 'deep learning': 0, 'tensorflow': 0,
    'pytorch': 0, 'pandas': 0, 'numpy': 0, 'scikit-learn': 0,
    'aws': 3, 'azure': 3, 'gcp': 3, 'docker': 3, 'kubernetes': 3,
    'spring': 1, 'spring boot': 1, 'kotlin': 1, 'scala': 1,
    'go': 0, 'rust': 7, 'swift': 6, 'c#': 7,
    'linux': 5, 'git': 2,
}

SKILL_PRIORITY = ['python', 'java', 'javascript', 'machine learning', 'sql', 'c++', 'php']

EDUCATION = ['CSE', 'EEE', 'ECE', 'IT', 'MCA', 'BCA', 'BTECH', 'MTECH', 'BSC', 'MSC', 'MBA', 'BE', 'ME']


def normalize_skill_text(text):
    for alias, canonical in SKILL_ALIASES.items():
        text = re.sub(r'\b' + re.escape(alias) + r'\b', canonical, text)
    return text


def extract_skills(resume_text):
    text_lower = resume_text.lower()
    text_clean = re.sub(r'[,;:\|\•\-–—]', ' ', text_lower)
    text_clean = re.sub(r'\s+', ' ', text_clean)
    text_clean = normalize_skill_text(text_clean)

    found_skills = set()
    for skill in COMMON_SKILLS:
        if ' ' in skill:
            if skill in text_clean:
                found_skills.add(skill.title())
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_clean):
                found_skills.add(skill.title())

    return [s for s in found_skills if len(s) >= 2 and not s.isdigit()][:25]


def extract_marks(resume_text):
    patterns = [
        r'cgpa[:\s]*(\d+(?:\.\d+)?)',
        r'gpa[:\s]*(\d+(?:\.\d+)?)',
        r'(\d{1,2}(?:\.\d+)?)\s*cgpa',
        r'(\d{1,2}(?:\.\d+)?)\s*%',
        r'percentage[:\s]*(\d{1,2}(?:\.\d+)?)',
    ]
    marks = []
    for pattern in patterns:
        for match in re.findall(pattern, resume_text.lower()):
            val_str = match if isinstance(match, str) else match[0]
            try:
                val = float(val_str)
                if 0 < val <= 10:
                    marks.append(f"CGPA: {val}")
                elif 30 <= val <= 100:
                    marks.append(f"{val}%")
            except Exception:
                pass
    return marks[:5]


def get_cgpa_value(marks):
    for m in marks:
        if 'CGPA' in m:
            try:
                return float(m.split(':')[1].strip())
            except Exception:
                pass
        elif '%' in m:
            try:
                val = float(m.replace('%', '').strip())
                return round(val / 10, 1)
            except Exception:
                pass
    return 7.0


def extract_education(resume_text):
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
        user_name = request.form['user_name'].strip()
        user_email = request.form['user_email'].strip().lower()
        password = request.form['password']
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('index'))
        hashed = generate_password_hash(password)
        try:
            with sqlite3.connect(DATABASE) as conn:
                conn.execute(
                    "INSERT INTO register (user_name, user_email, password) VALUES (?, ?, ?)",
                    (user_name, user_email, hashed)
                )
                conn.commit()
            flash('Account created! Please sign in.', 'success')
            return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            flash('Email already registered.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_email = request.form['user_email'].strip().lower()
        password = request.form['password']
        with sqlite3.connect(DATABASE) as conn:
            user = conn.execute(
                "SELECT * FROM register WHERE user_email=?", (user_email,)
            ).fetchone()
        if user and check_password_hash(user[3], password):
            session['user_email'] = user_email
            session['user_name'] = user[1]
            return render_template('upload.html', name=user[1], email=user_email)
        else:
            flash('Invalid email or password.', 'error')
            return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/back')
def back():
    if 'user_email' not in session:
        flash('Please sign in to continue.', 'error')
        return redirect(url_for('index'))
    return render_template('upload.html', name=session.get('user_name'), email=session.get('user_email'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_email' not in session:
        flash('Please sign in to continue.', 'error')
        return redirect(url_for('index'))

    if request.method == 'POST':
        start_time = time.time()

        if 'resume_file' not in request.files:
            flash('No file in request.', 'error')
            return redirect(url_for('upload'))

        i_f = request.files['resume_file']
        if not i_f.filename:
            flash('No file selected.', 'error')
            return redirect(url_for('upload'))

        if not i_f.filename.lower().endswith('.pdf'):
            flash('Only PDF files are allowed.', 'error')
            return redirect(url_for('upload'))

        file_data = i_f.read()
        if len(file_data) > MAX_FILE_SIZE:
            flash('File too large. Maximum size is 5 MB.', 'error')
            return redirect(url_for('upload'))
        i_f.seek(0)

        try:
            socketio.emit('progress', {'progress': 10, 'message': 'Extracting text from PDF...'})
            resMgr = PDFResourceManager()
            retData = io.StringIO()
            converter = TextConverter(resMgr, retData, laparams=LAParams())
            interpreter = PDFPageInterpreter(resMgr, converter)
            for page in PDFPage.get_pages(i_f):
                interpreter.process_page(page)
            txt = retData.getvalue()
            retData.close()
            converter.close()

            socketio.emit('progress', {'progress': 30, 'message': 'Analyzing skills...'})
            all_skills = extract_skills(txt)

            socketio.emit('progress', {'progress': 50, 'message': 'Processing academic info...'})
            marks = extract_marks(txt)
            education = extract_education(txt)

            marks_message = "Detected: " + ", ".join(marks[:5]) if marks else "No grades detected"

            socketio.emit('progress', {'progress': 70, 'message': 'Matching companies...'})
            mark_value = get_cgpa_value(marks)
            dominant_skill = next(
                (s.lower() for s in all_skills if s.lower() in SKILL_PRIORITY),
                all_skills[0].lower() if all_skills else 'python'
            )
            skill_encoded = SKILL_ENCODING.get(dominant_skill, 0)
            companies = predict(mark_value, skill_encoded, len(all_skills))

            socketio.emit('progress', {'progress': 82, 'message': 'Finding job matches...'})
            seen_roles = set()
            unique_roles = []
            for skill in all_skills[:10]:
                for role in suggest_roles(skill):
                    if role not in seen_roles:
                        unique_roles.append(role)
                        seen_roles.add(role)
            unique_roles = unique_roles[:8]

            socketio.emit('progress', {'progress': 92, 'message': 'Analyzing skill gaps...'})
            skill_gaps = analyze_skill_gaps(all_skills, unique_roles)

            has_certs = any(w in txt.lower() for w in ['certif', 'certified', 'certificate', 'certification'])
            has_projects = any(w in txt.lower() for w in ['project', 'developed', 'built', 'created', 'implemented'])
            score = calculate_resume_score(all_skills, len(education), has_certs, has_projects)

            socketio.emit('progress', {'progress': 100, 'message': 'Analysis complete!'})
            logger.info(f"Processing done in {time.time() - start_time:.2f}s")

            return render_template(
                'result.html',
                companies=companies,
                job=unique_roles,
                marks=marks_message,
                skill_gaps=skill_gaps,
                skills=all_skills,
                score=score,
                name=session.get('user_name', ''),
            )

        except Exception as e:
            logger.error(f"Error: {e}")
            socketio.emit('progress', {'progress': 0, 'message': f'Error: {str(e)}'})
            flash(f'Error processing resume: {str(e)}', 'error')
            return redirect(url_for('upload'))

    return render_template('upload.html', name=session.get('user_name'), email=session.get('user_email'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
