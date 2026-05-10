import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, session, redirect, url_for, flash
from flask_socketio import SocketIO
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

from main import (
    predict, suggest_roles, analyze_skill_gaps,
    calculate_score_breakdown, generate_insights,
    categorize_skills, normalize_aliases, role_match_scores,
    ALL_SKILLS, SKILL_CATEGORY,
)

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
    conn.execute('''
        CREATE TABLE IF NOT EXISTS register (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            user_email TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()

EDUCATION = ['CSE', 'EEE', 'ECE', 'IT', 'MCA', 'BCA', 'BTECH', 'MTECH', 'BSC', 'MSC', 'MBA', 'BE', 'ME', 'PHD']

SKILL_PRIORITY = ['python', 'java', 'javascript', 'machine learning', 'sql', 'c++', 'php']

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

# ─── Section parsing ────────────────────────────────────────────
SECTION_HEADERS = {
    'skills':         r'(?:technical\s+|core\s+|key\s+)?(?:skills?|technologies|competenc(?:e|ies)|expertise)',
    'experience':     r'(?:work\s+|professional\s+|industry\s+)?experience|employment(?:\s+history)?|career(?:\s+history)?',
    'education':      r'education(?:al(?:\s+background)?)?|academic(?:\s+background)?|qualifications?',
    'projects':       r'projects?|portfolio',
    'certifications': r'certif(?:ication)?s?(?:\s+(?:and|&)\s+(?:trainings?|internships?|courses?))?',
    'summary':        r'(?:professional\s+|career\s+)?(?:summary|profile|objective|about(?:\s+me)?|introduction)',
    'achievements':   r'achievements?|awards?|accomplishments',
}

def split_sections(text):
    """Split resume text into named sections by detecting section headers."""
    sections = {'_other': []}
    current = '_other'

    lines = text.split('\n')
    for line in lines:
        ll = line.strip().lower()
        if not ll or len(ll) > 60:
            sections.setdefault(current, []).append(line)
            continue

        matched = False
        for name, pattern in SECTION_HEADERS.items():
            if re.fullmatch(rf'\s*({pattern})\s*[:.\-—–]?\s*', ll):
                current = name
                sections.setdefault(current, [])
                matched = True
                break
        if not matched:
            sections.setdefault(current, []).append(line)

    return {k: '\n'.join(v) for k, v in sections.items()}


def extract_skills_smart(text):
    """Section-aware skill extraction with confidence ranking."""
    sections = split_sections(text)

    # Skills section gets highest weight; full text gets lower weight
    skills_section_text = normalize_aliases(sections.get('skills', '').lower())
    full_text = normalize_aliases(text.lower())
    full_text = re.sub(r'[,;:|•\-–—]', ' ', full_text)
    full_text = re.sub(r'\s+', ' ', full_text)

    found = {}  # skill -> confidence

    for skill in ALL_SKILLS:
        if ' ' in skill or '.' in skill or '/' in skill or '+' in skill or '#' in skill:
            # Multi-word or special-char skill: substring match
            if skill in full_text:
                found[skill] = 70
            if skill in skills_section_text:
                found[skill] = 100
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, full_text):
                found[skill] = 70
            if re.search(pattern, skills_section_text):
                found[skill] = 100

    # Sort by confidence then alphabetical, cap at 35
    sorted_skills = sorted(found.items(), key=lambda x: (-x[1], x[0]))
    return [s[0].title() for s in sorted_skills[:35]]


def extract_experience_years(text):
    """Detect total years of experience."""
    patterns = [
        r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)',
        r'(?:experience|exp)\s*[:|\-—]?\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)',
        r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:in|at|with|of)',
    ]
    max_years = 0
    text_lower = text.lower()
    for p in patterns:
        for m in re.findall(p, text_lower):
            try:
                v = float(m if isinstance(m, str) else m[0])
                if 0 < v <= 50 and v > max_years:
                    max_years = v
            except Exception:
                pass
    return int(max_years)


def extract_certifications(text):
    """Extract certifications from the certifications section or full text."""
    sections = split_sections(text)
    cert_text = sections.get('certifications', '') or text
    cert_lower = cert_text.lower()

    patterns = [
        r'aws certified [a-z\s]+(?:specialty|associate|professional|practitioner)?',
        r'(?:microsoft\s+)?azure (?:fundamentals|administrator|developer|architect|associate)',
        r'google cloud [a-z\s]+(?:professional|associate)?',
        r'oracle certified [a-z\s]+',
        r'cisco certified [a-z\s]+',
        r'comptia [a-z]+',
        r'certified [a-z\s]+(?:engineer|developer|administrator|architect|practitioner|associate|professional)',
        r'(?:pmp|csm|cspo|safe|itil) (?:certified|certification)?',
    ]
    found = []
    seen = set()
    for p in patterns:
        for m in re.findall(p, cert_lower):
            cert = m.strip().title()
            if cert.lower() not in seen and len(cert) > 5:
                found.append(cert)
                seen.add(cert.lower())
                if len(found) >= 8:
                    return found
    return found


def count_projects(text):
    sections = split_sections(text)
    proj = sections.get('projects', '')
    if not proj:
        return 0
    # Look for bullet markers OR numbered lines OR "Project N" patterns
    bullets = re.findall(r'(?:^|\n)\s*(?:[•●◆▸★\-\*\d]+\.?|\d+\))\s+\S', proj)
    named = re.findall(r'project\s+\d+|^\s*[A-Z][\w\s]{3,40}\n', proj, flags=re.M)
    return min(max(len(bullets), len(named)), 10)


def has_summary_section(text):
    sections = split_sections(text)
    s = sections.get('summary', '').strip()
    return len(s) > 40


def extract_marks(text):
    patterns = [
        r'cgpa[:\s]*(\d+(?:\.\d+)?)',
        r'gpa[:\s]*(\d+(?:\.\d+)?)',
        r'(\d{1,2}(?:\.\d+)?)\s*cgpa',
        r'(\d{1,2}(?:\.\d+)?)\s*%',
        r'percentage[:\s]*(\d{1,2}(?:\.\d+)?)',
    ]
    marks = []
    for p in patterns:
        for m in re.findall(p, text.lower()):
            v = m if isinstance(m, str) else m[0]
            try:
                val = float(v)
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
                return round(float(m.replace('%', '').strip()) / 10, 1)
            except Exception:
                pass
    return 7.0


def extract_education(text):
    edu = []
    upper = text.upper()
    for e in EDUCATION:
        if re.search(rf'\b{e}\b', upper):
            edu.append(e)
    return list(set(edu))


# ─── Routes ─────────────────────────────────────────────────────

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
            # ─── Auto-login on signup ───
            session['user_email'] = user_email
            session['user_name'] = user_name
            flash(f'Welcome, {user_name}! Upload your resume to begin.', 'success')
            return render_template('upload.html', name=user_name, email=user_email)
        except sqlite3.IntegrityError:
            flash('Email already registered. Please sign in.', 'error')
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
        t0 = time.time()

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
            socketio.emit('progress', {'progress': 8, 'message': 'Extracting text from PDF...'})
            resMgr = PDFResourceManager()
            ret = io.StringIO()
            converter = TextConverter(resMgr, ret, laparams=LAParams())
            interp = PDFPageInterpreter(resMgr, converter)
            for page in PDFPage.get_pages(i_f):
                interp.process_page(page)
            txt = ret.getvalue()
            ret.close()
            converter.close()

            socketio.emit('progress', {'progress': 22, 'message': 'Parsing resume sections...'})
            sections = split_sections(txt)

            socketio.emit('progress', {'progress': 36, 'message': 'Extracting skills...'})
            all_skills = extract_skills_smart(txt)

            socketio.emit('progress', {'progress': 48, 'message': 'Detecting experience...'})
            exp_years = extract_experience_years(txt)

            socketio.emit('progress', {'progress': 56, 'message': 'Reading academic info...'})
            marks = extract_marks(txt)
            education = extract_education(txt)
            marks_message = "Detected: " + ", ".join(marks[:5]) if marks else "Not detected"

            socketio.emit('progress', {'progress': 66, 'message': 'Finding certifications...'})
            certifications = extract_certifications(txt)

            socketio.emit('progress', {'progress': 72, 'message': 'Counting projects...'})
            project_count = count_projects(txt)

            socketio.emit('progress', {'progress': 80, 'message': 'Predicting companies...'})
            mark_value = get_cgpa_value(marks)
            dominant = next(
                (s.lower() for s in all_skills if s.lower() in SKILL_PRIORITY),
                all_skills[0].lower() if all_skills else 'python'
            )
            skill_encoded = SKILL_ENCODING.get(dominant, 0)
            companies = predict(mark_value, skill_encoded, len(all_skills))

            socketio.emit('progress', {'progress': 86, 'message': 'Matching career roles...'})
            seen = set()
            unique_roles = []
            for skill in all_skills[:12]:
                for role in suggest_roles(skill):
                    if role not in seen:
                        unique_roles.append(role)
                        seen.add(role)
            unique_roles = unique_roles[:8]

            # TF-IDF role match scores
            role_scores = role_match_scores(all_skills, unique_roles)
            roles_with_scores = [
                {'name': r, 'score': role_scores.get(r, 50)} for r in unique_roles
            ]
            roles_with_scores.sort(key=lambda x: -x['score'])

            socketio.emit('progress', {'progress': 92, 'message': 'Analyzing skill gaps...'})
            skill_gaps = analyze_skill_gaps(all_skills, unique_roles)

            socketio.emit('progress', {'progress': 96, 'message': 'Generating insights...'})
            skill_categories = categorize_skills(all_skills)
            score, breakdown = calculate_score_breakdown(
                all_skills,
                len(education),
                exp_years,
                certifications,
                project_count,
                has_summary_section(txt),
            )
            insights = generate_insights(
                all_skills, skill_categories, score, skill_gaps,
                exp_years, len(certifications), project_count
            )

            socketio.emit('progress', {'progress': 100, 'message': 'Analysis complete!'})
            logger.info(f"Total processing time: {time.time() - t0:.2f}s — score={score} skills={len(all_skills)}")

            return render_template(
                'result.html',
                companies=companies,
                roles=roles_with_scores,
                job=unique_roles,
                marks=marks_message,
                skill_gaps=skill_gaps,
                skills=all_skills,
                skill_categories=skill_categories,
                score=score,
                breakdown=breakdown,
                insights=insights,
                experience_years=exp_years,
                certifications=certifications,
                project_count=project_count,
                name=session.get('user_name', ''),
            )

        except Exception as e:
            logger.error(f"Error processing resume: {e}")
            socketio.emit('progress', {'progress': 0, 'message': f'Error: {str(e)}'})
            flash(f'Error processing resume: {str(e)}', 'error')
            return redirect(url_for('upload'))

    return render_template('upload.html', name=session.get('user_name'), email=session.get('user_email'))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
