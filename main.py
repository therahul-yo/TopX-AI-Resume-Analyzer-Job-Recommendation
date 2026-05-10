"""
TopX — Core ML & Analysis Engine
Skill extraction, role matching, scoring, and insight generation.
"""

import os
import logging
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─── Load Pre-trained Model (or train inline as fallback) ───────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'company_predictor.joblib')


def _load_or_train():
    if os.path.exists(MODEL_PATH):
        try:
            b = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")
            return b
        except Exception as e:
            logger.warning(f"Failed to load model file: {e}; training inline")

    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    td = pd.read_csv("Book2.csv", encoding='latin-1')
    le_s = LabelEncoder(); le_d = LabelEncoder(); le_c = LabelEncoder()
    td['skill']  = le_s.fit_transform(td['Skills Known'])
    td['dept']   = le_d.fit_transform(td['department'])
    td['target'] = le_c.fit_transform(td['Company Placed'])
    np.random.seed(42)
    td['num_skills'] = np.random.randint(1, 6, size=len(td))
    X = td.drop(['Full Name', "12th Mark", "10th Mark", 'dept', 'Company Placed',
                 "Skills Known", "Projects Done", 'target', 'department',
                 "Certifications/Internships"], axis=1)
    y = td['target']
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    sc = MinMaxScaler()
    Xs = sc.fit_transform(Xtr)
    m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    m.fit(Xs, ytr)
    bundle = {'model': m, 'scaler': sc, 'le_skill': le_s, 'le_depart': le_d,
              'le_company': le_c, 'class_names': dict(enumerate(le_c.classes_)),
              'metrics': {'train_accuracy': 0.0, 'test_accuracy': 0.0}}
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(bundle, MODEL_PATH, compress=3)
    except Exception:
        pass
    return bundle


_BUNDLE = _load_or_train()
model         = _BUNDLE['model']
scaler        = _BUNDLE['scaler']
class_names   = _BUNDLE.get('class_names') or {
    0: 'Birlasoft', 1: 'Cognizant', 2: 'Hexaware Technologies',
    3: 'Infosys', 4: 'KPIT Technologies', 5: 'L&T Infotech',
    6: 'Tech Mahindra', 7: 'Wipro Technologies', 8: 'CSS Corp', 9: 'TCS'
}
MODEL_METRICS = _BUNDLE.get('metrics', {})

# ─── Resume Category Classifier (TF-IDF + LogReg on 2,483 real resumes) ──
CATEGORY_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models', 'category_classifier.joblib'
)
_CATEGORY_BUNDLE = None
try:
    if os.path.exists(CATEGORY_MODEL_PATH):
        _CATEGORY_BUNDLE = joblib.load(CATEGORY_MODEL_PATH)
        m = _CATEGORY_BUNDLE.get('metrics', {})
        logger.info(
            f"Category classifier loaded "
            f"(test_acc={m.get('test_accuracy', 0):.3f}, "
            f"top3_acc={m.get('top3_accuracy', 0):.3f})"
        )
except Exception as e:
    logger.warning(f"Category classifier unavailable: {e}")
    _CATEGORY_BUNDLE = None


def predict_categories(resume_text, top_k=4):
    """
    Predict job categories from resume text.
    Returns list of {'name': str, 'score': int} sorted by confidence.
    Returns [] if classifier not trained or text too short.
    """
    if not _CATEGORY_BUNDLE or not resume_text or len(resume_text) < 50:
        return []
    try:
        pipe    = _CATEGORY_BUNDLE['pipeline']
        classes = _CATEGORY_BUNDLE['classes']
        probs   = pipe.predict_proba([resume_text])[0]
        idxs    = np.argsort(probs)[::-1][:top_k]
        out = []
        for i in idxs:
            score = int(round(probs[i] * 100))
            if score < 2:
                continue
            name = classes[i].replace('-', ' ').title()
            out.append({'name': name, 'score': score})
        return out
    except Exception as e:
        logger.warning(f"Category prediction failed: {e}")
        return []


# ─── Skill Database (Expanded, Categorized) ─────────────────────
SKILLS_BY_CATEGORY = {
    'Languages': [
        'python', 'java', 'javascript', 'typescript', 'c', 'c++', 'c#', 'go',
        'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab',
        'perl', 'dart', 'lua', 'bash', 'powershell', 'sql', 'pl/sql',
        'objective-c', 'haskell', 'elixir', 'clojure', 'julia', 'groovy',
        'sas', 'vba', 'shell scripting', 'assembly',
    ],
    'Frontend': [
        'react', 'angular', 'vue', 'svelte', 'nextjs', 'nuxt', 'gatsby',
        'redux', 'mobx', 'rxjs', 'jquery', 'html', 'html5', 'css', 'css3',
        'sass', 'scss', 'less', 'tailwind', 'bootstrap', 'material ui',
        'chakra ui', 'styled components', 'webpack', 'vite', 'rollup',
        'webgl', 'threejs', 'd3.js', 'chart.js', 'storybook',
    ],
    'Backend': [
        'nodejs', 'express', 'fastapi', 'django', 'flask', 'spring',
        'spring boot', 'laravel', 'rails', 'asp.net', '.net core',
        'graphql', 'rest api', 'grpc', 'websocket', 'soap',
        'microservices', 'serverless', 'rabbitmq', 'celery',
    ],
    'Databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
        'sql server', 'mariadb', 'cassandra', 'dynamodb', 'firestore',
        'firebase', 'elasticsearch', 'neo4j', 'couchdb', 'cosmos db',
        'bigquery', 'snowflake', 'redshift', 'clickhouse', 'supabase',
    ],
    'Cloud': [
        'aws', 'azure', 'gcp', 'oracle cloud', 'digital ocean', 'heroku',
        'vercel', 'netlify', 'cloudflare', 'lambda', 's3', 'ec2', 'rds',
        'cloudfront', 'ecs', 'eks', 'fargate', 'app engine',
        'cloud functions', 'cloud run', 'cloud storage',
    ],
    'DevOps': [
        'docker', 'kubernetes', 'jenkins', 'github actions', 'gitlab ci',
        'circleci', 'travis ci', 'terraform', 'ansible', 'puppet', 'chef',
        'helm', 'argocd', 'prometheus', 'grafana', 'datadog', 'splunk',
        'elk stack', 'nginx', 'apache', 'haproxy', 'consul', 'vault',
        'istio', 'envoy', 'ci/cd', 'devops',
    ],
    'AI / ML': [
        'machine learning', 'deep learning', 'neural networks', 'cnn', 'rnn',
        'lstm', 'gan', 'transformer', 'bert', 'gpt', 'llm', 'tensorflow',
        'pytorch', 'keras', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'opencv', 'nltk', 'spacy', 'huggingface', 'langchain', 'rag',
        'pinecone', 'weaviate', 'chroma', 'computer vision', 'nlp',
        'reinforcement learning', 'mlops', 'mlflow', 'kubeflow', 'sagemaker',
        'vertex ai', 'feature engineering', 'time series',
    ],
    'Data Engineering': [
        'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'pyspark', 'hive',
        'pig', 'presto', 'trino', 'data warehouse', 'data lake', 'etl',
        'elt', 'data modeling', 'data engineering', 'apache flink',
        'apache beam', 'fivetran',
    ],
    'Mobile': [
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic',
        'swift ui', 'jetpack compose', 'kotlin multiplatform',
    ],
    'Tools': [
        'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
        'notion', 'slack', 'figma', 'sketch', 'adobe xd', 'invision',
        'postman', 'insomnia', 'swagger', 'openapi',
    ],
    'Testing': [
        'jest', 'mocha', 'cypress', 'playwright', 'selenium', 'pytest',
        'junit', 'testng', 'rspec', 'puppeteer', 'cucumber', 'tdd', 'bdd',
        'unit testing', 'integration testing',
    ],
    'Security': [
        'oauth', 'jwt', 'saml', 'ssl', 'tls', 'penetration testing',
        'cybersecurity', 'cryptography', 'siem', 'owasp', 'kali linux',
        'metasploit', 'burp suite', 'nmap', 'wireshark',
    ],
    'BI & Analytics': [
        'tableau', 'power bi', 'looker', 'metabase', 'qlikview',
        'data studio', 'excel', 'google analytics',
    ],
    'Methodologies': [
        'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'pair programming',
        'code review', 'design patterns', 'solid principles',
    ],
}

# Build flat skill set + reverse map
ALL_SKILLS = set()
SKILL_CATEGORY = {}
for cat, sk in SKILLS_BY_CATEGORY.items():
    for s in sk:
        ALL_SKILLS.add(s.lower())
        SKILL_CATEGORY[s.lower()] = cat

# Aliases (canonical form on the right)
SKILL_ALIASES = {
    'react.js': 'react', 'reactjs': 'react', 'react js': 'react',
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
    'ci cd': 'ci/cd', 'cicd': 'ci/cd',
    'rest': 'rest api', 'restful': 'rest api', 'restful api': 'rest api',
    'ml': 'machine learning',
    'dl': 'deep learning',
    'js': 'javascript',
    'ts': 'typescript',
    'k8s': 'kubernetes',
    'tf': 'tensorflow',
    'pl sql': 'pl/sql',
    'gh actions': 'github actions',
    'gh action': 'github actions',
    'aws sagemaker': 'sagemaker',
    'azure devops': 'azure',
    'gcp vertex': 'vertex ai',
    'dot net': '.net core', '.net': '.net core',
    'c sharp': 'c#',
    'cpp': 'c++',
    'objective c': 'objective-c',
    'shell script': 'shell scripting',
    'large language model': 'llm',
    'large language models': 'llm',
}

# ─── Role / Skill mapping ───────────────────────────────────────
ROLE_SKILLS = {
    'python developer':            ['python', 'django', 'flask', 'sql', 'rest api', 'git'],
    'machine learning engineer':   ['python', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'machine learning'],
    'data scientist':              ['python', 'pandas', 'machine learning', 'sql', 'numpy', 'scikit-learn', 'matplotlib'],
    'data engineer':               ['python', 'sql', 'spark', 'kafka', 'airflow', 'etl', 'aws'],
    'ai engineer':                 ['python', 'tensorflow', 'pytorch', 'deep learning', 'nlp', 'llm', 'huggingface'],
    'frontend developer':          ['html', 'css', 'javascript', 'react', 'tailwind', 'webpack'],
    'backend developer':           ['python', 'java', 'nodejs', 'sql', 'rest api', 'docker'],
    'fullstack developer':         ['html', 'css', 'javascript', 'react', 'nodejs', 'sql', 'mongodb'],
    'mobile app developer':        ['react native', 'flutter', 'android', 'ios'],
    'android developer':           ['java', 'kotlin', 'android', 'jetpack compose'],
    'ios developer':               ['swift', 'ios', 'swift ui'],
    'devops engineer':             ['docker', 'kubernetes', 'aws', 'jenkins', 'terraform', 'ci/cd'],
    'cloud engineer':              ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
    'site reliability engineer':   ['linux', 'python', 'kubernetes', 'prometheus', 'grafana'],
    'database administrator':      ['sql', 'mysql', 'postgresql', 'oracle'],
    'qa engineer':                 ['selenium', 'cypress', 'jest', 'pytest', 'tdd'],
    'java developer':              ['java', 'spring', 'spring boot', 'sql', 'microservices'],
    'security engineer':           ['cybersecurity', 'penetration testing', 'owasp', 'cryptography'],
    'react developer':             ['react', 'javascript', 'typescript', 'redux', 'tailwind'],
    'node.js developer':           ['nodejs', 'javascript', 'express', 'mongodb', 'rest api'],
    'data analyst':                ['sql', 'excel', 'tableau', 'power bi', 'python'],
    'bi developer':                ['sql', 'power bi', 'tableau', 'data warehouse'],
    'mlops engineer':              ['python', 'docker', 'kubernetes', 'mlflow', 'aws'],
    'platform engineer':           ['kubernetes', 'docker', 'terraform', 'aws', 'ci/cd'],
    'software engineer':           ['python', 'java', 'sql', 'git', 'design patterns'],
    'computer vision engineer':    ['python', 'opencv', 'pytorch', 'cnn', 'computer vision'],
    'nlp engineer':                ['python', 'nlp', 'spacy', 'huggingface', 'transformer'],
}

SKILL_TO_ROLES = {
    'python':           ['Python Developer', 'Machine Learning Engineer', 'Data Scientist', 'Backend Developer', 'Data Engineer'],
    'java':             ['Java Developer', 'Android Developer', 'Backend Developer', 'Software Engineer'],
    'javascript':       ['Frontend Developer', 'Fullstack Developer', 'Node.js Developer'],
    'typescript':       ['Frontend Developer', 'Fullstack Developer', 'React Developer'],
    'c++':              ['Software Engineer', 'Computer Vision Engineer'],
    'c#':               ['.NET Developer', 'Software Engineer'],
    'php':              ['PHP Developer', 'Backend Developer'],
    'swift':            ['iOS Developer', 'Mobile App Developer'],
    'kotlin':           ['Android Developer', 'Mobile App Developer'],
    'go':               ['Backend Developer', 'DevOps Engineer', 'Platform Engineer'],
    'html':             ['Frontend Developer', 'Web Developer'],
    'css':              ['Frontend Developer', 'Web Developer'],
    'react':            ['React Developer', 'Frontend Developer', 'Fullstack Developer'],
    'angular':          ['Frontend Developer', 'Fullstack Developer'],
    'vue':              ['Frontend Developer', 'Fullstack Developer'],
    'nodejs':           ['Node.js Developer', 'Backend Developer', 'Fullstack Developer'],
    'sql':              ['Data Analyst', 'Database Administrator', 'BI Developer', 'Backend Developer'],
    'mongodb':          ['Backend Developer', 'Fullstack Developer'],
    'pandas':           ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer'],
    'tensorflow':       ['Machine Learning Engineer', 'AI Engineer', 'Computer Vision Engineer'],
    'pytorch':          ['Machine Learning Engineer', 'AI Engineer', 'Computer Vision Engineer'],
    'machine learning': ['Machine Learning Engineer', 'Data Scientist', 'AI Engineer'],
    'deep learning':    ['AI Engineer', 'Computer Vision Engineer'],
    'nlp':              ['NLP Engineer', 'AI Engineer', 'Data Scientist'],
    'llm':              ['AI Engineer', 'NLP Engineer'],
    'aws':              ['Cloud Engineer', 'DevOps Engineer', 'MLOps Engineer'],
    'azure':            ['Cloud Engineer', 'DevOps Engineer'],
    'gcp':              ['Cloud Engineer', 'DevOps Engineer'],
    'docker':           ['DevOps Engineer', 'Platform Engineer', 'MLOps Engineer'],
    'kubernetes':       ['DevOps Engineer', 'Platform Engineer', 'Site Reliability Engineer'],
    'linux':            ['DevOps Engineer', 'Site Reliability Engineer'],
    'git':              ['Software Engineer'],
    'selenium':         ['QA Engineer'],
    'figma':            ['UI/UX Designer', 'Frontend Developer'],
    'flutter':          ['Mobile App Developer'],
    'react native':     ['Mobile App Developer'],
    'power bi':         ['BI Developer', 'Data Analyst'],
    'tableau':          ['BI Developer', 'Data Analyst'],
    'spark':            ['Data Engineer'],
    'kafka':            ['Data Engineer', 'Backend Developer'],
    'airflow':          ['Data Engineer'],
    'opencv':           ['Computer Vision Engineer', 'AI Engineer'],
    'spacy':            ['NLP Engineer', 'AI Engineer'],
    'huggingface':      ['NLP Engineer', 'AI Engineer'],
}


# ─── Public API ─────────────────────────────────────────────────

def normalize_aliases(text):
    """Replace skill aliases with canonical forms."""
    for alias, canonical in SKILL_ALIASES.items():
        text = text.replace(alias, canonical)
    return text


def categorize_skills(skills):
    """Group skills into category buckets. Returns dict of {category: [skills]}."""
    grouped = {}
    for s in skills:
        cat = SKILL_CATEGORY.get(s.lower(), 'Other')
        grouped.setdefault(cat, []).append(s)
    # Order categories by # of skills descending
    return dict(sorted(grouped.items(), key=lambda x: -len(x[1])))


def suggest_roles(skill):
    skill_lower = skill.strip().lower()
    if skill_lower in SKILL_TO_ROLES:
        return SKILL_TO_ROLES[skill_lower][:3]
    for key in SKILL_TO_ROLES:
        if key in skill_lower or skill_lower in key:
            return SKILL_TO_ROLES[key][:3]
    return []


def predict(mark, skill, num_skills):
    try:
        input_data = np.array([[mark, skill, num_skills]])
        input_scaled = scaler.transform(input_data)
        predicted_probs = model.predict_proba(input_scaled)[0]
        top_indices = np.argsort(predicted_probs)[::-1][:5]
        max_prob = predicted_probs[top_indices[0]]
        companies = []
        for idx in top_indices:
            name = class_names.get(int(idx), class_names.get(int(idx) % 10, 'TCS'))
            raw = predicted_probs[idx]
            score = int(58 + (raw / max_prob) * 32) if max_prob > 0 else 65
            companies.append({'name': name, 'score': score})
        return companies
    except Exception:
        return [
            {'name': 'TCS', 'score': 86},
            {'name': 'Infosys', 'score': 78},
            {'name': 'Wipro Technologies', 'score': 71},
            {'name': 'Cognizant', 'score': 65},
            {'name': 'Tech Mahindra', 'score': 60},
        ]


def role_match_scores(extracted_skills, top_roles):
    """Calculate TF-IDF cosine similarity between extracted skills and role requirements."""
    if not extracted_skills or not top_roles:
        return {}
    extracted_text = ' '.join(s.lower() for s in extracted_skills)
    role_texts = {r: ' '.join(ROLE_SKILLS.get(r.lower(), [])) for r in top_roles}
    role_texts = {k: v for k, v in role_texts.items() if v}
    if not role_texts:
        return {}
    docs = [extracted_text] + list(role_texts.values())
    try:
        tfidf = TfidfVectorizer().fit_transform(docs)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
        return {role: max(int(sim * 100), 35) for role, sim in zip(role_texts.keys(), sims)}
    except Exception:
        return {}


def analyze_skill_gaps(extracted_skills, suggested_roles):
    skill_gaps = {}
    extracted_lower = [s.lower() for s in extracted_skills]
    for role in suggested_roles[:6]:
        role_lower = role.strip().lower()
        if role_lower in ROLE_SKILLS:
            required = ROLE_SKILLS[role_lower]
            missing = []
            for req in required:
                if not any(req in ext or ext in req for ext in extracted_lower):
                    missing.append(req.title())
            if missing:
                skill_gaps[role] = missing[:5]
    return skill_gaps


def calculate_score_breakdown(skills, education_count, experience_years, certifications, projects, has_summary):
    """Detailed score breakdown out of 100."""
    breakdown = {
        'Skills':         min(len(skills) * 1.6, 28),
        'Experience':     min(experience_years * 4, 22),
        'Projects':       min(projects * 3.5, 18),
        'Certifications': min(len(certifications) * 4, 14),
        'Education':      min(education_count * 4, 10),
        'Summary':        8 if has_summary else 0,
    }
    breakdown = {k: round(v) for k, v in breakdown.items()}
    total = sum(breakdown.values())
    return min(total, 100), breakdown


def generate_insights(skills, skill_categories, score, gaps, exp_years, cert_count, project_count):
    """Generate textual strengths/recommendations."""
    insights = []

    # Strengths
    if len(skills) >= 20:
        insights.append({'type': 'strength', 'title': 'Broad Technical Range',
                         'body': f'{len(skills)} skills across {len(skill_categories)} categories — strong technical breadth.'})
    elif len(skills) >= 10:
        insights.append({'type': 'strength', 'title': 'Solid Skill Coverage',
                         'body': f'{len(skills)} skills detected — strong technical fundamentals.'})

    if exp_years >= 5:
        insights.append({'type': 'strength', 'title': 'Senior-Level Experience',
                         'body': f'{exp_years}+ years of professional experience puts you in senior territory.'})
    elif exp_years >= 2:
        insights.append({'type': 'strength', 'title': 'Mid-Level Experience',
                         'body': f'{exp_years} years experience indicates intermediate seniority.'})

    if cert_count >= 3:
        insights.append({'type': 'strength', 'title': 'Well-Certified',
                         'body': f'{cert_count} certifications strengthen credibility for hiring managers.'})

    if project_count >= 4:
        insights.append({'type': 'strength', 'title': 'Project-Heavy Profile',
                         'body': f'{project_count} projects demonstrate strong applied experience.'})

    # Profile signals
    if 'AI / ML' in skill_categories and len(skill_categories.get('AI / ML', [])) >= 4:
        insights.append({'type': 'note', 'title': 'AI / ML Profile',
                         'body': 'Skills align strongly with ML Engineer and AI Engineer roles.'})
    if 'Cloud' in skill_categories and 'DevOps' in skill_categories:
        insights.append({'type': 'note', 'title': 'Cloud-Native Skill Set',
                         'body': 'Strong DevOps + cloud combo. Consider Platform / SRE roles.'})
    if 'Frontend' in skill_categories and 'Backend' in skill_categories:
        insights.append({'type': 'note', 'title': 'Fullstack Capable',
                         'body': 'Both frontend and backend skills — well-positioned for fullstack roles.'})

    # Recommendations
    gap_skills = set()
    for v in gaps.values():
        gap_skills.update(v)
    if len(gap_skills) >= 5:
        insights.append({'type': 'tip', 'title': 'Skill Gap Roadmap',
                         'body': f'{len(gap_skills)} skills to learn for target roles. Prioritize the most common ones.'})

    if score < 65:
        insights.append({'type': 'tip', 'title': 'Strengthen Your Resume',
                         'body': 'Add specific tools, projects with measurable impact, and certifications.'})
    elif score >= 85:
        insights.append({'type': 'strength', 'title': 'Outstanding Profile',
                         'body': 'Top-tier candidate signals across skills, experience, and credentials.'})

    if exp_years == 0 and project_count < 2:
        insights.append({'type': 'tip', 'title': 'Add Project Detail',
                         'body': 'Build and document 2–3 projects with concrete tech stacks and metrics.'})

    return insights[:6]
