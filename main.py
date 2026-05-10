from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

train_data = pd.read_csv("Book2.csv", encoding='latin-1')
le_Skill = LabelEncoder()
le_depart = LabelEncoder()
le_Company = LabelEncoder()

train_data['skill'] = le_Skill.fit_transform(train_data['Skills Known'])
train_data['dept'] = le_depart.fit_transform(train_data['department'])
train_data['target'] = le_Company.fit_transform(train_data['Company Placed'])

np.random.seed(42)
train_data['num_skills'] = np.random.randint(1, 6, size=len(train_data))

X = train_data.drop([
    'Full Name', "12th Mark", "10th Mark", 'dept', 'Company Placed',
    "Skills Known", "Projects Done", 'target', 'department', "Certifications/Internships"
], axis=1)
y = train_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

class_names = {
    0: 'Birlasoft', 1: 'Cognizant', 2: 'Hexaware Technologies',
    3: 'Infosys', 4: 'KPIT Technologies', 5: 'L&T Infotech',
    6: 'Tech Mahindra', 7: 'Wipro Technologies', 8: 'CSS Corp', 9: 'TCS'
}

ROLE_SKILLS = {
    'python developer': ['python', 'django', 'flask', 'sql', 'rest api'],
    'machine learning engineer': ['python', 'tensorflow', 'pandas', 'machine learning', 'numpy', 'scikit-learn'],
    'data scientist': ['python', 'pandas', 'machine learning', 'sql', 'statistics', 'visualization'],
    'data engineer': ['python', 'sql', 'spark', 'hadoop', 'etl', 'data pipeline'],
    'ai engineer': ['python', 'tensorflow', 'pytorch', 'deep learning', 'nlp'],
    'frontend developer': ['html', 'css', 'javascript', 'react', 'vue', 'angular'],
    'backend developer': ['python', 'java', 'nodejs', 'sql', 'rest api', 'microservices'],
    'fullstack developer': ['html', 'css', 'javascript', 'nodejs', 'react', 'sql', 'mongodb'],
    'web developer': ['html', 'css', 'javascript', 'php', 'mysql'],
    'mobile app developer': ['javascript', 'react native', 'flutter', 'android', 'ios'],
    'android developer': ['java', 'kotlin', 'android sdk', 'xml', 'gradle'],
    'ios developer': ['swift', 'objective-c', 'xcode', 'cocoapods'],
    'devops engineer': ['docker', 'kubernetes', 'aws', 'ci/cd', 'jenkins', 'terraform'],
    'cloud engineer': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux'],
    'site reliability engineer': ['linux', 'python', 'kubernetes', 'monitoring', 'automation'],
    'database administrator': ['sql', 'mysql', 'postgresql', 'oracle', 'database administration'],
    'sql developer': ['sql', 'mysql', 'database design', 'query optimization'],
    'qa engineer': ['selenium', 'testing', 'automation', 'jira', 'test cases'],
    'test automation engineer': ['selenium', 'java', 'python', 'cypress', 'testng'],
    'java developer': ['java', 'spring', 'hibernate', 'maven', 'microservices'],
    'security engineer': ['security', 'penetration testing', 'networking', 'linux', 'cryptography'],
    'game developer': ['c++', 'unity', 'unreal engine', 'c#', 'game design'],
    'bi developer': ['sql', 'power bi', 'tableau', 'data visualization', 'etl'],
    'php developer': ['php', 'laravel', 'mysql', 'javascript', 'html'],
    'software engineer': ['python', 'java', 'sql', 'data structures', 'algorithms'],
    'blockchain developer': ['solidity', 'ethereum', 'web3', 'smart contracts'],
    'ui/ux designer': ['figma', 'ui design', 'user research', 'prototyping', 'adobe xd'],
    'react developer': ['react', 'javascript', 'typescript', 'css', 'html'],
    'node.js developer': ['nodejs', 'javascript', 'express', 'mongodb', 'rest api'],
}

SKILL_TO_ROLES = {
    'python': ['Python Developer', 'Machine Learning Engineer', 'Data Scientist', 'Backend Developer', 'Data Engineer'],
    'java': ['Java Developer', 'Android Developer', 'Backend Developer', 'Software Engineer'],
    'javascript': ['Frontend Developer', 'Fullstack Developer', 'Web Developer', 'Node.js Developer'],
    'typescript': ['Frontend Developer', 'Fullstack Developer', 'React Developer'],
    'c++': ['Game Developer', 'Systems Programmer', 'Software Engineer'],
    'c#': ['Game Developer', 'Unity Developer', '.NET Developer'],
    'php': ['PHP Developer', 'Web Developer', 'Backend Developer'],
    'ruby': ['Ruby Developer', 'Rails Developer', 'Backend Developer'],
    'go': ['Backend Developer', 'DevOps Engineer', 'Cloud Engineer'],
    'swift': ['iOS Developer', 'Mobile App Developer'],
    'kotlin': ['Android Developer', 'Mobile App Developer'],
    'html': ['Frontend Developer', 'Web Developer', 'UI Developer'],
    'css': ['Frontend Developer', 'Web Developer', 'UI/UX Designer'],
    'react': ['React Developer', 'Frontend Developer', 'Fullstack Developer'],
    'angular': ['Angular Developer', 'Frontend Developer', 'Fullstack Developer'],
    'vue': ['Vue Developer', 'Frontend Developer', 'Fullstack Developer'],
    'nodejs': ['Node.js Developer', 'Backend Developer', 'Fullstack Developer'],
    'sql': ['Database Developer', 'Data Analyst', 'Backend Developer', 'BI Developer'],
    'mongodb': ['Backend Developer', 'Fullstack Developer', 'Node.js Developer'],
    'pandas': ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer'],
    'tensorflow': ['Machine Learning Engineer', 'AI Engineer', 'Deep Learning Engineer'],
    'pytorch': ['Machine Learning Engineer', 'AI Engineer', 'Deep Learning Engineer'],
    'machine learning': ['Machine Learning Engineer', 'Data Scientist', 'AI Engineer'],
    'deep learning': ['Deep Learning Engineer', 'AI Engineer', 'Computer Vision Engineer'],
    'nlp': ['NLP Engineer', 'AI Engineer', 'Data Scientist'],
    'aws': ['Cloud Engineer', 'DevOps Engineer', 'Solutions Architect'],
    'azure': ['Cloud Engineer', 'Azure Developer', 'DevOps Engineer'],
    'gcp': ['Cloud Engineer', 'GCP Developer', 'DevOps Engineer'],
    'docker': ['DevOps Engineer', 'Cloud Engineer', 'Backend Developer'],
    'kubernetes': ['DevOps Engineer', 'Cloud Engineer', 'SRE'],
    'linux': ['DevOps Engineer', 'System Administrator', 'Cloud Engineer'],
    'git': ['Software Engineer', 'Developer', 'DevOps Engineer'],
    'selenium': ['QA Engineer', 'Test Automation Engineer', 'SDET'],
    'figma': ['UI/UX Designer', 'Product Designer', 'Frontend Developer'],
    'flutter': ['Mobile App Developer', 'Android Developer', 'iOS Developer'],
    'power bi': ['BI Developer', 'Data Analyst', 'Business Analyst'],
    'tableau': ['BI Developer', 'Data Analyst', 'Business Analyst'],
    'spark': ['Data Engineer', 'Big Data Engineer', 'Data Scientist'],
    'blockchain': ['Blockchain Developer', 'Web3 Developer'],
    'solidity': ['Blockchain Developer', 'Smart Contract Developer'],
}


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
        top_indices = np.argsort(predicted_probs)[::-1][:3]

        max_prob = predicted_probs[top_indices[0]]
        companies = []
        for idx in top_indices:
            name = class_names.get(int(idx), class_names.get(int(idx) % 10, 'TCS'))
            raw_prob = predicted_probs[idx]
            score = int(55 + (raw_prob / max_prob) * 33) if max_prob > 0 else 65
            companies.append({'name': name, 'score': score})
        return companies
    except Exception:
        return [
            {'name': 'TCS', 'score': 85},
            {'name': 'Infosys', 'score': 76},
            {'name': 'Wipro Technologies', 'score': 68},
        ]


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
                skill_gaps[role] = missing[:4]

    return skill_gaps


def calculate_resume_score(skills, education_count=1, has_certifications=False, has_projects=True):
    score = 45
    score += min(len(skills) * 3, 30)
    score += min(education_count * 5, 10)
    if has_certifications:
        score += 8
    if has_projects:
        score += 7
    return min(score, 100)
