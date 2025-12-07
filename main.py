from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os

# Load and preprocess data once at startup
train_data = pd.read_csv("Book2.csv", encoding='latin-1')
le_Skill = LabelEncoder()
le_depart = LabelEncoder()
le_Company = LabelEncoder()

train_data['skill'] = le_Skill.fit_transform(train_data['Skills Known'])
train_data['dept'] = le_depart.fit_transform(train_data['department'])
train_data['target'] = le_Company.fit_transform(train_data['Company Placed'])

# Simulate num_skills feature in training data
np.random.seed(42)
train_data['num_skills'] = np.random.randint(1, 6, size=len(train_data))

# Features and target
X = train_data.drop([
    'Full Name', "12th Mark", "10th Mark", 'dept', 'Company Placed',
    "Skills Known", "Projects Done", 'target', 'department', "Certifications/Internships"
], axis=1)
y = train_data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use Random Forest instead of CNN (works with Python 3.14)
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Define company class mapping
class_names = {
    0: 'Birlasoft', 1: 'Cognizant', 2: 'Hexaware Technologies',
    3: 'Infosys', 4: 'KPIT Technologies', 5: 'L&T Infotech',
    6: 'Tech Mahindra', 7: 'Wipro Technologies', 8: 'CSS Corp', 9: 'TCS'
}

# Extended role mappings with more comprehensive skill coverage
ROLE_SKILLS = {
    # Python-based roles
    'python developer': ['python', 'django', 'flask', 'sql', 'rest api'],
    'machine learning engineer': ['python', 'tensorflow', 'pandas', 'machine learning', 'numpy', 'scikit-learn'],
    'data scientist': ['python', 'pandas', 'machine learning', 'sql', 'statistics', 'visualization'],
    'data engineer': ['python', 'sql', 'spark', 'hadoop', 'etl', 'data pipeline'],
    'ai engineer': ['python', 'tensorflow', 'pytorch', 'deep learning', 'nlp'],
    
    # Web Development
    'frontend developer': ['html', 'css', 'javascript', 'react', 'vue', 'angular'],
    'backend developer': ['python', 'java', 'nodejs', 'sql', 'rest api', 'microservices'],
    'fullstack developer': ['html', 'css', 'javascript', 'nodejs', 'react', 'sql', 'mongodb'],
    'web developer': ['html', 'css', 'javascript', 'php', 'mysql'],
    
    # Mobile Development
    'mobile app developer': ['javascript', 'react native', 'flutter', 'android', 'ios'],
    'android developer': ['java', 'kotlin', 'android sdk', 'xml', 'gradle'],
    'ios developer': ['swift', 'objective-c', 'xcode', 'cocoapods'],
    
    # DevOps & Cloud
    'devops engineer': ['docker', 'kubernetes', 'aws', 'ci/cd', 'jenkins', 'terraform'],
    'cloud engineer': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux'],
    'site reliability engineer': ['linux', 'python', 'kubernetes', 'monitoring', 'automation'],
    
    # Database
    'database administrator': ['sql', 'mysql', 'postgresql', 'oracle', 'database administration'],
    'database developer': ['sql', 'mysql', 'data modeling', 'etl', 'stored procedures'],
    'sql developer': ['sql', 'mysql', 'database design', 'query optimization'],
    
    # Testing & QA
    'qa engineer': ['selenium', 'testing', 'automation', 'jira', 'test cases'],
    'test automation engineer': ['selenium', 'java', 'python', 'cypress', 'testng'],
    'software tester': ['manual testing', 'test cases', 'bug tracking', 'agile'],
    
    # Java Ecosystem
    'java developer': ['java', 'spring', 'hibernate', 'maven', 'microservices'],
    'java software architect': ['java', 'microservices', 'design patterns', 'spring boot'],
    
    # Security
    'security engineer': ['security', 'penetration testing', 'networking', 'linux', 'cryptography'],
    'cybersecurity analyst': ['security', 'siem', 'incident response', 'networking'],
    
    # Other Roles
    'game developer': ['c++', 'unity', 'unreal engine', 'c#', 'game design'],
    'bi developer': ['sql', 'power bi', 'tableau', 'data visualization', 'etl'],
    'php developer': ['php', 'laravel', 'mysql', 'javascript', 'html'],
    'ruby developer': ['ruby', 'rails', 'sql', 'javascript'],
    'system administrator': ['linux', 'networking', 'security', 'bash', 'windows server'],
    'technical support engineer': ['troubleshooting', 'networking', 'customer support', 'windows', 'linux'],
    'ui/ux designer': ['figma', 'ui design', 'user research', 'prototyping', 'adobe xd'],
    'digital marketing specialist': ['seo', 'google analytics', 'content creation', 'social media'],
    'content manager': ['content strategy', 'seo', 'writing', 'cms'],
    'software engineer': ['python', 'java', 'sql', 'data structures', 'algorithms'],
    'blockchain developer': ['solidity', 'ethereum', 'web3', 'smart contracts'],
}

# Extended skill to role mapping
SKILL_TO_ROLES = {
    # Programming Languages
    'python': ['Python Developer', 'Machine Learning Engineer', 'Data Scientist', 'Backend Developer', 'Data Engineer'],
    'java': ['Java Developer', 'Android Developer', 'Backend Developer', 'Software Engineer', 'Java Software Architect'],
    'javascript': ['Frontend Developer', 'Fullstack Developer', 'Web Developer', 'Mobile App Developer', 'Node.js Developer'],
    'typescript': ['Frontend Developer', 'Fullstack Developer', 'Angular Developer', 'React Developer'],
    'c++': ['Game Developer', 'Systems Programmer', 'Embedded Developer', 'Software Engineer'],
    'c#': ['Game Developer', 'Unity Developer', '.NET Developer', 'Backend Developer'],
    'php': ['PHP Developer', 'Web Developer', 'Backend Developer', 'WordPress Developer'],
    'ruby': ['Ruby Developer', 'Rails Developer', 'Backend Developer'],
    'go': ['Backend Developer', 'DevOps Engineer', 'Cloud Engineer', 'Systems Programmer'],
    'rust': ['Systems Programmer', 'Blockchain Developer', 'Backend Developer'],
    'swift': ['iOS Developer', 'Mobile App Developer'],
    'kotlin': ['Android Developer', 'Mobile App Developer', 'Backend Developer'],
    
    # Web Technologies
    'html': ['Frontend Developer', 'Web Developer', 'UI Developer'],
    'css': ['Frontend Developer', 'Web Developer', 'UI Developer', 'UI/UX Designer'],
    'react': ['React Developer', 'Frontend Developer', 'Fullstack Developer'],
    'angular': ['Angular Developer', 'Frontend Developer', 'Fullstack Developer'],
    'vue': ['Vue Developer', 'Frontend Developer', 'Fullstack Developer'],
    'nodejs': ['Node.js Developer', 'Backend Developer', 'Fullstack Developer'],
    
    # Data & ML
    'sql': ['Database Developer', 'Data Analyst', 'Backend Developer', 'BI Developer', 'Data Engineer'],
    'mysql': ['Database Administrator', 'Backend Developer', 'Web Developer'],
    'mongodb': ['Backend Developer', 'Fullstack Developer', 'NoSQL Developer'],
    'pandas': ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer'],
    'tensorflow': ['Machine Learning Engineer', 'AI Engineer', 'Deep Learning Engineer'],
    'machine learning': ['Machine Learning Engineer', 'Data Scientist', 'AI Engineer', 'Research Scientist'],
    'deep learning': ['Deep Learning Engineer', 'AI Engineer', 'Computer Vision Engineer'],
    
    # Cloud & DevOps
    'aws': ['Cloud Engineer', 'DevOps Engineer', 'Solutions Architect', 'AWS Developer'],
    'azure': ['Cloud Engineer', 'Azure Developer', 'DevOps Engineer'],
    'gcp': ['Cloud Engineer', 'GCP Developer', 'DevOps Engineer'],
    'docker': ['DevOps Engineer', 'Cloud Engineer', 'Backend Developer', 'SRE'],
    'kubernetes': ['DevOps Engineer', 'Cloud Engineer', 'SRE', 'Platform Engineer'],
    
    # Other
    'linux': ['DevOps Engineer', 'System Administrator', 'Cloud Engineer', 'SRE'],
    'git': ['Software Engineer', 'Developer', 'DevOps Engineer'],
    'selenium': ['QA Engineer', 'Test Automation Engineer', 'SDET'],
    'figma': ['UI/UX Designer', 'Product Designer', 'Frontend Developer'],
}

def suggest_roles(skill):
    """
    Enhanced role suggestion based on skills.
    Returns comma-separated list of relevant job roles.
    """
    skill_lower = skill.strip().lower()
    
    # Check direct mapping first
    if skill_lower in SKILL_TO_ROLES:
        roles = SKILL_TO_ROLES[skill_lower]
        return ', '.join(roles[:3])  # Return top 3 roles
    
    # Fallback: check partial matches
    for key in SKILL_TO_ROLES:
        if key in skill_lower or skill_lower in key:
            roles = SKILL_TO_ROLES[key]
            return ', '.join(roles[:3])
    
    return 'Unknown Role'

def predict(mark, skill, num_skills):
    """
    Predict top 3 company matches based on resume features.
    """
    try:
        # Prepare input for prediction
        input_data = np.array([[mark, 0, skill, num_skills]])
        input_scaled = scaler.transform(input_data)

        # Predict probabilities
        predicted_probs = model.predict_proba(input_scaled)
        top_indices = np.argsort(predicted_probs[0])[::-1][:3]
        
        # Map to company names
        top_companies = []
        for idx in top_indices:
            if idx in class_names:
                top_companies.append(class_names[idx])
            else:
                top_companies.append(class_names[idx % 10])
        
        return top_companies
    except Exception as e:
        # Return default companies on error
        return ['TCS', 'Infosys', 'Wipro Technologies']

def analyze_skill_gaps(extracted_skills, suggested_roles):
    """
    Analyze what skills are missing for suggested job roles.
    """
    skill_gaps = {}
    extracted_skills_lower = [skill.lower() for skill in extracted_skills]

    for role_string in suggested_roles:
        # Handle comma-separated roles
        roles = [r.strip().lower() for r in role_string.split(',')]
        
        for role in roles:
            if role in ROLE_SKILLS:
                required_skills = ROLE_SKILLS[role]
                missing_skills = []
                
                for req_skill in required_skills:
                    # Check if any extracted skill matches
                    found = False
                    for ext_skill in extracted_skills_lower:
                        if req_skill in ext_skill or ext_skill in req_skill:
                            found = True
                            break
                    if not found:
                        missing_skills.append(req_skill.title())
                
                if missing_skills:
                    skill_gaps[role.title()] = missing_skills[:4]  # Limit to 4 gaps
                break  # Only process first matching role

    return skill_gaps

def calculate_resume_score(skills, education_count=1, has_certifications=False, has_projects=True):
    """
    Calculate a resume score from 0-100 based on various factors.
    """
    score = 50  # Base score
    
    # Skills contribution (up to 30 points)
    skill_count = len(skills) if skills else 0
    score += min(skill_count * 3, 30)
    
    # Education contribution (up to 10 points)
    score += min(education_count * 5, 10)
    
    # Certifications (up to 5 points)
    if has_certifications:
        score += 5
    
    # Projects (up to 5 points)
    if has_projects:
        score += 5
    
    return min(score, 100)