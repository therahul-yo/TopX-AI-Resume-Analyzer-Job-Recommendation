from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, BatchNormalization
import pandas as pd
import numpy as np

# Load and preprocess data once at startup
train_data = pd.read_csv("Book2.csv", encoding='latin-1')
le_Skill = LabelEncoder()
le_depart = LabelEncoder()
le_Company = LabelEncoder()

train_data['skill'] = le_Skill.fit_transform(train_data['Skills Known'])
train_data['dept'] = le_depart.fit_transform(train_data['department'])
train_data['target'] = le_Company.fit_transform(train_data['Company Placed'])

# Simulate num_skills feature in training data (assuming 1-5 skills per entry for demo purposes)
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

# Reshape for CNN
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Build optimized CNN model
num_classes = 10
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu',
                     input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), padding='same'))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling1D(pool_size=1))
cnn_model.add(Dropout(0.3))  # Reduced dropout rate

cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))  # Reduced dense layer size
cnn_model.add(Dropout(0.3))
cnn_model.add(Dense(num_classes, activation='softmax'))

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model once at startup
cnn_model.fit(X_train_reshaped, y_train, epochs=5, batch_size=64, validation_data=(X_test_reshaped, y_test), verbose=1)

# Define company class mapping
class_names = {
    0: 'Birlasoft', 1: 'Cognizant', 2: 'Hexaware Technologies',
    3: 'Infosys', 4: 'KPIT Technologies', 5: 'L&T Infotech',
    6: 'Tech Mahindra', 7: 'Wipro Technologies', 8: 'css corp', 9: 'TCS'
}

# Define required skills for roles
ROLE_SKILLS = {
    'python developer': ['python', 'django', 'flask', 'sql'],
    'machine learning engineer': ['python', 'tensorflow', 'pandas', 'machine learning'],
    'game developer': ['c++', 'unity', 'unreal engine'],
    'sql developer': ['sql', 'mysql', 'database design'],
    'database administrator (dba)': ['sql', 'mysql', 'database administration'],
    'database developer': ['sql', 'mysql', 'data modeling'],
    'bi developer': ['sql', 'power bi', 'tableau'],
    'javascript developer': ['javascript', 'react', 'nodejs'],
    'mobile app developer': ['javascript', 'react native', 'flutter'],
    'php developer': ['php', 'laravel', 'mysql'],
    'system administrator': ['linux', 'networking', 'security'],
    'web designer': ['html', 'css', 'javascript'],
    'java developer': ['java', 'spring', 'hibernate'],
    'java software architect': ['java', 'microservices', 'design patterns'],
    'android developer': ['java', 'kotlin', 'android sdk'],
    'ruby developer': ['ruby', 'rails', 'sql'],
    'test automation engineer': ['selenium', 'java', 'python'],
    'technical support engineer': ['troubleshooting', 'networking', 'customer support'],
    'frontend developer': ['html', 'css', 'javascript'],
    'digital marketing specialist': ['seo', 'google analytics', 'content creation'],
    'content manager': ['content strategy', 'seo', 'writing'],
    'ui/ux developer': ['figma', 'ui design', 'user research'],
    'software tester/qa engineer': ['selenium', 'junit', 'qa processes'],
    'software engineer': ['python', 'java', 'sql']
}

def suggest_roles(skill):
    skill_mapping = {
        'python': 'Python Developer, Machine Learning Engineer, Game Developer',
        'sql': 'SQL Developer, Database Administrator (DBA), Database Developer, BI Developer',
        'javascript': 'JavaScript Developer, Mobile App Developer, Game Developer',
        'php': 'PHP Developer, System Administrator, Web Designer',
        'java': 'Java Developer, Java Software Architect, Android Developer',
        'ruby': 'Ruby Developer, Test Automation Engineer, Technical Support Engineer',
        'html and css': 'FrontEnd Developer, Digital Marketing Specialist, Content Manager',
        'mobile applications developer': 'Mobile Applications Developer',
        'web developer': 'Web Developer',
        'network security engineer': 'Network Security Engineer',
        'technical support engineer': 'Technical Support Engineer',
        'ui/ux developer': 'UI/UX Developer',
        'software tester/Quality Assurance Engineer': 'Software Tester/QA Engineer',
        'database developer': 'Database Developer',
        'software engineer': 'Software Engineer'
    }
    return skill_mapping.get(skill.strip().lower(), 'Unknown Role')

def predict(mark, skill, num_skills):
    # Prepare input for prediction
    input_data = np.array([[mark, 0, skill, num_skills]])
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))

    # Predict
    predicted_probs = cnn_model.predict(input_reshaped, verbose=0)
    top_indices = np.argsort(predicted_probs[0])[::-1][:3]
    top_companies = [class_names[i] for i in top_indices]

    return top_companies

def analyze_skill_gaps(extracted_skills, suggested_roles):
    skill_gaps = {}
    extracted_skills_lower = [skill.lower() for skill in extracted_skills]

    for role in suggested_roles:
        role_key = role.lower().split(',')[0]  # Take the first role if multiple
        if role_key in ROLE_SKILLS:
            required_skills = ROLE_SKILLS[role_key]
            missing_skills = [skill for skill in required_skills if skill not in extracted_skills_lower]
            skill_gaps[role] = missing_skills

    return skill_gaps