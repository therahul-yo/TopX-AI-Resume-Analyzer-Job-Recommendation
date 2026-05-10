"""
TopX — Standalone Training Script

Run once locally:
    python ml/train.py

Trains the company predictor from Book2.csv and serializes it via joblib
to models/company_predictor.joblib. Commit that file to git so production
boots without re-training.

Re-run whenever Book2.csv is updated or you swap in a larger dataset.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Allow running from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'Book2.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'company_predictor.joblib')


def train(verbose=True):
    if verbose:
        print(f"→ Loading dataset: {DATA_PATH}")
    train_data = pd.read_csv(DATA_PATH, encoding='latin-1')

    if verbose:
        print(f"  rows: {len(train_data)}")
        print(f"  companies: {train_data['Company Placed'].nunique()}")

    le_skill   = LabelEncoder()
    le_depart  = LabelEncoder()
    le_company = LabelEncoder()

    train_data['skill']  = le_skill.fit_transform(train_data['Skills Known'])
    train_data['dept']   = le_depart.fit_transform(train_data['department'])
    train_data['target'] = le_company.fit_transform(train_data['Company Placed'])

    np.random.seed(42)
    train_data['num_skills'] = np.random.randint(1, 6, size=len(train_data))

    X = train_data.drop([
        'Full Name', "12th Mark", "10th Mark", 'dept', 'Company Placed',
        "Skills Known", "Projects Done", 'target', 'department',
        "Certifications/Internships"
    ], axis=1)
    y = train_data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    if verbose:
        print("→ Training RandomForest (n=200, max_depth=20)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train_s))
    test_acc  = accuracy_score(y_test,  model.predict(X_test_s))

    if verbose:
        print(f"  train accuracy: {train_acc:.3f}")
        print(f"  test accuracy:  {test_acc:.3f}")

    bundle = {
        'model':       model,
        'scaler':      scaler,
        'le_skill':    le_skill,
        'le_depart':   le_depart,
        'le_company':  le_company,
        'class_names': dict(enumerate(le_company.classes_)),
        'metrics': {
            'train_accuracy': float(train_acc),
            'test_accuracy':  float(test_acc),
            'n_classes':      int(len(le_company.classes_)),
            'n_train':        int(len(X_train)),
        }
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    if verbose:
        size_kb = os.path.getsize(MODEL_PATH) / 1024
        print(f"→ Saved {MODEL_PATH} ({size_kb:.1f} KB)")
    return bundle


if __name__ == '__main__':
    try:
        train()
        sys.exit(0)
    except Exception as e:
        print(f"✗ Training failed: {e}", file=sys.stderr)
        sys.exit(1)
