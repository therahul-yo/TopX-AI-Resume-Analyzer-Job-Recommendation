"""
TopX — Resume Category Classifier (real ML)

Trains a TF-IDF + Logistic Regression classifier on the Kaggle
"Resume Dataset" — 2,483 real resumes across 24 job categories.

Run once locally:
    python ml/train_classifier.py

Saves to:
    models/category_classifier.joblib

This is far more accurate than the original Book2.csv RandomForest because:
- Uses actual resume text (not synthetic CGPA + 1 skill)
- 2.5x more training rows
- 24 real-world job categories
- Proper TF-IDF feature extraction with bigrams
"""
import csv
import os
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.pipeline import Pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'resumes.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'category_classifier.joblib')

csv.field_size_limit(10_000_000)


def load_data():
    texts, labels = [], []
    with open(DATA_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = (row.get('Resume_str') or '').strip()
            c = (row.get('Category') or '').strip()
            if t and c:
                texts.append(t)
                labels.append(c)
    return texts, labels


def train(verbose=True):
    if verbose:
        print(f"→ Loading {DATA_PATH}")
    texts, labels = load_data()

    if verbose:
        print(f"  resumes: {len(texts)}")
        print(f"  categories: {len(set(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            stop_words='english',
            lowercase=True,
        )),
        ('clf', LogisticRegression(
            C=2.0,
            max_iter=1500,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=-1,
            random_state=42,
        )),
    ])

    if verbose:
        print("→ Training TF-IDF + LogisticRegression…")
    pipe.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, pipe.predict(X_train))
    test_acc  = accuracy_score(y_test,  pipe.predict(X_test))

    proba_test = pipe.predict_proba(X_test)
    top3 = top_k_accuracy_score(y_test, proba_test, k=3, labels=pipe.classes_)

    if verbose:
        print(f"  train accuracy:  {train_acc:.3f}")
        print(f"  test accuracy:   {test_acc:.3f}")
        print(f"  top-3 accuracy:  {top3:.3f}")

    bundle = {
        'pipeline': pipe,
        'classes':  list(pipe.classes_),
        'metrics': {
            'train_accuracy': float(train_acc),
            'test_accuracy':  float(test_acc),
            'top3_accuracy':  float(top3),
            'n_train':        len(X_train),
            'n_test':         len(X_test),
            'n_classes':      len(pipe.classes_),
        }
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    if verbose:
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"→ Saved {MODEL_PATH} ({size_mb:.1f} MB)")
    return bundle


if __name__ == '__main__':
    try:
        train()
        sys.exit(0)
    except Exception as e:
        print(f"✗ Training failed: {e}", file=sys.stderr)
        sys.exit(1)
