# TopX — AI Career Intelligence

An editorial dark-themed resume analyzer that combines real ML, transparent heuristics, and a print-ready career report. Built with Flask, scikit-learn, and a Fraunces × IBM Plex Mono design system.

**Live demo:** [topx-ai-resume-analyzer.onrender.com](https://topx-ai-resume-analyzer.onrender.com/)

![TopX Landing Page](screenshot.png?v=2)

---

## What it does

Upload a PDF resume. In under 10 seconds, you get:

- **Real-ML category prediction** — TF-IDF + Logistic Regression trained on **2,483 actual resumes** across 24 industry categories (89.5% top-3 accuracy).
- **Transparent company matching** — 35 companies (TCS, Infosys, Google, Amazon, EY, Hindustan Unilever, DHL...) scored by real skill overlap with their published tech stacks.
- **Skill extraction** — 411 skills across 14 categories (Languages, Frontend, Cloud, AI/ML, **Accounting & Finance**, **Business Operations**, **Marketing & Sales**, etc.) with section-aware ranking.
- **Resume score** — 0–100 with a transparent breakdown (Skills / Experience / Projects / Certifications / Education / Summary).
- **TF-IDF role matching** — semantic similarity between your skills and 47 role profiles.
- **Skill-gap roadmap** — exactly which skills to learn next for each target role.
- **Insights engine** — strengths, action items, and profile signals (e.g. "Cloud-Native Skill Set", "Operations / Supply Chain Profile").
- **Editorial PDF report** — native browser print with a dedicated `@media print` stylesheet. Vector text, magazine-style cover page, no flicker.
- **Resume archive** — every analysis saved per user, browseable at `/history`. Re-uploading the same PDF returns instantly via SHA-256 cache.

---

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Flask, Flask-SocketIO, Flask-Limiter, eventlet |
| ML | scikit-learn (TF-IDF + LogisticRegression, RandomForest), joblib |
| NLP | Section-aware extraction, alias normalization, ambiguous-skill disambiguation |
| PDF parsing | pdfminer.six |
| Auth | werkzeug bcrypt password hashing, session-based |
| Storage | SQLite (Postgres-ready via `DATABASE_PATH` env var) |
| Frontend | Vanilla HTML/CSS/JS, Fraunces (display) + IBM Plex Mono (body) |
| Print | Native `window.print()` + dedicated `@media print` styles |

---

## Quick start

```bash
git clone https://github.com/therahul-yo/TopX-AI-Resume-Analyzer-Job-Recommendation.git
cd TopX-AI-Resume-Analyzer-Job-Recommendation

python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open **http://localhost:5001** in your browser.

---

## Project structure

```
.
├── app.py                          # Flask app, routes, extraction pipeline
├── main.py                         # ML loaders, predictors, scoring, insights
├── ml/
│   ├── train.py                    # Trains the company predictor (RandomForest)
│   └── train_classifier.py         # Trains the resume category classifier (TF-IDF + LogReg)
├── models/
│   ├── company_predictor.joblib    # Pre-trained company model
│   └── category_classifier.joblib  # Pre-trained 24-category model
├── data/
│   └── resumes.csv                 # 2,483 Kaggle resumes (industry-labeled)
├── static/
│   └── style.css                   # Obelisk editorial dark design system
├── templates/
│   ├── index.html                  # Landing page (marquee + hero + features)
│   ├── upload.html                 # Drag-drop upload + loading skeleton
│   ├── result.html                 # Career report (with print styles)
│   └── history.html                # Archive of past analyses
├── Book2.csv                       # Legacy placement dataset (~950 rows)
├── Procfile                        # Render: python app.py
└── requirements.txt
```

---

## How the ML works

### 1. Resume Category Classifier

**Input:** raw resume text · **Output:** top-4 industry categories with confidence

```
TF-IDF (8000 features, ngram 1-2, sublinear)
   ↓
LogisticRegression (C=2.0, balanced class weights)
   ↓
24 categories: IT, Engineering, Finance, Healthcare, Sales, HR, ...
```

- **Train accuracy:** 95.1%
- **Test accuracy:** 68.8%
- **Top-3 accuracy:** 89.5%
- Trained on the public Kaggle "Resume Dataset" (2,483 resumes).

### 2. Company Match Predictor

Replaced a weak RandomForest (31% accuracy on synthetic data) with a transparent heuristic:

- 35 companies, each with curated **core skills** + **nice-to-have skills** based on their public job postings
- Score = `0.75 × (core_match / |core|) + 0.25 × (nice_match / |nice|)`
- Top-tier companies (Google, Meta, Apple) penalize candidates with <2 yrs experience
- Industry alignment from the category classifier gives a +4% bonus
- Output includes the actual match count: *"Matched 7/8 core skills · Top-tier"*

### 3. Skill Extraction

- 411 canonical skills across 14 categories
- Section-aware: skills found in the **Skills** section get 100% confidence; elsewhere 70%
- Alias normalization: `react.js → react`, `sklearn → scikit-learn`, `k8s → kubernetes`, `microsoft excel → ms excel`
- **Ambiguous skill protection:** words like *Swift, Go, R, sales, communication* must appear in the explicit Skills section, preventing false positives like "Swift" matching the company name "Swift ProSys".

### 4. Role Match Scoring

- 47 role profiles (Python Developer, Data Scientist, Supply Chain Analyst, Accountant, Tax Associate, ...)
- TF-IDF cosine similarity between extracted skills and role-required skills
- Replaces brittle keyword matching with proper semantic similarity

### 5. Resume Score Breakdown

Out of 100, fully transparent:

| Component | Max | Source |
|---|---|---|
| Skills | 28 | Skill count + density |
| Experience | 22 | Years detected from regex |
| Projects | 18 | Project section bullets |
| Certifications | 14 | Certification text patterns |
| Education | 10 | Department codes detected |
| Summary | 8 | Has dedicated About/Summary section |

---

## Design

- **Aesthetic:** "Obelisk" — editorial magazine on a pitch-black canvas
- **Typography:** Fraunces (italic display serif) × IBM Plex Mono (body & data)
- **Palette:** True black `#000`, cream `#F5F0E8` text, single molten orange accent `#FF6B35`
- **Layout:** Roman-numeral numbered sections (I, II, III…), magazine masthead, marquee ticker
- **Motion:** Mask-reveal hero, marquee scroll, IntersectionObserver-driven section reveals, score count-up, animated bars
- **PDF:** Native `window.print()` with `@media print` stylesheet — no flicker, vector text, page-break-aware

---

## Production-grade features

- **Bcrypt password hashing** via `werkzeug.security`
- **Auto-login** after registration
- **Rate limiting** — 12 uploads/hour per user (Flask-Limiter, in-memory)
- **PDF hash cache** — same resume re-uploaded within 24h returns instantly
- **Resume archive** — `/history` route, last 30 analyses per user
- **Serialized ML models** — joblib bundles ship with the repo for fast cold starts
- **Postgres-ready** — `DATABASE_PATH` env var falls back to SQLite
- **5 MB upload limit** + extension validation
- **Section parsing** — Skills / Experience / Education / Projects / Certifications / Summary detected from headers
- **Native PDF print** — no html2pdf.js, no canvas screenshot, no flicker

---

## Re-training the models

If you swap in a new dataset:

```bash
# Train the company predictor (uses Book2.csv)
python ml/train.py

# Train the category classifier (uses data/resumes.csv)
python ml/train_classifier.py

# Commit the updated joblib bundles
git add models/*.joblib
git commit -m "Retrain models"
```

The app loads the serialized bundles at startup. If they're missing, it trains inline as a fallback.

---

## Roadmap

Doable on the free tier with no external services:
- [x] Serialized ML models for fast cold starts
- [x] Resume archive with hash caching
- [x] Rate limiting
- [x] Editorial PDF report
- [x] Loading skeleton
- [x] Heuristic company matching with transparent scoring
- [x] Non-tech resume support (commerce, finance, operations)

Needs paid tier or external accounts:
- [ ] **Postgres** for persistent archive (Render free Postgres / Supabase)
- [ ] **sentence-transformers** for semantic skill matching (~500 MB model, needs paid Render)
- [ ] **spaCy NER** for proper resume entity extraction
- [ ] **Sentry** error tracking
- [ ] **SendGrid** email PDF reports
- [ ] **OG image generation** for social shares (Pillow)

---

## License

MIT License — free to use, modify, and ship.

---

Made by **Rahul** · 2025
