# Kenyan News Sentiment Analysis

A machine learning application for sentiment analysis of Kenyan news headlines. The project classifies headlines into **positive**, **neutral**, or **negative** sentiment using NLP and a supervised classifier, served through a FastAPI backend with a small React demonstration frontend.

---

## Features

* Data collection scripts for Kenyan news headlines (RSS + scraping)
* Weak-supervision / auto-labeling pipeline for building a labeled dataset
* Text preprocessing (wire-prefix/teaser/source stripping, tokenization, stopword removal, lemmatization)
* TF-IDF feature extraction
* Model comparison across Logistic Regression, Linear SVM, and Multinomial Naive Bayes
* A trained Linear SVM pipeline served via FastAPI (`POST /api/v1/predict`)
* React + Vite demonstration UI that calls the API

---

## Project Structure

```text
apt3010ua_sentiment_analysis/
│
├── app/                        # FastAPI backend
│   ├── main.py                 # app factory, CORS, router registration
│   ├── config.py                # settings (API prefix, CORS origins, model paths)
│   ├── exceptions.py            # global exception handler
│   ├── schemas.py                # request/response models
│   └── routers/
│       └── sentiment.py         # /, /health, /predict endpoints
│
├── frontend/
│   └── frontend/                # React + Vite demonstration UI (note the nested folder name)
│       ├── src/
│       │   ├── api.js            # fetch call to the backend
│       │   ├── Sentimentdemonstration.jsx
│       │   └── main.jsx
│       ├── package.json
│       └── vite.config.js
│
├── src/                          # ML pipeline
│   ├── preprocessing.py          # clean_text() shared by training + inference
│   ├── train.py                  # trains + saves the Linear SVM pipeline used by the API
│   ├── evaluate.py                # evaluates the saved pipeline on a held-out split
│   ├── compare_models.py         # trains/compares LogReg, Linear SVM, Naive Bayes
│   ├── predict.py                 # thin CLI-style wrapper around SentimentService
│   └── services.py                # SentimentService — loads the pickled pipeline
│
├── scripts/                       # data collection / dataset merging utilities
│   ├── rss_collector.py
│   ├── news_scraper.py
│   ├── dataset_extraction.py
│   ├── merge.py
│   └── final_merge.py
│
├── notebooks/                     # exploratory / labeling / evaluation notebooks
│
├── data/
│   ├── raw/                       # scraped/collected headline CSVs
│   ├── processed/                 # cleaned/merged headlines
│   └── labeled/                   # auto-labeled + final labeled datasets
│
├── models/                         # committed model artifacts (see note below)
│   ├── sentiment_pipeline.pkl
│   ├── label_encoder.pkl
│   └── model_comparison.csv
│
├── tests/                          # pytest suite (API + preprocessing + predict)
├── requirements.txt                 # pinned deps (Linux/macOS, includes uvloop)
├── requirements_windows.txt         # same deps, without the Unix-only uvloop
├── pyproject.toml                    # package metadata only — no dependency list
├── pytest.ini
└── README.md
```

---

## Technologies Used

* Python 3.10+
* FastAPI, Uvicorn, Pydantic
* scikit-learn, pandas, NumPy
* NLTK (tokenization, stopwords, lemmatization)
* Matplotlib, Seaborn (model comparison plots)
* Joblib (model persistence)
* React 18 + Vite (frontend demonstration)

---

## Backend Setup

See [app/README.md](app/README.md) for full backend instructions. Quick start:

```bash
git clone <repository-url>
cd apt3010ua_sentiment_analysis

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements_windows.txt   # Windows
# pip install -r requirements.txt         # Linux/macOS

uvicorn app.main:app --reload
```

The API will be live at `http://127.0.0.1:8000` (docs at `/docs`). The pretrained model files in `models/` are committed to the repo, so the API runs out of the box without retraining.

## Frontend Setup

See [frontend/frontend/README.md](frontend/frontend/README.md) for full frontend instructions. Quick start:

```bash
cd frontend/frontend
npm install
npm run dev
```

Opens at `http://localhost:5173` and talks to the backend at the URL in `VITE_API_BASE_URL` (defaults to `http://127.0.0.1:8000`, see `.env`). The backend must be running first.

---

## Machine Learning Pipeline

1. Collect news headlines (`scripts/`)
2. Auto-label / weak-supervise a training set (`notebooks/milestone2_labeling_pipeline.ipynb`)
3. Clean and preprocess text (`src/preprocessing.py`)
4. TF-IDF feature extraction
5. Train and tune Logistic Regression, Linear SVM, and Naive Bayes (`src/compare_models.py`)
6. Save the production pipeline — a **Linear SVM** (`src/train.py`)
7. Evaluate the saved pipeline (`src/evaluate.py`)
8. Serve it via FastAPI (`app/`)

### Training the model

```bash
python -m src.train
```

Reads `data/labeled/final_labeled_dataset.csv`, fits a `TfidfVectorizer` + `LinearSVC` pipeline (grid-searched over `C = [0.1, 1, 10]`), and saves:

```text
models/sentiment_pipeline.pkl
models/label_encoder.pkl
```

### Evaluating the model

```bash
python -m src.evaluate
```

Loads the saved pipeline, re-splits the same dataset (fixed `random_state=42`), and prints accuracy, a classification report, and a confusion matrix.

### Comparing models

```bash
python -m src.compare_models
```

Trains and tunes Logistic Regression, Linear SVM, and Multinomial Naive Bayes on the same split, prints per-model reports, shows confusion-matrix plots, and writes `models/model_comparison.csv`. **This script does not save a pickled pipeline** — only `src/train.py` produces the `.pkl` files the API loads.

### Running tests

```bash
pytest
```

---

## API Endpoints

Base prefix: **`/api/v1`**

### `GET /api/v1/`

```json
{
  "application": "Kenyan News Sentiment Analysis API",
  "version": "1.0.0",
  "status": "running"
}
```

### `GET /api/v1/health`

```json
{
  "status": "healthy",
  "model": "loaded"
}
```

### `POST /api/v1/predict`

Request:

```json
{
  "headline": "Government unveils affordable housing programme"
}
```

Response:

```json
{
  "sentiment": "positive"
}
```

`headline` must be non-empty (`min_length=1`); an empty string returns `422`.

---

## CORS

Configured in `app/config.py`, the backend allows:

* `http://localhost:3000`, `http://127.0.0.1:3000`
* `http://localhost:5173`, `http://127.0.0.1:5173` (the Vite dev server's default port)

---
