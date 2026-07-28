# apt3010ua_sentiment_analysis-
Sentiment analysis of Kenyan news headlines using NLP - RSS/scraping-based data collection, weak-supervision auto-labeling, and a trained classifier. APT3010 term project.


# Kenyan News Sentiment Analysis

A machine learning application for sentiment analysis of Kenyan news headlines. The project classifies headlines into **Positive**, **Neutral**, or **Negative** sentiments using Natural Language Processing (NLP) and supervised machine learning. The application includes a production-style FastAPI backend for real-time sentiment prediction.

---

## Features

* Data collection and preprocessing of Kenyan news headlines
* Exploratory Data Analysis (EDA)
* TF-IDF feature engineering
* Logistic Regression model with hyperparameter tuning
* Linear SVM model with hyperparameter tuning
* Multinomial Naive Bayes model with hyperparameter tuning
* Model comparison using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
* Reusable preprocessing pipeline
* Production-ready FastAPI backend
* REST API for real-time sentiment prediction

---

## Project Structure

```text
apt3010ua_sentiment_analysis/
│
├── app/
│   ├── config.py
│   ├── exceptions.py
│   ├── main.py
│   ├── routers/
│   └── schemas.py
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   ├── predict.py
│   └── services.py
│
├── models/
│   ├── sentiment_pipeline.pkl
│   ├── label_encoder.pkl
│   └── model_comparison.csv
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── labeled/
│
├── notebooks/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# Technologies Used

* Python 3.10+
* FastAPI
* scikit-learn
* Pandas
* NumPy
* NLTK
* Matplotlib
* Seaborn
* Joblib
* Uvicorn

---

# Machine Learning Pipeline

The project follows the standard machine learning workflow:

1. Collect news headlines
2. Clean and preprocess text
3. TF-IDF feature extraction
4. Train multiple machine learning models
5. Hyperparameter tuning using GridSearchCV
6. Evaluate and compare models
7. Save the best model
8. Deploy using FastAPI

---

# Text Preprocessing

The preprocessing pipeline performs:

* Removal of wire-service prefixes
* Removal of timestamps and teaser text
* Removal of source-name suffixes
* Lowercasing
* Removal of punctuation and numbers
* Tokenization
* Stopword removal
* Lemmatization

The same preprocessing pipeline is reused during inference to ensure consistency between training and prediction.

---

# Models

The following models were trained and evaluated:

* Logistic Regression
* Linear Support Vector Machine (Linear SVM)
* Multinomial Naive Bayes

Each model was tuned using GridSearchCV.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* Macro F1-score
* Confusion Matrix

The comparison results are exported to:

```text
models/model_comparison.csv
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd apt3010ua_sentiment_analysis
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the environment:

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the project in editable mode:

```bash
pip install -e .
```

---

# Training the Model

Run:

```bash
python -m src.train
```

Artifacts will be saved to:

```text
models/
├── sentiment_pipeline.pkl
└── label_encoder.pkl
```

---

# Evaluating the Model

Run:

```bash
python -m src.evaluate
```

This script reports:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

---

# Comparing Models

Run:

```bash
python -m src.compare_models
```

This script:

* Trains all three models
* Tunes hyperparameters
* Evaluates performance
* Generates confusion matrices
* Produces a comparison table
* Exports the comparison to:

```text
models/model_comparison.csv
```

---

# Running the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Alternative OpenAPI documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# API Endpoints

## Root

```http
GET /api/v1/
```

Example Response

```json
{
  "application": "Kenyan News Sentiment Analysis API",
  "version": "1.0.0",
  "status": "running"
}
```

---

## Health Check

```http
GET /api/v1/health
```

Example Response

```json
{
  "status": "healthy",
  "model": "loaded"
}
```

---

## Predict Sentiment

```http
POST /api/v1/predict
```

Request Body

```json
{
  "headline": "Government unveils affordable housing programme"
}
```

Example Response

```json
{
  "sentiment": "positive"
}
```

---

# CORS

The backend is configured to allow local frontend development for:

* http://localhost:3000
* http://127.0.0.1:3000
* http://localhost:5173
* http://127.0.0.1:5173


# License

This project is intended for academic and educational purposes.
