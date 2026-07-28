from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from src.preprocessing import clean_text



BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "labeled"
    / "final_labeled_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)



print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape : {df.shape}")
print(f"Columns       : {df.columns.tolist()}")

required_columns = ["headline", "sentiment"]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=["headline", "sentiment"])

print(f"After cleaning: {df.shape}")



print("\nCleaning headlines...")

df["clean_text"] = df["headline"].apply(clean_text)

print(df[["headline", "clean_text"]].head())



print("\nEncoding sentiment labels...")

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["sentiment"])

print("Sentiment classes:")
print(label_encoder.classes_)




X = df["clean_text"]
y = df["label"]



print("\nCreating training split...")

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")



pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LinearSVC(),
        ),
    ]
)



param_grid = {
    "classifier__C": [0.1, 1, 10]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1,
)



print("\nTraining model...")

grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_

print("\nTraining complete.")

print("\nBest Parameters")
print("----------------")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score")
print("---------------------------")
print(f"{grid_search.best_score_:.4f}")


print("\nSaving model artifacts...")

pipeline_path = MODEL_DIR / "sentiment_pipeline.pkl"
encoder_path = MODEL_DIR / "label_encoder.pkl"

joblib.dump(best_pipeline, pipeline_path)
joblib.dump(label_encoder, encoder_path)

print("\nSaved successfully.")

print(f"Pipeline : {pipeline_path}")
print(f"Encoder  : {encoder_path}")

print("\nTraining finished.")