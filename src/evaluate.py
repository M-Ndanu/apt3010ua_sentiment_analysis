from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text



BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "labeled"
    / "final_labeled_dataset.csv"
)

MODEL_DIR = BASE_DIR / "models"

PIPELINE_PATH = MODEL_DIR / "sentiment_pipeline.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"



print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

df = df.dropna(subset=["headline", "sentiment"])

df["clean_text"] = df["headline"].apply(clean_text)



label_encoder = joblib.load(ENCODER_PATH)

df["label"] = label_encoder.transform(df["sentiment"])

X = df["clean_text"]
y = df["label"]



_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)



pipeline = joblib.load(PIPELINE_PATH)



print("\nEvaluating model...")

y_pred = pipeline.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
    )
)

print("Confusion Matrix")
print("-" * 60)

print(confusion_matrix(y_test, y_pred))