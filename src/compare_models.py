from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "labeled"
    / "final_labeled_dataset.csv"
)


print("=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

df = df.dropna(subset=["headline", "sentiment"])

df["clean_text"] = df["headline"].apply(clean_text)


label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["sentiment"])

X = df["clean_text"]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


models = {
    "Logistic Regression": (
        Pipeline(
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
                    LogisticRegression(
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        {
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["liblinear", "saga"],
        },
    ),
    "Linear SVM": (
        Pipeline(
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
        ),
        {
            "classifier__C": [0.1, 1, 10],
        },
    ),
    "Naive Bayes": (
        Pipeline(
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
                    MultinomialNB(),
                ),
            ]
        ),
        {
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
        },
    ),
}


results = []


for model_name, (pipeline, params) in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=3,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    print("Best Parameters:")
    print(grid.best_params_)

    print()

    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
    ))

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "Macro F1": f1,
        }
    )

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )

    plt.title(f"Confusion Matrix - {model_name}")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()


comparison = (
    pd.DataFrame(results)
    .sort_values(
        by="Macro F1",
        ascending=False,
    )
)

print("\n")
print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

comparison = comparison.round(3)

comparison_path = (
    BASE_DIR
    / "models"
    / "model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)

print(comparison)

print(f"\nComparison saved to:\n{comparison_path}")