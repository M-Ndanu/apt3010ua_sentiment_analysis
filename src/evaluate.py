"""
Model evaluation script for the Kenyan News Sentiment Analysis project.

Evaluates the trained model on the held-out test set and reports
classification metrics.
"""

from __future__ import annotations

import logging

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.config import settings
from src.dataset import (
    load_dataset,
    prepare_dataset,
    encode_labels,
    split_dataset,
)
from src.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)



def evaluate() -> None:
    """
    Evaluate the trained models
    """

    loader = ModelLoader()

    df = load_dataset(settings.DATASET_PATH)

    df = prepare_dataset(df)

    df, _ = encode_labels(
        df,
        loader.encoder,
    )

    _, X_test, _, y_test = split_dataset(
        df,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )

    logger.info("Running inference...")

    y_pred = loader.pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Macro Precision    : {precision:.4f}")
    print(f"Macro Recall       : {recall:.4f}")
    print(f"Macro F1           : {macro_f1:.4f}")
    print(f"Weighted F1        : {weighted_f1:.4f}")

    print("\nClassification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=loader.encoder.classes_,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix")
    print("-" * 70)

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "Weighted F1",
            ],
            "Value": [
                accuracy,
                precision,
                recall,
                macro_f1,
                weighted_f1,
            ],
        }
    )

    output_path = (
        settings.MODEL_DIR
        / "evaluation_metrics.csv"
    )

    metrics.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Evaluation metrics saved to %s",
        output_path,
    )


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()