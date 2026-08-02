"""
Compare multiple machine learning models for Kenyan News Sentiment Analysis.

This script trains, tunes, evaluates and compares multiple classifiers
using identical preprocessing and TF-IDF feature engineering.

Models
------
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

Models are ranked using Macro F1 score.
"""

from __future__ import annotations

import logging
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections.abc import Mapping

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import settings

from src.dataset import (
    load_dataset,
    prepare_dataset,
    encode_labels,
    split_dataset,
)

from src.model_factory import build_models


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)



def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    param_grid: Mapping[str, object],
    X_train,
    X_test,
    y_train,
    y_test,
    encoder: LabelEncoder,
) -> tuple[dict, pd.DataFrame]:
    """
    Train, tune and evaluate a single model.
    """

    logger.info("=" * 70)
    logger.info("Training %s", model_name)
    logger.info("=" * 70)

    start_time = time.perf_counter()

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=settings.CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter() - start_time
    )

    best_model = grid.best_estimator_

    y_pred = best_model.predict(
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

    logger.info(
        "Best Parameters: %s",
        grid.best_params_,
    )

    logger.info(
        "Best CV Score: %.4f",
        grid.best_score_,
    )

    logger.info(
        "\n%s",
        classification_report(
            y_test,
            y_pred,
            target_names=encoder.classes_,
            digits=4,
            zero_division=0,
        ),
    )

    confusion = confusion_matrix(
        y_test,
        y_pred,
    )

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "CV Score": grid.best_score_,
        "Training Time (s)": round(
            training_time,
            2,
        ),
        "Best Parameters": str(
            grid.best_params_
        ),
    }

    return (
        metrics,
        pd.DataFrame(
            confusion,
            index=encoder.classes_,
            columns=encoder.classes_,
        ),
    )

def save_confusion_matrix(
    matrix: pd.DataFrame,
    model_name: str,
) -> None:
    """
    Save a confusion matrix as a PNG image.
    """

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    output = (
        settings.CONFUSION_MATRIX_DIR
        / f"{filename}_confusion_matrix.png"
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved %s",
        output.name,
    )

def compare_models() -> None:
    """
    Train, tune and compare all candidate models.
    """
    settings.CONFUSION_MATRIX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset(
        settings.DATASET_PATH,
    )

    df = prepare_dataset(df)

    df, encoder = encode_labels(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_dataset(
        df,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )

    models = build_models()

    results = []

    logger.info(
        "Comparing %d models...",
        len(models),
    )

    for (
        model_name,
        (
            pipeline,
            parameters,
        ),
    ) in models.items():

        metrics, confusion = evaluate_model(
            model_name=model_name,
            pipeline=pipeline,
            param_grid=parameters,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            encoder=encoder,
        )

        results.append(metrics)

        save_confusion_matrix(
            confusion,
            model_name,
        )

    comparison = (
        pd.DataFrame(results)
        SORT_METRIC = "Macro F1".reset_index(drop=True)
    )

    comparison = comparison.round(4)

    output_path = settings.MODEL_COMPARISON_PATH

    comparison.to_csv(
        output_path,
        index=False,
    )

    logger.info("=" * 70)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("=" * 70)

    print(comparison.to_string(index=False))

    logger.info(
        "Comparison saved to %s",
        output_path,
    )

    best_model = comparison.iloc[0]

    logger.info("")
    logger.info("BEST MODEL")
    logger.info("----------")
    logger.info(
        "Model      : %s",
        best_model["Model"],
    )
    logger.info(
        "Macro F1   : %.4f",
        best_model["Macro F1"],
    )
    logger.info(
        "Accuracy   : %.4f",
        best_model["Accuracy"],
    )
    logger.info(
        "CV Score   : %.4f",
        best_model["CV Score"],
    )

def main() -> None:
    compare_models()


if __name__ == "__main__":
    main()
