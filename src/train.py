from __future__ import annotations

import logging

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.model_selection import (
    GridSearchCV,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import settings

from src.model_factory import build_models
from src.dataset import (
    load_dataset,
    prepare_dataset,
    encode_labels,
    split_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)



def train_candidate(
    name: str,
    pipeline: Pipeline,
    param_grid: dict,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict[str, object]:
    """
    Train and evaluate a single candidate model.
    """

    logger.info("=" * 60)
    logger.info("Training %s", name)
    logger.info("=" * 60)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=settings.CV_FOLDS,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    best_pipeline = search.best_estimator_

    predictions = best_pipeline.predict(
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    logger.info(
        "Best Parameters : %s",
        search.best_params_,
    )

    logger.info(
        "Best CV Score   : %.4f",
        search.best_score_,
    )

    logger.info(
        "Accuracy        : %.4f",
        accuracy,
    )

    logger.info(
        "Macro F1        : %.4f",
        macro_f1,
    )

    return {
        "name": name,
        "pipeline": best_pipeline,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cv_score": search.best_score_,
        "best_params": search.best_params_,
        "predictions": predictions,
    }

def train_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
) -> list[dict[str, object]]:
    """
    Train every candidate model and rank them by Macro F1.
    """

    results: list[dict[str, object]] = []

    for (
        name,
        (
            pipeline,
            parameters,
        ),
    ) in build_models().items():

        result = train_candidate(
            name=name,
            pipeline=pipeline,
            param_grid=parameters,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    results.sort(
        key=lambda result: result["macro_f1"],
        reverse=True,
    )

    return results

def save_best_model(
    pipeline: Pipeline,
    encoder: LabelEncoder,
) -> None:
    """
    Save the best-performing pipeline and label encoder.
    """

    joblib.dump(
        pipeline,
        settings.PIPELINE_PATH,
    )

    joblib.dump(
        encoder,
        settings.ENCODER_PATH,
    )

    logger.info(
        "Saved pipeline to %s",
        settings.PIPELINE_PATH,
    )

    logger.info(
        "Saved label encoder to %s",
        settings.ENCODER_PATH,
    )

def save_comparison(
    results: list[dict[str, object]],
) -> None:
    """
    Save model comparison metrics.
    """

    comparison = pd.DataFrame(
        [
            {
                "Model": result["name"],
                "Accuracy": round(result["accuracy"], 4),
                "Macro F1": round(result["macro_f1"], 4),
                "CV Score": round(result["cv_score"], 4),
                "Best Parameters": result["best_params"],
            }
            for result in results
        ]
    )

    comparison.to_csv(
        settings.MODEL_COMPARISON_PATH,
        index=False,
    )

    logger.info(
        "Saved comparison to %s",
        settings.MODEL_COMPARISON_PATH,
    )


def print_results(
    results: list[dict[str, object]],
    y_test,
    encoder: LabelEncoder,
) -> None:
    """
    Print evaluation metrics for all models.
    """

    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    for index, result in enumerate(results, start=1):

        logger.info(
            "%d. %s",
            index,
            result["name"],
        )

        logger.info(
            "Accuracy : %.4f",
            result["accuracy"],
        )

        logger.info(
            "Macro F1 : %.4f",
            result["macro_f1"],
        )

        logger.info(
            "CV Score : %.4f",
            result["cv_score"],
        )

        print(
            classification_report(
                y_test,
                result["predictions"],
                target_names=encoder.classes_,
                digits=4,
                zero_division=0,
            )
        )

        print(
            confusion_matrix(
                y_test,
                result["predictions"],
            )
        )

        logger.info("-" * 70)

def main() -> None:
    """
    Train all candidate models, select the best model,
    and persist the training artifacts.
    """

    logger.info("=" * 70)
    logger.info("Kenyan News Sentiment Analysis Training")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    df = load_dataset(settings.DATASET_PATH)

    df = prepare_dataset(df)

    df, encoder = encode_labels(df)

    X_train, X_test, y_train, y_test = split_dataset(
        df,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )

    logger.info(
        "Training samples : %d",
        len(X_train),
    )

    logger.info(
        "Testing samples  : %d",
        len(X_test),
    )

    # ------------------------------------------------------------------
    # Train Candidate Models
    # ------------------------------------------------------------------

    results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    winner = results[0]

    # ------------------------------------------------------------------
    # Best Model
    # ------------------------------------------------------------------

    logger.info("=" * 70)
    logger.info("BEST MODEL")
    logger.info("=" * 70)

    logger.info(
        "Model      : %s",
        winner["name"],
    )

    logger.info(
        "Macro F1   : %.4f",
        winner["macro_f1"],
    )

    logger.info(
        "Accuracy   : %.4f",
        winner["accuracy"],
    )

    logger.info(
        "CV Score   : %.4f",
        winner["cv_score"],
    )

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    print_results(
        results=results,
        y_test=y_test,
        encoder=encoder,
    )

    # ------------------------------------------------------------------
    # Save Artifacts
    # ------------------------------------------------------------------

    save_best_model(
        pipeline=winner["pipeline"],
        encoder=encoder,
    )

    save_comparison(
        results,
    )

    logger.info("=" * 70)
    logger.info("Training completed successfully.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()