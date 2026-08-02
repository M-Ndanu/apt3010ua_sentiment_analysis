"""
Model artifact loader for the Kenyan News Sentiment Analysis project.

This module is responsible for loading and validating the trained
machine learning artifacts. The loaded objects are cached so they are
loaded only once during the application's lifetime.
"""

from __future__ import annotations

import logging
from functools import cached_property

import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and validates trained model artifacts.

    The pipeline and label encoder are loaded lazily and cached
    for the lifetime of the application.
    """

    @cached_property
    def pipeline(self) -> Pipeline:
        """Load the trained sklearn pipeline."""

        path = settings.PIPELINE_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"Pipeline not found: {path}"
            )

        pipeline = joblib.load(path)

        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                "Loaded object is not a sklearn Pipeline."
            )

        expected_steps = {
            "tfidf",
            "classifier",
        }

        if not expected_steps.issubset(
            pipeline.named_steps.keys()
        ):
            raise ValueError(
                "Pipeline is missing required steps."
            )

        logger.info(
            "Loaded pipeline from %s",
            path,
        )

        return pipeline

    @cached_property
    def encoder(self) -> LabelEncoder:
        """Load the fitted LabelEncoder."""

        path = settings.ENCODER_PATH

        if not path.exists():
            raise FileNotFoundError(
                f"Label encoder not found: {path}"
            )

        encoder = joblib.load(path)

        if not isinstance(
            encoder,
            LabelEncoder,
        ):
            raise TypeError(
                "Loaded object is not a LabelEncoder."
            )

        if not hasattr(
            encoder,
            "classes_",
        ):
            raise ValueError(
                "LabelEncoder has not been fitted."
            )

        logger.info(
            "Loaded label encoder from %s",
            path,
        )

        return encoder