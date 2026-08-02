"""
Model factory for the Kenyan News Sentiment Analysis project.

Provides reusable model pipelines and hyperparameter grids used during
training and model comparison.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.config import settings


def build_models() -> dict:
    """
    Create all candidate models and their hyperparameter grids.

    Returns
    -------
    dict
        Dictionary mapping model names to
        (Pipeline, parameter grid).
    """

    return {

        "Logistic Regression": (

            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=settings.MAX_FEATURES,
                            ngram_range=settings.NGRAM_RANGE,
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
                "classifier__C": [
                    0.1,
                    1,
                    10,
                ],

                "classifier__solver": [
                    "liblinear",
                    "saga",
                ],
            },

        ),

        "Linear SVM": (

            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=settings.MAX_FEATURES,
                            ngram_range=settings.NGRAM_RANGE,
                        ),
                    ),
                    (
                        "classifier",
                        LinearSVC(),
                    ),
                ]
            ),

            {
                "classifier__C": [
                    0.1,
                    1,
                    10,
                ],
            },

        ),

        "Naive Bayes": (

            Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=settings.MAX_FEATURES,
                            ngram_range=settings.NGRAM_RANGE,
                        ),
                    ),
                    (
                        "classifier",
                        MultinomialNB(),
                    ),
                ]
            ),

            {
                "classifier__alpha": [
                    0.01,
                    0.1,
                    0.5,
                    1.0,
                    2.0,
                ],
            },

        ),

    }