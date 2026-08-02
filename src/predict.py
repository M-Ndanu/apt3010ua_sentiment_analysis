"""
Prediction utilities for the Kenyan News Sentiment Analysis project.

This module provides reusable prediction functions that can be used by
the FastAPI application, unit tests, notebooks, and other clients.
"""

from __future__ import annotations

from typing import Iterable

from src.model_loader import ModelLoader
from src.services import SentimentService

_loader = ModelLoader()
_service = SentimentService(_loader)


def predict_sentiment(headline: str) -> str:
    """
    Predict the sentiment of a single news headline.

    Parameters
    ----------
    headline : str
        News headline.

    Returns
    -------
    str
        Predicted sentiment.
    """

    return _service.predict(headline)


def predict_batch(headlines: Iterable[str]) -> list[str]:
    """
    Predict sentiments for multiple headlines.

    Parameters
    ----------
    headlines : Iterable[str]

    Returns
    -------
    list[str]
        Predicted sentiment labels.
    """

    return [
        _service.predict(headline)
        for headline in headlines
    ]