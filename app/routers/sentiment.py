"""
API routes for the Kenyan News Sentiment Analysis service.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from src.model_loader import ModelLoader
from src.services import SentimentService

router = APIRouter(
    prefix=settings.API_PREFIX,
    tags=["Sentiment Analysis"],
)

loader = ModelLoader()
service = SentimentService(loader)


@router.get(
    "/",
    summary="API information",
)
def root() -> dict[str, str]:
    """
    Return basic API information.
    """

    return {
        "application": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
    }


@router.get(
    "/health",
    summary="Health check",
)
def health() -> dict[str, str]:
    """
    Verify that the application and model artifacts are available.
    """

    # Trigger lazy loading of model artifacts.
    loader.pipeline
    loader.encoder

    return {
        "status": "healthy",
        "model": "loaded",
    }


@router.post(
    "/predict",
    summary="Predict headline sentiment",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Predict the sentiment of a news headline.
    """

    sentiment = service.predict(
        request.headline,
    )

    return PredictionResponse(
        sentiment=sentiment,
    )