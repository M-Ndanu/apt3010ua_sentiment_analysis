from fastapi import APIRouter

from app.config import settings
from app.schemas import PredictionRequest, PredictionResponse
from src.services import SentimentService

router = APIRouter(
    prefix=settings.API_PREFIX,
    tags=["Sentiment Analysis"],
)

service = SentimentService()


@router.get("/")
def root():
    return {
        "application": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running",
    }


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded",
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    sentiment = service.predict(request.headline)

    return PredictionResponse(
        sentiment=sentiment
    )