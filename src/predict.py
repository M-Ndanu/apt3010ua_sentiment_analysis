from src.services import SentimentService

service = SentimentService()


def predict_sentiment(headline: str) -> str:
    return service.predict(headline)