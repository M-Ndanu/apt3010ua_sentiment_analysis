from pathlib import Path

import joblib

from src.preprocessing import clean_text


class SentimentService:
    """
    Loads the trained model once and serves predictions.
    """

    def __init__(self):

        base_dir = Path(__file__).resolve().parent.parent

        self.pipeline = joblib.load(
            base_dir / "models" / "sentiment_pipeline.pkl"
        )

        self.encoder = joblib.load(
            base_dir / "models" / "label_encoder.pkl"
        )

    def predict(self, headline: str) -> str:

        cleaned = clean_text(headline)

        prediction = self.pipeline.predict([cleaned])

        return self.encoder.inverse_transform(prediction)[0]