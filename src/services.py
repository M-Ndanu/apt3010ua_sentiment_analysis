from __future__ import annotations

from src.model_loader import ModelLoader
from src.preprocessing import clean_text


class SentimentService:
    """
    Performs sentiment inference using loaded model artifacts.
    """

    def __init__(
        self,
        loader: ModelLoader,
    ) -> None:

        self.pipeline = loader.pipeline
        self.encoder = loader.encoder

    def predict(
        self,
        headline: str,
    ) -> str:

        cleaned = clean_text(headline)

        prediction = self.pipeline.predict(
            [cleaned]
        )

        return self.encoder.inverse_transform(
            prediction
        )[0]