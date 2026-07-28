from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):


    PROJECT_NAME: str = "Kenyan News Sentiment Analysis API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Sentiment Analysis API for Kenyan News Headlines"
    )


    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    MODEL_DIR: Path = BASE_DIR / "models"

    PIPELINE_PATH: Path = MODEL_DIR / "sentiment_pipeline.pkl"

    LABEL_ENCODER_PATH: Path = MODEL_DIR / "label_encoder.pkl"



    API_PREFIX: str = "/api/v1"


    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"


settings = Settings()