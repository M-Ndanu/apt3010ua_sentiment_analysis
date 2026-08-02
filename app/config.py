"""
Application configuration.

Central configuration shared by the API,
training scripts and evaluation scripts.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global application settings.
    """

    # ==========================================================
    # API
    # ==========================================================

    PROJECT_NAME: str = "Kenyan News Sentiment Analysis API"

    VERSION: str = "1.0.0"

    DESCRIPTION: str = (
        "Sentiment Analysis API for Kenyan News Headlines"
    )

    API_PREFIX: str = "/api/v1"

    # ==========================================================
    # Directories
    # ==========================================================

    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = BASE_DIR / "data"

    RAW_DATA_DIR: Path = DATA_DIR / "raw"

    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    LABELED_DATA_DIR: Path = DATA_DIR / "labeled"

    MODEL_DIR: Path = BASE_DIR / "models"

    # ==========================================================
    # Files
    # ==========================================================

    DATASET_PATH: Path = (
        LABELED_DATA_DIR
        / "final_labeled_dataset.csv"
    )

    PIPELINE_PATH: Path = (
        MODEL_DIR
        / "sentiment_pipeline.pkl"
    )

    ENCODER_PATH: Path = (
        MODEL_DIR
        / "label_encoder.pkl"
    )

    MODEL_COMPARISON_PATH: Path = (
        MODEL_DIR
        / "model_comparison.csv"
    )

    CONFUSION_MATRIX_DIR = MODEL_DIR / "confusion_matrices"
    CONFUSION_MATRIX_DIR: Path = (
        MODEL_DIR / "confusion_matrices"
    )
    AUDIT_DIR: Path = DATA_DIR / "audit"

    AUDIT_REPORT_PATH: Path = (
        AUDIT_DIR
        / "suspicious_labels.csv"
    )
    

    # ==========================================================
    # Machine Learning
    # ==========================================================

    TEST_SIZE: float = 0.20

    RANDOM_STATE: int = 42

    CV_FOLDS: int = 3

    MAX_FEATURES: int = 5000

    NGRAM_RANGE: tuple[int, int] = (1, 2)

    # ==========================================================
    # CORS
    # ==========================================================

    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = {
        "env_file": ".env",
    }


settings = Settings()

settings.MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

settings.AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)