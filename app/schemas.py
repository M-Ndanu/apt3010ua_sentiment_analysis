from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    headline: str = Field(
        ...,
        min_length=1,
        description="News headline to analyze"
    )


class PredictionResponse(BaseModel):
    sentiment: str