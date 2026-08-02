from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    headline: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=[
            "Kenya's inflation falls to 3.8%"
        ],
        description="News headline to analyze",
    )


class PredictionResponse(BaseModel):
    sentiment: str = Field(
        description="Predicted sentiment label"
    )