# Backend — Kenyan News Sentiment Analysis API

FastAPI service that loads a pretrained TF-IDF + Linear SVM pipeline and classifies news headlines as `positive`, `neutral`, or `negative`.

This README covers only the `app/` package. For the ML training pipeline see the root [README.md](../README.md); for the demo UI see [frontend/frontend/README.md](../frontend/frontend/README.md).

## Requirements

* Python 3.10+
* The model artifacts `models/sentiment_pipeline.pkl` and `models/label_encoder.pkl` must exist (they're committed to the repo, or run `python -m src.train` from the project root to regenerate them)

## Setup

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements_windows.txt   # Windows
# pip install -r requirements.txt         # Linux/macOS
```

## Running the server

From the project root (the app is imported as `app.main:app`, so it must run from there, not from inside `app/`):

```bash
uvicorn app.main:app --reload
```

* API base: `http://127.0.0.1:8000`
* Interactive docs: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`

## Configuration

Settings live in [config.py](config.py) (`pydantic-settings`, loads an optional `.env` in the project root):

| Setting | Value |
|---|---|
| `API_PREFIX` | `/api/v1` |
| `MODEL_DIR` | `<project root>/models` |
| `PIPELINE_PATH` | `models/sentiment_pipeline.pkl` |
| `LABEL_ENCODER_PATH` | `models/label_encoder.pkl` |
| `ALLOWED_ORIGINS` | `http://localhost:3000`, `http://127.0.0.1:3000`, `http://localhost:5173`, `http://127.0.0.1:5173` |

## Structure

```text
app/
├── main.py            # creates the FastAPI app, registers CORS + exception handlers + router
├── config.py           # Settings (API prefix, CORS origins, model paths)
├── exceptions.py        # catch-all handler -> 500 {"detail": "Internal server error"}
├── schemas.py            # PredictionRequest / PredictionResponse pydantic models
└── routers/
    └── sentiment.py       # GET /, GET /health, POST /predict (all under /api/v1)
```

The router itself (`routers/sentiment.py`) delegates prediction to `src.services.SentimentService`, which loads the pickled pipeline once at import time and applies the same `clean_text()` preprocessing used during training (`src/preprocessing.py`).

## Endpoints

All routes are under the `/api/v1` prefix.

### `GET /api/v1/`

```json
{ "application": "Kenyan News Sentiment Analysis API", "version": "1.0.0", "status": "running" }
```

### `GET /api/v1/health`

```json
{ "status": "healthy", "model": "loaded" }
```

### `POST /api/v1/predict`

Request:

```json
{ "headline": "Government unveils affordable housing programme" }
```

Response:

```json
{ "sentiment": "positive" }
```

`headline` is required and must be non-empty; an empty string returns `422 Unprocessable Entity`.

## Tests

From the project root:

```bash
pytest
```

`tests/test_api.py` exercises all three endpoints via FastAPI's `TestClient` against the real pickled pipeline (no mocking).
