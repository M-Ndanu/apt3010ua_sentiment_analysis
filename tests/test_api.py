from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/api/v1/")

    assert response.status_code == 200


def test_health():
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_predict():
    response = client.post(
        "/api/v1/predict",
        json={
            "headline":
            "Government launches affordable housing programme"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "sentiment" in data

    assert data["sentiment"] in [
        "negative",
        "neutral",
        "positive",
    ]


def test_blank_headline():
    response = client.post(
        "/api/v1/predict",
        json={
            "headline": ""
        },
    )

    assert response.status_code == 422