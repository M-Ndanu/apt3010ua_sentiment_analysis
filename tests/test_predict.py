from src.predict import predict_sentiment


def test_prediction_returns_string():
    prediction = predict_sentiment(
        "Fuel prices expected to reduce next month"
    )

    assert isinstance(prediction, str)


def test_prediction_is_valid_class():
    prediction = predict_sentiment(
        "Three killed in highway accident"
    )

    assert prediction in [
        "negative",
        "neutral",
        "positive",
    ]