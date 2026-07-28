from src.preprocessing import clean_text


def test_remove_country_prefix():
    text = "Kenya: Fuel prices rise"

    cleaned = clean_text(text)

    assert "kenya" not in cleaned


def test_remove_source_suffix():
    text = "Fuel prices rise - Citizen Digital"

    cleaned = clean_text(text)

    assert "citizen" not in cleaned


def test_lowercase():
    cleaned = clean_text("HELLO WORLD")

    assert cleaned == "hello world"


def test_return_string():
    assert isinstance(clean_text("Hello"), str)