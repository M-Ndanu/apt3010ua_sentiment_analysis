from src.preprocessing import (
    normalize_unicode,
    remove_html,
    remove_urls,
    remove_emails,
    expand_contractions,
    strip_wire_prefix,
    strip_teaser_junk,
    strip_source_suffix,
    normalize_whitespace,
    clean_text,
)


def test_unicode_normalization():
    text = "Nairobi–Mombasa"
    assert normalize_unicode(text) == "Nairobi–Mombasa"


def test_html_removal():
    assert remove_html("<b>Breaking News</b>") == " Breaking News "


def test_url_removal():
    text = "Read more at https://citizen.digital"
    assert remove_urls(text) == "Read more at "


def test_email_removal():
    text = "Contact news@citizen.digital today"
    assert remove_emails(text) == "Contact   today"


def test_expand_contractions():
    assert expand_contractions("can't") == "cannot"


def test_strip_wire_prefix():
    assert strip_wire_prefix(
        "Kenya: Fuel prices increase"
    ) == "Fuel prices increase"


def test_strip_teaser_junk():
    text = "News • 4 minutes ago Fuel prices increase"
    assert strip_teaser_junk(text) == "Fuel prices increase"


def test_strip_source_suffix():
    text = "Fuel prices increase - Citizen Digital"
    assert strip_source_suffix(text) == "Fuel prices increase"


def test_whitespace_normalization():
    assert normalize_whitespace(
        "Fuel    prices     increase"
    ) == "Fuel prices increase"


def test_clean_text_returns_string():
    result = clean_text(
        "Kenya: News • 4 minutes ago Fuel prices rise - Citizen Digital"
    )

    assert isinstance(result, str)


def test_clean_text_expected_output():
    text = "Kenya: News • 4 minutes ago Fuel prices rise - Citizen Digital"

    assert clean_text(text) == "fuel price rise"


def test_negation_preserved():
    result = clean_text(
        "Government does not increase taxes"
    )

    assert "not" in result


def test_empty_string():
    assert clean_text("") == ""


def test_none():
    assert clean_text(None) == ""


def test_integer():
    assert clean_text(12345) == ""