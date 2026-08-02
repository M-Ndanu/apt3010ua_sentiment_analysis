"""
Text preprocessing utilities for the Kenyan News Sentiment Analysis project.

This module performs deterministic preprocessing of news headlines before
feature extraction using TF-IDF. The same preprocessing pipeline is reused
during training, evaluation, and inference to ensure consistency.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Final

import contractions
import nltk

from nltk.corpus import stopwords
from nltk.data import find
from nltk.stem import WordNetLemmatizer

# ============================================================================
# NLTK Resource Management
# ============================================================================


def ensure_nltk_resource(resource_path: str, package: str) -> None:
    """
    Ensure an NLTK resource exists.

    If the resource is missing it is downloaded once.
    """

    try:
        find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True)


ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
ensure_nltk_resource("corpora/stopwords", "stopwords")
ensure_nltk_resource("corpora/wordnet", "wordnet")

# ============================================================================
# Constants
# ============================================================================

COUNTRY_PREFIXES: Final[tuple[str, ...]] = (
    "Kenya",
    "Uganda",
    "Tanzania",
    "Africa",
    "Rwanda",
    "Burundi",
    "Ethiopia",
    "Somalia",
    "Nigeria",
    "Ghana",
    "Zambia",
    "Zimbabwe",
)

SOURCE_NAMES: Final[tuple[str, ...]] = (
    "Citizen Digital",
    "The Standard",
    "The Star",
    "NTV Kenya",
    "K24",
    "KBC",
)

NEGATION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "no",
        "not",
        "nor",
        "never",
    }
)

STOP_WORDS: Final[frozenset[str]] = frozenset(
    set(stopwords.words("english")) - NEGATION_WORDS
)

LEMMATIZER: Final = WordNetLemmatizer()

# ============================================================================
# Compiled Regular Expressions
# ============================================================================

WIRE_PREFIX_REGEX = re.compile(
    rf"^({'|'.join(map(re.escape, COUNTRY_PREFIXES))}):\s*",
    flags=re.IGNORECASE,
)

TEASER_REGEX = re.compile(
    r"^(?:[A-Za-z]+\s*[•\-]\s*)?"
    r"\d+\s*"
    r"(?:minute|hour|day)s?\s*ago\s*",
    flags=re.IGNORECASE,
)

BYLINE_REGEX = re.compile(
    r"By\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?"
)

SOURCE_SUFFIX_REGEX = re.compile(
    rf"\s*-\s*({'|'.join(map(re.escape, SOURCE_NAMES))})\s*$",
    flags=re.IGNORECASE,
)

HTML_REGEX = re.compile(r"<.*?>")

URL_REGEX = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

EMAIL_REGEX = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

NON_ALPHA_REGEX = re.compile(r"[^a-z\s]+")

WHITESPACE_REGEX = re.compile(r"\s+")

# ============================================================================
# Helper Functions
# ============================================================================


def _ensure_string(text: object) -> str:
    """Return a valid string."""

    return text if isinstance(text, str) else ""


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters."""

    return unicodedata.normalize("NFKC", text)


def remove_html(text: str) -> str:
    """Remove HTML tags."""

    text = html.unescape(text)
    return HTML_REGEX.sub(" ", text)


def remove_urls(text: str) -> str:
    """Remove URLs."""

    return URL_REGEX.sub(" ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses."""

    return EMAIL_REGEX.sub(" ", text)


def expand_contractions(text: str) -> str:
    """Expand English contractions."""

    return contractions.fix(text)


def strip_wire_prefix(text: str) -> str:
    """Remove leading country prefixes."""

    return WIRE_PREFIX_REGEX.sub("", text)


def strip_teaser_junk(text: str) -> str:
    """Remove teaser timestamps and bylines."""

    text = TEASER_REGEX.sub("", text)
    text = BYLINE_REGEX.split(text)[0]

    return text.strip()


def strip_source_suffix(text: str) -> str:
    """Remove trailing publisher names."""

    return SOURCE_SUFFIX_REGEX.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""

    return WHITESPACE_REGEX.sub(" ", text).strip()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "clean_text",
]


def clean_text(text: object) -> str:
    """
    Clean a news headline for sentiment classification.

    Pipeline
    --------
    1. Unicode normalization
    2. HTML decoding and removal
    3. URL removal
    4. Email removal
    5. Contraction expansion
    6. Wire prefix removal
    7. Timestamp/byline removal
    8. Publisher suffix removal
    9. Unicode-aware lowercase conversion
    10. Remove numbers and punctuation
    11. Tokenization
    12. Stopword removal (preserving negations)
    13. Lemmatization
    14. Whitespace normalization
    """

    text = _ensure_string(text)

    if not text:
        return ""

    text = normalize_unicode(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = expand_contractions(text)
    text = strip_wire_prefix(text)
    text = strip_teaser_junk(text)
    text = strip_source_suffix(text)

    text = text.casefold()

    text = NON_ALPHA_REGEX.sub(" ", text)

    tokens = nltk.word_tokenize(text)

    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if len(token) > 1 and token not in STOP_WORDS
    ]

    return normalize_whitespace(" ".join(cleaned_tokens))