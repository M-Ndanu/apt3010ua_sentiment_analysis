import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources once
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

COUNTRY_PREFIXES = [
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
]

PREFIX_PATTERN = r"^(" + "|".join(COUNTRY_PREFIXES) + r"):\s*"

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def strip_wire_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(PREFIX_PATTERN, "", text)


def strip_teaser_junk(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"^(?:[A-Za-z]+\s*[•\-]\s*)?\d+\s*(?:minute|hour|day)s?\s*ago",
        "",
        text,
    )

    text = re.split(
        r"By [A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?",
        text,
    )[0]

    return text.strip()


def strip_source_suffix(text: str) -> str:
    if not isinstance(text, str):
        return ""

    return re.sub(
        r"\s*-\s*(Citizen Digital|The Standard|The Star|NTV Kenya|K24|KBC)\s*$",
        "",
        text,
    )


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = strip_wire_prefix(text)
    text = strip_teaser_junk(text)
    text = strip_source_suffix(text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    tokens = nltk.word_tokenize(text.lower())

    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(tokens)
