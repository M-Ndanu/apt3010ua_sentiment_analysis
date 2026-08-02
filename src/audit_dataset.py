"""
Dataset auditing tool.

This module performs deterministic rule-based auditing of the
labelled Kenyan News Sentiment dataset.

The auditor NEVER changes labels automatically.
It only identifies suspicious samples for manual review.

Pipeline

Dataset
    ↓
Normalization
    ↓
Tokenization
    ↓
Domain Detection
    ↓
Event Detection
    ↓
Sentiment Inference
    ↓
Human Label Comparison
    ↓
CSV Audit Report
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass
from collections import defaultdict

import pandas as pd

from app.config import settings

from src.dataset import (
    load_dataset,
    prepare_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Event Definition
# ==========================================================


@dataclass(frozen=True)
class Event:

    """
    Represents one semantic event.
    """

    category: str

    name: str

    sentiment: str

    confidence: float

    required: tuple[str, ...]

    optional: tuple[str, ...] = ()


# ==========================================================
# Sentiment Priority
# ==========================================================

SENTIMENT_PRIORITY = {

    "negative": 3,

    "positive": 2,

    "neutral": 1,

}

# ==========================================================
# Event Knowledge Base
# ==========================================================

EVENTS: list[Event] = [

    # ======================================================
    # ECONOMY
    # ======================================================

    Event(
        category="Economy",
        name="Fuel Price Increase",
        sentiment="negative",
        confidence=0.99,
        required=("fuel",),
        optional=(
            "increase",
            "increases",
            "increased",
            "rise",
            "rises",
            "hike",
            "higher",
        ),
    ),

    Event(
        category="Economy",
        name="Fuel Price Reduction",
        sentiment="positive",
        confidence=0.99,
        required=("fuel",),
        optional=(
            "reduce",
            "reduced",
            "reduces",
            "drop",
            "drops",
            "fall",
            "falls",
            "lower",
        ),
    ),

    Event(
        category="Economy",
        name="Inflation Increase",
        sentiment="negative",
        confidence=0.98,
        required=("inflation",),
        optional=(
            "increase",
            "rise",
            "higher",
        ),
    ),

    Event(
        category="Economy",
        name="Inflation Reduction",
        sentiment="positive",
        confidence=0.98,
        required=("inflation",),
        optional=(
            "drop",
            "decline",
            "falls",
            "reduced",
        ),
    ),

    Event(
        category="Economy",
        name="Economic Growth",
        sentiment="positive",
        confidence=0.95,
        required=("economy",),
        optional=(
            "growth",
            "grow",
            "expands",
            "improves",
        ),
    ),

    Event(
        category="Economy",
        name="Economic Decline",
        sentiment="negative",
        confidence=0.95,
        required=("economy",),
        optional=(
            "recession",
            "decline",
            "slowdown",
            "collapse",
        ),
    ),

    # ======================================================
    # TRANSPORT
    # ======================================================

    Event(
        category="Transport",
        name="Road Fatality",
        sentiment="negative",
        confidence=0.99,
        required=("accident",),
        optional=(
            "killed",
            "dead",
            "dies",
            "fatal",
            "crash",
            "collision",
        ),
    ),

    Event(
        category="Transport",
        name="Road Injury",
        sentiment="negative",
        confidence=0.97,
        required=("accident",),
        optional=(
            "injured",
            "hospitalised",
            "hurt",
        ),
    ),

    Event(
        category="Transport",
        name="Road Safety",
        sentiment="positive",
        confidence=0.92,
        required=("road",),
        optional=(
            "completed",
            "opened",
            "expanded",
            "upgrade",
        ),
    ),

    # ======================================================
    # CRIME
    # ======================================================

    Event(
        category="Crime",
        name="Murder",
        sentiment="negative",
        confidence=0.99,
        required=(),
        optional=(
            "murder",
            "killed",
            "shot",
            "stabbed",
            "dead",
        ),
    ),

    Event(
        category="Crime",
        name="Fraud",
        sentiment="negative",
        confidence=0.96,
        required=(),
        optional=(
            "fraud",
            "corruption",
            "bribery",
            "graft",
            "embezzlement",
        ),
    ),

    # ======================================================
    # HEALTH
    # ======================================================

    Event(
        category="Health",
        name="Disease Outbreak",
        sentiment="negative",
        confidence=0.98,
        required=(),
        optional=(
            "cholera",
            "malaria",
            "covid",
            "outbreak",
            "virus",
        ),
    ),

    Event(
        category="Health",
        name="Vaccination",
        sentiment="positive",
        confidence=0.93,
        required=(),
        optional=(
            "vaccination",
            "vaccinated",
            "immunisation",
        ),
    ),

    # ======================================================
    # SPORTS
    # ======================================================

    Event(
        category="Sports",
        name="Victory",
        sentiment="positive",
        confidence=0.96,
        required=(),
        optional=(
            "wins",
            "won",
            "beats",
            "victory",
            "champion",
            "qualified",
        ),
    ),

    Event(
        category="Sports",
        name="Defeat",
        sentiment="negative",
        confidence=0.96,
        required=(),
        optional=(
            "loses",
            "lost",
            "eliminated",
            "knocked",
        ),
    ),

    # ======================================================
    # GOVERNMENT
    # ======================================================

    Event(
        category="Government",
        name="Administrative Action",
        sentiment="neutral",
        confidence=0.90,
        required=(),
        optional=(
            "appoints",
            "appointed",
            "announces",
            "meeting",
            "committee",
            "summit",
            "conference",
            "gazette",
            "parliament",
            "cabinet",
        ),
    ),

]


# ==========================================================
# Tokenization
# ==========================================================

TOKEN_PATTERN = re.compile(r"[a-z]+")


def normalize_text(text: str) -> str:
    """
    Normalize headline text.
    """

    return (
        text.lower()
        .replace("-", " ")
        .replace("/", " ")
        .replace("'", "")
    )


def tokenize(text: str) -> list[str]:
    """
    Convert headline into normalized tokens.

    Example
    -------
    Twin killed, brother injured

    becomes

    ["twin", "killed", "brother", "injured"]
    """

    text = normalize_text(text)

    return TOKEN_PATTERN.findall(text)


# ==========================================================
# Event Detection
# ==========================================================

def detect_events(headline: str) -> list[Event]:
    """
    Detect semantic events from a headline.

    Matching is performed using WHOLE TOKENS only.

    A required keyword MUST exist.

    If optional keywords are supplied,
    at least one optional keyword must also exist.
    """

    tokens = set(
        tokenize(headline)
    )

    detected: list[Event] = []

    for event in EVENTS:

        # --------------------------------------------------
        # Required keywords
        # --------------------------------------------------

        if event.required:

            if not all(
                keyword in tokens
                for keyword in event.required
            ):
                continue

        # --------------------------------------------------
        # Optional keywords
        # --------------------------------------------------

        if event.optional:

            if not any(
                keyword in tokens
                for keyword in event.optional
            ):
                continue

        detected.append(event)

    return detected


# ==========================================================
# Human-readable event summary
# ==========================================================

def build_match_string(
    events: list[Event],
) -> str:
    """
    Convert detected events into a readable string.
    """

    return ", ".join(
        f"{event.category}: {event.name}"
        for event in events
    )


# ==========================================================
# Domain Statistics
# ==========================================================

def summarize_domains(
    events: list[Event],
) -> dict[str, int]:
    """
    Count events detected per domain.
    """

    counts: dict[str, int] = defaultdict(int)

    for event in events:

        counts[event.category] += 1

    return dict(counts)





# ==========================================================
# Sentiment Inference Engine
# ==========================================================

MIN_CONFIDENCE = 0.90


def score_headline(
    headline: str,
) -> tuple[str, float, str]:
    """
    Infer the sentiment of a headline from detected events.

    Returns
    -------
    tuple
        (
            predicted_sentiment,
            confidence,
            matched_rule,
        )
    """

    events = detect_events(
        headline,
    )

    if not events:
        return (
            "unknown",
            0.0,
            "",
        )

    # ------------------------------------------------------
    # Domain events override administrative announcements.
    #
    # Example:
    #
    # Government announces fuel price increase
    #
    # should be NEGATIVE rather than NEUTRAL.
    # ------------------------------------------------------

    domain_events = [

        event

        for event in events

        if event.category != "Government"

    ]

    if domain_events:
        events = domain_events

    # ------------------------------------------------------
    # Score sentiments
    # ------------------------------------------------------

    sentiment_scores = {
        "positive": 0.0,
        "negative": 0.0,
        "neutral": 0.0,
    }

    for event in events:

        sentiment_scores[
            event.sentiment
        ] += event.confidence

    # ------------------------------------------------------
    # Resolve ties
    # ------------------------------------------------------

    ordered = sorted(

        sentiment_scores.items(),

        key=lambda item: (
            item[1],
            SENTIMENT_PRIORITY[item[0]],
        ),

        reverse=True,

    )

    predicted = ordered[0][0]

    confidence = ordered[0][1]

    # ------------------------------------------------------
    # Explanation
    # ------------------------------------------------------

    explanation = build_match_string(
        events,
    )

    return (
        predicted,
        confidence,
        explanation,
    )



# ==========================================================
# Audit
# ==========================================================

def audit_dataset() -> None:
    """
    Audit the labelled dataset.

    This tool NEVER changes labels.
    It only reports high-confidence disagreements.
    """

    logger.info("Loading dataset...")

    df = load_dataset(
        settings.DATASET_PATH,
    )

    df = prepare_dataset(df)

    findings: list[dict] = []

    for _, row in df.iterrows():

        headline = row["headline"]
        human_label = row["sentiment"]

        predicted, confidence, rule = score_headline(
            headline,
        )

        if predicted == "unknown":
            continue

        if confidence < MIN_CONFIDENCE:
            continue

        if predicted == human_label:
            continue

        findings.append(
            {
                "headline": headline,
                "human_label": human_label,
                "predicted_label": predicted,
                "confidence": round(
                    confidence,
                    2,
                ),
                "matched_rule": rule,
            }
        )

    report = pd.DataFrame(
        findings,
    )

    report = report.sort_values(
        by="confidence",
        ascending=False,
    )

    report.to_csv(
        settings.AUDIT_REPORT_PATH,
        index=False,
    )

    logger.info(
        "Flagged %d possible label errors.",
        len(report),
    )

    logger.info(
        "Audit report saved to %s",
        settings.AUDIT_REPORT_PATH,
    )


def main() -> None:
    audit_dataset()


if __name__ == "__main__":
    main()