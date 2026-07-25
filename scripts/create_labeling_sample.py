"""
Create Manual Labeling Sample
-------------------------------
Pulls a random sample from the master headline dataset for manual
sentiment labeling (ground truth), trying to spread the sample across
sources so no single outlet dominates the manually-labeled set.

Requirements:
    pip install pandas

Usage:
    python create_labeling_sample.py
Run this from the scripts/ folder (expects ../data/processed/master_headlines.csv
and writes to ../data/labeled/manual_labels_sample.csv)
"""

import pandas as pd
import os

INPUT_FILE = "../data/processed/master_headlines.csv"
OUTPUT_DIR = "../data/labeled"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "manual_labels_sample.csv")

SAMPLE_SIZE = 350  # adjust as needed (150-300 is a reasonable range)
RANDOM_SEED = 50   # fixed seed so the sample is reproducible if re-run


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} total headlines from {INPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Try to sample proportionally across sources so no outlet dominates
    # the manually-labeled ground truth set. If a source has fewer rows
    # than its proportional share, just take all of them.
    n_sources = df["source"].nunique()
    per_source_target = max(1, SAMPLE_SIZE // n_sources)

    sampled_parts = []
    for source, group in df.groupby("source"):
        n = min(len(group), per_source_target)
        sampled_parts.append(group.sample(n=n, random_state=RANDOM_SEED))

    sample = pd.concat(sampled_parts, ignore_index=True)

    # If proportional sampling came up short of SAMPLE_SIZE (common when
    # many sources have very few rows), top up randomly from the rest
    if len(sample) < SAMPLE_SIZE:
        remaining = df.drop(sample.index, errors="ignore")
        top_up_n = min(SAMPLE_SIZE - len(sample), len(remaining))
        if top_up_n > 0:
            top_up = remaining.sample(n=top_up_n, random_state=RANDOM_SEED)
            sample = pd.concat([sample, top_up], ignore_index=True)

    # Shuffle so rows aren't grouped by source when labeling
    sample = sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Add empty columns for the actual labeling work
    sample["sentiment"] = ""          # to be filled in manually: positive / negative / neutral
    sample["labeled_by"] = ""         # optional: initials of whoever labels each row
    sample["notes"] = ""              # optional: flag tricky/ambiguous cases here

    sample.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(sample)} headlines for manual labeling: {OUTPUT_FILE}")
    print("\nSource distribution in sample:")
    print(sample["source"].value_counts())


if __name__ == "__main__":
    main()
