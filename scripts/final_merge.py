"""
final_merge.py

Run this LOCALLY, from inside your project folder (wherever semi_final_dataset.csv
and the labeled still_needs_review.csv both live -- move them into the same folder
as this script if they aren't already).

What it does:
- Loads semi_final_dataset.csv        (~1,662 rows: trusted + cardiffnlp + bootstrap)
- Loads still_needs_review.csv        (874 rows, now manually labeled by the team)
- Renames 'manual sentiment' -> 'sentiment' in the reviewed file
- Drops any row where sentiment == 'remove' (case-insensitive, whitespace-trimmed)
- Keeps only the shared columns: source, headline, published, link, sentiment
- Stacks everything into: final_labeled_dataset.csv

Usage:
    python final_merge.py
"""

import pandas as pd
from pathlib import Path

SEMI_FINAL_FILE = "semi_final_dataset.csv"
REVIEWED_FILE = "still_needs_review.csv"
OUTPUT_FILE = "final_labeled_dataset.csv"

OUTPUT_COLUMNS = ["source", "headline", "published", "link", "sentiment"]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find '{path.name}' in this folder. "
            f"Put this script in the same folder as your CSVs."
        )
    return pd.read_csv(path)


def main():
    folder = Path(__file__).parent

    semi_final = load_csv(folder / SEMI_FINAL_FILE)
    reviewed = load_csv(folder / REVIEWED_FILE)

    print(f"semi_final_dataset:    {len(semi_final)} rows")
    print(f"still_needs_review:    {len(reviewed)} rows (before dropping 'remove')")

    # Standardize the reviewed file's label column
    if "manual sentiment" in reviewed.columns:
        reviewed = reviewed.rename(columns={"manual sentiment": "sentiment"})
    elif "sentiment" not in reviewed.columns:
        raise ValueError(
            "Couldn't find a 'manual sentiment' or 'sentiment' column in "
            f"{REVIEWED_FILE}. Check the column names."
        )

    # Clean + drop 'remove' rows
    reviewed["sentiment"] = reviewed["sentiment"].astype(str).str.strip().str.lower()
    remove_count = (reviewed["sentiment"] == "remove").sum()
    reviewed = reviewed[reviewed["sentiment"] != "remove"].copy()

    print(f"Dropped {remove_count} rows flagged 'remove'")
    print(f"still_needs_review after cleanup: {len(reviewed)} rows")

    # Keep only shared columns present in both files
    common_cols = [c for c in OUTPUT_COLUMNS if c in semi_final.columns and c in reviewed.columns]
    missing_semi = set(OUTPUT_COLUMNS) - set(semi_final.columns)
    missing_reviewed = set(OUTPUT_COLUMNS) - set(reviewed.columns)
    if missing_semi:
        print(f"Note: semi_final_dataset is missing columns {missing_semi} (dropped from merge)")
    if missing_reviewed:
        print(f"Note: still_needs_review is missing columns {missing_reviewed} (dropped from merge)")

    merged = pd.concat(
        [semi_final[common_cols], reviewed[common_cols]],
        ignore_index=True,
    )

    dupe_mask = merged.duplicated(subset="headline", keep=False)
    dupes = merged[dupe_mask].sort_values("headline")
    if len(dupes):
        n_dupe_headlines = dupes["headline"].nunique()
        print(f"\nWarning: {n_dupe_headlines} headline(s) appear more than once ({len(dupes)} rows total):")
        for headline, group in dupes.groupby("headline"):
            print(f"\n  \"{headline}\"")
            for _, row in group.iterrows():
                print(f"    -> sentiment: {row['sentiment']}, source: {row.get('source', 'n/a')}")
        dupes.to_csv(folder / "duplicate_headlines_review.csv", index=False)
        print(f"\n  (also saved to duplicate_headlines_review.csv for a closer look)")

    merged.to_csv(folder / OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Total final rows: {len(merged)}")
    print("\nsentiment breakdown:")
    print(merged["sentiment"].value_counts())


if __name__ == "__main__":
    main()