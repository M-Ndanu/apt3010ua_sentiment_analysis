import pandas as pd
import glob
import os
import re

RAW_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "master_headlines.csv")

# The final lean schema -- extra columns
# get dropped if present, since they're not needed for labeling/training
KEEP_COLUMNS = ["source", "headline", "published", "link"]

SYNTHETIC_PATTERN = re.compile(r"\bvariant\s*\d+\b", re.IGNORECASE)

#remove AI generated content from the dataset
def is_synthetic(text):
    """Flag likely placeholder/auto-generated rows, e.g. 'Maize subsidies variant 905'."""
    if not isinstance(text, str):
        return False
    return bool(SYNTHETIC_PATTERN.search(text))


def normalize_headline(text):
    """Lowercase, strip punctuation/extra whitespace -- for near-duplicate matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_all_csvs(raw_dir):
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. "
                                f"Put your raw headline CSVs there first.")

    frames = []
    for path in csv_files:
        print(f"Loading: {path}")
        df = pd.read_csv(path)

        
        # try to  map common variants to our standard schema
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ("title", "headline", "news_headline"):
                rename_map[col] = "headline"
            elif col_lower in ("source", "outlet", "site"):
                rename_map[col] = "source"
            elif col_lower in ("published", "pubdate", "date", "published_at"):
                rename_map[col] = "published"
            elif col_lower in ("link", "url"):
                rename_map[col] = "link"
        df = df.rename(columns=rename_map)

        # Ensure all expected columns exist, even if empty, so concat works cleanly
        for col in KEEP_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[KEEP_COLUMNS]
        df["_source_file"] = os.path.basename(path)  # track provenance while debugging
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean_and_dedup(df):
    before = len(df)

    # Drop rows with no headline text at all -- useless for labeling
    df = df[df["headline"].notna() & (df["headline"].str.strip() != "")]

    # Drop likely synthetic/placeholder rows (e.g. "Maize subsidies variant 905")
    synthetic_mask = df["headline"].apply(is_synthetic)
    num_synthetic = synthetic_mask.sum()
    if num_synthetic > 0:
        print(f"\nWarning: found {num_synthetic} likely synthetic/placeholder rows "
            f"(matching pattern like 'variant 123') -- excluding them.")
        df = df[~synthetic_mask]

    #drop exact duplicate links (skip blank links, can't compare those)
    has_link = df["link"].notna() & (df["link"].str.strip() != "")
    with_link = df[has_link].drop_duplicates(subset="link", keep="first")
    without_link = df[~has_link]
    df = pd.concat([with_link, without_link], ignore_index=True)
    after_link_dedup = len(df)

    #drop near-identical headlines (normalized comparison)
    df["_normalized"] = df["headline"].apply(normalize_headline)
    df = df.drop_duplicates(subset="_normalized", keep="first")
    df = df.drop(columns=["_normalized"])
    after_headline_dedup = len(df)

    print(f"\nRows before cleaning: {before}")
    print(f"After dropping empty headlines: {len(df) + (before - after_link_dedup)}")
    print(f"After link-based dedup: {after_link_dedup}")
    print(f"After headline-based dedup: {after_headline_dedup}")
    print(f"Total duplicates removed: {before - after_headline_dedup}")

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined = load_all_csvs(RAW_DIR)
    cleaned = clean_and_dedup(combined)
    cleaned = cleaned.drop(columns=["_source_file"])  # drop provenance col from final output
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved master dataset: {OUTPUT_FILE} ({len(cleaned)} unique headlines)")


if __name__ == "__main__":
    main()
