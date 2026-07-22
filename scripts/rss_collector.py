import csv
import os
import feedparser
from datetime import datetime, timezone

# Add/remove feeds here 
FEEDS = {
    "Standard - Headlines": "https://www.standardmedia.co.ke/rss/headlines.php",
    "Standard - Kenya News": "https://www.standardmedia.co.ke/rss/kenya.php",
    "Standard - Politics": "https://www.standardmedia.co.ke/rss/politics.php",
    "Standard - Business": "https://www.standardmedia.co.ke/rss/business.php",
    "Business Daily Africa": "https://www.businessdailyafrica.com/service/rss/bd/1939132/feed.rss",
    "Capital FM Kenya": "https://www.capitalfm.co.ke/news/feed/",
    "The East African": "https://www.theeastafrican.co.ke/service/rss/tea/1289142/feed.rss",
    "Kenyans.co.ke": "https://www.kenyans.co.ke/feeds/news?_wrapper_format=html",
    "NTV Kenya": "https://ntvkenya.co.ke/feed/",
    "Citizen Digital (via Google News)": "https://news.google.com/rss/search?q=site:citizen.digital&hl=en-KE&gl=KE&ceid=KE:en",
    "K24": "https://k24.digital/feed",
    "KBC": "https://www.kbc.co.ke/feed/",
    "Kenya News Agency": "https://www.kenyanews.go.ke/feed/",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "raw", "kenyan_headlines.csv")

def load_existing_links(filepath):
    """Return the set of article links already saved, so we don't duplicate."""
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["link"] for row in reader}


def collect():
    existing_links = load_existing_links(OUTPUT_FILE)
    new_rows = []

    for source_name, feed_url in FEEDS.items():
        print(f"Fetching: {source_name} ...")
        parsed = feedparser.parse(feed_url)

        if parsed.bozo:
            # bozo=True usually means the feed URL is wrong/broken/redirected
            print(f"  Warning: could not parse feed cleanly ({parsed.bozo_exception})")

        for entry in parsed.entries:
            link = entry.get("link", "")
            if not link or link in existing_links:
                continue  # skip duplicates or malformed entries

            new_rows.append({
                "source": source_name,
                "headline": entry.get("title", "").strip(),
                "summary": entry.get("summary", "").strip(),
                "published": entry.get("published", ""),
                "link": link,
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })
            existing_links.add(link)

        print(f"  {source_name}: found {len(parsed.entries)} entries in feed")

    # Append to CSV 
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
        fieldnames = ["source", "headline", "summary", "published", "link", "collected_at"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"\nDone. Added {len(new_rows)} new headlines. Total unique so far: {len(existing_links)}")


if __name__ == "__main__":
    collect()