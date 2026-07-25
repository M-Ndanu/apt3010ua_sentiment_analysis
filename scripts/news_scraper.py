
import csv
import os
import re
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "raw", "scraped_headlines.csv")

# Be a polite, identifiable scraper -- some sites block generic/blank user agents
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# Seconds to wait between requests -- avoid hammering the server / getting blocked
REQUEST_DELAY = 2

# Site configuration
# Each site has:
#   - "pages": list of section/category URLs to scrape (breadth strategy)
#   - "article_pattern": regex that a real article URL matches (filters out
#     nav links, ads, category links, etc.)
SITES = {
    "The Star": {
        "pages": [
            "https://www.the-star.co.ke/news/world",
            "https://www.the-star.co.ke/news/africa",
            "https://www.the-star.co.ke/news/corridors-of-power",
            "https://www.the-star.co.ke/news/big-read",
            "https://www.the-star.co.ke/news/news-brief",
            "https://www.the-star.co.ke/counties/rift-valley",
            "https://www.the-star.co.ke/counties/nairobi",
            "https://www.the-star.co.ke/counties/north-eastern",
            "https://www.the-star.co.ke/counties/coast",
            "https://www.the-star.co.ke/counties/central",
            "https://www.the-star.co.ke/counties/nyanza",
            "https://www.the-star.co.ke/counties/western",
            "https://www.the-star.co.ke/counties/eastern",
            "https://www.the-star.co.ke/business/kenya",
            "https://www.the-star.co.ke/business/markets",
            "https://www.the-star.co.ke/business/commentary",
            "https://www.the-star.co.ke/opinion/columnists",
            "https://www.the-star.co.ke/opinion/leader",
            "https://www.the-star.co.ke/sports/football",
            "https://www.the-star.co.ke/sports/athletics",
            "https://www.the-star.co.ke/sports/rugby",
            "https://www.the-star.co.ke/sasa/lifestyle",
            "https://www.the-star.co.ke/sasa/technology",
            "https://www.the-star.co.ke/sasa/entertainment",
            "https://www.the-star.co.ke/sasa/society",
            "https://www.the-star.co.ke/health",
            "https://www.the-star.co.ke/siasa",
            "https://www.the-star.co.ke/climate-change",
        ],
    
        "article_pattern": re.compile(r"/\d{4}-\d{2}-\d{2}-[a-z0-9\-]+"),
        "paginate": None,  # no clean pagination found -- rely on breadth instead
    },
    "NTV Kenya": {
        # Base category pages 
        "pages": [
            "https://ntvkenya.co.ke/newsfeatures/",
            "https://ntvkenya.co.ke/politics/",
            "https://ntvkenya.co.ke/business/",
            "https://ntvkenya.co.ke/ntvsports/",
            "https://ntvkenya.co.ke/news/"
        ],
        # NTV article URLs look like: ntvkenya.co.ke/news/some-headline-slug/
        "article_pattern": re.compile(
            r"ntvkenya\.co\.ke/(news|politics|business|entertainment|ntvsports)/[a-z0-9\-]+/?$"
        ),
        # NTV has real pagination: /news/page/2/, /news/page/3/, etc.
        # Scrape first N pages per category -- plenty for volume without overdoing it
        "paginate": 15,
    },
    "Citizen Digital": {
        "pages": [
            "https://citizen.digital/",
            "https://citizen.digital/news",
            "https://citizen.digital/news/tech/26",
            "https://citizen.digital/news/sports/7",
            "https://citizen.digital/news/entertainment/4",
            "https://citizen.digital/news/business/3",
            "https://citizen.digital/news/wananchi-reporting/19",
            "https://citizen.digital/news/lifestyle/8"
        ],
        # Citizen article URLs look like: /article/some-headline-slug-n386709
        "article_pattern": re.compile(r"/article/[a-z0-9\-]+-n\d+"),
        "paginate": None,  # no confirmed pagination pattern -- rely on breadth
    },
    "The Standard": {
        "pages": [
            "https://www.standardmedia.co.ke/category/588/national",
            "https://www.standardmedia.co.ke/category/1/counties",
            "https://www.standardmedia.co.ke/category/3/politics",
            "https://www.standardmedia.co.ke/business",
            "https://www.standardmedia.co.ke/category/5/world",
            "https://www.standardmedia.co.ke/health",
            "https://www.standardmedia.co.ke/sports",
            "https://www.standardmedia.co.ke/entertainment",
            "https://www.standardmedia.co.ke/category/56/education",
            "https://www.standardmedia.co.ke/category/63/environment",
            "https://www.standardmedia.co.ke/category/7/opinion",
            "https://www.standardmedia.co.ke/category/14/nairobi",
        ],
        # Standard article URLs look like: /national/article/2001553615/slug-text
        "article_pattern": re.compile(r"/article/\d+/[a-z0-9\-]+"),
        "paginate": None,  # rely on breadth across many category pages
    },

    "KTN News": {
        "pages": [
            "https://www.standardmedia.co.ke/ktnnews/category/1/ktn-news",
            "https://www.standardmedia.co.ke/ktnnews/category/22/business",
            "https://www.standardmedia.co.ke/ktnnews/category/23/sports",
            "https://www.standardmedia.co.ke/ktnnews/category/34/checkpoint",
            "https://www.standardmedia.co.ke/ktnnews/category/13/news-features",
            "https://www.standardmedia.co.ke/ktnnews/category/28/morning-express",
            "https://www.standardmedia.co.ke/ktnnews/category/2/ktn-leo",
            "https://www.standardmedia.co.ke/ktnnews/category/134/leo-mashinani",
            "https://www.standardmedia.co.ke/ktnnews/category/162/the-big-story",
        ],
        # Same underlying site as Standard, so same article URL pattern
        "article_pattern": re.compile(r"/article/\d+/[a-z0-9\-]+"),
        "paginate": None,
    },
}


def is_article_link(href, pattern):
    """Check whether a link's href matches the article URL pattern for a site."""
    if not href:
        return False
    return bool(pattern.search(href))


def load_existing_links(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["link"] for row in reader}


def scrape_page(url, base_domain, article_pattern):
    """Fetch one page and return a list of (headline, link) tuples."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    seen_on_page = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if not is_article_link(href, article_pattern):
            continue

        # Normalize relative URLs to absolute
        if href.startswith("/"):
            href = base_domain + href
        elif not href.startswith("http"):
            continue

        headline = a_tag.get_text(strip=True)
        # Skip empty text (e.g. image-only links) and obvious junk
        if not headline or len(headline) < 15:
            continue

        if href in seen_on_page:
            continue
        seen_on_page.add(href)
        results.append((headline, href))

    return results


def build_page_list(config):
    """Expand base 'pages' into full list of URLs to fetch, adding
    pagination URLs (e.g. /page/2/, /page/3/) if configured."""
    base_pages = config["pages"]
    paginate = config.get("paginate")

    if not paginate:
        return base_pages

    all_pages = list(base_pages)  # page 1 of each category
    for base_url in base_pages:
        base_url = base_url.rstrip("/")
        for page_num in range(2, paginate + 1):
            all_pages.append(f"{base_url}/page/{page_num}/")
    return all_pages


def scrape_all():
    existing_links = load_existing_links(OUTPUT_FILE)
    new_rows = []

    for source_name, config in SITES.items():
        page_list = build_page_list(config)
        base_domain = "https://" + config["pages"][0].split("/")[2]
        print(f"\n=== Scraping {source_name} ({len(page_list)} pages) ===")

        for page_url in page_list:
            print(f"Fetching: {page_url}")
            found = scrape_page(page_url, base_domain, config["article_pattern"])
            print(f"  Found {len(found)} article links on this page")

            for headline, link in found:
                if link in existing_links:
                    continue
                new_rows.append({
                    "source": source_name,
                    "headline": headline,
                    "summary": "",  # not available from listing pages
                    "published": "",  # not reliably available without visiting each article
                    "link": link,
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                })
                existing_links.add(link)

            time.sleep(REQUEST_DELAY)  # be polite between requests

    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
        fieldnames = ["source", "headline", "summary", "published", "link", "collected_at"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    print(f"\nDone. Added {len(new_rows)} new headlines. Total unique so far: {len(existing_links)}")


if __name__ == "__main__":
    scrape_all()
