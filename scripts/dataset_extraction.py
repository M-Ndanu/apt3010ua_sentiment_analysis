import feedparser
import pandas as pd
import requests
import time
from datetime import datetime
# from google.colab import files

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "raw", "headlines_kenya.csv")

print(" Setup done!\n")


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

RSS_FEEDS = {
    "Capital FM": "https://www.capitalfm.co.ke/news/feed/",
    "KBC": "https://www.kbc.co.ke/feed/",
    "The Standard": "https://www.standardmedia.co.ke/rss/kenya.php",
    "The Standard Headlines": "https://www.standardmedia.co.ke/rss/headlines.php",
    "Business Daily": "https://www.businessdailyafrica.com/service/rss/bd/1939132/feed.rss",
    "People Daily": "https://www.pd.co.ke/feed/",
    "Nairobi Wire": "https://nairobiwire.com/feed",
    "Kahawa Tungu": "https://www.kahawatungu.com/feed/",
    "Kenyan Wallstreet": "https://kenyanwallstreet.com/feed/",
    "Citizen Digital": "https://citizentv.co.ke/feed/",
    "Tuko": "https://www.tuko.co.ke/rss/",
    "Kenya News Agency": "https://www.kenyanews.go.ke/feed/",
}
WORLD_NEWS_API_KEY = ""
NEWSDATA_IO_API_KEY = ""
CURRENTS_API_KEY = ""

try:
    from google.colab import userdata
    if not WORLD_NEWS_API_KEY: WORLD_NEWS_API_KEY = userdata.get('WORLD_NEWS_API_KEY')
    if not NEWSDATA_IO_API_KEY: NEWSDATA_IO_API_KEY = userdata.get('NEWSDATA_IO_API_KEY')
    if not CURRENTS_API_KEY: CURRENTS_API_KEY = userdata.get('CURRENTS_API_KEY')
except:
    pass

def fetch_rss():
    print(" Fetching RSS Feeds (Higher Volume)...")
    dataset = []
    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            added = 0
            for entry in feed.entries[:50]:
                title = entry.get('title', '').strip()
                if len(title) < 15: continue
                dataset.append({
                    "headline": title,
                    "source": name,
                    "link": entry.get('link', ''),
                    "published": entry.get('published', entry.get('pubDate', '')),
                    "fetch_method": "RSS"
                })
                added += 1
            print(f"    {name} → {added}")
            time.sleep(0.7)
        except:
            print(f"   ✗ {name}")
    return dataset

def fetch_worldnews(key):
    if not key: return []
    print(" World News API (max 100)...")
    try:
        r = requests.get("https://api.worldnewsapi.com/search-news",
            params={"api-key": key, "text": "Kenya", "language": "en", "number": 100},
            headers=HEADERS, timeout=20)
        if r.status_code == 200:
            articles = r.json().get("news", [])
            print(f"   ✓ {len(articles)} from World News API")
            return [{
                "headline": a.get("title"), "source": a.get("source", "World News API"),
                "link": a.get("url"), "published": a.get("published_date"), "fetch_method": "API"
            } for a in articles if a.get("title")]
    except: pass
    return []

def fetch_newsdata(key):
    if not key: return []
    print(" NewsData.io (multiple pages)...")
    dataset = []
    next_page = None
    pages = 0
    while pages < 8:
        try:
            params = {"apikey": key, "q": "Kenya", "language": "en"}
            if next_page:
                params["page"] = next_page
            r = requests.get("https://newsdata.io/api/1/news",
                params=params,
                headers=HEADERS, timeout=15)
            if r.status_code != 200: break
            data = r.json()
            articles = data.get("results", [])
            for a in articles:
                if a.get("title"):
                    dataset.append({
                        "headline": a.get("title"),
                        "source": a.get("source_name", "NewsData.io"),
                        "link": a.get("link"),
                        "published": a.get("pubDate"),
                        "fetch_method": "API"
                    })
            next_page = data.get("nextPage")
            pages += 1
            print(f"   Page {pages}: {len(articles)} articles")
            if not next_page: break
            time.sleep(1.2)
        except: break
    return dataset

def fetch_currents(key):
    if not key: return []
    print(" Currents API...")
    try:
        r = requests.get("https://api.currentsapi.services/v1/search",
            params={"apiKey": key, "language": "en", "country": "KE", "page_size": 50},
            headers=HEADERS, timeout=15)
        print(f"   Currents API Status Code: {r.status_code}")
        if r.status_code == 200:
            articles = r.json().get("news", [])
            print(f"   ✓ {len(articles)} articles from Currents API (raw)")
            valid_articles = [{
                "headline": a.get("title"), "source": a.get("author", "Currents API"),
                "link": a.get("url"), "published": a.get("published"), "fetch_method": "API"
            } for a in articles if a.get("title")]
            print(f"   ✓ {len(valid_articles)} articles from Currents API (after title check)")
            return valid_articles
        else:
            print(f"   ✗ Currents API Error: {r.text}")
    except Exception as e:
        print(f"   ✗ Currents API Exception: {e}")
    return []


print(" Starting High-Volume Kenyan News Collection...\n")

master = fetch_rss()
master.extend(fetch_worldnews(WORLD_NEWS_API_KEY))
master.extend(fetch_newsdata(NEWSDATA_IO_API_KEY))
master.extend(fetch_currents(CURRENTS_API_KEY))

df = pd.DataFrame(master)
df = df.dropna(subset=['headline'])
df['clean'] = df['headline'].str.lower().str.strip()
df = df.drop_duplicates(subset=['clean', 'source']).drop(columns=['clean'])
df = df.sort_values(by=['source', 'headline']).reset_index(drop=True)


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "="*70)
print(" COLLECTION COMPLETE!")
print(f"Total unique records: {len(df)}")
print(f"Saved as: {OUTPUT_FILE}")
print("="*70)
print("\nTop Sources:")
print(df['source'].value_counts().head(12))

# files.download(filename)
