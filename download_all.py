import os
import requests
import pandas as pd
import time
import re

# Create folders
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/posters", exist_ok=True)

# Clean HTML tags from description
def clean_html(text):
    if not text:
        return ""
    text = re.sub('<.*?>', '', text)
    return text.replace("&quot;", '"').replace("&amp;", "&").strip()

# Fetch anime page from AniList
def fetch_anilist_page(page):
    query = """
    query ($page: Int) {
      Page(page: $page, perPage: 50) {
        media(type: ANIME, sort: POPULARITY_DESC) {
          id
          idMal
          title {
            romaji
            english
            native
          }
          description
          episodes
          genres
          popularity
          averageScore
          coverImage {
            extraLarge
            large
          }
        }
      }
    }
    """
    variables = {"page": page}

    for attempt in range(3):  # retry mechanism
        try:
            res = requests.post(
                "https://graphql.anilist.co",
                json={"query": query, "variables": variables},
                timeout=10
            ).json()
            return res["data"]["Page"]["media"]
        except:
            print(f"Retrying page {page}... attempt {attempt+1}")
            time.sleep(1)

    print(f"Failed to fetch page {page}")
    return []

# Download poster by AniList ID
def download_poster(url, anilist_id):
    if not url:
        return None

    filepath = f"data/posters/{anilist_id}.jpg"

    # Skip if exists
    if os.path.exists(filepath):
        return filepath

    try:
        img = requests.get(url, timeout=10).content
        with open(filepath, "wb") as f:
            f.write(img)
        return filepath
    except:
        return None


anime_list = []
target = 1500
per_page = 50
pages = target // per_page

processed = 0
posters_downloaded = 0
posters_failed = 0

print("Fetching AniList anime and downloading posters...\n")

for page in range(1, pages + 1):

    data = fetch_anilist_page(page)
    if not data:
        continue

    for a in data:
        processed += 1

        anilist_id = a["id"]
        cover_url = a["coverImage"].get("extraLarge") or a["coverImage"].get("large")

        poster_path = download_poster(cover_url, anilist_id)
        if poster_path:
            posters_downloaded += 1
        else:
            posters_failed += 1

        anime_list.append({
            "anilist_id": anilist_id,
            "mal_id": a["idMal"],
            "title_romaji": a["title"]["romaji"] or "",
            "title_english": a["title"].get("english") or a["title"]["romaji"],
            "title_native": a["title"].get("native"),
            "description": clean_html(a.get("description")),
            "episodes": a.get("episodes"),
            "genres": a.get("genres"),
            "popularity": a.get("popularity"),
            "averageScore": a.get("averageScore"),
            "coverImage": cover_url
        })

        print(
            f"Processed {processed}/{target} | Posters OK: {posters_downloaded} | Failed: {posters_failed}",
            end="\r"
        )

        if processed >= target:
            break
        time.sleep(0.2)

    if processed >= target:
        break

print("\n\nSaving CSV files...")

# Save anime metadata
df = pd.DataFrame(anime_list)
df.to_csv("data/raw/anime_data.csv", index=False)

# Save ratings
ratings = pd.DataFrame({
    "anime_id": df["anilist_id"],
    "rating_count": df["popularity"],
    "avg_rating": df["averageScore"] / 10
})
ratings.to_csv("data/raw/ratings.csv", index=False)

print("\n=== DOWNLOAD COMPLETE ===")
print(f"Total anime:           {processed}")
print(f"Posters downloaded:    {posters_downloaded}")
print(f"Failed poster downloads: {posters_failed}")
print("Saved: data/raw/anime_data.csv")
print("Saved: data/raw/ratings.csv")