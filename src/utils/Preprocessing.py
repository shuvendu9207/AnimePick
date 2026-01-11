import pandas as pd
import os

# INPUT
RAW_ANIME = "data/raw/anime_data.csv"
RAW_RATINGS = "data/raw/ratings.csv"

# OUTPUT
CLEAN_ANIME = "data/processed/anime_data_clean.csv"
CLEAN_RATINGS = "data/processed/ratings_clean.csv"
FINAL_MERGED = "data/processed/final_merged.csv"


def preprocess_anime():
    df = pd.read_csv(RAW_ANIME)

    needed = ["anilist_id", "title_english", "genres", "description"]
    for col in needed:
        if col not in df.columns:
            df[col] = ""

    df["combined"] = (
        df["title_english"].fillna("") + " " +
        df["genres"].fillna("") + " " +
        df["description"].fillna("")
    )

    df.to_csv(CLEAN_ANIME, index=False)
    print("Anime cleaned and saved.")


def preprocess_ratings():
    df = pd.read_csv(RAW_RATINGS)

    # Rename anime_id to anilist_id to match anime_data.csv
    if "anime_id" in df.columns:
        df = df.rename(columns={"anime_id": "anilist_id"})

    # Ensure required columns exist
    if "avg_rating" not in df.columns:
        df["avg_rating"] = 5.0  # default fallback
    
    if "rating_count" not in df.columns:
        df["rating_count"] = 1

    # Clean and validate numeric values
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(5.0)
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce").fillna(1)

    # Clip extreme values
    df["avg_rating"] = df["avg_rating"].clip(1, 10)

    df.to_csv(CLEAN_RATINGS, index=False)
    print("Ratings file standardized and saved.")




def merge_processed_data():
    anime = pd.read_csv(CLEAN_ANIME)
    ratings = pd.read_csv(CLEAN_RATINGS)

    # Direct merge because ratings already has avg_rating + rating_count
    merged = anime.merge(ratings, on="anilist_id", how="left")

    # Fill missing values
    merged["avg_rating"] = merged["avg_rating"].fillna(5.0)
    merged["rating_count"] = merged["rating_count"].fillna(1)

    merged.to_csv(FINAL_MERGED, index=False)
    print("Final merged dataset saved.")


def run_all():
    os.makedirs("data/processed", exist_ok=True)

    preprocess_anime()
    preprocess_ratings()
    merge_processed_data()

    print("All preprocessing completed.")
