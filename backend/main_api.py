from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from recommender import get_recommendations
from utils import parse_genres, load_poster_path

app = FastAPI()

# allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load data
df = pd.read_csv("../data/raw/anime_data.csv")
df["genres_clean"] = df["genres"].apply(parse_genres)


@app.get("/anime")
def all_anime():
    data = df.to_dict(orient="records")
    for item in data:
        item["poster"] = load_poster_path(item["anilist_id"])
    return data


@app.get("/search")
def search(q: str):
    query = q.lower()
    def match(row):
        name = str(row["title_english"]).lower()
        genres = ", ".join(row["genres_clean"]).lower()
        return query in name or query in genres
    results = df[df.apply(match, axis=1)].copy()
    results["poster"] = results["anilist_id"].apply(load_poster_path)
    return results.to_dict(orient="records")


@app.get("/popular")
def popular():
    pop = df.sort_values("averageScore", ascending=False).head(20).copy()
    pop["poster"] = pop["anilist_id"].apply(load_poster_path)
    return pop.to_dict(orient="records")


@app.get("/details/{anime_id}")
def details(anime_id: int):
    row = df[df["anilist_id"] == anime_id].iloc[0]
    data = row.to_dict()
    data["poster"] = load_poster_path(anime_id)
    return data


@app.get("/recommend/{anime_id}")
def recommend(anime_id: int):
    recs = get_recommendations(anime_id, df)
    recs["poster"] = recs["anilist_id"].apply(load_poster_path)
    return recs.to_dict("records")