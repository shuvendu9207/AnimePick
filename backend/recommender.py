import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# build TF-IDF vectorizer only once
def build_model(df):
    text = df["description"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(text)
    sim = cosine_similarity(matrix)
    return sim

_sim_matrix = None

def get_recommendations(anime_id, df):
    global _sim_matrix
    if _sim_matrix is None:
        _sim_matrix = build_model(df)

    index = df.index[df["anilist_id"] == anime_id][0]
    scores = list(enumerate(_sim_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top = scores[1:11]  # skip itself
    ids = [i for i, _ in top]

    return df.iloc[ids].copy()