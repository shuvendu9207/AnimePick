import re
import numpy as np
import pandas as pd

def get_base_title(title):
    if not isinstance(title, str):
        return ""
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"Final Season.*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"The Final Chapters.*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"Season\s*\d+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"S\d+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"Part\s*\d+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"Special\s*\d*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"Movie\s*\d*", "", title, flags=re.IGNORECASE)
    return title.strip()

class HybridRecommender:
    def __init__(self, cm, cf, alpha=0.7):
        self.cm = cm
        self.cf = cf
        self.df = cm.df.copy()
        self.alpha = alpha

    def recommend(self, anime_index, top_n=10):
        content_scores = self.cm.sim_matrix[anime_index]
        cf_scores = self.cf.encoded

        content_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-9)
        cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-9)

        final_score = self.alpha * content_norm + (1 - self.alpha) * cf_norm

        df = self.df.copy()
        df["hybrid_score"] = final_score
        df = df[df.index != anime_index]

        df["base_title"] = df["title_english"].apply(get_base_title)
        df = df.sort_values("hybrid_score", ascending=False)

        df_unique = df.groupby("base_title", as_index=False).first()
        df_unique = df_unique[df_unique["avg_rating"] >= 7]

        return df_unique.sort_values("hybrid_score", ascending=False).head(top_n)
