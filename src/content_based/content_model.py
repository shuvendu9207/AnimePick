import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentModel:
    def __init__(self):
        self.df = pd.read_csv("data/processed/final_merged.csv")
        self.vectorizer = None
        self.sim_matrix = None

    def build(self):
        corpus = self.df["combined"].fillna("").astype(str).tolist()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )

        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.sim_matrix = cosine_similarity(tfidf_matrix)

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.vectorizer, "models/tfidf_vectorizer.pkl")
        np.save("models/similarity_matrix.npy", self.sim_matrix)

        print("TF-IDF model built.")
        return self.sim_matrix

    def load_saved(self):
        self.vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        self.sim_matrix = np.load("models/similarity_matrix.npy")
        print("Loaded saved TF-IDF content model.")
