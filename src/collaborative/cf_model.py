import pandas as pd
import numpy as np
import os

class CollaborativeModel:
    def __init__(self):
        self.df = pd.read_csv("data/processed/final_merged.csv")
        self.encoded = None

    def build(self):
        df = self.df.copy()

        df["log_count"] = np.log1p(df["rating_count"])

        r = df["avg_rating"]
        c = df["rating_count"]
        l = df["log_count"]

        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)
        c_norm = (c - c.min()) / (c.max() - c.min() + 1e-9)
        l_norm = (l - l.min()) / (l.max() - l.min() + 1e-9)

        self.encoded = (0.6 * r_norm) + (0.3 * c_norm) + (0.1 * l_norm)
        self.encoded = self.encoded.values

        os.makedirs("model", exist_ok=True)
        np.save("model/cf_features.npy", self.encoded)

        print("Popularity-based model built and saved.")

    def load_saved(self):
        self.encoded = np.load("model/cf_features.npy")
        print("Loaded popularity-based CF features.")
