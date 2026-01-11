import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Evaluator:
    def __init__(self, cf_model, content_model):
        self.cf = cf_model
        self.cm = content_model
        self.df = pd.read_csv("data/processed/final_merged.csv")

    def rmse(self):
        preds = self.cf.encoded[:, 0]
        true = self.df["avg_rating"].fillna(0).values
        return np.sqrt(mean_squared_error(true, preds))

    def mae(self):
        preds = self.cf.encoded[:, 0]
        true = self.df["avg_rating"].fillna(0).values
        return mean_absolute_error(true, preds)

    def precision_at_k(self, index, k=10):
        relevant = self.df[self.df["avg_rating"] >= 7.0].index
        predicted = np.argsort(self.cm.sim_matrix[index])[::-1][:k]
        hits = len(set(predicted) & set(relevant))
        return hits / k

    def recall_at_k(self, index, k=10):
        relevant = self.df[self.df["avg_rating"] >= 7.0].index
        if len(relevant) == 0:
            return 0
        predicted = np.argsort(self.cm.sim_matrix[index])[::-1][:k]
        hits = len(set(predicted) & set(relevant))
        return hits / len(relevant)
