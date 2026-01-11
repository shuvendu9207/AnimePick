import pandas as pd
from sentence_transformers import SentenceTransformer, util

class SemanticSearch:
    def __init__(self, content_model):
        self.df = content_model.df
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(self.df["combined"].tolist(), convert_to_tensor=True)

    def search_by_text(self, query, top_n=10):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_idx = scores.topk(top_n).indices.tolist()
        return self.df.iloc[top_idx][["title_english", "avg_rating", "genres", "description"]]
