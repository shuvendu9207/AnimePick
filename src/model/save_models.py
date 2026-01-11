import joblib
import numpy as np
from src.content_based.content_model import ContentModel
from src.collaborative.cf_model import CollaborativeModel

def save_all_models():
    print("Saving models...")

    cm = ContentModel()
    cm.build()

    cf = CollaborativeModel()
    cf.build()

    joblib.dump(cm.vectorizer, "model/tfidf_vectorizer.pkl")
    joblib.dump(cm.sim_matrix, "model/content_sim.pkl")
    np.save("model/cf_features.npy", cf.encoded)

    print("All models saved successfully.")
