import os
import sys
import shutil

def clear_pycache():
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# Add src folder to Python path so imports work cleanly
sys.path.append("src")

from utils.Preprocessing import run_all
from content_based.content_model import ContentModel
from collaborative.cf_model import CollaborativeModel
from hybrid.hybrid_recommender import HybridRecommender
from evaluation.evaluator import Evaluator
from model import save_all_models


def file_exists(path):
    return os.path.exists(path)


def main():
    print("PIPELINE STARTED\n")

    clear_pycache()

    # Step 1: Preprocessing
    print("Running preprocessing...")
    run_all()

    # Step 2: Save all models (TF-IDF + CF)
    print("\nSaving models...")
    save_all_models()

    # Step 3: Load Content Model
    cm = ContentModel()
    if file_exists("model/tfidf_vectorizer.pkl") and file_exists("model/similarity_matrix.npy"):
        cm.load_saved()
    else:
        cm.build()

    # Step 4: Load Collaborative Model
    cf = CollaborativeModel()
    if file_exists("model/cf_features.npy"):
        cf.load_saved()
    else:
        cf.build()

    # Step 5: Initialize Hybrid Recommender
    hybrid = HybridRecommender(cm, cf)
    evaluator = Evaluator(cf, cm)

    # # Step 6: Model Performance
    # print("\nMODEL PERFORMANCE:")
    # print(f"RMSE: {evaluator.rmse():.3f}")
    # print(f"MAE: {evaluator.mae():.3f}")
    # print(f"Precision: {evaluator.precision_at_k(0):.3f}")
    # print(f"Recall: {evaluator.recall_at_k(0):.3f}")

    # Step 7: Recommendations
    
    def print_recommendations(df):
        import ast
        from tabulate import tabulate

        rows = []

        for idx, row in df.iterrows():
            title = row["title_english"]
            rating = row["avg_rating"]

            # Parse genres safely
            genres_raw = row["genres"]
            try:
                genres_list = ast.literal_eval(genres_raw) if isinstance(genres_raw, str) else genres_raw
            except:
                genres_list = [genres_raw]

            genres = ", ".join(genres_list) if isinstance(genres_list, list) else genres_raw

            score = row["hybrid_score"]

            rows.append([title, rating, genres, f"{score:.3f}"])

        print("\nTOP RECOMMENDATIONS:\n")
        print(tabulate(rows, headers=["Title", "Rating", "Genres", "Hybrid Score"], tablefmt="fancy_grid"))

    recs = hybrid.recommend(0, top_n=10)
    print_recommendations(recs)


    print("\nPIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
