import ast
import os

def parse_genres(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def load_poster_path(anilist_id):
    path = f"data/posters/{anilist_id}.jpg"
    if os.path.exists(path):
        return path
    return "assets/placeholder.jpg"