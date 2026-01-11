import os

folders = [
    "data/raw",
    "data/processed",
    "data/posters",
    "src",
    "src/model",
    "src/logger",
    "src/evaluation",
    "src/collaborative",
    "src/content_based",
    "src/hybrid",
    "src/utils",
    "src/semantic",
    "backend",
    "notebooks",
    "models",
    "styles"
]

files = {
    "src/model/__init__.py": "",
    "src/model/save_models.py": "",
    "src/logger/__init__.py": "",
    "src/logger/logger.py": "",
    "src/evaluation/__init__.py": "",
    "src/evaluation/evaluator.py": "",
    "src/collaborative/__init__.py": "",
    "src/collaborative/cf_model.py": "",
    "src/content_based/__init__.py": "",
    "src/content_based/content_model.py": "",
    "src/hybrid/__init__.py": "",
    "src/hybrid/hybrid_recommender.py": "",
    "src/utils/__init__.py": "",
    "src/utils/Preprocessing.py": "",
    "src/semantic/semantic_search.py": "",
    "notebooks/EDA.ipynb": "",
    "notebooks/content_model.ipynb": "",
    "notebooks/collaborative_model.ipynb": "",
    "notebooks/hybrid_model.ipynb": "",
    "models/.gitkeep": "",
    "backend/main_api.py": "",
    "backend/recommender.py": "",
    "backend/utils.py": "",
    "backend/reqirements.txt": "",
    "README.md": "",
    ".gitignore": "",
    "app.py": "",
    "config.py": "",
    "render.yaml": "",
    "Procfile": "",
    "requirements.txt": "",
    "download_all.py": "",
    "main.py": "",
    "styles/style.css": ""
}

for a in folders:
    if not os.path.exists(a):
        os.makedirs(a)

for a, b in files.items():
    d = os.path.dirname(a)
    if d == "" or os.path.exists(d):
        if not os.path.exists(a):
            with open(a, "w", encoding="utf-8") as f:
                f.write(b)

print("Project structure created successfully.")
