# AnimePick : Anime Recommendation System

A complete Machine Learning powered Anime Recommendation System combining Content-Based Filtering, Collaborative Filtering, Hybrid Scoring, and Semantic Search (NLP). Includes FastAPI backend, Streamlit frontend, and full cloud deployment.

## Streamlit url: https://animepick.streamlit.app/

---

##Workflow Diagram
```txt
                  ┌────────────────────────┐
                  │        User UI         │
                  │   (Streamlit Frontend) │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │      Search Query      │
                  │ (title/genre/keywords) │
                  └───────────┬────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │         FastAPI Backend    │
                │   /search endpoint         │
                └─────────────┬──────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
 ┌──────────────────┐  ┌──────────────┐  ┌────────────────────┐
 │ Content Model    │  │ Ratings Data │  │ Poster Fetch/Local │
 │ TF-IDF Similarity│  │ avg ratings  │  │ Local image lookup │
 └──────────────────┘  └──────────────┘  └────────────────────┘
              └───────────────┬────────────────────┘
                              ▼
                ┌────────────────────────────┐
                │      Filter by Genre       │
                │        + Score Sorting     │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │      Top Anime Results     │
                │   + Local Poster Loading   │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌────────────────────────────┐
                │ Streamlit Visual Display   │
                │  - Horizontal Scroll       │
                │  - Modal Details Popup     │
                └────────────────────────────┘
```

## Features

### Hybrid Recommendation Engine
- TF-IDF content similarity
- Latent collaborative features (SVD)
- Weighted hybrid fusion
- Series-level deduplication (removes multiple seasons)

### FastAPI Backend
GET /recommend?anime_id=123
GET /search?q=time travel

### Streamlit Web App
- Anime dropdown autocomplete  
- Poster + rating + genre + description  
- Genre filtering  
- Minimum rating slider  
- “More Like This” recommendations  
- Semantic search bar  

### Semantic Search (NLP)
Uses SentenceTransformer model:
all-MiniLM-L6-v2

---

## 📁 Project Structure

```text
📁 Anime-Recommendation/
 │
 ├── 📁 backend/
 │    ├── 🧑‍💻 main_api.py
 │    ├── 🧑‍💻 recommender.py
 │    ├── 📄 requirements.txt
 │    └── 🧑‍💻 utils.py
 │
 ├── 📁 data/
 │    ├── 📦 raw/
 │    │    ├── 🖼 anime_data.csv
 │    │    └── 🖼 ratings.csv
 │    │
 │    └── 📦 processed/
 │         ├── 🖼 anime_data_clean.csv
 │         ├── 🖼 ratings_clean.csv
 │         └── 🖼 final_merged.csv
 │
 ├── 📁 logs/
 │    ├── 🌐 log_20260110.txt
 │    └── 🌐 log_20260111.txt
 │
 ├── 📁 model/
 │    ├── 📄 __init__.py
 │    └── ⚙️ cf_features.npy
 │
 ├── 📁models/
 │    ├── ⚙️ tfidf_vectorizer.pkl
 │    ├── ⚙️ similarity_matrix.npy
 │    ├── ⚙️ cf_features.npy
 │    └── ⚙️ hybrid_alpha.pkl
 │
 ├── 📁 notebooks/
 │    ├── 🔄 EDA.ipynb
 │    ├── 🔄 content_model.ipynb
 │    ├── 🔄 collaborative_model.ipynb
 │    └── 🔄 hybrid_model.ipynb
 │
 ├── 📁 src/
 │    ├── 📦 collaborative/
 │    │    ├── 📄 __init__.py
 │    │    └── 📄 cf_model.py
 │    │
 │    ├── 📦 content_based/
 │    │    ├── 📄 __init__.py
 │    │    └── 🖼 content_model.py
 │    │
 │    ├── 📦 semantic/
 │    │    ├── 📄 __init__.py
 │    │    └── 📄 semantic_search.py
 │    │
 │    ├── 📦 hybrid/
 │    │    ├── 📄  __init__.py
 │    │    └── 📄 hybrid_recommender.py
 │    │
 │    ├── 📦 evaluation/
 │    │    ├── 📄 __init__.py
 │    │    └── 📄 evaluator.py
 │    │
 │    ├── 📦 logger/
 │    │     ├── 📄 __init__.py
 │    │     └── 📄 logger.py
 │    │
 │    ├── 📦 model/
 │    │     ├── 📄 __init__.py
 │    │     └── 📄 save_models.py
 │    │
 │    └── 📦 utils/
 │         ├── 📄 __init__.py
 │         └── 📄 Preprocessing.py
 │
 ├── 📁 styles/
 │    └── 🎨 style.css
 │
 ├── 📄 app.py
 ├── 📄 main.py
 ├── 📄 config.py
 ├── 📄 requirements.txt
 ├── 🧑‍💻 Procfile
 ├── 🧑‍💻 render.yaml
 ├── 📄 README.md
 └── 🧑‍💻 .gitignore

```

---


## Tech Stack

### Machine Learning
- scikit-learn
- sentence-transformers
- numpy / pandas

### Backend
- FastAPI
- Uvicorn

### Frontend
- Streamlit

### Deployment
- Render (backend)
- Streamlit Cloud (frontend)

---

## Running Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

2️⃣ Run backend (FastAPI)
```bash
uvicorn backend.main_api:app --reload
```
Test in browser:
```bash
http://localhost:8000/recommend?anime_id=1
```
3️⃣ Run frontend (Streamlit)
```bash
streamlit run app.py
```

🌐 Deployment
Backend (Render)
Use this start command:
```nginx
uvicorn backend.main_api:app --host 0.0.0.0 --port $PORT
```
Frontend (Streamlit Cloud)

Update config.py:
```python
API_URL = "https://your-render-backend-url.onrender.com"
```

📝 Example API Output
```json
{
  "recommendations": [
    {
      "title_english": "Attack on Titan",
      "avg_rating": 8.9,
      "genres": ["Action", "Drama"],
      "description": "Humans fight Titans...",
      "image_url": "https://..."
    }
  ]
}
```
---

##👤 Author

Shuvendu Kumar Mohapatra

Machine Learning Engineer

GitHub: https://github.com/shuvendu9207

LinkedIn: https://www.linkedin.com/in/shuvendu-kumar-mohapatra













