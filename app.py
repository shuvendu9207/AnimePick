import streamlit as st
import pandas as pd

st.set_page_config(page_title="AnimePick", layout="wide")

# Load CSS
with open("styles/style.css", "r") as f:
    st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)

LOGO_URL = "https://i.ibb.co/tF5Fg6k/logo.png"

# Load dataset
df = pd.read_csv("data/processed/final_merged.csv")

def clean_genres(g):
    if isinstance(g, str):
        return g.replace("[","").replace("]","").replace("'","")
    return g

df["genres_clean"] = df["genres"].apply(clean_genres)

# Featured Anime
featured = df.iloc[0]
featured_img = featured["coverImage"]
rating = featured["averageScore"] / 10 if "averageScore" in featured else 0

# ---------------- HEADER ----------------
st.markdown(
    f"""
    <div class="header">
        <img src="{LOGO_URL}" class="logo-img">
        <div class="search-container">
            <input id="searchInput" type="text" placeholder="Search anime..." class="search-box"
                   onkeypress="if(event.key === 'Enter') runSearch()">
        </div>
    </div>

    <script>
    function runSearch() {{
        let v = document.getElementById("searchInput").value;
        const u = new URL(window.location.href);
        if(v) {{
            u.searchParams.set("q", v);
        }} else {{
            u.searchParams.delete("q");
        }}
        window.location.href = u;
    }}
    </script>

    <div class="page-spacer"></div>
    """,
    unsafe_allow_html=True
)

# ---------------- SEARCH LOGIC ----------------
query = st.query_params.get("q", "")

if isinstance(query, list):
    query = query[0]

query = query.strip()

if query:
    q = query.lower()
    results = df[
        df["title_english"].str.lower().str.contains(q, na=False)
        | df["genres_clean"].str.lower().str.contains(q, na=False)
        | df["description"].str.lower().str.contains(q, na=False)
    ]
else:
    results = df.iloc[1:]


# ---------------- HERO BOX ----------------
hero_html = f"""
<div class="hero-box">
    <div class="hero-left">
        <h1 class="hero-title">{featured['title_english']}</h1>
        <div class="hero-rating">⭐ {rating:.1f}/10</div>
        <p class="hero-desc">{featured['description'][:260]}...</p>
        <div class="hero-buttons">
            <button class="explore-btn">Explore</button>
            <button class="details-btn">Details</button>
        </div>
    </div>
    <div class="hero-right">
        <img src="{featured_img}" class="hero-poster">
    </div>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

# FEATURED SECTION (FINAL FIX)
st.markdown("<h2 class='section-title'>All Anime</h2>", unsafe_allow_html=True)

anime_list = results.copy()

html = """<div class='anime-list-container'>"""

for _, row in anime_list.iterrows():
    img_url = row["coverImage"] if isinstance(row["coverImage"], str) else LOGO_URL
    title = row["title_english"]
    rating = row["averageScore"] / 10 if "averageScore" in row else 0

    html += f"""
<div class='anime-row'>
    <img src="{img_url}" class="row-img"/>
    <div class='row-info'>
        <div class='row-title'>{title}</div>
        <div class='row-rating'>⭐ {rating:.1f}</div>
    </div>
</div>
"""

html += "</div>"

st.markdown(html, unsafe_allow_html=True)