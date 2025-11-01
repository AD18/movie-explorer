# app.py — Robust OMDb + local-genre recommender
import os
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# -------------------------
# Config & secrets
# -------------------------
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

DATA_PATH = Path("data/cleaned_movies.csv")  # adjust if needed

st.set_page_config(page_title="🎬 Movie Explorer", layout="wide")

# -------------------------
# Helper: safe OMDb fetch (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_omdb(title):
    """Query OMDb by title. Returns JSON dict or None."""
    if not OMDB_API_KEY:
        return None
    try:
        resp = requests.get(
            "http://www.omdbapi.com/",
            params={"t": title, "apikey": OMDB_API_KEY},
            timeout=6
        )
        data = resp.json()
        if data.get("Response") == "True":
            return data
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def fetch_omdb_by_title(title):
    return fetch_omdb(title)


# -------------------------
# Load dataset and normalize
# -------------------------
@st.cache_data
def load_local_movies(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Save your cleaned CSV at this path.")
    df = pd.read_csv(path)

    # Try to find the title column (case-insensitive)
    title_col = None
    for c in df.columns:
        if c.lower() == "title":
            title_col = c
            break
    if title_col is None:
        for alt in ("movie_title", "name"):
            if alt in df.columns:
                title_col = alt
                break
    if title_col is None:
        raise ValueError("Could not find a 'title' column in cleaned_movies.csv.")

    if title_col != "title":
        df = df.rename(columns={title_col: "title"})

    # Detect genre columns
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    if genre_cols:
        u_genre_path = Path("data/ml-100k/u.genre")
        idx_to_name = {}
        if u_genre_path.exists():
            try:
                with open(u_genre_path, "r", encoding="latin-1") as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split("|")
                            if len(parts) == 2 and parts[1].isdigit():
                                idx_to_name[int(parts[1])] = parts[0]
            except Exception:
                idx_to_name = {}

        genre_labels = []
        for col in genre_cols:
            m = re.search(r"genre_(\d+)$", col)
            if m:
                idx = int(m.group(1))
                label = idx_to_name.get(idx, f"genre_{idx}")
            else:
                label = col
            genre_labels.append((col, label))

        def row_to_genre_list(row):
            out = []
            for col, label in genre_labels:
                try:
                    if int(row.get(col, 0)) == 1:
                        out.append(label)
                except Exception:
                    if row.get(col):
                        out.append(label)
            return out

        df["genres"] = df.apply(row_to_genre_list, axis=1)

    elif any(c.lower() == "genres" for c in df.columns):
        gcol = next(c for c in df.columns if c.lower() == "genres")
        df["genres"] = df[gcol].fillna("").apply(
            lambda x: [g.strip() for g in str(x).split(",") if g.strip()]
        )
    else:
        df["genres"] = [[] for _ in range(len(df))]

    df["title_clean"] = (
        df["title"]
        .astype(str)
        .str.lower()
        .apply(lambda x: re.sub(r"[^a-z0-9 ]", "", x))
    )

    df = df.reset_index(drop=True)
    return df


try:
    movies = load_local_movies(DATA_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()


# -------------------------
# Recommendation by genre overlap
# -------------------------
@st.cache_data
def build_genre_matrix(df):
    all_genres = sorted({g for gs in df["genres"] for g in gs})
    if not all_genres:
        return None, []
    mat = np.zeros((len(df), len(all_genres)), dtype=int)
    for i, gs in enumerate(df["genres"]):
        for g in gs:
            if g in all_genres:
                mat[i, all_genres.index(g)] = 1
    return mat, all_genres


genre_matrix, genre_labels = build_genre_matrix(movies)


def recommend_by_genres_from_title(title_search, df, matrix, labels, top_n=5):
    if matrix is None or matrix.shape[1] == 0:
        return []
    title_clean = re.sub(r"[^a-z0-9 ]", "", title_search.lower().strip())
    matches = df[df["title_clean"].str.contains(title_clean, na=False)]
    if matches.empty:
        matches = df[df["title_clean"] == title_clean]
        if matches.empty:
            return []
    idx = matches.index[0]
    sims = cosine_similarity(matrix, matrix[idx : idx + 1]).flatten()
    sims[idx] = -1
    top_idx = sims.argsort()[::-1][:top_n]
    return df.loc[top_idx, "title"].tolist()


# -------------------------
# UI
# -------------------------
st.title("🎥 Movie Explorer — The Classic 90s Movie Recommender")

st.markdown("""
Welcome to **Movie Explorer: The 90s Edition** 🎥  
If you're a true movie buff who loves the charm, nostalgia, and storytelling of the 90s —  
you’re in the right place!  
Search for your favorite movie from the 90s and discover similar gems from that golden era.
""")

st.write(
    "Type any movie name.We'll recommend genre-similar movies from your dataset."
)

col_search, col_options = st.columns([3, 1])
with col_search:
    query = st.text_input("Search movie (any title)", value="")
with col_options:
    top_n = st.slider("Number of recommendations", 1, 12, 5)

if st.button("Search") or (query and not st.session_state.get("_queried")):
    st.session_state["_queried"] = True

    if not query.strip():
        st.warning("Please enter a movie title.")
    else:
        omdb = fetch_omdb_by_title(query.strip())

        # ✅ NEW: Handle invalid or nonsense titles gracefully
        if not omdb or omdb.get("Response") == "False":
            st.error("⚠️ Please enter a valid movie name.")
        else:
            # Display movie details
            left, right = st.columns([1, 2])
            with left:
                poster = omdb.get("Poster")
                if poster and poster != "N/A":
                    st.image(poster, width=250)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Image", width=250)
            with right:
                st.header(f"{omdb.get('Title')} ({omdb.get('Year')})")
                st.markdown(f"**Genres:** {omdb.get('Genre')}")
                st.markdown(f"**IMDB Rating:** {omdb.get('imdbRating')}")
                st.markdown(f"**Plot:** {omdb.get('Plot')}")

            # Recommendations
            with st.spinner("Finding similar movies from local dataset..."):
                recs = recommend_by_genres_from_title(
                    omdb.get("Title", ""), movies, genre_matrix, genre_labels, top_n=top_n
                )

            if not recs:
                st.info("No similar movies found in the local dataset. (Try another title.)")
            else:
                st.subheader("🎯 Recommended Movies")
                cols = st.columns(4)
                for i, title in enumerate(recs):
                    col = cols[i % 4]
                    with col:
                        st.markdown(f"🎬 **{title}**")
