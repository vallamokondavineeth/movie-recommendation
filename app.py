import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Netflix Style Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #141414;
    color: white;
}

h1 {
    color: #E50914;
    text-align: center;
    font-size: 50px;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 220px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #b20710;
    color: white;
}

.movie-card {
    background-color: #222222;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    color: white;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

movies = pd.read_csv(
    'dataset/movies_metadata.csv',
    low_memory=False
)

# Important columns
movies = movies[['title', 'overview', 'genres']]

# Remove nulls
movies.dropna(inplace=True)

# Remove duplicates
movies.drop_duplicates(subset='title', inplace=True)

# Keep good overviews
movies = movies[movies['overview'].str.len() > 20]

# Random sample
movies = movies.sample(20000, random_state=42)

# Reset index
movies.reset_index(drop=True, inplace=True)

# ---------------------------------------------------
# CLEAN GENRES
# ---------------------------------------------------

def clean_genres(text):

    try:
        genres_list = eval(text)

        names = []

        for genre in genres_list:
            names.append(genre['name'])

        return " ".join(names)

    except:
        return ""

movies['genres'] = movies['genres'].apply(clean_genres)

# ---------------------------------------------------
# CREATE TAGS
# ---------------------------------------------------

movies['tags'] = (
    movies['overview'] + " " + movies['genres']
)

# ---------------------------------------------------
# TF-IDF
# ---------------------------------------------------

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(
    movies['tags']
)

# ---------------------------------------------------
# RECOMMEND FUNCTION
# ---------------------------------------------------

def recommend(movie_name):

    if movie_name not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_name].index[0]

    cosine_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    )

    sim_scores = list(
        enumerate(cosine_sim[0])
    )

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:11]

    recommended_movies = []

    for movie in sim_scores:

        movie_title = movies.iloc[movie[0]].title

        recommended_movies.append(movie_title)

    return recommended_movies

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("🎬 Netflix Style Movie Recommender")

st.write(
    "Discover Hollywood, Bollywood, Tollywood, "
    "and international movies using AI."
)

selected_movie = st.selectbox(
    "Search Movie",
    sorted(movies['title'].unique())
)

if st.button("Recommend"):

    recommendations = recommend(selected_movie)

    st.subheader("🔥 Recommended Movies")

    for movie in recommendations:

        st.markdown(
            f"""
            <div class="movie-card">
                🎥 {movie}
            </div>
            """,
            unsafe_allow_html=True
        )
