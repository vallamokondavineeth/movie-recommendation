import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
credits = pd.read_csv('dataset/tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select important columns
movies = movies[['movie_id',
                 'title',
                 'overview',
                 'genres',
                 'keywords',
                 'cast',
                 'crew']]

# Remove null values
movies.dropna(inplace=True)

# Convert genres and keywords
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Extract top 3 cast members
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# Extract director
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Apply functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview into list
movies['overview'] = movies['overview'].apply(lambda x:x.split())

# Remove spaces
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Final dataframe
new_df = movies[['movie_id','title','tags']]

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

# Lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

# Text vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x:x[1])[1:6]

    print("\nRecommended Movies:\n")

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

# Example
recommend('Avatar')
