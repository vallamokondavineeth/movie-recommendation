AI-Powered Global Movie Recommendation Engine
Intern Details
Name: Vineeth vallamkonda
Intern ID: CITS1645
Internship Duration: 4 Weeks
Project Overview

The AI-Powered Global Movie Recommendation Engine is a Machine Learning and NLP-based web application that recommends similar movies to users based on movie content and genres.

The system supports:

Hollywood movies
Bollywood movies
Tollywood movies
International cinema

The application analyzes movie descriptions and genres using Natural Language Processing techniques and recommends related movies using cosine similarity.

Features
AI-based movie recommendations
Supports global movie datasets
Interactive Streamlit web application
Fast movie search system
NLP-powered recommendation engine
Modern dark-themed UI
Real-time recommendations
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Streamlit
NLP (TF-IDF Vectorization)
Cosine Similarity
Machine Learning Concepts Used
1. TF-IDF Vectorization

Converts movie descriptions into numerical feature vectors.

2. Natural Language Processing (NLP)

Processes movie overviews and genres for similarity analysis.

3. Cosine Similarity

Measures similarity between movies and recommends related content.

Dataset Used

Dataset Source:

TMDB Movies Dataset
Kaggle Movie Metadata Dataset

Dataset includes:

movie titles
genres
overviews
international movies
ratings and metadata
Project Workflow
Load movie dataset
Clean and preprocess data
Extract movie features
Apply TF-IDF vectorization
Compute cosine similarity
Recommend similar movies
Display results using Streamlit
Project Structure
movie-recommendation-engine/
│
├── dataset/
│   ├── movies_metadata.csv
│   ├── credits.csv
│   └── keywords.csv
│
├── app.py
├── requirements.txt
└── README.md
Installation Steps
Step 1: Clone Repository
git clone YOUR_GITHUB_REPOSITORY_LINK
Step 2: Open Project Folder
cd movie-recommendation-engine
Step 3: Install Required Packages
pip install -r requirements.txt
Required Libraries
streamlit
pandas
numpy
scikit-learn
Run the Project
streamlit run app.py
Output

The application displays:

selected movie
recommended similar movies
AI-generated recommendations
interactive user interface
Sample Recommended Movies

Input Movie:

Interstellar

Recommended Movies:

The Martian
Gravity
Moon
Avatar
2001: A Space Odyssey
Future Enhancements
Movie posters integration
User login system
Personalized recommendations
Collaborative filtering
Deep learning recommendation models
Movie rating prediction
Watchlist feature
Learning Outcomes

Through this project, the following concepts were learned:

Machine Learning fundamentals
Recommendation systems
NLP techniques
Feature engineering
Similarity algorithms
Streamlit web app development
Data preprocessing
Conclusion

This project successfully demonstrates an AI-powered recommendation system capable of suggesting movies from multiple film industries using NLP and Machine Learning techniques.

The system provides an interactive and scalable solution for movie recommendation applications.
