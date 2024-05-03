from flask import Flask, render_template, request
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

moviesData = []

with open('movies.json') as moviesFile:
    for line in moviesFile:
        moviesData.append(json.loads(line))
        
moviesDF = pd.DataFrame.from_records(moviesData)

plotDF = moviesDF.loc[:,['name','description']]
plotDF.loc[:, 'name'] = plotDF.loc[:, 'name'].apply(lambda x: x.lower())
plotDF.loc[:, 'description'] = plotDF.loc[:, 'description'].apply(lambda x: x.lower())

# Drop all the rows which are empty
plotDF = plotDF[plotDF['description'] != '']

# Now drop duplicates in the plotDF
plotDF = plotDF.drop_duplicates()

# Define a TF IDF Vectorizer Object with the removal of english stopwords turned on
tfidf = TfidfVectorizer(stop_words = 'english')

# Now costruct the TF-IDF Matrix by applying the fit_transform method on the description feature
tfidf_matrix = tfidf.fit_transform(plotDF['description'])

# Compute the cosine similarity matrix by using linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(plotDF.index, index = plotDF['name'])

def plot_based_recommender(title, df = plotDF, cosine_sim = cosine_sim, indices = indices):
    title = title.lower()
    try:
        idx = indices[title]
    except KeyError:
        return 'Movie does not exist :('
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    movie_indices = [sim_score[0] for sim_score in sim_scores]
    return df['name'].iloc[movie_indices]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    recommended_movies = plot_based_recommender(movie_name)
    return render_template('recommend.html', movie_name=movie_name, recommended_movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
