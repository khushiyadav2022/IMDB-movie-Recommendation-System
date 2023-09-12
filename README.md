# IMDB-movie-Recommendation-System
This project is designed to build a movie recommendation system using IMDb dataset. The recommendation system suggests movies similar to the user's preferences based on content similarity.

## Project Steps

### Step 1 - Dataset from IMDb Database
- Obtain the IMDb dataset from IMDb's official website or a third-party source.

### Step 2 - Check for Null Values and Duplicates
- Clean the dataset by checking for and handling any missing values or duplicate entries.

### Step 3 - Create CSV Files According to Project Needs
- Organize the dataset into multiple CSV files, each serving a specific purpose in the recommendation system.

### Step 4 - Merging Data
- Merge relevant CSV files to create a comprehensive dataset for building the recommendation system.

### Step 5 - Create Lists According to Primary Key
- Organize the data into lists, where each list corresponds to a unique movie identifier (e.g., movie ID or title).

### Step 6 - Create a Single Column "Tags"
- Combine relevant information about movies, such as genres, keywords, and other attributes, into a single column called "Tags."

### Step 7 - Create a New Data Frame
- Build a new data frame that includes the movie title, unique identifier, and the "Tags" column.

### Step 8 - Convert Lists into String
- Convert the lists of movie attributes into strings to facilitate text processing.

### Step 9 - Convert All Strings to Lowercase
- Standardize text data by converting all strings to lowercase.

### Step 10 - Text Vectorization
- Implement text vectorization techniques to convert textual data into numerical representations suitable for machine learning.

### Step 11 - Convert into Arrays using Transform
- Transform the text vectorized data into arrays to prepare for similarity calculation.

### Step 12 - Stemming
- Apply text processing techniques such as stemming to further refine text data.

### Step 13 - Calculate Distance for Similarity
- Calculate similarity scores between movies based on their content attributes (e.g., tags) using distance metrics like cosine similarity.

## Building the model
Here, is what we are gonna actually do, we are recommending the user based upon these ‘tags’. Now how we will know that these 5 movies are similar? We will find similarities between the tags of the two movies. Manually it is difficult to calculate similar words between two movies and in other words, it is inefficient.

Hence here we will vectorize each of the ‘tags’. And we will find similarities between them by finding similarities in the text. The 5 nearest vectors will be the answer for each vector.

![](https://copyassignment.com/wp-content/uploads/2022/07/image-136-300x174.png)

## What is Vectorization?
Vectorization is jargon for a classic approach of converting input data from its raw format (i.e. text ) into vectors of real numbers which is the format that ML models support. This approach has been there ever since computers were first built, it has worked wonderfully across various domains, and it’s now used in NLP.

In Machine Learning, vectorization is a step in feature extraction. The idea is to get some distinct features out of the text for the model to train on, by converting text to numerical vectors.

### Vectorization Techniques:

Bag of Words
TF-IDF
Word2Vec
GloVe
FastText
For more info, read here.

We are using the ‘Bag of Words’ technique in our project.

How we are using **‘Bag of Words‘**?

We will combine each of the tags to form a large text. And from this text, we will calculate 5000 most frequent words. Then, for every word, we will calculate its frequency with each of the rows in ‘tags’. These frequency rows are nothing but vectors in 5000×5000 geometry space. (5000 – no. of words and 5000 – no. of movies)

For eg.
- vectors in streamlit in ml or machine learning
- Each of the vectors will plot against each with words as axes and most similar vectors will be considered as the result.

For eg. if we consider 2 words then the space will be 5000×2, for understanding, we can plot the graph.

![](https://copyassignment.com/wp-content/uploads/2022/07/image-138.png)

Before performing vectorization, stop words must be removed. Stop words are words useful in constructing the sentence for eg. and, or, to, is, etc.

# Integrating movie recommendation system with Streamlit
For this, you need to open any python IDE and create a virtual environment.
![](https://copyassignment.com/wp-content/uploads/2022/07/image-149-675x129.png)

### What is streamlit? How to install it?
Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps. 

To install streamlit, type the following command into your cmd:
---
pip install streamlit
---
For more info, visit documentation: https://docs.streamlit.io

Create streamlit model
Inside the app.py file, import streamlit first and start with the code.

import  streamlit as st
st.title('Movie Recommender System')
Now, to run the streamlit app, go to cmd with the current location and type:

streamlit run filename.py
On your local host, the app will be deployed.

![](https://copyassignment.com/wp-content/uploads/2022/07/image-150.png)

# Complete code for Movie Recommendation system in Python using Streamlit

---
import  streamlit as st

import pickle
import pandas as pd
import requests

st.title('Movie Recommender System')


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key='YOUR API KEY'&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movie_posters = []

    for i in movies_list:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movies,recommended_movie_posters

movies_dict = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl','rb'))


selected_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)
if st.button('Show Recommendation'):
    names,posters = recommend(selected_movie_name)

    #display with the columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
        ---

        Enter your API key here and see the magic. The code is super simple. We just integrate the recommended function with the fetch poster and are done! Now, let’s see the final output.

To run the app, from your current working directory, open the terminal and type:

---
streamlit run filename.py
---

## Let’s run:
[![Watch the video]()](https://youtu.be/T-D1KVIuvjA)

