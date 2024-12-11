import nltk

import streamlit as st
import psutil
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import zstandard as zstd
from sentence_transformers import SentenceTransformer
import faiss


def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2 

st.write(f"Initial memory usage: {get_memory_usage():.2f} MB")

@st.cache_resource
def download_nltk_resources():
    resources = ['punkt_tab', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

st.write(f"Memory usage after NLTK dl: {get_memory_usage():.2f} MB")

@st.cache_resource
def load_knn():
    with open('models/nearest_neighbors_model.pkl', 'rb') as f:
        nearest_neighbors = pickle.load(f)
    
    return nearest_neighbors

@st.cache_resource
def load_matrix():  
    tfidf_matrix = scipy.sparse.load_npz('models/tfidf_matrix.npz')
    return tfidf_matrix

@st.cache_resource
def load_vectorizer():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

@st.cache_resource
def load_model():
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_index():
    index = faiss.read_index("faiss_index.bin")

@st.cache_resource
def load_data():
    df1 = pd.read_csv('Data/revised_recipes_1_1.csv.zst', compression="zstd")
    df2 = pd.read_csv('Data/revised_recipes_1_2.csv.zst', compression="zstd")
    df6 = pd.read_csv('Data/revised_recipes_1_3.csv.zst', compression="zstd")
    df1 = pd.concat([df1, df2], ignore_index=True)
    del df2
    df1 = pd.concat([df1, df6], ignore_index=True)
    del df6

    df3 = pd.read_csv('Data/revised_recipes_2_1.csv.zst', compression='zstd')
    df4 = pd.read_csv('Data/revised_recipes_2_2.csv.zst', compression='zstd')
    df5 = pd.read_csv('Data/revised_recipes_2_3.csv.zst', compression='zstd')
    df5 = pd.concat([df5, df4], ignore_index=True)
    del df4
    df5 = pd.concat([df5, df3], ignore_index=True)
    del df3
    
    data = pd.merge(df1, df5, on = 'ID', how = 'inner')
    del df5
    del df1

    return data

nearest_neighbors = load_knn()
tfidf_matrix = load_matrix()
vectorizer = load_vectorizer()
data = load_data()
model = load_model()
index = load_index()
st.write(f"Memory usage after initializing: {get_memory_usage():.2f} MB")

lemmatizer = WordNetLemmatizer()

def lemmatize_string(string):
    string_lower = string.lower()
    tokens = word_tokenize(string_lower)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def lemmatize_list(list):
    return [lemmatizer.lemmatize(item.lower()) for item in list]

st.write(f"Memory usage after cache nlp: {get_memory_usage():.2f} MB")

def recommend_by_ingredients(ingredients_list, excluded_ingredients=None):
    if not preferred_ingredients or not preferred_ingredients.strip():
        return pd.DataFrame()
     
    excluded_ingredients = None if not excluded_ingredients.strip() else excluded_ingredients

    if excluded_ingredients is None:
        excluded_ingredients = []
    # if excluded_ingredients is a string, convert to list
    elif isinstance(excluded_ingredients, str):
        excluded_ingredients = excluded_ingredients.lower().split(', ')

    # preprocess input by lemmatizing
    lemmatized_ingredients = lemmatize_string(ingredients_list)

    # Transform the user's input ingredients into the vector space
    user_vector = vectorizer.transform([lemmatized_ingredients])
    
    # Find the top N nearest neighbors
    distances, indices = nearest_neighbors.kneighbors(user_vector, n_neighbors=50)
    
    # Retrieve recommended recipes and their similarity scores
    recommendations['IngredientSimilarity'] = 1 - distances[0]  # Similarity = 1 - distance (cosine)

    # filter out recipes containing excluded ingredients
    lemmatized_excluded_ingredients = lemmatize_list(excluded_ingredients)
    def contains_excluded(ingredients):
        for excluded in lemmatized_excluded_ingredients:
            if excluded in ingredients:
                return True
        return False
    
    recommendations = recommendations[~recommendations['NLP_Ingredients'].apply(contains_excluded)].reset_index()
    
    return recommendations

def recommend_by_description(user_query):
    # encode the query
    query_embedding = model.encode(user_query, convert_to_tensor=False)

    # normalize query embedding
    faiss.normalize_L2(np.array([query_embedding]))

    # query the faiss index
    distances, indices = index.search(np.array([query_embedding]), df.shape[0])

    # retrieve recommended rows
    recommendations = df.iloc[indices[0]].copy().reset_index()

    # add similarity scores from faiss to the dataframe
    recommendations['DescriptionSimilarity'] = distances[0]

    return recommendations

def recommend_combined(ingredients_list=None, user_query=None, ingredients_weight=0.5, description_weight=0.5, excluded_ingredients=None, top_n=5):
    if ingredients_list is not None and user_query is None:
        return recommend_by_ingredients(ingredients_list=ingredients_list, excluded_ingredients=excluded_ingredients).head(top_n)

    if ingredients_list is None and user_query is not None:
        return recommend_by_description(user_query=user_query).head(top_n)

    if ingredients_list is not None and user_query is not None:
        if ingredients_weight + description_weight != 1:
            print("Please make sure ingredients_weight and description_weight add up to 1.")
        else:
            # get recommendations from both models
            ingredient_recs = recommend_by_ingredients(ingredients_list=ingredients_list, excluded_ingredients=excluded_ingredients)
            description_recs = recommend_by_description(user_query)

            # merge the two recommendation lists on ID
            combined_recs = pd.merge(ingredient_recs, description_recs, on=['ID','Name'], how='outer')

            # normalize scores to make sure they are on the same scale
            scaler = MinMaxScaler()
            combined_recs[['IngredientSimilarity', 'DescriptionSimilarity']] = scaler.fit_transform(combined_recs[['IngredientSimilarity','DescriptionSimilarity']])

            # combine the scores
            combined_recs['WeightedScore'] = (combined_recs['IngredientSimilarity'] * ingredients_weight) + (combined_recs['DescriptionSimilarity'] * description_weight)

            # sort by combined score and return top recommendations
            combined_recs = combined_recs.sort_values(by='WeightedScore', ascending=False).head(top_n).reset_index()

            return combined_recs


st.write(f"Memory usage after cache recommendations: {get_memory_usage():.2f} MB")

st.title('Dishcovery')
ingredients_list = st.text_input("Which ingredients are you using?")
exclude_list = st.text_input("Any allergies or exceptions?")
ingredients_weight = st.slider(
    "Select weight for ingredients",  
    min_value=0,                
    max_value=100,             
    value=50,                   
)
query_weight = st.slider(
    "Select weight for query",  
    min_value=0,                
    max_value=100,             
    value=50,                   
)
user_query = st.text_input("Input query")

st.write(f'Your Ingredients List is: {ingredients_list}')
st.write(f'Things to Exclude are: {exclude_list}')

# Display each recipe in an expander
if st.button('Get Recommendations', key = 'Recommendations'):
    
    st.write(f"Memory usage after pressing button: {get_memory_usage():.2f} MB")
    with st.spinner('Recommending...'):
        recommendations = recommend_by_ingredients(ingredients_list, excluded_ingredients = exclude_list)
        st.write(f"Memory usage after recommending: {get_memory_usage():.2f} MB")
        for index, row in recommendations.iterrows():
            with st.expander(row['Name']):
                st.markdown(f"## {row['Name']}")
                st.write(f"**Similarity:** {row['Similarity']:.2f}")
                st.write("**Ingredients:**")
                raw_ingredients_list = row['IngredientsRaw'].split("', '")
                for i, step in enumerate(raw_ingredients_list, 1):
                    step = step.replace("'", '').replace('(', '').replace(')', '')
                    if not step.strip():  
                        continue
                    st.write(f"{i}. {step}")

                # Display nutrition
                st.write("**Nutrition**")
                st.write(f"**Calories:** {row['Calories']:.2f}")
                st.write(f"**Fat:** {row['FatContent']:.2f}g")
                st.write(f"**Saturated Fat:** {row['SaturatedFatContent']:.2f}g")
                st.write(f"**Cholesterol:** {row['CholesterolContent']:.2f}mg")
                st.write(f"**Sodium:** {row['SodiumContent']:.2f}mg")
                st.write(f"**Total Carbohydrates:** {row['CarbohydrateContent']:.2f}g")
                st.write(f"**Dietary Fiber:** {row['FiberContent']:.2f}g")
                st.write(f"**Sugars:** {row['FiberContent']:.2f}g")
                st.write(f"**Protein:** {row['ProteinContent']:.2f}g")
                
                # Parse instructions into separate steps
                st.write("**Instructions:**")
                instructions_list = row['Instructions'].split("', '")
                for i, step in enumerate(instructions_list, 1):
                    step = step.replace("'", '').replace('(', '').replace(')', '')
                    if not step.strip():  
                        continue
                    st.write(f"{i}. {step}")
                st.write("------")

