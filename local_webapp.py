import nltk

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import pickle
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import faiss
from sklearn.preprocessing import MinMaxScaler

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
    index = faiss.read_index("models/faiss_index.bin")
    return index

@st.cache_resource
def load_data():
    data = pd.read_csv('Data/recipes_food_com_combinedinfo.csv')
    return data

nearest_neighbors = load_knn()
tfidf_matrix = load_matrix()
vectorizer = load_vectorizer()
data = load_data()
model = SentenceTransformer('all-MiniLM-L6-v2')
index = load_index()

lemmatizer = WordNetLemmatizer()

def lemmatize_string(string):
    string_lower = string.lower()
    tokens = word_tokenize(string_lower)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def lemmatize_list(list):
    return [lemmatizer.lemmatize(item.lower()) for item in list]

def recommend_by_ingredients(ingredients_list, excluded_ingredients=None):
    if not ingredients_list or not ingredients_list.strip():
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
    distances, indices = nearest_neighbors.kneighbors(user_vector, data.shape[0])

    recommendations = data.iloc[indices[0]].copy().reset_index()
    # Retrieve recommended recipes and their similarity scores
    recommendations['IngredientSimilarity'] = 1 - distances[0]  

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
    distances, indices = index.search(np.array([query_embedding]), data.shape[0])

    # retrieve recommended rows
    recommendations = data.iloc[indices[0]].copy().reset_index()

    # add similarity scores from faiss to the dataframe
    recommendations['DescriptionSimilarity'] = distances[0]

    return recommendations

@st.cache_resource
def recommend_combined(ingredients_list=None, user_query=None, ingredients_weight=0.5, description_weight=0.5, excluded_ingredients=None, top_n=5):
    if ingredients_list is None and user_query is None:
        st.warning("Please provide ingredients or a search query.")
        return pd.DataFrame()
    
    if ingredients_list is not None and user_query is None:
        return recommend_by_ingredients(ingredients_list=ingredients_list, excluded_ingredients=excluded_ingredients).head(top_n)

    if ingredients_list is None and user_query is not None:
        return recommend_by_description(user_query=user_query).head(top_n)

    if ingredients_list is not None and user_query is not None:
        if ingredients_weight + description_weight != 1:
            st.write("Please make sure ingredients_weight and query_weight add up to 100.")
            return pd.DataFrame()
        else:
            # get recommendations from both models
            ingredient_recs = recommend_by_ingredients(ingredients_list=ingredients_list, excluded_ingredients=excluded_ingredients)
            description_recs = recommend_by_description(user_query)

            # merge the two recommendation lists on ID
            ingredient_recs = ingredient_recs[['ID','Name', 'IngredientSimilarity', 'IngredientsRaw', 'Calories', 
                                                'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
                                                'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent',
                                                'Instructions']]
            description_recs = description_recs[['ID','Name', 'DescriptionSimilarity']]
            combined_recs = pd.merge(ingredient_recs, description_recs, on=['ID','Name'], how='outer')

            # normalize scores to make sure they are on the same scale
            scaler = MinMaxScaler()
            combined_recs[['IngredientSimilarity', 'DescriptionSimilarity']] = scaler.fit_transform(combined_recs[['IngredientSimilarity','DescriptionSimilarity']])

            # combine the scores
            combined_recs['WeightedScore'] = (combined_recs['IngredientSimilarity'] * ingredients_weight) + (combined_recs['DescriptionSimilarity'] * description_weight)

            # sort by combined score and return top recommendations
            combined_recs = combined_recs.sort_values(by='WeightedScore', ascending=False).head(top_n).reset_index()

            return combined_recs

st.title('Dishcovery')
ingredients_list = st.text_input("Which ingredients are you using?")
exclude_list = st.text_input("Any allergies or exceptions?")
user_query = st.text_input("Input search query")
     
st.write('Adjust weights, please check that they add up to 100')
query_weight = st.number_input('Search query weight', value = 0, key = 'query_numeric', step = 10)
ingredients_weight = st.number_input('Ingredients weight', value = 0, key = 'ingredients_numeric', step = 10)

query_weight = query_weight / 100
ingredients_weight = ingredients_weight / 100

# Display each recipe in an expander
if st.button('Get Recommendations', key = 'Recommendations'):
    
    with st.spinner('Recommending...'):
        recommendations = recommend_combined(
            ingredients_list = ingredients_list,
            excluded_ingredients = exclude_list,
            user_query = user_query,
            ingredients_weight = ingredients_weight,
            description_weight = query_weight
        )

    if recommendations.empty:
        st.write('No recommendations were made, try again!')
    else:
        for index, row in recommendations.iterrows():
            with st.expander(row['Name']):
                st.markdown(f"## {row['Name']}")
                st.write(f"**Similarity:**")
                st.write(f"Ingredient Similarity: {row['IngredientSimilarity']:.2f}")
                st.write(f"Description Similarity: {row['DescriptionSimilarity']:.2f}")
                st.write(f"WeightedScore: {row['WeightedScore']:.2f}")
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
                st.write(f"**Sugars:** {row['SugarContent']:.2f}g")
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
