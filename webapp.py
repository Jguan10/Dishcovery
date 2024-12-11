import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import zstandard as zstd

with st.spinner('Loading models...'):
    with open('models/nearest_neighbors_model.pkl', 'rb') as f:
        nearest_neighbors = pickle.load(f)
    tfidf_matrix = scipy.sparse.load_npz('models/tfidf_matrix.npz')
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

with st.spinner('Loading data...'):
    df1 = pd.read_csv('Data/revised_recipes_1_1.csv.zst', compression="zstd")
    df2 = pd.read_csv('Data/revised_recipes_1_2.csv.zst', compression="zstd")
    df1 = pd.concat([df1, df2], ignore_index=True)

    df3 = pd.read_csv('Data/revised_recipes_2_1.csv.zst', compression='zstd')
    df4 = pd.read_csv('Data/revised_recipes_2_2.csv.zst', compression='zstd')
    df5 = pd.read_csv('Data/revised_recipes_2_3.csv.zst', compression='zstd')
    df5 = pd.concat([df5, df4], ignore_index=True)
    df5 = pd.concat([df5, df3], ignore_index=True)
    
    data = pd.merge(df1, df5, on = 'ID', how = 'inner')

lemmatizer = WordNetLemmatizer()

@st.cache_data
def lemmatize_string(string):
    string_lower = string.lower()
    tokens = word_tokenize(string_lower)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

@st.cache_data
def lemmatize_list(list):
    return [lemmatizer.lemmatize(item.lower()) for item in list]

@st.cache_data
def recommend(preferred_ingredients, top_n=5, excluded_ingredients=None):
    if not preferred_ingredients or not preferred_ingredients.strip():
        return pd.DataFrame()
     
    excluded_ingredients = None if not excluded_ingredients.strip() else excluded_ingredients

    if excluded_ingredients is None:
        excluded_ingredients = []
    elif isinstance(excluded_ingredients, str):
        excluded_ingredients = excluded_ingredients.lower().split(', ')

    lemmatized_ingredients = lemmatize_string(preferred_ingredients)

    user_vector = vectorizer.transform([lemmatized_ingredients])
    
    distances, indices = nearest_neighbors.kneighbors(user_vector, n_neighbors=top_n)
    
    recommendations = data.iloc[indices[0]].copy()
    recommendations['Similarity'] = 1 - distances[0]  

    lemmatized_excluded_ingredients = lemmatize_list(excluded_ingredients)
    def contains_excluded(ingredients):
        for excluded in lemmatized_excluded_ingredients:
            if excluded in ingredients:
                return True
        return False
    
    recommendations = recommendations[~recommendations['NLP_Ingredients'].apply(contains_excluded)]
    
    return recommendations

st.title('Dishcovery')
ingredients_list = st.text_input("Which Ingredients Are You Using?")
exclude_list = st.text_input("Any Allergies or Exceptions?")

st.write(f'Your Ingredients List is: {ingredients_list}')
st.write(f'Things to Exclude are: {exclude_list}')

# Display each recipe in an expander
if st.button('Get Recommendations', key = 'Recommendations'):
    with st.spinner('Recommending...'):
        recommendations = recommend(ingredients_list, excluded_ingredients = exclude_list)
        for index, row in recommendations.iterrows():
            with st.expander(row['Name']):
                st.markdown(f"## {row['Name']}")
                st.write(desc)
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

