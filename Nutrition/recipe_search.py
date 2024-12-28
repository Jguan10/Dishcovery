import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#making smaller files
@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

recipe_ingredients_tags = load_data('recipes_ingredients_tags.parquet')
recipe_steps = load_data('recipe_steps.parquet')
recipe_features = load_data('recipe_features.parquet')

st.title('Dishcovery!')

#Some important tags, not all to make it simpler for now
tag_options = ['60-minutes-or-less','30-minutes-or-less', '15-minutes-or-less', 'poultry', 'meat', 'vegetables', 'fruits', 'pasta-rice-and-grains', 'dietary',
                'healthy', 'low-carb', 'low-sodium', 'low-saturated-fat', 'low-cholesterol', 'low-fat', 'low-sugar', 'beginner-cook', 'sweet', 'savory']

tags_selected = st.multiselect('Select tags you are interested in:', tag_options)

def present_tags(item_tags, selected):
    return all(string in item_tags for string in selected)

ingredients_selected = st.text_input('Enter ingredients seperated by commas:')
ingredients_selected = ingredients_selected.split(',')

def ingredients_row(row):
    ingredients_matched = row
    ingredients_str = ' '.join(str(ingredients).lower() for ingredients in ingredients_matched)
    
    for item in ingredients_selected:
        item = item.strip('s')
        if item not in ingredients_str:
            return False
    return True

search = st.button('Search for matching recipes!')

if search:
    recipes_rec = recipe_ingredients_tags.copy()
    if tags_selected:
        recipes_rec['tag_match'] = recipes_rec['tags'].apply(present_tags, selected = tags_selected)
        recipes_rec = recipes_rec[recipes_rec['tag_match']==True]
    if ingredients_selected:
        recipes_rec['ingredients_match'] = recipes_rec['ingredients'].apply(ingredients_row)
        recipes_rec = recipes_rec[recipes_rec['ingredients_match']==True]
    recipes_id = recipes_rec['id'].values
    recipe_steps_rec = recipe_steps[recipe_steps['id'].isin(recipes_id)][['id', 'name', 'description']]
    st.write(recipe_steps_rec)

id_num = st.number_input('Does one of these recipes catch your eye? Enter the id number here.', value=0)
get_recipe = st.button('Get Recipe')

if get_recipe:
    recipe = recipe_steps[recipe_steps['id'] == id_num]
    rec_name = recipe['name'].values[0]
    rec_steps = recipe['steps'].values[0]
    rec_ingredients = recipe['ingredients'].values[0]
    link_name = rec_name.replace(' ', '-')
    link_url = f"https://www.food.com/recipe/{link_name}-{id_num}"
    st.write(f"Link to Recipe: [link_url]({link_url})")
    st.write(f"Recipe Name: {rec_name.title()}")
    st.write(f"Ingredients: {rec_ingredients}")
    num = 1
    for step in rec_steps:
        st.write(f"Step {num}: {step.capitalize()}")
        num += 1
else:
    st.write('No recipe selected yet')

# Get similar recipes
sim = st.button('Take a look at some similar recipes')

if sim:
    rec_feat = recipe_features[recipe_features['id'] == id_num]
    rec_feat = rec_feat.drop(columns=['id']).values.reshape(1, -1)

    cosine_sim = cosine_similarity(rec_feat, recipe_features.drop(columns=['id']))
    sim_scores = list(zip(recipe_features['id'].values, cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    rec_indices = [i[0] for i in sim_scores]

    recs = recipe_steps[recipe_steps['id'].isin(rec_indices)][['id', 'name', 'description']]
    st.write(recs)