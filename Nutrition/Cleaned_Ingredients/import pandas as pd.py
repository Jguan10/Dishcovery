# import pandas as pd
# import re
# from fractions import Fraction
# import os

# file_path = 'Cleaned_Ingredients/recipes_food_com_cleaned.csv'

# if not os.path.exists(file_path):
#     print(f"File not found: {file_path}")
# else:
#     df = pd.read_csv(file_path)

#     conversion_factors = {
#         'lb': 453.592,
#         'tsp': 4.92892,
#         'Tbsp': 14.7868,
#         'cup': 240,
#         'oz': 28.3495,
#         'pinch': 0.31,
#     }

#     # Function to convert ingredient quantities to grams
#     def convert_to_grams(ingredient):
#         for unit, factor in conversion_factors.items():
#             if unit in ingredient:
#                 match = re.search(r'(\d+[\d\/\s]*)', ingredient)
#                 if match:
#                     quantity_str = match.group(1).strip()
#                     try:
#                         if '/' in quantity_str:
#                             quantity = sum(float(Fraction(s)) for s in quantity_str.split())
#                         else:
#                             quantity = sum(float(part) for part in re.split(r'\s+', quantity_str))
#                         return quantity * factor
#                     except ValueError:
#                         return ingredient
#                 elif unit == 'pinch':
#                     return factor
#                 elif unit == 'small red onion':
#                     quantity = float(ingredient.split()[0].replace('½', '0.5'))
#                     return quantity * factor
#         return ingredient

#     df['IngredientsRawInGrams'] = df['IngredientsRaw'].apply(convert_to_grams)

#     df.to_csv(file_path, index=False)

#     print(f"The ingredients have been converted to grams and saved to '{file_path}'.")


import pandas as pd
import re
from fractions import Fraction

# Load the CSV file named 'Cleaned_Ingredients/recipes_food_com_cleaned.csv'
df = pd.read_csv('Cleaned_Ingredients/recipes_food_com_cleaned.csv')

# Conversion factors (example for common ingredients)
conversion_factors = {
    'lb': 453.592,
    'tsp': 4.92892,
    'Tbsp': 14.7868,
    'cup': 240,
    'oz': 28.3495,
    'pinch': 0.31,
}

# Function to convert ingredient quantities to grams
def convert_to_grams(ingredient):
    for unit, factor in conversion_factors.items():
        if unit in ingredient:
            # Extract the quantity using regex to handle fractions and other formats
            match = re.search(r'(\d+[\d\/\s]*)', ingredient)
            if match:
                quantity_str = match.group(1).strip()
                # Convert fractions to float
                try:
                    if '/' in quantity_str:
                        quantity = sum(float(Fraction(s)) for s in quantity_str.split())
                    else:
                        quantity = sum(float(part) for part in re.split(r'\s+', quantity_str))
                    converted_quantity = quantity * factor
                    # Replace the original quantity with the converted quantity in grams
                    ingredient = ingredient.replace(match.group(1), f"{converted_quantity:.2f}g")
                except ValueError:
                    # Handle cases where the quantity is not a valid float
                    return ingredient
            elif unit == 'pinch':
                ingredient = ingredient.replace('pinch', f"{factor:.2f}g")
            elif unit == 'small red onion':
                quantity = float(ingredient.split()[0].replace('½', '0.5'))
                converted_quantity = quantity * factor
                ingredient = ingredient.replace(ingredient.split()[0], f"{converted_quantity:.2f}g")
    return ingredient

# Apply conversion
df['IngredientsRawInGrams'] = df['IngredientsRaw'].apply(convert_to_grams)
df['IngredientsRawInGrams'].remove('lb',)
# Save the updated dataframe to a new CSV file for testing purposes
df.to_csv('test.csv', index=False)

print("The ingredients have been converted to grams and saved to 'test.csv'.")