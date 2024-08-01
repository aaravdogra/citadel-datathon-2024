import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the food access state data
df_food_access_state = pd.read_csv('food_access_state_data_2019.csv')

# Remove columns that have "half" in their name
columns_to_drop = [col for col in df_food_access_state.columns if 'half' in col]
df_food_access_state_cleaned = df_food_access_state.drop(columns=columns_to_drop)

# Group by 'State' and calculate the mean for numeric columns
df_food_access_state_cleaned_mean = df_food_access_state_cleaned.groupby('State').mean(numeric_only=True)

# Load the newly uploaded files
df_veggie_state = pd.read_csv('veggie-state-2019-cleaned.csv')
df_obesity_state = pd.read_csv('obesity-state-2019-cleaned.csv')
df_fruit_state = pd.read_csv('fruit-state-2019-cleaned.csv')

# Rename 'LocationAbbr' to 'State'
df_veggie_state.rename(columns={'LocationAbbr': 'State'}, inplace=True)
df_obesity_state.rename(columns={'LocationAbbr': 'State'}, inplace=True)
df_fruit_state.rename(columns={'LocationAbbr': 'State'}, inplace=True)

# Group by 'State' and calculate the mean for numeric columns
df_veggie_state_mean = df_veggie_state.groupby('State').mean(numeric_only=True)
df_obesity_state_mean = df_obesity_state.groupby('State').mean(numeric_only=True)
df_fruit_state_mean = df_fruit_state.groupby('State').mean(numeric_only=True)

# Add lapop1share_Avg and Urban_Percentage to the cleaned food access data
urban_percentage = df_food_access_state_cleaned_mean['Urban_Percentage']

# Create new lists from specific columns in the food_access_state_data_2019 dataset
pop2010_sum = df_food_access_state_cleaned_mean['Pop2010_Sum']
ohu2010_sum = df_food_access_state_cleaned_mean['OHU2010_Sum']
lapop1_sum = df_food_access_state_cleaned_mean['lapop1_Sum']
lalowi1_sum = df_food_access_state_cleaned_mean['lalowi1_Sum']
lahunv1_sum = df_food_access_state_cleaned_mean['lahunv1_Sum']
median_family_income_avg = df_food_access_state_cleaned_mean['MedianFamilyIncome_Avg']

# Combine the new lists with the existing combined DataFrame
df_combined = pd.DataFrame({
    'Veggie': df_veggie_state_mean.mean(axis=1),
    'Obesity Rate': df_obesity_state_mean.mean(axis=1),
    'Fruit Rate': df_fruit_state_mean.mean(axis=1),
    'Population': pop2010_sum,
    'Occupied House': ohu2010_sum,
    'Population 1 mile far away': lapop1_sum,
    'Poor population': lalowi1_sum,
    'Population with no car and more than 1 mile away': lahunv1_sum,
    'Median income': median_family_income_avg,
    'Urban Percentage': urban_percentage
})

# Fill any missing values with 0
df_combined.fillna(0, inplace=True)

# Create a correlation matrix
correlation_matrix = df_combined.corr()

# Generate a triangular mask for the heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='crest', center=0, mask=mask, square=True)
plt.title('Correlation Matrix Heatmap (Triangular)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Save the cleaned DataFrame to CSV for review
df_combined.to_csv('/mnt/data/combined_data_final.csv', index=True)
