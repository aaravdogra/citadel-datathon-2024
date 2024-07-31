import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_clean_food_access = pd.read_csv('/data/other-data-clean/clean_food_access_2019.csv')
df_food_environ_county = pd.read_csv('/data/other-data-clean/food_environ_county_data_clean.csv')
df_food_environ_state = pd.read_csv('/data/other-data-clean/food_environ_state_data_clean.csv')
df_obesity_2011 = pd.read_csv('/data/provided-data-clean/obesity-state-2011-cleaned.csv')
df_obesity_2015 = pd.read_csv('data/provided-data-clean/obesity-state-2015-cleaned.csv')
df_nutrition_and_obesity = pd.read_csv('/data/provided-data-clean/nutrition-and-obesity-cleaned.csv')
df_obesity_2011.rename(columns={'LocationAbbr': 'State'}, inplace=True)
df_obesity_2015.rename(columns={'LocationAbbr': 'State'}, inplace=True)
df_nutrition_and_obesity.rename(columns={'LocationAbbr': 'State'}, inplace=True)
df_clean_food_access_mean = df_clean_food_access.groupby('State').mean(numeric_only=True)
df_food_environ_county_mean = df_food_environ_county.groupby('State').mean(numeric_only=True)
df_food_environ_state_mean = df_food_environ_state.groupby('State').mean(numeric_only=True)
df_obesity_2011_mean = df_obesity_2011.groupby('State').mean(numeric_only=True)
df_obesity_2015_mean = df_obesity_2015.groupby('State').mean(numeric_only=True)
df_nutrition_and_obesity_mean = df_nutrition_and_obesity.groupby('State').mean(numeric_only=True)
df_combined = pd.DataFrame({
    'clean_food_access_2019': df_clean_food_access_mean.mean(axis=1),
    'food_environ_county_data_clean': df_food_environ_county_mean.mean(axis=1),
    'food_environ_state_data_clean': df_food_environ_state_mean.mean(axis=1),
    'obesity_state_2011': df_obesity_2011_mean['Mean_Obesity_Pct'],
    'obesity_state_2015': df_obesity_2015_mean['Mean_Obesity_Pct'],
    'nutrition_and_obesity_cleaned': df_nutrition_and_obesity_mean.mean(axis=1)
})

df_combined.fillna(0, inplace=True)

correlation_matrix = df_combined.corr()

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='crest', center=0, mask=mask, square=True)
plt.title('Correlation Matrix Heatmap (Triangular)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
