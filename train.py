"""
Training script for coffee rating prediction models.
Exercises 1 and 2.
"""

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load the coffee analysis data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Exercise 1: Train linear regression model with single feature
print("Training Model 1: Linear Regression (100g_USD -> rating)")
X1 = df[['100g_USD']]
y = df['rating']

model_1 = LinearRegression()
model_1.fit(X1, y)

# Save model 1
with open('model_1.pickle', 'wb') as f:
    pickle.dump(model_1, f)
print("Model 1 saved as model_1.pickle")

# Exercise 2: Train decision tree regressor with two features
print("\nTraining Model 2: Decision Tree Regressor (100g_USD, roast -> rating)")

# Create a dictionary mapping roast categories to numbers
roast_categories = df['roast'].unique()
roast_cat = {category: idx for idx, category in enumerate(roast_categories)}
print(f"Roast categories: {roast_cat}")

# Create numerical column for roast
df['roast_code'] = df['roast'].map(roast_cat)

# Prepare features for model 2
X2 = df[['100g_USD', 'roast_code']]

model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(X2, y)

# Save model 2 directly (for autograder)
with open('model_2.pickle', 'wb') as f:
    pickle.dump(model_2, f)
print("Model 2 saved as model_2.pickle")

# Save roast category dictionary separately
with open('roast_categories.pickle', 'wb') as f:
    pickle.dump(roast_cat, f)
print("Roast categories saved as roast_categories.pickle")

print("\nTraining complete!")
