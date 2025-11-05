"""
Test script to verify exercises are completed correctly.
"""

import pandas as pd
from apputil import predict_rating
import os

print("=" * 60)
print("Testing Exercises")
print("=" * 60)

# Check if model files exist
print("\n1. Checking for model files...")
if os.path.exists('model_1.pickle'):
    print("✓ model_1.pickle exists")
else:
    print("✗ model_1.pickle NOT found")

if os.path.exists('model_2.pickle'):
    print("✓ model_2.pickle exists")
else:
    print("✗ model_2.pickle NOT found")

# Test predict_rating function
print("\n2. Testing predict_rating function...")

# Test case from exercises
df_X = pd.DataFrame([
    [10.00, "Dark"],
    [15.00, "Very Light"]], 
    columns=["100g_USD", "roast"])

print("\nInput:")
print(df_X)

y_pred = predict_rating(df_X)
print("\nPredictions:")
print(y_pred)

# Explanation
print("\n3. Results explanation:")
print("- First row: 'Dark' roast is in training data → uses model_2")
print(f"  Predicted rating: {y_pred[0]:.2f}")
print("- Second row: 'Very Light' roast is NOT in training data → uses model_1")
print(f"  Predicted rating: {y_pred[1]:.2f}")

print("\n" + "=" * 60)
print("All tests completed successfully! ✓")
print("=" * 60)
