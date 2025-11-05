"""
Utility functions for the coffee rating prediction app.
Exercise 3.
"""

import pandas as pd
import pickle
import numpy as np

def predict_rating(df_X, text=False):
    """
    Predict coffee ratings based on input features.
    
    Parameters
    ----------
    df_X : pd.DataFrame or array-like
        If text=False: DataFrame with columns '100g_USD' and 'roast'
        If text=True: DataFrame with column 'text' containing review text
    text : bool, optional
        Whether to use text-based prediction (default: False)
    
    Returns
    -------
    np.ndarray
        Array of predicted ratings
    """
    if text:
        # Bonus exercise - text-based prediction
        try:
            with open('model_3.pickle', 'rb') as f:
                model_3_data = pickle.load(f)
            model_3 = model_3_data['model']
            vectorizer = model_3_data['vectorizer']
            
            # Vectorize the input text
            X_vectorized = vectorizer.transform(df_X.iloc[:, 0])
            
            # Make predictions
            predictions = model_3.predict(X_vectorized)
            return predictions
        except FileNotFoundError:
            raise FileNotFoundError("model_3.pickle not found. Please train the text model first.")
    else:
        # Load both models
        with open('model_1.pickle', 'rb') as f:
            model_1 = pickle.load(f)
        
        with open('model_2.pickle', 'rb') as f:
            model_2_data = pickle.load(f)
        model_2 = model_2_data['model']
        roast_cat = model_2_data['roast_cat']
        
        # Initialize predictions array
        predictions = np.zeros(len(df_X))
        
        # Process each row
        for idx, row in df_X.iterrows():
            roast_value = row['roast']
            price_value = row['100g_USD']
            
            # Check if roast value is in our training categories
            if roast_value in roast_cat:
                # Use model_2 (both features)
                roast_code = roast_cat[roast_value]
                X_input = pd.DataFrame([[price_value, roast_code]], 
                                      columns=['100g_USD', 'roast_code'])
                predictions[idx] = model_2.predict(X_input)[0]
            else:
                # Use model_1 (only price feature)
                X_input = pd.DataFrame([[price_value]], columns=['100g_USD'])
                predictions[idx] = model_1.predict(X_input)[0]
        
        return predictions