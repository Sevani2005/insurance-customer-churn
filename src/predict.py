import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/churn_model.pkl")

def predict_churn(input_df: pd.DataFrame):
    
    probabilities = model.predict_proba(input_df)[:, 1]
    return probabilities
