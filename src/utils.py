"""
Utility functions for the Insurance Churn Prediction project.
Contains business logic, classification, validation, and UI helper functions.
"""
from typing import Dict, Any
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from .config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM, FEATURE_NAMES_MAP

class CustomerInputSchema(BaseModel):
    """
    Pydantic schema for validating single customer input data.
    Ensures data types and ranges are correct before passing to the model.
    """
    feature_0: int = Field(..., ge=18, le=100, description="Age")
    feature_1: int = Field(..., ge=0, le=120, description="Tenure")
    feature_2: float = Field(..., ge=0.0, description="Monthly Premium")
    feature_3: float = Field(..., ge=0.0, description="Total Charges")
    feature_4: int = Field(..., ge=1, le=10, description="Number of Policies")
    feature_5: int = Field(..., ge=0, description="Claim Count")
    feature_6: int = Field(..., ge=0, description="Support Calls")
    feature_7: int = Field(..., description="Payment Method ID")
    feature_8: int = Field(..., description="Auto Renewal ID")
    feature_9: int = Field(..., description="Policy Type ID")
    feature_10: int = Field(..., description="Gender ID")
    feature_11: int = Field(..., ge=0, description="Late Payments")
    feature_12: int = Field(..., ge=0, description="Complaints Raised")
    feature_13: int = Field(..., description="Region ID")
    feature_14: int = Field(..., ge=0, description="Online Login Count")
    feature_15: int = Field(..., description="Discount ID")

def validate_input(data: Dict[str, Any]) -> bool:
    """
    Validates the customer input dictionary against the schema.
    
    Args:
        data (dict): Raw input data from the UI.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        CustomerInputSchema(**data)
        return True
    except ValidationError as e:
        st.error(f"Input Validation Error: {e.errors()[0]['msg']} for {e.errors()[0]['loc'][0]}")
        return False

def classify_risk(prob: float) -> str:
    """
    Classifies the churn risk based on probability.
    
    Args:
        prob (float): Churn probability (0.0 to 1.0)
        
    Returns:
        str: Risk category (High Risk, Medium Risk, Low Risk)
    """
    if prob >= RISK_THRESHOLD_HIGH:
        return "High Risk"
    elif prob >= RISK_THRESHOLD_MEDIUM:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_risk_badge_html(risk: str) -> str:
    """
    Returns HTML for a risk-specific badge based on the classified risk.
    
    Args:
        risk (str): The classified risk string.
        
    Returns:
        str: HTML string for the badge.
    """
    if risk == "High Risk":
        return '<div class="next-step-badge badge-high">NEXT STEP: Send Discount Coupon</div>'
    elif risk == "Medium Risk":
        return '<div class="next-step-badge badge-medium">NEXT STEP: Schedule Follow-up Call</div>'
    else:
        return '<div class="next-step-badge badge-low">NEXT STEP: Offer Loyalty Program</div>'

def apply_custom_styles(css_content: str) -> None:
    """
    Applies custom CSS to the Streamlit app.
    
    Args:
        css_content (str): The CSS string to inject.
    """
    st.markdown(css_content, unsafe_allow_html=True)

def plot_feature_importance(model, feature_names):
    """
    Plots the top feature importances for the model.
    """
    import pandas as pd
    import plotly.express as px
    
    importance = model.feature_importances_
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False).head(10)
    
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title="Key Drivers of Prediction (Top 10)",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
    return fig

def generate_excel_report(df: pd.DataFrame) -> bytes:
    """
    Generates an Excel file in memory from the provided DataFrame.
    
    Args:
        df (pd.DataFrame): Prediction results to export.
        
    Returns:
        bytes: The Excel file content as bytes.
    """
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Churn Predictions')
    return output.getvalue()

def plot_prediction_explanation(model, input_df):
    """
    Plots a SHAP-based local explanation for a single prediction.
    Shows how much each feature contributed to this specific customer's risk.
    """
    import shap
    import plotly.graph_objects as go
    import numpy as np

    # Calculate SHAP values for the single row
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # For LightGBM classification, shap_values is a list for [class0, class1] or just class1.
    # Handle both cases.
    if isinstance(shap_values, list):
        # We want SHAP for class 1 (churn)
        sv = shap_values[1][0]
    else:
        # Some LightGBM versions return values for class 1 only if binary
        sv = shap_values[0]

    # Prepare data for plotting
    raw_features = input_df.columns
    contributions = sv
    
    # Sort by magnitude of contribution
    indices = np.argsort(np.abs(contributions))
    top_indices = indices[-10:] # Top 10 factors

    # Map raw names to readable names
    sorted_features = [FEATURE_NAMES_MAP.get(raw_features[i], raw_features[i]) for i in top_indices]
    sorted_contributions = [contributions[i] for i in top_indices]

    # Create waterfall-like bar chart
    colors = ['#ef4444' if c > 0 else '#10b981' for c in sorted_contributions]
    
    fig = go.Figure(go.Bar(
        x=sorted_contributions,
        y=sorted_features,
        orientation='h',
        marker_color=colors,
        text=[f"{'+ ' if c > 0 else ''}{c:.3f}" for c in sorted_contributions],
        textposition='auto',
    ))

    fig.update_layout(
        title="Why this prediction? Local Explained Factors",
        xaxis_title="Impact on Probability (High = Increases Risk)",
        yaxis_title="Customer Feature",
        height=450,
        margin=dict(l=20, r=20, t=40, b=40)
    )
    
    return fig
