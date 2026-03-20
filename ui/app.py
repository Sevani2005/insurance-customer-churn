import streamlit as st
import pandas as pd
import lightgbm as lgb
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
from typing import Dict, Any


# Add project root to sys.path for modular imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import (
    GENDER_MAP, AUTO_RENEWAL_MAP, DISCOUNT_MAP, PAYMENT_MAP,
    POLICY_TYPE_MAP, REGION_MAP, PAGE_TITLE, PAGE_ICON, CUSTOM_CSS
)
from src.utils import classify_risk, get_risk_badge_html, apply_custom_styles, validate_input, plot_feature_importance, generate_excel_report
from src.logger import logger

# -----------------------------
# Load data & model
# -----------------------------
@st.cache_resource
def load_model():
    """
    Loads the trained LightGBM model from the local disk.
    Uses st.cache_resource to avoid reloading the model on every rerun.
    """
    logger.info("Starting model loading process...")

    try:
        model_path = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
        model = joblib.load(model_path)
        feature_cols = [f"feature_{i}" for i in range(16)]
        logger.info(f"Model successfully loaded from {model_path}")
        return model, feature_cols
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        st.error("Critical Error: AI model file not found or corrupted.")
        return None, []

model, feature_cols = load_model()

# -----------------------------
# Load test data for batch predictions
# -----------------------------
@st.cache_data
def load_test_data():
    """
    Loads the test CSV dataset for batch analysis.
    Uses st.cache_data for faster performance on subsequent loads.
    """
    test_path = os.path.join(PROJECT_ROOT, "data", "Insurance_Churn_ParticipantsData", "Test.csv")

    test_data = pd.read_csv(test_path)
    return test_data

# -----------------------------
# UI Initialization
# -----------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_styles(CUSTOM_CSS)


# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🛡️ Insurance Churn Prediction</div>
    <div class="hero-subtitle">AI-Powered Customer Retention Intelligence System</div>
</div>
""", unsafe_allow_html=True)

# Initialize Session State for History
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### About This Tool")
    st.markdown("""
    <div class="glass-card">
    This application uses <strong>LightGBM</strong> machine learning to predict customer churn risk with high accuracy.
    
    <br><br><strong>Features:</strong>
    <ol>
        <li>Single customer prediction</li>
        <li>Batch customer analysis</li>
        <li>Risk assessment & recommendations</li>
        <li>Interactive visualizations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Risk Categories")
    st.markdown("""
    <div class="glass-card">
    <span style="color: #ef4444; font-size: 1.2rem;">●</span> <strong>High Risk</strong> (≥70%)<br>
    <span style="color: #94a3b8; font-size: 0.9rem;">→ Immediate action needed</span><br><br>
    
    <span style="color: #f59e0b; font-size: 1.2rem;">●</span> <strong>Medium Risk</strong> (40-69%)<br>
    <span style="color: #94a3b8; font-size: 0.9rem;">→ Monitor closely</span><br><br>
    
    <span style="color: #10b981; font-size: 1.2rem;">●</span> <strong>Low Risk</strong> (<40%)<br>
    <span style="color: #94a3b8; font-size: 0.9rem;">→ Stable customer</span>
    </div>
    """, unsafe_allow_html=True)

    # NEW: Recent Checks Section
    st.markdown("---")
    st.markdown("### Recent Checks")
    if not st.session_state.prediction_history:
        st.info("No predictions yet.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.prediction_history[-5:])):
            color = "#ef4444" if entry['risk'] == "High Risk" else "#f59e0b" if entry['risk'] == "Medium Risk" else "#10b981"
            st.markdown(f"""
            <div style="background: #1e293b; border-left: 5px solid {color}; padding: 10px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05);">
                <div style="font-size: 0.7rem; color: #94a3b8;">Check #{len(st.session_state.prediction_history)-idx}</div>
                <div style="font-weight: 700; color: #f1f5f9;">{entry['risk']}</div>
                <div style="font-size: 0.8rem; color: #6366f1;">{entry['prob']:.1f}% Probability</div>
            </div>
            """, unsafe_allow_html=True)

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Single Customer Prediction", "Batch Customer Analysis", "What-If Simulation"])


# ============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ============================================================================
with tab1:
    input_data: Dict[str, Any] = {}


    
    # Demographics Section (3 features)
    st.markdown('<div class="section-header">Customer Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data["feature_0"] = st.number_input("Age", min_value=18, max_value=100, value=35, help="Customer's age in years")
    
    with col2:
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()), help="Customer's gender")
        input_data["feature_10"] = GENDER_MAP[gender]
    
    with col3:
        region = st.selectbox("Region", list(REGION_MAP.keys()), help="Customer's geographic region")
        input_data["feature_13"] = REGION_MAP[region]
    
    # Policy Information Section (3 features)
    st.markdown('<div class="section-header">Policy Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data["feature_1"] = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, help="How long customer has been with us")
    
    with col2:
        policy = st.selectbox("Policy Type", list(POLICY_TYPE_MAP.keys()), help="Type of insurance policy")
        input_data["feature_9"] = POLICY_TYPE_MAP[policy]
    
    with col3:
        input_data["feature_4"] = st.number_input("Number of Policies", min_value=1, max_value=10, value=1, help="Total active policies held")
    
    # Financial Information Section (3 features)
    st.markdown('<div class="section-header">Financial Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data["feature_2"] = st.number_input("Monthly Premium ($)", min_value=0.0, max_value=10000.0, value=100.0, step=10.0, help="Monthly premium amount")
    
    with col2:
        input_data["feature_3"] = st.number_input("Total Charges ($)", min_value=0.0, max_value=100000.0, value=1200.0, step=100.0, help="Total amount charged to date")
    
    with col3:
        payment = st.selectbox("Payment Method", list(PAYMENT_MAP.keys()), help="Preferred payment method")
        input_data["feature_7"] = PAYMENT_MAP[payment]
    
    # Engagement & Service Section (3 features)
    st.markdown('<div class="section-header">Customer Engagement</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data["feature_5"] = st.number_input("Claim Count", min_value=0, max_value=20, value=0, help="Number of insurance claims filed")
    
    with col2:
        input_data["feature_6"] = st.number_input("Support Calls", min_value=0, max_value=20, value=0, help="Number of customer support calls made")
    
    with col3:
        auto = st.selectbox("Auto Renewal", list(AUTO_RENEWAL_MAP.keys()), help="Is auto-renewal enabled?")
        input_data["feature_8"] = AUTO_RENEWAL_MAP[auto]
    
    # Behavior Section (4 features)
    st.markdown('<div class="section-header">Customer Behavior</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        input_data["feature_14"] = st.number_input("Online Login Count", min_value=0, max_value=100, value=5, help="Number of times logged into online portal")
    
    with col2:
        discount = st.selectbox("Discount Availed", list(DISCOUNT_MAP.keys()), help="Has customer used any discounts?")
        input_data["feature_15"] = DISCOUNT_MAP[discount]
    
    with col3:
        input_data["feature_11"] = st.number_input("Late Payments", min_value=0, max_value=20, value=0, help="Number of late payment instances")
    
    with col4:
        input_data["feature_12"] = st.number_input("Complaints Raised", min_value=0, max_value=10, value=0, help="Number of formal complaints filed")
    
    # Ensure all features are present
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = 0
    
    input_df = pd.DataFrame([input_data])
    
    # Prediction Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("Predict Churn Risk", use_container_width=True, key="single_predict")
    if predict_button:
        if not validate_input(input_data):
            st.stop()
            
        with st.spinner('Analyzing customer data with AI...'):
            time.sleep(0.8)

            
            try:
                prob = model.predict_proba(input_df)[0][1]
                risk = classify_risk(prob)
                
                logger.info(f"Single Prediction: Probability={prob:.4f}, Risk={risk}")
                
                # Save to History
                st.session_state.prediction_history.append({
                    'prob': prob * 100,
                    'risk': risk
                })
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                st.error("An error occurred during prediction. Check logs for details.")
                prob, risk = 0.0, "Error"
        
        # Results Section
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Churn Probability",
                value=f"{prob*100:.1f}%",
                delta=f"{prob*100 - 50:.1f}% vs avg" if prob > 0.5 else f"{50 - prob*100:.1f}% below avg",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Risk Level",
                value=f"{risk}"
            )
        
        with col3:
            retention_score = int((1 - prob) * 100)
            st.metric(
                label="Retention Score",
                value=f"{retention_score}/100"
            )
        
        # Visual Risk Indicator
        st.markdown("<br>", unsafe_allow_html=True)
        if risk == "High Risk":
            st.error("HIGH RISK CUSTOMER - Immediate action required!")
        elif risk == "Medium Risk":
            st.warning("MEDIUM RISK CUSTOMER - Monitor closely")
        else:
            st.success("LOW RISK CUSTOMER - Customer is stable")
        
        # Progress bar
        st.markdown("**Churn Risk Visualization:**")
        st.progress(prob)

        # Automated Next Step Badge
        st.markdown("<br>", unsafe_allow_html=True)
        if risk == "High Risk":
            badge_html = '<div class="next-step-badge badge-high">NEXT STEP: Send Discount Coupon</div>'
        elif risk == "Medium Risk":
            badge_html = '<div class="next-step-badge badge-medium">NEXT STEP: Schedule Follow-up Call</div>'
        else:
            badge_html = '<div class="next-step-badge badge-low">NEXT STEP: Offer Loyalty Program</div>'
        
        st.markdown(badge_html, unsafe_allow_html=True)

        # Feature Importance section
        st.markdown('<div class="section-header">Decision Factors</div>', unsafe_allow_html=True)
        fig_imp = plot_feature_importance(model, feature_cols)
        st.plotly_chart(fig_imp, use_container_width=True)


# ============================================================================
# TAB 2: BATCH CUSTOMER ANALYSIS
# ============================================================================
with tab2:
    st.markdown('<div class="section-header">Batch Customer Analysis</div>', unsafe_allow_html=True)
    
    # Load test data
    test_data = load_test_data()
    total_customers = len(test_data)
    
    st.info(f"Total Customers Available: {total_customers:,}")
    
    # User controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_customers = st.number_input(
    "How many customers to analyze?",
    min_value=1,
    max_value=total_customers,
    value=20,
    step=1,
    help="Enter number of customers to analyze"
)

    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("Analyze Customers", use_container_width=True, key="batch_analyze")
    
    if analyze_button:
        with st.spinner(f'Analyzing {num_customers} customers...'):
            import time
            time.sleep(1)
            
            # Get subset of data
            subset_data = test_data.head(num_customers)
            
            # Make predictions
            predictions = model.predict_proba(subset_data)[:, 1]
            risks = [classify_risk(p) for p in predictions]
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Customer_ID': range(1, len(subset_data) + 1),
                'Churn_Probability': predictions,
                'Risk_Level': risks,
                'Probability_%': (predictions * 100).round(1)
            })
            
            # Add some key features for display
            results_df['Age'] = subset_data['feature_0'].values
            results_df['Tenure'] = subset_data['feature_1'].values
            results_df['Monthly_Premium'] = subset_data['feature_2'].values
            
            # Summary Statistics
            st.markdown('<div class="section-header">Summary Statistics</div>', unsafe_allow_html=True)
            
            high_risk_count = sum(1 for r in risks if r == "High Risk")
            medium_risk_count = sum(1 for r in risks if r == "Medium Risk")
            low_risk_count = sum(1 for r in risks if r == "Low Risk")
            avg_prob = predictions.mean() * 100
            
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">High Risk</div>
                    <div class="metric-value color-high">{high_risk_count}</div>
                    <div class="metric-delta color-low">↑ {high_risk_count/len(risks)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Medium Risk</div>
                    <div class="metric-value color-medium">{medium_risk_count}</div>
                    <div class="metric-delta color-low">↑ {medium_risk_count/len(risks)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Low Risk</div>
                    <div class="metric-value color-low">{low_risk_count}</div>
                    <div class="metric-delta color-low">↑ {low_risk_count/len(risks)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Churn Prob</div>
                    <div class="metric-value color-neutral">{avg_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualizations
            st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                risk_counts = pd.Series(risks).value_counts()
                fig_pie = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Customer Risk Distribution",
                    color=risk_counts.index,
                    color_discrete_map={
                        "High Risk": "#ef4444",
                        "Medium Risk": "#f59e0b",
                        "Low Risk": "#10b981"
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart
                fig_bar = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title="Customer Count by Risk Level",
                    labels={'x': 'Risk Level', 'y': 'Number of Customers'},
                    color=risk_counts.index,
                    color_discrete_map={
                        "High Risk": "#ef4444",
                        "Medium Risk": "#f59e0b",
                        "Low Risk": "#10b981"
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            
            # Results Table
            st.markdown('<div class="section-header">Customer Results</div>', unsafe_allow_html=True)
            
            # Display table
            display_df = results_df[['Customer_ID', 'Risk_Level', 'Probability_%', 'Age', 'Tenure', 'Monthly_Premium']]
            
            # Color code the dataframe
            def highlight_risk(row):
                if row['Risk_Level'] == 'High Risk':
                    return ['background-color: #fee2e2; color: #991b1b; font-weight: 600'] * len(row)
                elif row['Risk_Level'] == 'Medium Risk':
                    return ['background-color: #fef3c7; color: #92400e; font-weight: 600'] * len(row)
                else:
                    return ['background-color: #d1fae5; color: #065f46; font-weight: 600'] * len(row)
            
            styled_df = display_df.style.apply(highlight_risk, axis=1)
            st.dataframe(
                styled_df, 
                use_container_width=True, 
                height=450,
                hide_index=True
            )
            
            # Download buttons
            c1, c2 = st.columns(2)
            with c1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"churn_predictions_{num_customers}_customers.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c2:
                excel_data = generate_excel_report(results_df)
                st.download_button(
                    label="Download Results as Excel",
                    data=excel_data,
                    file_name=f"churn_predictions_{num_customers}_customers.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

            
            # High Risk Customers Alert
            if high_risk_count > 0:
                st.markdown('<div class="section-header">High Risk Customers Alert</div>', unsafe_allow_html=True)
                high_risk_df = results_df[results_df['Risk_Level'] == 'High Risk']
                st.error(f"**{high_risk_count} customers require immediate attention!**")
                st.dataframe(high_risk_df[['Customer_ID', 'Probability_%', 'Age', 'Tenure', 'Monthly_Premium']], use_container_width=True)

# ============================================================================
# TAB 3: WHAT-IF SIMULATION
# ============================================================================
with tab3:
    st.markdown('<div class="section-header">What-If Retention Simulator</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
    Simulate how business interventions (like discounts or policy changes) impact a customer's churn risk.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Base Customer Profile")
        base_age = st.slider("Customer Age", 18, 100, 45)
        base_tenure = st.slider("Tenure (Months)", 0, 120, 24)
        base_premium = st.number_input("Current Monthly Premium ($)", value=250.0)
        base_discount = st.radio("Has Discount?", ["No", "Yes"], index=0)
    
    with col2:
        st.subheader("Proposed Interventions")
        new_premium = st.number_input("Adjusted Premium ($)", value=base_premium * 0.9, help="Try lowering the premium to see the impact")
        new_discount = st.radio("Apply New Discount?", ["No", "Yes"], index=1)
        improve_support = st.checkbox("Improve Support (Zero Complaints/Calls)", value=True)

    if st.button("Run Simulation", use_container_width=True):
        # Create base and simulated dataframes
        base_data = {f"feature_{i}": 0.0 for i in range(16)}
        base_data["feature_0"] = base_age
        base_data["feature_1"] = base_tenure
        base_data["feature_2"] = base_premium
        base_data["feature_15"] = DISCOUNT_MAP[base_discount]
        
        sim_data = base_data.copy()
        sim_data["feature_2"] = new_premium
        sim_data["feature_15"] = DISCOUNT_MAP[new_discount]
        if improve_support:
            sim_data["feature_6"] = 0 # Support calls
            sim_data["feature_12"] = 0 # Complaints
            
        base_df = pd.DataFrame([base_data])
        sim_df = pd.DataFrame([sim_data])
        
        base_prob = model.predict_proba(base_df)[0][1]
        sim_prob = model.predict_proba(sim_df)[0][1]
        
        # Display Results
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        diff = (sim_prob - base_prob) * 100
        
        with c1:
            st.metric("Base Churn Risk", f"{base_prob*100:.1f}%")
        with c2:
            st.metric("Simulated Risk", f"{sim_prob*100:.1f}%", delta=f"{diff:.1f}%", delta_color="inverse")
            
        if diff < 0:
            st.success(f"Success! The proposed changes reduce churn risk by **{abs(diff):.1f}%**.")
        else:
            st.warning("Warning: The proposed changes do not significantly reduce churn risk.")

        # Comparison Chart
        fig = go.Figure(data=[
            go.Bar(name='Base', x=['Churn Risk'], y=[base_prob*100], marker_color='#94a3b8'),
            go.Bar(name='Simulated', x=['Churn Risk'], y=[sim_prob*100], marker_color='#5d5fef')
        ])
        fig.update_layout(title="Impact Visualization", barmode='group', height=300)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <p style="font-size: 1.1rem; color: #667eea; font-weight: 600;">
        🛡️ Insurance Churn Prediction System v2.1
    </p>
    <p style="color: #94a3b8;">
        Powered by LightGBM Machine Learning | Built with Streamlit
    </p>
    <p style="color: #94a3b8; font-size: 0.9rem;">
        Analyzing 16 customer features for accurate churn prediction
    </p>
</div>
""", unsafe_allow_html=True)
