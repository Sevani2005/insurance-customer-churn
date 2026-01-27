import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------
# Load data & train model
# -----------------------------
import joblib

@st.cache_resource
def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        script_dir, "..", "models", "churn_model.pkl"
    )

    model = joblib.load(model_path)

    feature_cols = [f"feature_{i}" for i in range(16)]

    return model, feature_cols


model, feature_cols = load_model()

# -----------------------------
# Load test data for batch predictions
# -----------------------------
@st.cache_data
def load_test_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(script_dir, "..", "data", "Insurance_Churn_ParticipantsData", "Test.csv")
    test_data = pd.read_csv(test_path)
    return test_data

# -----------------------------
# Risk classification
# -----------------------------
def classify_risk(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

# -----------------------------
# Categorical mappings (UI → Model)
# -----------------------------
GENDER_MAP = {"Male": 0, "Female": 1}

AUTO_RENEWAL_MAP = {"No": 0, "Yes": 1}

DISCOUNT_MAP = {"No": 0, "Yes": 1}

PAYMENT_MAP = {
    "Credit Card": 0,
    "Debit Card": 1,
    "UPI": 2,
    "Net Banking": 3
}

POLICY_TYPE_MAP = {
    "Basic": 0,
    "Silver": 1,
    "Gold": 2
}

REGION_MAP = {
    "North": 0,
    "South": 1,
    "East": 2,
    "West": 3
}

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="Insurance Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero section with solid background */
    .hero-section {
        background: #5d5fef;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(93, 95, 239, 0.2);
    }
    
    .hero-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Section headers with solid underline */
    .section-header {
        color: #5d5fef;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #5d5fef;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Info card with solid border */
    .info-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #5d5fef;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .info-card h4 {
        color: #667eea;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .info-card ul {
        margin-left: 1rem;
    }
    
    .info-card li {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    /* Enhanced button with solid color */
    .stButton>button {
        width: 100%;
        background: #5d5fef;
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 8px 25px rgba(93, 95, 239, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(93, 95, 239, 0.5);
        background: #4749d4;
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Metric card colors */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-label { font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.5rem; }
    .metric-value { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .metric-delta { font-size: 0.8rem; margin-top: 0.5rem; }
    
    .color-high { color: #ef4444; }
    .color-medium { color: #f59e0b; }
    .color-low { color: #10b981; }
    .color-neutral { color: #667eea; }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: #5d5fef;
    }
    
    /* Sidebar styling with solid color */
    [data-testid="stSidebar"] {
        background: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Input field enhancements */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    /* Next Step Badge styling */
    .next-step-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .badge-high {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .badge-medium {
        background: rgba(245, 158, 11, 0.1);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    .badge-low {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

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
            <div style="background: white; border-left: 5px solid {color}; padding: 10px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">Check #{len(st.session_state.prediction_history)-idx}</div>
                <div style="font-weight: 700; color: #1e293b;">{entry['risk']}</div>
                <div style="font-size: 0.9rem; color: #5d5fef;">{entry['prob']:.1f}% Probability</div>
            </div>
            """, unsafe_allow_html=True)

# Create tabs for different features
tab1, tab2 = st.tabs(["Single Customer Prediction", "Batch Customer Analysis"])

# ============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ============================================================================
with tab1:
    input_data = {}
    
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
        with st.spinner('Analyzing customer data with AI...'):
            import time
            time.sleep(0.8)
            
            prob = model.predict_proba(input_df)[0][1]
            risk = classify_risk(prob)
            
            # Save to History
            st.session_state.prediction_history.append({
                'prob': prob * 100,
                'risk': risk
            })
        
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
                    return ['background-color: #fee2e2'] * len(row)
                elif row['Risk_Level'] == 'Medium Risk':
                    return ['background-color: #fef3c7'] * len(row)
                else:
                    return ['background-color: #d1fae5'] * len(row)
            
            styled_df = display_df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"churn_predictions_{num_customers}_customers.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # High Risk Customers Alert
            if high_risk_count > 0:
                st.markdown('<div class="section-header">High Risk Customers Alert</div>', unsafe_allow_html=True)
                high_risk_df = results_df[results_df['Risk_Level'] == 'High Risk']
                st.error(f"**{high_risk_count} customers require immediate attention!**")
                st.dataframe(high_risk_df[['Customer_ID', 'Probability_%', 'Age', 'Tenure', 'Monthly_Premium']], use_container_width=True)

# Footer
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
