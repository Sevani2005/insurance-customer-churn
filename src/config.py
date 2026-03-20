"""
Configuration constants and mappings for the Insurance Churn Prediction project.
This centralizes all domain-specific mappings and UI styles.
"""

# Categorical mappings (UI -> Model)
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

# Risk Thresholds
RISK_THRESHOLD_HIGH = 0.7
RISK_THRESHOLD_MEDIUM = 0.4

# Feature names mapping for better UI display
FEATURE_NAMES_MAP = {
    "feature_0": "Age",
    "feature_1": "Tenure (Months)",
    "feature_2": "Monthly Premium ($)",
    "feature_3": "Total Charges ($)",
    "feature_4": "Number of Policies",
    "feature_5": "Claim Count",
    "feature_6": "Support Calls",
    "feature_7": "Payment Method",
    "feature_8": "Auto Renewal",
    "feature_9": "Policy Type",
    "feature_10": "Gender",
    "feature_11": "Late Payments",
    "feature_12": "Complaints Raised",
    "feature_13": "Region",
    "feature_14": "Online Login Count",
    "feature_15": "Discount Availed"
}

# Updated UI Configuration to Pastel
PAGE_TITLE = "Insurance Churn AI"
PAGE_ICON = "🛡️"

# Pastel CSS Styles
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Force Light/Pastel Theme */
    .stApp, .stAppViewContainer, .stHeader {
        background: linear-gradient(160deg, #f0f7ff 0%, #f5f3ff 100%) !important;
        color: #1e293b !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
        background-color: #fcfdff !important;
        border-right: 1px solid #eef2ff !important;
    }

    /* Text Colors */
    h1, h2, h3, p, span, label, .stMarkdown {
        color: #1e293b !important;
    }

    .hero-section {
        background: linear-gradient(135deg, #c7d2fe 0%, #a5b4fc 100%);
        padding: 3.5rem 2rem;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(129, 140, 248, 0.15);
    }
    
    .hero-title {
        color: white; font-size: 3.2rem; font-weight: 800; margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.9); font-size: 1.2rem; font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .glass-card {
        background: white; 
        border-radius: 20px; 
        border: 1px solid #e2e8f0;
        padding: 1.8rem; 
        margin: 1.2rem 0; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: #1e293b;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }
    
    .section-header {
        color: #6366f1; font-size: 1.4rem; font-weight: 700; margin-top: 2.5rem;
        margin-bottom: 1.5rem; padding-bottom: 0.8rem; 
        border-bottom: 2px solid #e2e8f0;
        display: flex; align-items: center; gap: 12px;
    }
    
    .next-step-badge {
        display: inline-block; padding: 0.7rem 1.8rem; border-radius: 16px;
        font-weight: 600; font-size: 0.95rem; margin-top: 1.5rem;
        text-align: center; width: 100%;
    }
    
    .badge-high { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .badge-medium { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .badge-low { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }

    /* Streamlit specific pastel overrides */
    .stButton>button {
        background-color: #6366f1 !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background-color: #4f46e5 !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
</style>
"""
