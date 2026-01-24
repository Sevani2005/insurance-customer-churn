import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# -----------------------------
# Load data & train model
# -----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        base_dir, "..", "data", "Insurance_Churn_ParticipantsData", "Train.csv"
    )
    data = pd.read_csv(data_path)

    X = data.drop(columns=["labels"])
    y = data["labels"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


model = load_model()

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
# UI → Model mappings
# -----------------------------
GENDER_MAP = {"Male": 0, "Female": 1}
REGION_MAP = {"North": 0, "South": 1, "East": 2, "West": 3}
POLICY_MAP = {"Basic": 0, "Silver": 1, "Gold": 2}
PAYMENT_MAP = {"Credit Card": 0, "Debit Card": 1, "UPI": 2, "Net Banking": 3}
AUTO_RENEW_MAP = {"No": 0, "Yes": 1}
DISCOUNT_MAP = {"No": 0, "Yes": 1}

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Insurance Churn Prediction",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Insurance Customer Churn Prediction")
st.write("Enter customer details to predict churn risk")

# -----------------------------
# UI INPUTS (16 FEATURES)
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (months)", 0, 120, 12)
    num_policies = st.number_input("Number of Policies", 1, 10, 1)
    claim_count = st.number_input("Claim Count", 0, 20, 0)
    late_payments = st.number_input("Late Payments", 0, 20, 0)
    complaints = st.number_input("Complaints Raised", 0, 10, 0)

with col2:
    gender = st.selectbox("Gender", GENDER_MAP.keys())
    region = st.selectbox("Region", REGION_MAP.keys())
    policy = st.selectbox("Policy Type", POLICY_MAP.keys())
    payment = st.selectbox("Payment Method", PAYMENT_MAP.keys())
    auto_renew = st.selectbox("Auto Renewal", AUTO_RENEW_MAP.keys())
    discount = st.selectbox("Discount Availed", DISCOUNT_MAP.keys())

with col3:
    monthly_premium = st.number_input("Monthly Premium", 0.0, 100000.0, 1000.0)
    total_charges = st.number_input("Total Charges", 0.0, 1000000.0, 12000.0)
    support_calls = st.number_input("Support Calls", 0, 20, 0)
    login_count = st.number_input("Online Login Count", 0, 100, 5)

# -----------------------------
# Convert UI → model format
# (ORDER MUST MATCH TRAIN DATA)
# -----------------------------
input_data = [[
    age,                               # feature_0
    tenure,                            # feature_1
    monthly_premium,                  # feature_2
    total_charges,                    # feature_3
    num_policies,                     # feature_4
    claim_count,                      # feature_5
    support_calls,                    # feature_6
    PAYMENT_MAP[payment],             # feature_7
    AUTO_RENEW_MAP[auto_renew],        # feature_8
    POLICY_MAP[policy],               # feature_9
    GENDER_MAP[gender],               # feature_10
    0,                                # feature_11 (reserved / safe default)
    0,                                # feature_12 (reserved / safe default)
    REGION_MAP[region],               # feature_13
    login_count,                      # feature_14
    DISCOUNT_MAP[discount]             # feature_15
]]

input_df = pd.DataFrame(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]
    risk = classify_risk(prob)

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {prob:.2f}")
    st.write(f"**Risk Level:** {risk}")

    st.subheader("Suggested Action")

    if risk == "High Risk":
        st.error("Immediate retention action required")
        st.write("• Offer discounts or personalized plans")
        st.write("• Priority customer support")

    elif risk == "Medium Risk":
        st.warning("Monitor customer closely")
        st.write("• Send engagement offers")

    else:
        st.success("Customer is stable")
        st.write("• Normal follow-up")
