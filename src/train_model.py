import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb



data = pd.read_csv("data/Insurance_Churn_ParticipantsData/Train.csv")

X = data.drop(columns=["labels"])
y = data["labels"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)



y_prob = model.predict_proba(X_test)[:, 1]

print("Churn probability of first 6 customers:")
print(y_prob[:6])



def classify_risk(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


risk_levels = [classify_risk(p) for p in y_prob]


risk_summary = pd.Series(risk_levels).value_counts()

total_customers = len(X_test)

print("\nCustomer Risk Summary")
print(f"Total Customers Analysed: {total_customers}")

for risk, count in risk_summary.items():
    print(f"{risk}: {count} customers")

print("\nRisk levels of first 15 customers:")
for i in range(15):
    print(f"Customer {i+1}: {risk_levels[i]} (prob={y_prob[i]:.2f})")



y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



customer_id = 44


if customer_id > len(X_test):
    print("Customer ID out of range")
    exit()

customer_data = X_test.iloc[customer_id - 1]
customer_prob = model.predict_proba(
    customer_data.values.reshape(1, -1)
)[0][1]

risk = classify_risk(customer_prob)

print("\n----------------------------------")
print(f"Customer {customer_id} Detailed Analysis")
print("----------------------------------")
print(customer_data.to_dict())


print(f"\nChurn Probability: {customer_prob:.2f}")
print("Risk Level:", risk)


print("\nSuggested Action:")

if risk == "High Risk":
    print("→ Immediate retention action required")
    print("→ Offer discounts or personalized plans")
    print("→ Priority customer support")

elif risk == "Medium Risk":
    print("→ Monitor customer closely")
    print("→ Send engagement offers")
    print("→ Improve service communication")

else:
    print("→ Customer is stable")
    print("→ Normal follow-up and engagement")


print("\nNew Customer Prediction ")

new_customer = pd.DataFrame([{
    "feature_0": 0.2,
    "feature_1": -0.4,
    "feature_2": -1.0,
    "feature_3": 0.8,
    "feature_4": -0.3,
    "feature_5": 0.6,
    "feature_6": 0.9,
    "feature_7": 2,
    "feature_8": 1,
    "feature_9": 5,
    "feature_10": 0,
    "feature_11": 1,
    "feature_12": 1,
    "feature_13": 0,
    "feature_14": 1,
    "feature_15": 1
}])

new_prob = model.predict_proba(new_customer)[0][1]
new_risk = classify_risk(new_prob)

print(f"Churn Probability: {new_prob:.2f}")
print("Risk Level:", new_risk)

if new_risk == "High Risk":
    print("Suggested Action: Immediate retention offer")
elif new_risk == "Medium Risk":
    print("Suggested Action: Engagement and follow-up")
else:
    print("Suggested Action: Customer is stable")

import joblib
joblib.dump(model, "models/churn_model.pkl")
