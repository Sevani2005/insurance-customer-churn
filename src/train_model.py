import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from src.utils import classify_risk

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Insurance_Churn_ParticipantsData", "Train.csv")

# Load data
data = pd.read_csv(data_path)
X = data.drop(columns=["labels"])
y = data["labels"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_prob = model.predict_proba(X_test)[:, 1]
risk_levels = [classify_risk(p) for p in y_prob]

print("\nCustomer Risk Summary")
risk_summary = pd.Series(risk_levels).value_counts()
for risk, count in risk_summary.items():
    print(f"{risk}: {count} customers")

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
model_save_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
joblib.dump(model, model_save_path)
print(f"\nModel saved to {model_save_path}")
