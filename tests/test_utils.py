import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import classify_risk

def test_classify_risk_high():
    assert classify_risk(0.85) == "High Risk"
    assert classify_risk(0.70) == "High Risk"

def test_classify_risk_medium():
    assert classify_risk(0.55) == "Medium Risk"
    assert classify_risk(0.40) == "Medium Risk"

def test_classify_risk_low():
    assert classify_risk(0.10) == "Low Risk"
    assert classify_risk(0.39) == "Low Risk"

if __name__ == "__main__":
    print("Running tests...")
    try:
        test_classify_risk_high()
        test_classify_risk_medium()
        test_classify_risk_low()
        print("✅ All tests passed!")
    except AssertionError as e:
        print(f"❌ Test failed: {str(e)}")
