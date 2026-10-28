#JIAO JIE 0395413
"""T2-to-T3 contract: run with `python example_inference.py`."""
from pathlib import Path
import joblib
from feature_engineering import single_prediction_features

# 1. Load the trained machine learning model
ROOT = Path(__file__).resolve().parent
# Only open the trusted ev_model.pkl from the T2 teammate.
model = joblib.load(ROOT / "ev_model.pkl")

# 2. Prepare sample charging session inputs
# Feature engineering ensures consistency with the training pipeline
row = single_prediction_features(
    charger_type="DC",
    user_type="Student",
    start_hour=17,
    duration_mins=60.0,
    energy_kwh=18.0,
    ambient_temp_c=35.5,
    overstay_mins=5,
)

# 3. Predict grid status using the trained model
predicted = str(model.predict(row)[0])
print("Predicted grid status:", predicted)

# 4. Display input features for integration verification
print("Feature columns supplied:", row.columns.tolist())

# 5. Display model class scores if supported
# These scores are not calibrated real-world risk probabilities
if hasattr(model, "predict_proba"):
    scores = dict(zip(model.classes_, map(float, model.predict_proba(row)[0])))
    print("Model class scores (NOT calibrated risk probabilities):", scores)
