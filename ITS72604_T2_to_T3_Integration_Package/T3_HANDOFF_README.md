# T2 → T3 integration handoff (ITS72604 Assessment 2)

**Prepared for the T3 Streamlit teammate.** This is a handoff package, **not** a replacement for your Streamlit app. The dataset was supplied by the T1 group member. The T2 module trained and saved a `scikit-learn` **Pipeline** (preprocessing plus classifier) as `ev_model.pkl`.

## Required to run T3

- `ev_model.pkl`: saved trained classifier pipeline (do not retrain it just to use it).
- `feature_engineering.py`: call `single_prediction_features(...)` for interactive predictions and `add_engineered_features(...)` for charts requiring `Power_kW`.
- `cleaned_ev_dataset.csv`: team-cleaned input data for KPI cards, charts, and tables. It is already shared by T1, use **one authoritative copy**.
- `requirements.txt`: align the team environment with these requirements; **merge** rather than blindly overwrite your app dependencies.

## Supporting evidence

- `train_model.py`: shows 80/20 stratified split, training-only 5-fold CV, Decision Tree vs Random Forest, and serialization.
- `docs/model_metrics.json`, `docs/model_comparison.csv`: numerical evaluation.
- `figures/confusion_matrix_best.png`: confusion matrix of selected model.
- `docs/TEAM_DATA_PROVENANCE.md`: limitations and source data details.

## Interface: how to call from `app.py`

```python
from pathlib import Path
import joblib
from feature_engineering import single_prediction_features

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "ev_model.pkl")  # trusted teammate artifact only
row = single_prediction_features(
    charger_type="DC", user_type="Student", start_hour=17,
    duration_mins=60.0, energy_kwh=18.0,
    ambient_temp_c=35.5, overstay_mins=5,
)
status = str(model.predict(row)[0])
```

All parameter names in the call must match. The helper produces a `pandas.DataFrame` that includes required engineered fields. **Do not manually run one-hot encoding**; this is inside the saved model. Do **not** pass `Grid_Status_Label` as a predictor.

Valid classes are `Normal`, `Grid_Surge`, `Overheated`. `Charger_Type` in the team dataset is `AC` or `DC` (not `AC_Fast` / `DC_Ultra`). The raw CSV labels the location-temperature measurement `Battery_Temp_C`; per the project owner's clarification, Task 1 exports it as `Ambient_Temp_C`. The saved T2 model retains `Battery_Temp_C` as its internal feature name, and the helper maps `ambient_temp_c` to that feature. The start-hour convention computes `Peak_Hour_Flag=1` from 12:00 through 16:59 and 0 from 17:00 onward. `Power_kW = Energy_kWh / (Duration_Mins/60)`.
