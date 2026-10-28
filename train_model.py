#JIAO JIE 0395413
"""ITS72604 Assessment 2 Task 2: stratified ML comparison + saved pipeline.

Usage: python train_model.py
The persisted object contains preprocessing as well as the classifier, so the
Streamlit app accepts a raw feature-row DataFrame without manual encoding.
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from feature_engineering import add_engineered_features, validate_group_data
import hashlib

# 1. Define project paths, target label, and input features
ROOT = Path(__file__).resolve().parent
LABEL = "Grid_Status_Label"
NUMERIC_FEATURES = ["Start_Hour", "Duration_Mins", "Energy_kWh", "Overstay_Mins",
                    "Battery_Temp_C", "Peak_Hour_Flag", "Power_kW"]
CATEGORICAL_FEATURES = ["Charger_Type", "User_Type", "Charging_Efficiency_Group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DISPLAY_ORDER = ["Normal", "Grid_Surge", "Overheated"]

# 2. Build the preprocessing pipeline
def build_preprocessor() -> ColumnTransformer:
    """Fit only on training samples, including one-hot coding of Charger_Type."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES)
        ], remainder="drop"
    )

# 3. Execute the machine learning workflow
def main() -> dict:

    # Load and validate the team-cleaned dataset
    path = ROOT / "cleaned_ev_dataset.csv"
    if not path.is_file():
        raise FileNotFoundError("Missing team-provided cleaned_ev_dataset.csv")
    source = pd.read_csv(path)
    validate_group_data(source)

    # Generate additional features without modifying the original CSV
    data = add_engineered_features(source)
    absent = sorted(set(FEATURES + [LABEL]) - set(data.columns))
    if absent: raise ValueError(f"Expected columns missing: {absent}")
    X, y = data[FEATURES], data[LABEL]

    # 4. Split data into 80% training and 20% testing
    # Stratification preserves the target class distribution
    # Deterministic, stratified, 80/20 holdout; never train on test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Define two classification algorithms for comparison
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, min_samples_leaf=4, max_depth=9),
        "Random Forest": RandomForestClassifier(n_estimators=240, max_depth=14,
                                                min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    candidate_results = {}
    trained = {}

    # Configure 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 6. Train and evaluate each classification model
    for name, classifier in models.items():

        # Combine preprocessing and classification into one pipeline
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])

        # Perform cross-validation using training data only
        # CV runs ONLY on training data, so test data remain untouched.
        cv_f1 = cross_val_score(pipeline, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=1)

        # Fit the model and predict the held-out test set
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)

        # Calculate classification metrics and confusion matrix
        candidate_results[name] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision_macro": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
            "train_f1_macro": float(f1_score(y_train, pipeline.predict(X_train), average="macro", zero_division=0)),
            "cv_f1_macro_mean": float(cv_f1.mean()),
            "cv_f1_macro_std": float(cv_f1.std()),
            "per_class": classification_report(y_test, prediction, labels=DISPLAY_ORDER,
                                               output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=DISPLAY_ORDER).tolist()
        }
        trained[name] = pipeline

         # Display model performance in the terminal
        print(name, 'Accuracy:',round(candidate_results[name]['accuracy'],4),
              'macro-F1:',round(candidate_results[name]['f1_macro'],4),
              'train-only CV macro-F1:',round(candidate_results[name]['cv_f1_macro_mean'],4))

     # 7. Select the best model using training-only cross-validation
    # Test metrics are reported but not used for model selection
    winner = max(models, key=lambda n: (candidate_results[n]["cv_f1_macro_mean"],
                                         candidate_results[n]["cv_f1_macro_std"] * -1))
    best = trained[winner]
    
     # Save the complete preprocessing and classification pipeline
    joblib.dump(best, ROOT / "ev_model.pkl", compress=3)

     # Verify that the saved model can be loaded and used
    saved = joblib.load(ROOT / "ev_model.pkl")
    sample_pred = saved.predict(X_test.iloc[:3]); assert len(sample_pred) == 3

     # 8. Generate confusion matrix visualizations
    figures = ROOT / "figures"; figures.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    for name, results in candidate_results.items():
        fig, ax = plt.subplots(figsize=(6.6, 5.1))
        sns.heatmap(np.array(results["confusion_matrix"]), annot=True, fmt="d", cmap="Blues",
                    xticklabels=DISPLAY_ORDER, yticklabels=DISPLAY_ORDER, ax=ax, cbar=True)
        ax.set(xlabel="Predicted grid status", ylabel="Actual grid status",
               title=f"{name} — held-out confusion matrix")
        fig.tight_layout()
        fig.savefig(figures / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=190)

        # Save an additional figure for the selected model
        if name == winner:
            fig.savefig(figures / "confusion_matrix_best.png", dpi=190)
        plt.close(fig)
    cm = candidate_results[winner]["confusion_matrix"]

    # 9. Prepare the model evaluation report
    report = {
        "selected_model": winner,
        "selection_rule": "Highest 5-fold training-only stratified CV macro F1 (test set not used for selection)",
        "seed": 42,
        "input_data": "cleaned_ev_dataset.csv (provided by Task 1 group member; preserved unchanged)",
        "input_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "engineered_only_in_memory": ["Power_kW", "Charging_Efficiency_Group"],
        "n_total": int(len(data)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "classes": DISPLAY_ORDER,
        "train_distribution": {str(k): int(v) for k, v in y_train.value_counts().items()},
        "test_distribution": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_from_features": ["Session_ID", LABEL],
        "candidates": candidate_results,
        "caveats": [
            "The raw CSV names the location-temperature field Battery_Temp_C; Task 1 exports it as Ambient_Temp_C per the data owner's clarification. The model helper maps it to the saved feature name Battery_Temp_C.",
            "Power_kW and Charging_Efficiency_Group are derived from Energy_kWh and Duration_Mins.",
            "Task 1 median imputation was performed by the group before this stage and may cause "
            "mild cross-split leakage; model preprocessing itself is fit only on training folds.",
            "One holdout split cannot establish generalization to new chargers, semesters, or sites.",
            "Model predicts labels found in historical data; it is not a safety-critical control system."
        ]
    }

    # 10. Export model metrics and comparison results
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = [{"Model": k, **{m: round(v[m],4) for m in
                          ("accuracy", "precision_macro", "recall_macro", "f1_macro", "cv_f1_macro_mean", "cv_f1_macro_std", "train_f1_macro")}}
            for k, v in candidate_results.items()]
    pd.DataFrame(rows).to_csv(ROOT / "docs" / "model_comparison.csv", index=False)

    # Save test predictions for reproducibility and error analysis
    pd.DataFrame({"Session_ID": data.loc[X_test.index, "Session_ID"].to_numpy(),
                  "Actual": y_test.to_numpy(), "Predicted": saved.predict(X_test)}).to_csv(
        ROOT / "docs" / "holdout_predictions.csv", index=False)

    # 11. Display the final model selection and output status
    print('Best model selected using train-only CV:', winner)
    print('Held-out best confusion matrix:',cm)
    print('Task 2 complete: ev_model.pkl and docs/model_metrics.json')
    return report

# Run the training workflow when executed directly
if __name__ == "__main__":
    main()
