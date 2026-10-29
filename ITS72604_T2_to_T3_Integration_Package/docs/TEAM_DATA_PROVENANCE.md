# Team-cleaned dataset provenance and integration notes

## Source and attribution
`ev_campus_dataset.csv` is the raw 1,200-session input. `data_pipeline.py` generates `cleaned_ev_dataset.csv` and `docs/eda_summary_by_charger_peak.csv` from it.

## Validated contract
- 1,200 charging sessions; the cleaned output includes the raw columns, a derived peak flag, charging power, and a power-rate group
- No missing cells or duplicate `Session_ID` values
- Status distribution: Normal 898; Grid_Surge 172; Overheated 130
- `Peak_Hour_Flag` is derived using `12 <= Start_Hour < 17`
- The raw source has 25 missing duration and 35 missing energy values; the pipeline median-imputes them

## Important task-brief mismatch
The pipeline saves `Power_kW` and `Charging_Efficiency_Group` in the cleaned output. The group labels are power-rate bands, not independently measured conversion efficiency.

The raw source column is named `Battery_Temp_C`; the project owner clarified that these readings measure temperature at the charging location. The cleaned output therefore names the column `Ambient_Temp_C`. The trained model keeps `Battery_Temp_C` internally for compatibility. Charger types are `AC` and `DC`, not illustrative `AC_Fast` and `DC_Ultra`.

`Charging_Efficiency_Group` represents charging **power rate bands**, not independently measured energy-conversion efficiency.

## Handoff for Task 3
Load `ev_model.pkl` as a `Pipeline` and call `single_prediction_features(...)` before `.predict()`. Do not separately one-hot encode categorical variables. The saved model includes fitted preprocessing.

## Model-scoping limitation
Data were median-imputed by the Task 1 author before ML train/test splitting; for strict causal or production validation, fit imputation only within training folds. This report does not claim to validate that upstream transformation from the cleaned file alone.
