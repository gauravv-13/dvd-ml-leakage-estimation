import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

config_path = Path("config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# 1. Load data and split
df    = pd.read_csv("data/transistor/nmos_on.csv")
split = pd.read_csv("data/transistor/split_nmos_on.csv")

train = df[split["set"] == "train"]
val   = df[split["set"] == "val"]

inputs  = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
outputs = ["id", "ig", "is", "ib"]

# 2. Extract features and arcsinh-scaled targets
X_train = train[inputs]
y_train = np.arcsinh(train[outputs] / 1e-18)

X_val = val[inputs]
y_val = np.arcsinh(val[outputs] / 1e-18)


best_params = {
    "n_estimators": 100,
    "max_depth": None,
    "max_features": 1.0,
    "min_samples_leaf": 1
}

def tune_param(param_name, candidates):
    """Evaluates candidates for a single 
    parameter and updates best_params."""
    best_val, best_r2 = None, -float("inf")
    for val in candidates:
        model_kwargs = {** best_params, param_name: val}
        rf = RandomForestRegressor(
            **model_kwargs,
            random_state = cfg["rf"]["seed"]
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        score = r2_score(y_val, y_pred, multioutput="uniform_average")

        # Invert arcsinh back to Amperes to compute Deliverable 4 metrics
        i_pred = np.sinh(y_pred) * 1e-18
        i_true = np.sinh(y_val.to_numpy()) * 1e-18
        rel_error = np.abs(i_pred - i_true) / (np.abs(i_true) + 1e-18)

        med_err = np.median(rel_error)
        p95_err = np.percentile(rel_error, 95)

        if score > best_r2:
            best_r2, best_val = score, val
            best_med, best_p95 = med_err, p95_err

    best_params[param_name] = best_val
    print(f"Optimal {param_name:16s}: {best_val} | Mean Val R2: {best_r2:.5f}")

tune_param("max_features",    [3, 4, 6, 0.5, 1.0])
tune_param("n_estimators",    [10, 50, 100, 200])
tune_param("max_depth",       [8, 12, 16, 20, None])
tune_param("min_samples_leaf", [1, 2, 5, 10])

print("\n--- Final Tuned Hyperparameters ---")
print(best_params)


final_rf = RandomForestRegressor(
    **best_params,
    random_state=cfg["rf"]["seed"]
)
final_rf.fit(X_train, y_train)

# Evaluate on validation set for Deliverable 4
y_val_pred = final_rf.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred, multioutput="uniform_average")

i_pred = np.sinh(y_val_pred) * 1e-18
i_true = np.sinh(y_val.to_numpy()) * 1e-18
rel_error = np.abs(i_pred - i_true) / (np.abs(i_true) + 1e-18)

val_median_err = np.median(rel_error)
val_p95_err = np.percentile(rel_error, 95)

print("\n--- Validation Performance ---")
print(f"R2: {val_r2:.5f} | Median Error: {val_median_err*100:.2f}% | 95th-Percentile Error: {val_p95_err*100:.2f}%")

# Save results to a CSV, like the MLP script does
results = {
    "model": "RF",
    "case": "nmos_on",
    **best_params,
    "val_r2": val_r2,
    "val_median_err_pct": val_median_err * 100,
    "val_p95_err_pct": val_p95_err * 100,
}
Path("results").mkdir(exist_ok=True)
pd.DataFrame([results]).to_csv("results/rf_validation.csv", index=False)
print("Saved results/rf_validation.csv")

