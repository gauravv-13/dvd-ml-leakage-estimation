import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

config_path = Path("config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

cases   = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
# Terminal-relative voltages replace the absolute vd/vs/vg/vb (which are correlated with them)
inputs  = ["vdd", "temp", "w", "vgs", "vds", "vbs", "vth0", "toxe", "ndep", "u0", "xj"]
outputs = ["id", "ig", "is", "ib"]


inputs_raw = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]


def add_voltage_features(d):
    d = d.copy()
    d["vgs"] = d["vg"] - d["vs"]
    d["vds"] = d["vd"] - d["vs"]
    d["vbs"] = d["vb"] - d["vs"]
    return d.drop(columns=["vd", "vs", "vg", "vb"])


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


results = []   # one row per case

for case in cases:
    print(f"\n===== {case} =====")

    # 1. Load data and split
    df    = pd.read_csv(f"data/transistor/{case}.csv")
    split = pd.read_csv(f"data/transistor/split_{case}.csv")

    # targets are read from the raw columns; features use vgs/vds/vbs instead of vd/vs/vg/vb
    df = pd.concat([add_voltage_features(df[inputs_raw]), df[outputs]], axis=1)

    train = df[split["set"] == "train"]
    val   = df[split["set"] == "val"]

    # 2. Extract features and arcsinh-scaled targets
    X_train = train[inputs]
    y_train = np.arcsinh(train[outputs] / 1e-18)

    X_val = val[inputs]
    y_val = np.arcsinh(val[outputs] / 1e-18)


    # fresh start for every case
    best_params = {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": 1.0,
        "min_samples_leaf": 1
    }

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
    r2_each = r2_score(y_val, y_val_pred, multioutput="raw_values")   # one R2 per current

    i_pred = np.sinh(y_val_pred) * 1e-18
    i_true = np.sinh(y_val.to_numpy()) * 1e-18
    rel_error = np.abs(i_pred - i_true) / np.abs(i_true) * 100        # in %, same as the MLP

    median_each = np.median(rel_error, axis=0)                        # one value per current
    p95_each    = np.percentile(rel_error, 95, axis=0)

    print("\n--- Validation Performance ---")
    for j, c in enumerate(outputs):
        print(f"{c:3s} | R2: {r2_each[j]:.5f} | Median Error: {median_each[j]:.2f}% "
              f"| 95th-Percentile Error: {p95_each[j]:.2f}%")

    row = {"model": "RF", "case": case, **best_params, "val_r2": val_r2}
    for j, c in enumerate(outputs):
        row[f"{c}_r2"]             = r2_each[j]
        row[f"{c}_median_err_pct"] = median_each[j]
        row[f"{c}_p95_err_pct"]    = p95_each[j]
    results.append(row)

# Save results to a CSV, like the MLP script does
Path("results").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv("results/rf_validation.csv", index=False)
print("\nSaved results/rf_validation.csv")


# Tables: validation metrics per current, as RF / MLP, better value in bold
mlp = pd.read_csv("results/mlp_validation.csv").set_index("case")
rf  = pd.DataFrame(results).set_index("case")

BOLD, END = "\033[1m", "\033[0m"

def fmt_r2(x):
    return f"{x:.4f}".replace("0.", ".", 1).rjust(6)   # 0.9991 -> .9991

def fmt_pct(x):
    return f"{x:.2f}%".rjust(8)                         # 1.234 -> 1.23%

def print_table(title, metric, fmt, higher_is_better):
    width = 2 * len(fmt(0)) + 3
    print(f"\n{title}, as RF / MLP, with the better value in bold:\n")
    print(f"{'Condition':10s}   " + "  ".join(f"{c.capitalize():^{width}s}" for c in outputs))
    rf_wins = 0
    for case in cases:
        cells = []
        for c in outputs:
            a, b = rf.loc[case, f"{c}_{metric}"], mlp.loc[case, f"{c}_{metric}"]
            rf_better = a >= b if higher_is_better else a <= b
            if rf_better:
                rf_wins += 1
                cells.append(f"{BOLD}{fmt(a)}{END} / {fmt(b)}")
            else:
                cells.append(f"{fmt(a)} / {BOLD}{fmt(b)}{END}")
        print(f"{case:10s}   " + "  ".join(cells))
    print(f"\nRF is better on {rf_wins} of {len(cases) * len(outputs)}.")

print_table("Validation R2 on asinh(I/1e-18)", "r2", fmt_r2, higher_is_better=True)
print("1.0000 values are rounded, not exact.")
print_table("Validation median relative error", "median_err_pct", fmt_pct, higher_is_better=False)
print_table("Validation 95th-percentile relative error", "p95_err_pct", fmt_pct, higher_is_better=False)
