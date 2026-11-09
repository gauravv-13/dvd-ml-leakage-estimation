"""Phase 3 (Part A step 2), model type: gradient-boosted trees (XGBoost). Validation results and saved models.

Per case (nmos_on, nmos_off, pmos_on, pmos_off), 4 boosters: 16 in total.
  Inputs  : features() below, built from the 12 input columns of the dataset CSV. Drain and source
            enter only through min/max(Vd, Vs), since swapping them mirrors the device
  Outputs : Ig, Ib, Ich = (Id - Is) / 2 and Icm = (Id + Is) / 2, each as asinh(I / (1e-18 * W / Lmin)),
            with Ich multiplied by sign(Vd - Vs) so it does not jump where Vd - Vs crosses 0.
            Then Id = Icm + Ich and Is = Icm - Ich. Icm gets its own booster instead of -(Ig + Ib) / 2
            because the PMOS OFF Ig and Ib labels carry ~1e-16 A simulator noise
  Boosting: a least-squares line through the inputs is the starting point (base_margin),
            and the trees fit what it leaves over
  Split   : data/transistor/split_<case>.csv (03_make_split.py); trained on train, scored on val
  Settings: fixed per case (SETTINGS below), chosen on the validation set; seed 42
  Loss    : squared error, except L1 (absolute error) for PMOS OFF: squared error averages the mixed signs
            of its noisy Ig and Ib labels towards 0, while L1 follows the median
  Metrics (validation, per current, all rows), same as 03_train_mlp.py:
            R^2 on asinh(I / 1e-18)
            median and 95th-percentile relative error, |I_pred - I_SPICE| / |I_SPICE| * 100 [%]

Outputs: results/xgboost_validation.csv
         saved_models/xgboost/<case>.joblib   the 4 fitted boosters of each case (src/xgb_model.py, XGBFET)
Reproducibility: xgboost is pinned in requirements.txt and runs with a fixed thread count (NTHREAD)
Run:     python 03_train_xgboost.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

from src.config import CASES, DATA_DIR, RESULTS_DIR, SAVED_MODELS_DIR, load_config
from src.xgb_model import XGBFET, features, linear_baseline, terminal_currents  # noqa: F401 (used by tests)

CURRENTS = ["id", "ig", "is", "ib"]
I_SCALE = 1e-18                       # target = asinh(I / I_SCALE)
SEED = 42

COMMON = {"objective": "reg:squarederror", "rounds": 6000, "eta": 0.015, "max_depth": 6,
          "subsample": 0.8, "colsample_bytree": 0.8}
PLAIN = {**COMMON, "min_child_weight": 1, "lambda": 1.0}
PLAIN_L1 = {**PLAIN, "objective": "reg:absoluteerror"}
REGULARISED = {**COMMON, "min_child_weight": 5, "lambda": 5.0}
SETTINGS = {"nmos_on": REGULARISED, "nmos_off": REGULARISED, "pmos_on": REGULARISED, "pmos_off": PLAIN_L1}


# shortcut: load() and metrics() are copied from 03_train_mlp.py because importing it needs torch;
# move both to src/ once all four model scripts exist. tests/test_train_xgboost.py catches drift.
def load(case: str):
    df = pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv")
    split = pd.read_csv(DATA_DIR / "transistor" / f"split_{case}.csv")
    assert (split.row.to_numpy() == np.arange(len(df))).all()
    df["vds"] = df.vd - df.vs
    df["vgs"] = df.vg - df.vs
    df["vbs"] = df.vb - df.vs
    df["inv_t"] = 1.0 / (df.temp + 273.15)
    return {s: df[split.set.to_numpy() == s] for s in ["train", "val"]}


def fit_predict(tr: pd.DataFrame, va: pd.DataFrame, s: dict) -> np.ndarray:
    """Train on tr; return asinh(I / I_SCALE) on va, one column per entry of CURRENTS."""
    return np.arcsinh(XGBFET("", s, load_config().lmin).fit(tr).predict(va) / I_SCALE)


def metrics(I_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    I_pred = I_SCALE * np.sinh(y_pred)
    y_true = np.arcsinh(I_true / I_SCALE)
    for j, cur in enumerate(CURRENTS):
        rel = np.abs(I_pred[:, j] - I_true[:, j]) / np.abs(I_true[:, j]) * 100
        ss_res = ((y_true[:, j] - y_pred[:, j]) ** 2).sum()
        ss_tot = ((y_true[:, j] - y_true[:, j].mean()) ** 2).sum()
        out[f"{cur}_r2"] = 1 - ss_res / ss_tot
        out[f"{cur}_median_err_pct"] = np.median(rel)
        out[f"{cur}_p95_err_pct"] = np.quantile(rel, 0.95)
    out["mean_median_err_pct"] = np.mean([out[f"{c}_median_err_pct"] for c in CURRENTS])
    return out


def run_case(case: str) -> dict:
    s = SETTINGS[case]
    d = load(case)
    t0 = time.time()
    model = XGBFET(case, s, load_config().lmin).fit(d["train"])
    train_seconds = round(time.time() - t0)
    y_pred = np.arcsinh(model.predict(d["val"]) / I_SCALE)
    SAVED_MODELS_DIR.joinpath("xgboost").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODELS_DIR / "xgboost" / f"{case}.joblib", compress=3)   # ~28 MB instead of ~85 MB
    return {"model": "XGBoost", "case": case, **s, "train_seconds": train_seconds,
            **metrics(d["val"][CURRENTS].to_numpy(), y_pred)}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    rows = Parallel(n_jobs=len(CASES))(delayed(run_case)(c) for c in CASES)
    print(f"trained {4 * len(CASES)} boosters in {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "xgboost_validation.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.T.to_string())


if __name__ == "__main__":
    main()
