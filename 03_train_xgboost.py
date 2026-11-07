"""Phase 3 (Part A step 2), model type: gradient-boosted trees (XGBoost). Validation results only.

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

Output: results/xgboost_validation.csv
Run:    python 03_train_xgboost.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

from src.config import CASES, DATA_DIR, RESULTS_DIR, load_config

CURRENTS = ["id", "ig", "is", "ib"]
I_SCALE = 1e-18                       # target = asinh(I / I_SCALE)
SEED = 42
NTHREAD = 2
BOLTZMANN_OVER_Q = 8.617333e-5        # V / K
MIN_GAP = 1e-9                        # V, keeps log() finite when Vd = Vs

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


def features(df: pd.DataFrame) -> pd.DataFrame:
    vt = BOLTZMANN_OVER_Q * (df.temp + 273.15)
    lo = df[["vd", "vs"]].min(axis=1)
    hi = df[["vd", "vs"]].max(axis=1)
    gap = np.maximum(hi - lo, MIN_GAP)
    tox_nm = df.toxe * 1e9
    return pd.DataFrame({
        "vdd": df.vdd, "temp": df.temp, "inv_t": df.inv_t, "w": df.w, "vg": df.vg, "vb": df.vb,
        "vth0": df.vth0, "toxe": df.toxe, "ndep": df.ndep, "u0": df.u0, "xj": df.xj,
        "lo": lo, "hi": hi, "gap": gap, "ln_sat": np.log(-np.expm1(-gap / vt)),
        "vg_lo": df.vg - lo, "vg_hi": df.vg - hi, "vb_lo": df.vb - lo, "vb_hi": df.vb - hi,
        "vov_lo": (df.vg - lo - df.vth0) / vt, "vov_hi": (df.vg - hi - df.vth0) / vt,
        "vb_lo_vt": (df.vb - lo) / vt, "vb_hi_vt": (df.vb - hi) / vt,
        "vg_lo_tox": (df.vg - lo) / tox_nm, "vg_hi_tox": (df.vg - hi) / tox_nm,
        "vgb_tox": (df.vg - df.vb) / tox_nm,
        "ln_w": np.log(df.w), "ln_tox": np.log(tox_nm), "ln_ndep": np.log(df.ndep), "ln_u0": np.log(np.abs(df.u0)),
        "vth_vt": df.vth0 / vt,
    }, index=df.index)


def linear_baseline(X_tr: np.ndarray, y: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit on X_tr; returns its values on X_tr and X_va."""
    x_mean, x_std = X_tr.mean(0), X_tr.std(0)
    x_std = np.where(x_std == 0, 1.0, x_std)      # constant column (e.g. vb = 0 for nmos_on)
    A_tr = np.column_stack([np.ones(len(X_tr)), (X_tr - x_mean) / x_std])
    A_va = np.column_stack([np.ones(len(X_va)), (X_va - x_mean) / x_std])
    coef = np.linalg.lstsq(A_tr, y, rcond=None)[0]
    return A_tr @ coef, A_va @ coef


def boost(X_tr: np.ndarray, y: np.ndarray, X_va: np.ndarray, s: dict) -> np.ndarray:
    base_tr, base_va = linear_baseline(X_tr, y, X_va)
    params = {**{k: v for k, v in s.items() if k != "rounds"}, "seed": SEED, "nthread": NTHREAD}
    booster = xgb.train(params, xgb.DMatrix(X_tr, label=y, base_margin=base_tr), num_boost_round=s["rounds"])
    return booster.predict(xgb.DMatrix(X_va, base_margin=base_va)).astype(float)


def terminal_currents(ich: np.ndarray, icm: np.ndarray, ig: np.ndarray, ib: np.ndarray) -> dict[str, np.ndarray]:
    """Id and Is from Ich = (Id - Is) / 2 and Icm = (Id + Is) / 2."""
    return {"id": icm + ich, "ig": ig, "is": icm - ich, "ib": ib}


def fit_predict(tr: pd.DataFrame, va: pd.DataFrame, s: dict) -> np.ndarray:
    """Train on tr; return asinh(I / I_SCALE) on va, one column per entry of CURRENTS."""
    X_tr, X_va = features(tr).to_numpy(), features(va).to_numpy()
    lmin = load_config().lmin
    w_tr, w_va = tr.w.to_numpy() / lmin, va.w.to_numpy() / lmin

    def predict(I_tr: np.ndarray, sign_tr: np.ndarray | float = 1.0, sign_va: np.ndarray | float = 1.0) -> np.ndarray:
        y = sign_tr * np.arcsinh(I_tr / (I_SCALE * w_tr))
        return I_SCALE * w_va * np.sinh(sign_va * boost(X_tr, y, X_va, s))

    id_tr, is_tr = tr.id.to_numpy(), tr["is"].to_numpy()
    ich = predict((id_tr - is_tr) / 2, np.sign(tr.vds.to_numpy()), np.sign(va.vds.to_numpy()))
    icm = predict((id_tr + is_tr) / 2)
    cur = terminal_currents(ich, icm, predict(tr.ig.to_numpy()), predict(tr.ib.to_numpy()))
    return np.column_stack([np.arcsinh(cur[c] / I_SCALE) for c in CURRENTS])


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
    y_pred = fit_predict(d["train"], d["val"], s)
    return {"model": "XGBoost", "case": case, **s, "train_seconds": round(time.time() - t0),
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
