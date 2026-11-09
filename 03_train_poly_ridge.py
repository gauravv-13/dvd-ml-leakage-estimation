"""Phase 3 (Part A step 2), model type: polynomial regression with ridge (scikit-learn). Validation results only.

Per case (nmos_on, nmos_off, pmos_on, pmos_off), one PolyRidgeFET (src/poly_ridge.py) made of three
polynomial ridge regressions, on log scale, of D/Vds = (Id - Is)/(Vd - Vs), Ig and Ib; Id and Is
follow from D and Kirchhoff (Id + Ig + Is + Ib = 0).
  Inputs  : the dataset input columns, turned into 20 features in a symmetric NMOS-like frame
            (10 base + 10 physics-based, see src/poly_ridge.py), standardised on the training set
  Model   : StandardScaler -> PolynomialFeatures(degree) -> StandardScaler -> Ridge(alpha)
  Search  : degree in DEGREES x alpha in ALPHAS x junction feature fjun on/off, chosen separately
            for each of the three parts by
            the validation RMSE of its log target (the ridge objective on held-out data; unlike the
            median error it penalises the rare wild extrapolations of a high-degree polynomial)
  Split   : data/transistor/split_<case>.csv (03_make_split.py); trained on train, scored on val
  Metrics (validation, per current, all rows), same definitions as 03_train_mlp.py:
            R^2 on asinh(I / 1e-18)
            median and 95th-percentile relative error, |I_pred - I_SPICE| / |I_SPICE| * 100 [%]

Outputs: results/poly_ridge_sweep.csv        every (part, degree, alpha) tried, train and val error
         results/poly_ridge_validation.csv   chosen setting per case, validation metrics
         saved_models/poly_ridge/<case>.joblib
Run:     python 03_train_poly_ridge.py
"""
from __future__ import annotations

import itertools
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning

from src.config import CASES, DATA_DIR, RESULTS_DIR, SAVED_MODELS_DIR
from src.poly_ridge import PARTS, PolyRidgeFET, features, pipeline, targets

CURRENTS = ["id", "ig", "is", "ib"]
I_SCALE = 1e-18                       # R^2 scale: asinh(I / I_SCALE), as in 03_train_mlp.py
DEGREES = [2, 3, 4]                   # degree 4: 10.6k terms for 7k training rows; 5 has 53k
ALPHAS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def load(case: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv")
    split = pd.read_csv(DATA_DIR / "transistor" / f"split_{case}.csv")
    assert (split.row.to_numpy() == np.arange(len(df))).all()
    return {s: df[split.set.to_numpy() == s] for s in ["train", "val"]}


def rel_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true) / np.abs(true) * 100


def metrics(I_true: np.ndarray, I_pred: np.ndarray) -> dict:
    out = {}
    y_true, y_pred = np.arcsinh(I_true / I_SCALE), np.arcsinh(I_pred / I_SCALE)
    for j, cur in enumerate(CURRENTS):
        rel = rel_err(I_pred[:, j], I_true[:, j])
        ss_res = ((y_true[:, j] - y_pred[:, j]) ** 2).sum()
        ss_tot = ((y_true[:, j] - y_true[:, j].mean()) ** 2).sum()
        out[f"{cur}_r2"] = 1 - ss_res / ss_tot
        out[f"{cur}_median_err_pct"] = np.median(rel)
        out[f"{cur}_p95_err_pct"] = np.quantile(rel, 0.95)
    out["mean_median_err_pct"] = np.mean([out[f"{c}_median_err_pct"] for c in CURRENTS])
    return out


def sweep(case: str, d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fit every (junction, degree, alpha) for the three parts; return train/val error of each part."""
    dev = CASES[case]["dev"]
    X = {(s, j): features(d[s], dev, j) for s in d for j in (True, False)}
    t = {s: targets(d[s]) for s in d}
    rows = []
    for part in PARTS:
        sign = 1.0 if np.mean(t["train"][part] > 0) >= 0.5 else -1.0
        y_tr = np.log(np.abs(t["train"][part]))
        for junction, degree, alpha in itertools.product((True, False), DEGREES, ALPHAS):
            pipe = pipeline(degree, alpha).fit(X["train", junction], y_tr)
            y_hat = {s: np.clip(pipe.predict(X[s, junction]), -300, 300) for s in d}
            err = {s: rel_err(sign * np.exp(y_hat[s]), t[s][part]) for s in d}
            rows.append({"case": case, "part": part, "junction": junction, "degree": degree, "alpha": alpha,
                         "n_terms": pipe[1].n_output_features_,
                         "val_rmse_log": np.sqrt(np.mean((y_hat["val"] - np.log(np.abs(t["val"][part]))) ** 2)),
                         "train_median_err_pct": np.median(err["train"]),
                         "val_median_err_pct": np.median(err["val"]),
                         "val_p95_err_pct": np.quantile(err["val"], 0.95)})
    return pd.DataFrame(rows)


def run_case(case: str) -> tuple[dict, pd.DataFrame]:
    warnings.filterwarnings("ignore", category=LinAlgWarning)      # tiny alphas: near-singular solve
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    d = load(case)
    t0 = time.time()
    sw = sweep(case, d)
    best = sw.loc[sw.groupby("part").val_rmse_log.idxmin()].set_index("part")
    settings = {p: (int(best.degree[p]), float(best.alpha[p]), bool(best.junction[p])) for p in PARTS}

    model = PolyRidgeFET(case, CASES[case]["dev"], settings).fit(d["train"])
    SAVED_MODELS_DIR.joinpath("poly_ridge").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODELS_DIR / "poly_ridge" / f"{case}.joblib")

    I_pred = model.predict(d["val"])
    row = {"model": "PolyRidge", "case": case,
           **{f"{p}_{k}": v for p in PARTS for k, v in zip(["degree", "alpha", "junction"], settings[p])},
           "train_seconds": round(time.time() - t0),
           **metrics(d["val"][CURRENTS].to_numpy(), I_pred)}
    return row, sw


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    out = Parallel(n_jobs=len(CASES))(delayed(run_case)(c) for c in CASES)
    print(f"swept and trained {len(CASES)} cases in {time.time() - t0:.0f} s")
    pd.concat([sw for _, sw in out]).to_csv(RESULTS_DIR / "poly_ridge_sweep.csv", index=False)
    res = pd.DataFrame([row for row, _ in out])
    res.to_csv(RESULTS_DIR / "poly_ridge_validation.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.T.to_string())


if __name__ == "__main__":
    main()
