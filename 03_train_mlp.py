"""Phase 3 (Part A step 2), model type: neural network (MLP, PyTorch). Validation results and saved models.

Per case (nmos_on, nmos_off, pmos_on, pmos_off), one MLP predicts all four currents
(src/mlp_model.py, target recipes in src/target_transform.py).
  Inputs  : the 12 input columns of the dataset CSV plus Vd-Vs, Vg-Vs, Vb-Vs and 1/T (T in kelvin);
            with the "extra" recipe also Vds/Vt, Vgs/Vt, Vbs/Vt and log10(1 - exp(-|Vds|/Vt) + 1e-12);
            standardised on the training set
  Outputs : log10 of the current magnitudes (per recipe, see RECIPES), each standardised, plus a
            sign output for every current whose sign is not fixed
  Recipes : chosen per case on the validation set by experiments/target_transforms.py
            (report appendix "Target-transform study"): canonical drain/source swap, per-width
            targets, and per case the saturation target, extra inputs, Is from Kirchhoff or clipped
            labels
  Split   : data/transistor/split_<case>.csv (03_make_split.py); trained on train, scored on val
  Settings: fixed per case (SETTINGS below); SiLU activations, L1 loss on the magnitudes plus
            logistic loss on the signs, AdamW, one-cycle learning-rate schedule (peak 3e-3),
            batch 256, seed 42
  Metrics (validation, per current, all rows):
            R^2 on asinh(I / 1e-18)
            median and 95th-percentile relative error, |I_pred - I_SPICE| / |I_SPICE| * 100 [%]

Outputs: results/mlp_validation.csv
         saved_models/mlp/<case>.joblib   trained network + scaling (src/mlp_model.py, MLPFET)
Run:     python 03_train_mlp.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import joblib
import torch
from joblib import Parallel, delayed

from src.config import CASES, DATA_DIR, RESULTS_DIR, SAVED_MODELS_DIR
from src.mlp_model import MLPFET
CURRENTS = ["id", "ig", "is", "ib"]
I_SCALE = 1e-18                       # R^2 scale: asinh(I / I_SCALE)
SEED = 42

COMMON = {"activation": "silu", "loss": "l1", "lr": 3e-3, "batch": 256}
SMALL = {**COMMON, "width": 128, "depth": 3, "epochs": 6000, "weight_decay": 0.0}
LARGE = {**COMMON, "width": 256, "depth": 4, "epochs": 3000, "weight_decay": 1e-3}
SETTINGS = {"nmos_on": SMALL, "pmos_on": SMALL, "nmos_off": LARGE, "pmos_off": LARGE}
# target recipe per case (flags: src/target_transform.py); names from the transform study
BASE = {"swap": 1, "perw": 1}
RECIPES = {
    "nmos_on":  {**BASE, "sat": 1},                  # T3b
    "nmos_off": {**BASE, "extra": 1, "kcl": 1},      # T4 on T3a
    "pmos_on":  {**BASE, "sat": 1},                  # T3b
    "pmos_off": {**BASE, "extra": 1, "clip": 1},     # T5 on T3a
}


def load(case: str):
    df = pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv")
    split = pd.read_csv(DATA_DIR / "transistor" / f"split_{case}.csv")
    assert (split.row.to_numpy() == np.arange(len(df))).all()
    df["vds"] = df.vd - df.vs
    df["vgs"] = df.vg - df.vs
    df["vbs"] = df.vb - df.vs
    df["inv_t"] = 1.0 / (df.temp + 273.15)
    return {s: df[split.set.to_numpy() == s] for s in ["train", "val"]}


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
    torch.set_num_threads(2)
    s, recipe = SETTINGS[case], RECIPES[case]
    d = load(case)
    t0 = time.time()
    model = MLPFET.fit(case, s, recipe, d["train"], SEED)
    train_seconds = round(time.time() - t0)
    I_pred = model.predict(d["val"])

    SAVED_MODELS_DIR.joinpath("mlp").mkdir(parents=True, exist_ok=True)
    path = SAVED_MODELS_DIR / "mlp" / f"{case}.joblib"
    joblib.dump(model, path)
    assert np.array_equal(joblib.load(path).predict(d["val"]), I_pred)   # saved model = scored network
    return {"model": "MLP", "case": case, **s, "recipe": "+".join(recipe), "train_seconds": train_seconds,
            **metrics(d["val"][CURRENTS].to_numpy(), np.arcsinh(I_pred / I_SCALE))}


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    rows = Parallel(n_jobs=len(CASES))(delayed(run_case)(c) for c in CASES)
    print(f"trained {len(CASES)} networks in {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "mlp_validation.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.T.to_string())


if __name__ == "__main__":
    main()
