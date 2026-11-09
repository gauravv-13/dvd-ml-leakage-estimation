"""Phase 3 (Part A step 2), model type: neural network (MLP, PyTorch). Validation results and saved models.

Per case (nmos_on, nmos_off, pmos_on, pmos_off), one MLP predicts all four currents.
  Inputs  : the 12 input columns of the dataset CSV plus 4 derived inputs
            Vd-Vs, Vg-Vs, Vb-Vs and 1/T (T in kelvin), standardised on the training set
  Outputs : asinh(I / 1e-18) for Id, Ig, Is, Ib, each standardised on the training set
  Split   : data/transistor/split_<case>.csv (03_make_split.py); trained on train, scored on val
  Settings: fixed per case (SETTINGS below); SiLU activations, L1 loss, AdamW,
            one-cycle learning-rate schedule (peak 3e-3), batch 256, seed 42
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
from torch import nn

from src.config import CASES, DATA_DIR, RESULTS_DIR, SAVED_MODELS_DIR
from src.mlp_model import DERIVED, INPUTS, MLPFET, make_net
CURRENTS = ["id", "ig", "is", "ib"]
I_SCALE = 1e-18                       # target = asinh(I / I_SCALE)
SEED = 42

COMMON = {"activation": "silu", "loss": "l1", "lr": 3e-3, "batch": 256}
SMALL = {**COMMON, "width": 128, "depth": 3, "epochs": 6000, "weight_decay": 0.0}
LARGE = {**COMMON, "width": 256, "depth": 4, "epochs": 3000, "weight_decay": 1e-3}
SETTINGS = {"nmos_on": SMALL, "pmos_on": SMALL, "nmos_off": LARGE, "pmos_off": LARGE}


def load(case: str):
    df = pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv")
    split = pd.read_csv(DATA_DIR / "transistor" / f"split_{case}.csv")
    assert (split.row.to_numpy() == np.arange(len(df))).all()
    df["vds"] = df.vd - df.vs
    df["vgs"] = df.vg - df.vs
    df["vbs"] = df.vb - df.vs
    df["inv_t"] = 1.0 / (df.temp + 273.15)
    return {s: df[split.set.to_numpy() == s] for s in ["train", "val"]}


def train_net(X: torch.Tensor, Y: torch.Tensor, s: dict, seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    net = make_net(X.shape[1], s["width"], s["depth"])
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    steps = int(np.ceil(len(X) / s["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s["lr"], epochs=s["epochs"], steps_per_epoch=steps)
    loss_fn = nn.L1Loss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(s["epochs"]):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), s["batch"]):
            idx = perm[i:i + s["batch"]]
            opt.zero_grad()
            loss_fn(net(X[idx]), Y[idx]).backward()
            opt.step()
            sched.step()
    return net.eval()


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
    s = SETTINGS[case]
    d = load(case)
    F = INPUTS + DERIVED
    X_tr, X_va = d["train"][F].to_numpy(), d["val"][F].to_numpy()
    Y_tr = np.arcsinh(d["train"][CURRENTS].to_numpy() / I_SCALE)
    x_mean, x_std = X_tr.mean(0), X_tr.std(0)
    x_std[x_std == 0] = 1.0                       # constant column (e.g. vb = 0 for nmos_on)
    y_mean, y_std = Y_tr.mean(0), Y_tr.std(0)
    f32 = lambda a: torch.tensor(a, dtype=torch.float32)
    X, Y, Xv = f32((X_tr - x_mean) / x_std), f32((Y_tr - y_mean) / y_std), f32((X_va - x_mean) / x_std)

    t0 = time.time()
    net = train_net(X, Y, s, SEED)
    with torch.no_grad():
        y_pred = net(Xv).numpy() * y_std + y_mean
    train_seconds = round(time.time() - t0)

    model = MLPFET(case, s, net, x_mean, x_std, y_mean, y_std)
    assert np.array_equal(model.predict_asinh(d["val"]), y_pred)   # saved model = scored network
    SAVED_MODELS_DIR.joinpath("mlp").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODELS_DIR / "mlp" / f"{case}.joblib")
    return {"model": "MLP", "case": case, **s, "train_seconds": train_seconds,
            **metrics(d["val"][CURRENTS].to_numpy(), y_pred)}


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
