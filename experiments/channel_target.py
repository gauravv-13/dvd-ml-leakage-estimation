"""Optional study: physics-constrained channel target ("chan") for the MLP transistor models, against
the current saved MLPs (saved_models/mlp/<case>.joblib). Validation set and gate data only; the test
set is never loaded.

Problem. In a gate, transistors sit at |Vds| ~ 1e-7 .. 3e-5 V, which the training data barely covers
(1-2 samples in 10,000 per case below 0.1 mV). The current OFF recipes take log10(1 - exp(-|Vds|/Vt))
as an input; at such Vds it lies ~30 training standard deviations out and the networks return
currents up to 1e265 A.

Approaches, each taken from the literature (src/target_transform.py):
  chan  zero channel current at Vds = 0 by construction: the network predicts
        Ich = (Id - Is)/2 divided by a factor that vanishes at Vds = 0, as Ids = Vds * exp(NN) in
        Guo et al., Micromachines 14:1150 (2023), Eq. 3 and Novkin et al., arXiv:2611.06127 (2026), Eq. 1.
        The factor here is W (1 - exp(-|Vds|/Vt)), the weak-inversion form of the EKV core model used
        as physics term by Kao, Kam, Hu, IEEE EDL 43:974 (2022) (equal to W Vds/Vt as Vds -> 0).
        Drain/source symmetry from |Vds| plus the canonical drain/source swap (as Novkin et al. 2026:
        sign from Vds, magnitude from |Vds|). The common part Icm = (Id + Is)/2 = -(Ig + Ib)/2 is taken
        from conservation (as Q_B = -(Q_D + Q_S + Q_G) in Novkin & Amrouch, arXiv:2503.15209, Eq. 18).
        Network inputs: the existing raw voltages incl. |Vds| and Vgs (Novkin et al. 2026 feed
        |Vds| and Vgs unchanged); the old log10(1 - exp(-|Vds|/Vt)) input is not used.
  dt    chan plus the Vds/Vgs domain transform of Guo et al. (2023), Eqs. 4-5.

Checks:
  (a) validation set: R^2, median and p95 relative error per current (metrics of 03_train_mlp.py)
  (b) Vds sweep 1 uV .. 10 mV at the median validation row, plus drain/source swap symmetry
  (c) all 3,600 gate rows (data/gates/gate_spice.csv): leakage = sum V*I / VDD over every terminal of
      every transistor, no |Vds| floor; transistor list from 06_gate_preds.py (imported, not modified)

Outputs (results/experiments/channel_target/): validation.csv, vds_sweep.csv, symmetry.csv,
gate_errors.csv, saved_models/<case>__<variant>.joblib
Run: python experiments/channel_target.py   (~10 min, 6 networks in parallel)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xgboost  # noqa: F401, E402  (before torch, see 03_select_and_test.py)
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from src.config import DATA_DIR, RESULTS_DIR, SAVED_MODELS_DIR  # noqa: E402
from src.mlp_model import MLPFET  # noqa: E402

CURRENTS = ["id", "ig", "is", "ib"]
INPUTS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
I_SCALE = 1e-18
SEED = 42
COMMON = {"activation": "silu", "loss": "l1", "lr": 3e-3, "batch": 256}      # = 03_train_mlp.py settings
SMALL = {**COMMON, "width": 128, "depth": 3, "epochs": 6000, "weight_decay": 0.0}
LARGE = {**COMMON, "width": 256, "depth": 4, "epochs": 3000, "weight_decay": 1e-3}
SETTINGS = {"nmos_on": SMALL, "pmos_on": SMALL, "nmos_off": LARGE, "pmos_off": LARGE}
CHAN = {"swap": 1, "perw": 1, "chan": 1}
VARIANTS = {("nmos_on", "chan"): CHAN, ("pmos_on", "chan"): CHAN,
            ("nmos_off", "chan"): CHAN, ("nmos_off", "chan_dt"): {**CHAN, "dt": 1},
            ("pmos_off", "chan"): CHAN, ("pmos_off", "chan_dt"): {**CHAN, "dt": 1}}
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
OUT = RESULTS_DIR / "experiments" / "channel_target"


def load(case: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv")
    split = pd.read_csv(DATA_DIR / "transistor" / f"split_{case}.csv")
    return {s: df[split.set.to_numpy() == s] for s in ["train", "val"]}


def metrics(I_true: np.ndarray, I_pred: np.ndarray) -> dict:
    out = {}
    y_true, y_pred = np.arcsinh(I_true / I_SCALE), np.arcsinh(I_pred / I_SCALE)
    for j, cur in enumerate(CURRENTS):
        rel = np.abs(I_pred[:, j] - I_true[:, j]) / np.abs(I_true[:, j]) * 100
        out[f"{cur}_r2"] = 1 - ((y_true[:, j] - y_pred[:, j]) ** 2).sum() / ((y_true[:, j] - y_true[:, j].mean()) ** 2).sum()
        out[f"{cur}_median_err_pct"] = np.median(rel)
        out[f"{cur}_p95_err_pct"] = np.quantile(rel, 0.95)
    out["mean_median_err_pct"] = np.mean([out[f"{c}_median_err_pct"] for c in CURRENTS])
    out["mean_p95_err_pct"] = np.mean([out[f"{c}_p95_err_pct"] for c in CURRENTS])
    return out


def train(case: str, variant: str, recipe: dict) -> tuple[str, str, MLPFET, int]:
    torch.set_num_threads(2)
    t0 = time.time()
    model = MLPFET.fit(case, SETTINGS[case], recipe, load(case)["train"], SEED)
    return case, variant, model, round(time.time() - t0)


def sweep_and_symmetry(models: dict, case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vds sweep at the median validation row (drain on the high-current side, as in gates) and the
    drain/source swap check: predict(vd, vs) must equal predict(vs, vd) with Id and Is exchanged."""
    val = load(case)["val"]
    base = val[INPUTS].median()
    sweep, sym = [], []
    for vds in np.logspace(-6, -2, 9):
        r = base.copy()
        r["vd"] = base.vs + vds if case.startswith("nmos") else base.vs - vds
        X = pd.DataFrame([r])[INPUTS]
        Xs = X.copy()
        Xs[["vd", "vs"]] = X[["vs", "vd"]].to_numpy()
        for name, m in models.items():
            I, Is_ = m.predict(X)[0], m.predict(Xs)[0]
            sweep.append({"case": case, "model": name, "vds_V": vds, "id_A": I[0], "ig_A": I[1], "is_A": I[2],
                          "ib_A": I[3], "ich_A": (I[0] - I[2]) / 2, "kcl_residual_A": I.sum()})
            sym.append({"case": case, "model": name, "vds_V": vds,
                        "max_rel_asym": float(np.max(np.abs(I - Is_[[2, 1, 0, 3]]) / np.abs(I)))})
    return pd.DataFrame(sweep), pd.DataFrame(sym)


def gate_errors(models: dict) -> pd.DataFrame:
    """Sum V*I / VDD over all terminals, no Vds floor; models = case -> model."""
    spec = importlib.util.spec_from_file_location("gate_preds", ROOT / "06_gate_preds.py")
    gp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gp)                       # read-only use of its transistor builder
    rows = []
    for i, r in gp.df.iterrows():
        for tx in gp.get_gate_transistors(r):
            rows.append({"row": i, "case": tx["case"], **tx["tx_data"]})
    tx = pd.DataFrame(rows)
    I = np.zeros((len(tx), 4))
    for case, m in models.items():                   # one batched predict per case
        sel = (tx.case == case).to_numpy()
        I[sel] = m.predict(tx.loc[sel, INPUTS])
    tx["p"] = (tx[["vd", "vg", "vs", "vb"]].to_numpy() * I).sum(1)
    leak = tx.groupby("row").p.sum() / gp.df.vdd
    d = gp.df[["gate", "inputs", "corner", "leak"]].assign(ml=leak)
    d["err_pct"] = (d.ml - d.leak).abs() / d.leak * 100
    return d


def main() -> None:
    (OUT / "saved_models").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    trained = Parallel(n_jobs=len(VARIANTS))(delayed(train)(c, v, r) for (c, v), r in VARIANTS.items())
    print(f"trained {len(trained)} networks in {time.time() - t0:.0f} s")
    torch.set_num_threads(4)
    new = {}
    for case, variant, model, secs in trained:
        joblib.dump(model, OUT / "saved_models" / f"{case}__{variant}.joblib")
        new[(case, variant)] = (model, secs)
    current = {c: joblib.load(SAVED_MODELS_DIR / "mlp" / f"{c}.joblib") for c in CASES}
    cands = {c: {"current": current[c], **{v: new[(cc, v)][0] for (cc, v) in new if cc == c}} for c in CASES}

    # (a) validation
    rows = []
    for case in CASES:
        val = load(case)["val"]
        for name, m in cands[case].items():
            secs = new[(case, name)][1] if (case, name) in new else None
            rows.append({"case": case, "model": name, "recipe": "+".join(k for k in m.recipe if k != "dt_gamma"),
                         "train_seconds": secs, **metrics(val[CURRENTS].to_numpy(), m.predict(val))})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "validation.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n(a) validation [%]\n", res[["case", "model", "id_median_err_pct", "ig_median_err_pct", "is_median_err_pct",
                                        "ib_median_err_pct", "mean_median_err_pct", "mean_p95_err_pct"]].round(3).to_string(index=False))

    # (b) Vds sweep + drain/source symmetry
    parts = [sweep_and_symmetry(cands[c], c) for c in CASES]
    sw, sym = pd.concat([p[0] for p in parts]), pd.concat([p[1] for p in parts])
    sw.to_csv(OUT / "vds_sweep.csv", index=False)
    sym.to_csv(OUT / "symmetry.csv", index=False)
    with pd.option_context("display.float_format", "{:.3g}".format, "display.width", 250):
        print("\n(b) Ich [A] vs Vds at the median validation row\n",
              sw.pivot_table(index=["case", "vds_V"], columns="model", values="ich_A").to_string())
        print("\n    max relative drain/source-swap asymmetry\n",
              sym.groupby(["case", "model"]).max_rel_asym.max().unstack().to_string())

    # (c) gates, sum V*I / VDD, no floor
    combos = {"current": current,
              "chan": {c: new[(c, "chan")][0] for c in CASES},
              "chan_dt_off": {"nmos_on": new[("nmos_on", "chan")][0], "pmos_on": new[("pmos_on", "chan")][0],
                              "nmos_off": new[("nmos_off", "chan_dt")][0], "pmos_off": new[("pmos_off", "chan_dt")][0]},
              "chan_off_only": {**current, "nmos_off": new[("nmos_off", "chan")][0],
                                "pmos_off": new[("pmos_off", "chan")][0]}}
    g = pd.concat([gate_errors(m).assign(models=name) for name, m in combos.items()])
    g.to_csv(OUT / "gate_errors.csv", index=False)
    t = g.groupby(["gate", "inputs", "models"], sort=False).err_pct.agg(
        median="median", p95=lambda x: x.quantile(0.95), max="max").unstack("models")
    with pd.option_context("display.float_format", "{:.3g}".format, "display.width", 300):
        print("\n(c) gate leakage error [%], sum V*I / VDD, no Vds floor\n", t.to_string())
        print("\nall rows:\n", g.groupby("models").err_pct.agg(
            median="median", p95=lambda x: x.quantile(0.95), max="max").to_string())


if __name__ == "__main__":
    main()
