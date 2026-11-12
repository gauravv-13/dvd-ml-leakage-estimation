"""Phase 5 (Part B steps 4-5): ML leakage of the five gates, 18 input cases x 200 corners, vs SPICE.

For every row of data/gates/gate_spice.csv, every transistor of the gate is run through the chosen
transistor model of its case, with terminal voltages from the SPICE node voltages of that row
(spec step 4), and the gate leakage is formed with the SPICE definition (method: src/gate_ml.py).

Outputs
  data/gates/gate_ml.csv        one row per (gate, inputs, corner): SPICE node voltages (the ML side
                                uses the same ones), output voltage, SPICE leakage, ML leakage, relative
                                error, ML leakage with KCL for ON transistors, number of floored OFF
                                transistors, and each transistor's case and predicted Id, Ig, Is, Ib
  results/gate_results.csv      deliverable 6, per gate and inputs: SPICE median, ML median, bias
                                (median signed error), median and p95 |error|, mean accuracy
                                (100 % - mean |error|), the same errors for leak_kcl, and the median
                                SPICE node voltages v_out, v_x, v_y, v_nd, v_nr (empty if the gate has none)
  figures/gate_parity.png       deliverable 7: ML vs SPICE leakage of all 3,600 rows, log axes
Run:  python 06_gate_ml.py      (after 05_gate_spice.py)
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from src.gate_ml import apply_kcl, gate_currents, leakage  # noqa: E402

NODES = ["v_out", "v_x", "v_y", "v_nd", "v_nr"]
CURRENT_NAMES = {"d": "id", "g": "ig", "s": "is", "b": "ib"}


def errors(ml: pd.Series, spice: pd.Series) -> pd.Series:
    return (ml - spice) / spice * 100


def summary(g: pd.DataFrame) -> pd.Series:
    e, ek = g.err_pct, errors(g.ml_leak_kcl, g.spice_leak)
    return pd.Series({"n": len(g), "spice_median_A": g.spice_leak.median(), "ml_median_A": g.ml_leak.median(),
                      "bias_pct": e.median(), "median_err_pct": e.abs().median(),
                      "p95_err_pct": e.abs().quantile(0.95), "mean_accuracy_pct": 100 - e.abs().mean(),
                      "max_err_pct": e.abs().max(), "n_floored": int((g.n_floored > 0).sum()),
                      "kcl_bias_pct": ek.median(), "kcl_median_err_pct": ek.abs().median(),
                      "kcl_p95_err_pct": ek.abs().quantile(0.95), "kcl_mean_accuracy_pct": 100 - ek.abs().mean(),
                      **{f"{n}_median": g[n].median() if n in g else np.nan for n in NODES}})


def main() -> None:
    spice = pd.read_csv(DATA_DIR / "gates" / "gate_spice.csv", dtype={"inputs": str})
    corners = pd.read_csv(DATA_DIR / "gates" / "corners.csv").drop(columns=["vdd", "temp"])
    out = []
    for (gate, inputs), rows in spice.groupby(["gate", "inputs"], sort=False):
        rows = rows.merge(corners, on="corner", how="left", validate="one_to_one")
        devs = gate_currents(gate, inputs, rows)
        res = rows[["gate", "inputs", "corner", "vdd", "temp", *[n for n in NODES if n in rows]]].copy()
        res["spice_leak"] = rows.leak.to_numpy()
        res["ml_leak"] = leakage(devs, inputs)
        res["err_pct"] = errors(res.ml_leak, res.spice_leak)
        res["ml_leak_kcl"] = leakage(apply_kcl(devs), inputs)
        res["n_floored"] = sum(d["floored"].astype(int) for d in devs)
        for d in devs:
            res[f"{d['name']}_case"] = d["case"]
            for k, c in CURRENT_NAMES.items():
                res[f"{d['name']}_{c}"] = d["I"][k]
        out.append(res)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(DATA_DIR / "gates" / "gate_ml.csv", index=False, float_format="%.12e")

    table = df.groupby(["gate", "inputs"], sort=False).apply(summary, include_groups=False).reset_index()
    table.to_csv(RESULTS_DIR / "gate_results.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(table.drop(columns=[c for c in table if c.startswith("kcl_") or c.startswith("v_")]).to_string(index=False))
    e = df.err_pct.abs()
    print(f"all 3,600 rows: median |error| {e.median():.2f} %, p95 {e.quantile(0.95):.2f} %, "
          f"mean accuracy {100 - e.mean():.2f} %, max {e.max():.1f} % | with KCL: median "
          f"{errors(df.ml_leak_kcl, df.spice_leak).abs().median():.2f} % | rows with a floored OFF "
          f"transistor: {(df.n_floored > 0).sum()}")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for gate, g in df.groupby("gate", sort=False):
        ax.scatter(g.spice_leak, g.ml_leak, s=6, alpha=0.5, label=gate.upper())
    lo, hi = df[["spice_leak", "ml_leak"]].min().min() / 1.5, df[["spice_leak", "ml_leak"]].max().max() * 1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ML = SPICE")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="SPICE leakage [A]",
           ylabel="ML leakage [A]",
           title=f"Gate leakage, 18 cases x 200 corners: median |error| {e.median():.2f} %")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gate_parity.png", dpi=200)


if __name__ == "__main__":
    main()
