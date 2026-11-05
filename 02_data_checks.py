"""Phase 2 (deliverable 3): data check of the four transistor datasets.

1. Kirchhoff check per case: sum of the four terminal currents, Id + Ig + Is + Ib.
2. Three trend plots of the dataset values:
     OFF current (|Id| of nmos_off, pmos_off) vs temperature
     OFF current (|Id| of nmos_off, pmos_off) vs |vth0|
     gate current (|Ig| of all four cases) vs toxe

Outputs: results/kcl_check.csv, figures/kcl_residual.png,
         figures/trend_ioff_vs_temp.png, figures/trend_ioff_vs_vth0.png, figures/trend_ig_vs_toxe.png
Run:     python 02_data_checks.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CASES, DATA_DIR, FIGURES_DIR, RESULTS_DIR

COLORS = {"nmos_on": "#1d4ed8", "nmos_off": "#60a5fa", "pmos_on": "#b91c1c", "pmos_off": "#f87171"}


def kcl_check(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, df in data.items():
        resid = np.abs(df[["id", "ig", "is", "ib"]].sum(axis=1).to_numpy())
        rows.append({"case": case, "median_abs_sum_A": np.median(resid), "max_abs_sum_A": resid.max()})
        ax.hist(np.log10(np.maximum(resid, 1e-24)), bins=60, histtype="step", lw=1.5,
                color=COLORS[case], label=case)
    ax.set_xlabel(r"$\log_{10}\,|I_d+I_g+I_s+I_b|$  [A]")
    ax.set_ylabel("samples")
    ax.set_title("Kirchhoff check: sum of the four terminal currents")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kcl_residual.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(rows)


def trend_plot(data, cases, x, y, xlabel, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for case in cases:
        ax.scatter(np.abs(data[case][x]), np.abs(data[case][y]), s=3, alpha=0.3,
                   color=COLORS[case], label=case, rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=200)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    data = {case: pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv") for case in CASES}

    kcl = kcl_check(data)
    kcl.to_csv(RESULTS_DIR / "kcl_check.csv", index=False)
    print(kcl.to_string(index=False, float_format=lambda v: f"{v:.2e}"))

    off = ["nmos_off", "pmos_off"]
    trend_plot(data, off, "temp", "id", "Temperature [°C]", r"OFF current $|I_d|$ [A]",
               "OFF current vs temperature", "trend_ioff_vs_temp.png")
    trend_plot(data, off, "vth0", "id", r"$|v_{th0}|$ [V]", r"OFF current $|I_d|$ [A]",
               "OFF current vs vth0", "trend_ioff_vs_vth0.png")
    trend_plot(data, list(CASES), "toxe", "ig", r"$t_{oxe}$ [m]", r"Gate current $|I_g|$ [A]",
               "Gate current vs toxe", "trend_ig_vs_toxe.png")


if __name__ == "__main__":
    main()
