"""Phase 2 (deliverable 3): data check of the four transistor datasets.

1. Kirchhoff check per case: residual |Id + Ig + Is + Ib| (absolute and relative).
2. Trend plots from the datasets, with a clean nominal SPICE sweep overlaid:
     a. OFF current vs temperature
     b. OFF current vs vth0
     c. gate current vs toxe
   OFF current = |Id| per unit width of the OFF devices, restricted to samples with
   |Vds| > VDD/2 so the scatter shows the trend rather than the Vds spread.

Outputs: results/kcl_check.csv, figures/kcl_residual.png, figures/trend_*.png
Run:     python 02_data_checks.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CASES, DATA_DIR, FIGURES_DIR, RESULTS_DIR, load_config
from src.sampling import nominal_sweep
from src.spice import simulate_single_fet

ABS_FLOOR = 1e-15   # A, see README "Kirchhoff floor"
REL_TOL = 1e-6
COLORS = {"nmos": "#2563eb", "pmos": "#dc2626"}


def load(case: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv")


def kcl_check(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for case, df in data.items():
        I = df[["id", "ig", "is", "ib"]].to_numpy()
        resid = np.abs(I.sum(1))
        rel = resid / np.abs(I).max(1)
        passed = resid <= np.maximum(REL_TOL * np.abs(I).max(1), ABS_FLOOR)
        rows.append({"case": case, "n": len(df),
                     "median_abs_residual_A": np.median(resid), "max_abs_residual_A": resid.max(),
                     "median_rel_residual": np.median(rel), "p95_rel_residual": np.quantile(rel, 0.95),
                     "pass_fraction": passed.mean()})
        ax.hist(np.log10(np.maximum(resid, 1e-24)), bins=60, histtype="step", lw=1.5, label=case)
    ax.axvline(np.log10(ABS_FLOOR), color="k", ls="--", lw=1, label="1e-15 A floor")
    ax.set_xlabel(r"$\log_{10}\,|I_d+I_g+I_s+I_b|$  [A]")
    ax.set_ylabel("samples")
    ax.set_title("Kirchhoff residual per sample")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kcl_residual.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(rows)


def binned_median(x: np.ndarray, y: np.ndarray, bins: int = 15):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    med = np.array([np.median(y[idx == i]) if np.any(idx == i) else np.nan for i in range(bins)])
    return centers, med


def trend_plot(cfg, data, *, cases, x, y, xlabel, ylabel, title, fname, xscale=1.0, filt=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for case in cases:
        dev = CASES[case]["dev"]
        df = data[case]
        if filt is not None:
            df = df[filt(df)]
        xv = df[x].to_numpy() * xscale
        yv = np.abs(df[y].to_numpy()) / df.w.to_numpy() * 1e-6   # A per um of width
        ax.scatter(xv, yv, s=3, alpha=0.15, color=COLORS[dev], rasterized=True)
        cx, my = binned_median(xv, yv)
        ax.plot(cx, my, color=COLORS[dev], lw=2, label=f"{case}: dataset (binned median)")

        lo, hi = data[case][x].min(), data[case][x].max()
        sweep = simulate_single_fet(case, nominal_sweep(case, x, np.linspace(lo, hi, 25), cfg), cfg)
        ax.plot(sweep[x] * xscale, np.abs(sweep[y]) / sweep.w * 1e-6, color=COLORS[dev], ls="--", lw=1.5,
                label=f"{case}: SPICE sweep, rest nominal")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    FIGURES_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    data = {case: load(case) for case in CASES}

    kcl = kcl_check(data)
    kcl.to_csv(RESULTS_DIR / "kcl_check.csv", index=False)
    print("Kirchhoff check")
    print(kcl.to_string(index=False, float_format=lambda v: f"{v:.2e}"))

    big_vds = lambda df: np.abs(df.vd - df.vs) > 0.5 * df.vdd
    trend_plot(cfg, data, cases=["nmos_off", "pmos_off"], x="temp", y="id", filt=big_vds,
               xlabel="Temperature [°C]", ylabel=r"OFF current $|I_d|/W$  [A/µm]",
               title=f"OFF current vs temperature ({cfg.model_path.stem})", fname="trend_ioff_vs_temp.png")
    for case in ["nmos_off", "pmos_off"]:   # vth0 differs in sign/scale per device -> one plot each
        trend_plot(cfg, data, cases=[case], x="vth0", y="id", filt=big_vds,
                   xlabel=r"$v_{th0}$ [V]", ylabel=r"OFF current $|I_d|/W$  [A/µm]",
                   title=f"OFF current vs vth0, {case} ({cfg.model_path.stem})",
                   fname=f"trend_ioff_vs_vth0_{case}.png")
    trend_plot(cfg, data, cases=["nmos_on", "pmos_on"], x="toxe", y="ig", xscale=1e9,
               xlabel=r"$t_{oxe}$ [nm]", ylabel=r"Gate current $|I_g|/W$  [A/µm]",
               title=f"Gate current vs toxe, ON devices ({cfg.model_path.stem})", fname="trend_ig_vs_toxe.png")
    print(f"\nFigures written to {FIGURES_DIR.relative_to(FIGURES_DIR.parent)}/")


if __name__ == "__main__":
    main()
