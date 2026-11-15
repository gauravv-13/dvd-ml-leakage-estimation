"""Report extras for the error analysis: tables and figures the earlier scripts do not write.

Outputs
  results/circuit_results_per_vector.csv  deliverable 8 per input vector (256 rows, 200 corners each): SPICE
                                          median, ML median, bias, median and p95 |error|, mean accuracy
  figures/gate_parity_by_case.png         deliverable 7, coloured by whether OFF PMOS set most of the leakage
  figures/error_vs_corner.png             median |error| of all gates, the NAND2 11 gates and the circuit,
                                          against temperature and PMOS |vth0| (corner quantile bins)
  figures/off_id_error.png                test-set |error| of the OFF-case Id against |Id| and temperature
Prints the circuit error with the NAND2 11 gate errors removed (accuracy if the OFF-PMOS error were zero)
and the OFF-case Id errors quoted in the report.
Run:  python 09_report_figures.py     (after 08_circuit_ml.py)
"""
from __future__ import annotations

import importlib

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, SAVED_MODELS_DIR  # noqa: E402

circuit_ml = importlib.import_module("08_circuit_ml")
select = importlib.import_module("03_select_and_test")

# cases whose leakage is 89-99 % OFF-PMOS current (report, gate results)
PMOS_OFF_SET = {("inv", "1"), ("nand2", "11"), ("and2", "00"), ("nor2", "01"), ("nor2", "10")}
NOISE_FLOOR_A = 1e-15
plt.rcParams.update({"font.size": 11})


def per_vector(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"vector": v, **circuit_ml.summary(g.ml_leak, g.spice_leak)}
                         for v, g in df.groupby("vector")])


def binned_median(x: pd.Series, err: pd.Series, bins: int = 8) -> pd.DataFrame:
    b = pd.qcut(x, bins)
    return pd.DataFrame({"x": x.groupby(b, observed=True).median(),
                         "err": err.abs().groupby(b, observed=True).median()})


def gate_parity(gml: pd.DataFrame) -> None:
    dominated = np.array([(g, i) in PMOS_OFF_SET for g, i in zip(gml.gate, gml.inputs)])
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for mask, label, colour in [(~dominated, "other 13 cases", "tab:blue"),
                                (dominated, "5 cases set by OFF PMOS", "tab:red")]:
        ax.scatter(gml.leak[mask], gml.leak_preds[mask], s=6, alpha=0.4, c=colour, label=label)
    lo, hi = gml.leak.min() / 1.5, gml.leak.max() * 1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ML = SPICE")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="SPICE leakage [A]",
           ylabel="ML leakage [A]", title="Gate leakage, 18 cases x 200 corners")
    ax.legend(markerscale=3)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "gate_parity_by_case.png", dpi=200)


def error_vs_corner(gml: pd.DataFrame, circ: pd.DataFrame) -> None:
    gate_err = (gml.leak_preds - gml.leak) / gml.leak * 100
    n11 = (gml.gate == "nand2") & (gml.inputs == "11")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, xlabel in [(axes[0], "temp", "temperature [°C]"), (axes[1], "p_abs_vth0", "PMOS |vth0| [V]")]:
        for data, err, label, style in [(gml, gate_err, "all gates (3,600 rows)", "o-"),
                                        (gml[n11], gate_err[n11], "NAND2 11 (OFF PMOS)", "s--"),
                                        (circ, circ.ml_err_pct, "parity checker (51,200 rows)", "^-")]:
            b = binned_median(data[col], err)
            ax.plot(b.x, b.err, style, label=label)
        ax.set(xlabel=xlabel, ylabel="median |error| [%]", ylim=(0, None))
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_vs_corner.png", dpi=200)


def off_id_error() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for case, colour in [("nmos_off", "tab:blue"), ("pmos_off", "tab:red")]:
        te = select.rows(case, "test")
        model = joblib.load(SAVED_MODELS_DIR / "mlp" / f"{case}.joblib")
        err = pd.Series(select.rel_err(model.predict(te)[:, 0], te.id.to_numpy()), index=te.index)
        i_abs = te.id.abs()
        by_dec = err.groupby(np.floor(np.log10(i_abs))).median()
        label = case.replace("_", " ").upper()
        axes[0].plot(10 ** (by_dec.index + 0.5), by_dec, "o-", c=colour, label=label)
        b = binned_median(te.temp, err)
        axes[1].plot(b.x, b.err, "o-", c=colour, label=label)
        above = i_abs >= NOISE_FLOOR_A
        print(f"{case} Id test median |error|: all {err.median():.2f} %, |Id| >= 1e-15 A {err[above].median():.2f} %, "
              f">= 1e-12 A {err[i_abs >= 1e-12].median():.2f} % (n={int((i_abs >= 1e-12).sum())}), "
              f"share below 1e-15 A {(~above).mean():.2f}")
    axes[0].axvline(NOISE_FLOOR_A, c="k", ls=":", lw=1, label="noise floor")
    axes[0].set(xscale="log", yscale="log", xlabel="SPICE |Id| [A]", ylabel="median |error| of Id [%]")
    axes[1].set(yscale="log", xlabel="temperature [°C]", ylabel="median |error| of Id [%]")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "off_id_error.png", dpi=200)


def main() -> None:
    gml = pd.read_csv(DATA_DIR / "gates" / "gate_ml.csv", dtype={"inputs": str})
    corners = pd.read_csv(DATA_DIR / "gates" / "corners.csv", usecols=["corner", "p_vth0"])
    corners = corners.assign(p_abs_vth0=corners.p_vth0.abs()).drop(columns="p_vth0")
    gml = gml.merge(corners, on="corner")
    circ = pd.read_csv(DATA_DIR / "circuit" / "circuit_ml.csv").merge(corners, on="corner")

    pv = per_vector(circ)
    pv.to_csv(RESULTS_DIR / "circuit_results_per_vector.csv", index=False)
    print(f"per vector: median |error| {pv.median_err_pct.min():.2f}-{pv.median_err_pct.max():.2f} %, "
          f"p95 {pv.p95_err_pct.min():.1f}-{pv.p95_err_pct.max():.1f} %, "
          f"mean accuracy {pv.mean_accuracy_pct.min():.1f}-{pv.mean_accuracy_pct.max():.1f} %")

    # circuit leakage with the NAND2 11 gates at their SPICE leakage, all other gates ML
    nand = gml[gml.gate == "nand2"].rename(columns={"leak": "spice_leak", "leak_preds": "ml_leak"})
    nand = nand.assign(e_rest=lambda g: (g.ml_leak - g.spice_leak) * (g.inputs != "11"))
    idx = pd.MultiIndex.from_frame(circ[["vector", "corner"]])
    rest = circuit_ml.circuit_sum(circuit_ml.input_counts(), nand, "e_rest").reindex(idx).to_numpy()
    e = circuit_ml.errors(circ.gate_spice_sum + rest, circ.spice_leak)
    print(f"circuit with exact NAND2 11 gates: median |error| {e.abs().median():.2f} %, "
          f"p95 {e.abs().quantile(0.95):.2f} %, mean accuracy {100 - e.abs().mean():.2f} %")

    gate_parity(gml)
    error_vs_corner(gml, circ)
    off_id_error()


if __name__ == "__main__":
    main()
