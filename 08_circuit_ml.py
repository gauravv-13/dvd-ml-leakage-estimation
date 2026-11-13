"""Phase 6 (Part C steps 7-8): hierarchical ML leakage of the test circuit, 256 input vectors x 200 corners, vs SPICE.

For every input vector, src/circuit.py gives the logic value of every net, so each of the 28 NAND2 has
known input bits (pin order a b; A is the NMOS at ground, as in netlists/gates/nand2.cir). The circuit
corners are the Part B corners, so each gate's ML leakage is the Phase 5 result for that gate, inputs
and corner (data/gates/gate_ml.csv), and the circuit leakage is their sum (spec step 7). This assumes
every gate input sits at exactly 0 or VDD and ignores loading between gates. The same sum of the gate
SPICE leakages measures that assumption alone, without ML.

Outputs
  data/circuit/circuit_ml.csv     one row per (vector, corner): circuit SPICE leakage, ML sum, gate SPICE sum,
                                  the relative error of both sums [%], and the share of the gate SPICE sum that
                                  comes from NAND2 gates at 11
  results/circuit_results.csv     deliverable 8, for the ML sum, the gate SPICE sum, and as reference the Phase 5
                                  NAND2 cases and all Phase 5 gates: SPICE median, estimate median, bias (median
                                  signed error), median and p95 |error|, mean accuracy (100 % - mean |error|), max
  results/circuit_error_breakdown.csv   circuit-level error analysis: bias, median and p95 |error| of the ML sum
                                  by temperature band, PMOS |vth0|, NMOS vth0 and VDD quartile, and quartile of
                                  the leakage share of NAND2 gates at 11 (the case dominated by OFF-PMOS current);
                                  n_corners, since each corner contributes all 256 vectors
  results/circuit_error_drivers.csv     Spearman correlation of the per-corner median |error| with each corner
                                  condition (200 corners), to rank the drivers without repeating corners
  figures/circuit_parity.png      deliverable 8: ML vs SPICE leakage of all 51,200 rows, log axes
Run:  python 08_circuit_ml.py     (after 06_gate_ml.py and 07_circuit_spice.py)
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.circuit import evaluate, gate_list, vectors  # noqa: E402
from src.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR  # noqa: E402

INPUT_CASES = ["00", "01", "10", "11"]


def input_counts() -> pd.DataFrame:
    """Per input vector: how many NAND2 of the circuit see each input combination (pins a b)."""
    gates = gate_list()
    assert set(gates.type) == {"nand2"}, "only NAND2 leakages are looked up; add the other gate types first"
    rows = []
    for v in vectors():
        val = evaluate(gates, v)
        cases = [f"{val[g.a]}{val[g.b]}" for g in gates.itertuples()]
        rows.append([cases.count(c) for c in INPUT_CASES])
    return pd.DataFrame(rows, columns=INPUT_CASES).rename_axis("vector")


def circuit_sum(counts: pd.DataFrame, gates: pd.DataFrame, column: str) -> pd.Series:
    """Sum of the gate leakages `column` over the circuit, indexed by (vector, corner)."""
    per_corner = gates.pivot(index="corner", columns="inputs", values=column)[INPUT_CASES]
    return (counts @ per_corner.T).stack().rename(column)


def errors(est: pd.Series, spice: pd.Series) -> pd.Series:
    return (est - spice) / spice * 100


def summary(est: pd.Series, spice: pd.Series) -> dict:
    """The metrics of results/gate_results.csv (06_gate_ml.py), for one estimate against SPICE."""
    e = errors(est, spice)
    return {"n": len(e), "spice_median_A": spice.median(), "estimate_median_A": est.median(),
            "bias_pct": e.median(), "median_err_pct": e.abs().median(), "p95_err_pct": e.abs().quantile(0.95),
            "mean_accuracy_pct": 100 - e.abs().mean(), "max_err_pct": e.abs().max()}


def breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Bias, median and p95 |error| of the ML sum per bin of the conditions that drive the error."""
    groups = {"temperature_C": pd.cut(df.temp, [-float("inf"), 0, 40, float("inf")]),
              "pmos_abs_vth0_quartile": pd.qcut(df.p_vth0.abs(), 4),
              "nmos_vth0_quartile": pd.qcut(df.n_vth0, 4),
              "vdd_quartile": pd.qcut(df.vdd, 4),
              "nand2_11_share_quartile": pd.qcut(df.nand2_11_share, 4)}
    return pd.DataFrame([{"by": name, "bin": str(b), "n": len(e), "n_corners": df.corner[e.index].nunique(),
                          "bias_pct": e.median(), "median_err_pct": e.abs().median(),
                          "p95_err_pct": e.abs().quantile(0.95)}
                         for name, bins in groups.items()
                         for b, e in df.ml_err_pct.groupby(bins, observed=True)])


def drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of the per-corner median |error| with the corner conditions (one point per corner)."""
    per_corner = df.groupby("corner").agg(median_abs_err_pct=("ml_err_pct", lambda s: s.abs().median()),
                                          temp=("temp", "first"), vdd=("vdd", "first"),
                                          pmos_abs_vth0=("p_vth0", lambda s: s.abs().iloc[0]),
                                          nmos_vth0=("n_vth0", "first"))
    return pd.DataFrame([{"condition": c, "n_corners": len(per_corner),
                          "spearman_r": per_corner[c].corr(per_corner.median_abs_err_pct, method="spearman")}
                         for c in ["pmos_abs_vth0", "temp", "nmos_vth0", "vdd"]])


def main() -> None:
    gml = pd.read_csv(DATA_DIR / "gates" / "gate_ml.csv", dtype={"inputs": str},
                      usecols=["gate", "inputs", "corner", "spice_leak", "ml_leak"])
    # w_abs_err: leakage-weighted |gate error|, whose sum over the circuit gives the error without cancellation
    nand = gml[gml.gate == "nand2"].assign(w_abs_err=lambda g: (g.ml_leak - g.spice_leak).abs() * 100)
    counts = input_counts()
    corners = pd.read_csv(DATA_DIR / "gates" / "corners.csv", usecols=["corner", "p_vth0", "n_vth0"])
    spice = pd.read_csv(DATA_DIR / "circuit" / "circuit_spice.csv", usecols=["vector", "corner", "vdd", "temp", "leak"])
    sums = pd.concat([circuit_sum(counts, nand, "ml_leak"),
                      circuit_sum(counts, nand, "spice_leak").rename("gate_spice_sum")], axis=1)
    df = spice.rename(columns={"leak": "spice_leak"}).join(sums, on=["vector", "corner"])
    assert len(df) == len(counts) * len(corners) and df.notna().all().all(), "missing vector or corner"
    df["ml_err_pct"] = errors(df.ml_leak, df.spice_leak)
    df["gate_sum_err_pct"] = errors(df.gate_spice_sum, df.spice_leak)
    sp11 = nand[nand.inputs == "11"].set_index("corner").spice_leak
    df["nand2_11_share"] = (counts["11"].reindex(df.vector).to_numpy() * sp11.reindex(df.corner).to_numpy()
                            / df.gate_spice_sum)
    df.to_csv(DATA_DIR / "circuit" / "circuit_ml.csv", index=False, float_format="%.12e")

    table = pd.DataFrame([
        {"estimate": "ML sum of gate leakages", **summary(df.ml_leak, df.spice_leak)},
        {"estimate": "gate SPICE sum (no ML)", **summary(df.gate_spice_sum, df.spice_leak)},
        {"estimate": "Phase 5 NAND2 cases (reference)", **summary(nand.ml_leak, nand.spice_leak)},
        {"estimate": "Phase 5 all gates (reference)", **summary(gml.ml_leak, gml.spice_leak)},
    ])
    table.to_csv(RESULTS_DIR / "circuit_results.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))

    with_corners = df.merge(corners, on="corner")
    breakdown(with_corners).to_csv(RESULTS_DIR / "circuit_error_breakdown.csv", index=False)
    drv = drivers(with_corners)
    drv.to_csv(RESULTS_DIR / "circuit_error_drivers.csv", index=False)
    e = df.ml_err_pct.abs()
    worst = with_corners[e >= e.quantile(0.99)].drop_duplicates("corner")
    # gates at one input combination share the corner's error, so only the different combinations can cancel
    corner_share = df.groupby("corner").ml_err_pct.transform("mean").var() / df.ml_err_pct.var()
    per_vector = df.groupby("vector").ml_err_pct.apply(lambda s: s.abs().median())
    no_cancel = circuit_sum(counts, nand, "w_abs_err").reindex(pd.MultiIndex.from_frame(df[["vector", "corner"]]))
    no_cancel = no_cancel.to_numpy() / df.gate_spice_sum.to_numpy()
    print(f"error variance explained by the corner: {corner_share:.3f} | per-vector median |error| "
          f"{per_vector.min():.2f}-{per_vector.max():.2f} % | without cancellation between gates: median "
          f"{pd.Series(no_cancel).median():.2f} %, p95 {pd.Series(no_cancel).quantile(0.95):.2f} %")
    print(f"worst 1 % of rows: {(e >= e.quantile(0.99)).sum()} rows from corners {worst.corner.tolist()} "
          f"(T {worst.temp.round(1).tolist()} C, PMOS |vth0| {worst.p_vth0.abs().round(3).tolist()} V)")
    print("drivers (Spearman of per-corner median |error|):", drv.set_index("condition").spearman_r.round(2).to_dict())
    r = nand.assign(err=errors(nand.ml_leak, nand.spice_leak)).pivot(index="corner", columns="inputs", values="err").corr()
    print("correlation of NAND2 gate errors across corners, by input case:\n" + r.round(2).to_string())

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(df.spice_leak, df.ml_leak, s=2, alpha=0.3, label="256 vectors x 200 corners")
    lo, hi = df[["spice_leak", "ml_leak"]].min().min() / 1.5, df[["spice_leak", "ml_leak"]].max().max() * 1.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ML = SPICE")
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi), xlabel="SPICE leakage [A]",
           ylabel="ML leakage [A]", title=f"8-bit parity checker leakage: median |error| {e.median():.2f} %")
    ax.legend(markerscale=4, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "circuit_parity.png", dpi=200)


if __name__ == "__main__":
    main()
