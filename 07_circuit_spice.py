"""Phase 6 (Part C step 6): SPICE reference leakage of the test circuit, 256 input vectors x 200 corners.

The 8-bit parity checker has a hand-written netlist netlists/circuit/parity8.cir (deliverable 2;
28 NAND2, out = 1 for an odd number of 1s, see src/circuit.py). For every corner of data/gates/corners.csv (the same 200
corners as Part B) and every input vector, the script replaces the netlist's `.param` values
(corner, inputs at exactly 0 or VDD, widths from config gates.widths_lmin), runs ngspice and
records the output voltage and the current of every source.
Vector k sets input in_i to bit i of k.

Leakage = total static power / VDD = current delivered by the VDD supply plus the currents
delivered by the inputs held at VDD (inputs at 0 V deliver no power): the same definition as the
gate reference (05_gate_spice.py). The individual source currents are stored too.
These results are for evaluation only and are not used for training.

Outputs
  data/circuit/circuit_spice.csv       one row per (vector, corner): inputs, corner, v_out, out_logic,
                                       i_vdd, i_in0..i_in7 (current delivered by each source), leak [A]
  data/circuit/metadata.json           simulator version, widths, timing, checks
  results/circuit_spice_summary.csv    per input vector: median / min / max SPICE leakage
Run:  python 07_circuit_spice.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.circuit import INPUTS, NETLIST, evaluate, gate_list, vectors
from src.config import DATA_DIR, RESULTS_DIR, load_config
from src.spice import apply_params, ngspice_version, run_print

PROCESS = ["vth0", "toxe", "ndep", "u0", "xj"]
# netlist width param -> (cell, n/p) in config gates.widths_lmin
WIDTHS = {"wn_nand": ("nand2", "n"), "wp_nand": ("nand2", "p")}


def simulate(k: int, v: dict[str, int], out_logic: int, corner: pd.Series, netlist: str, widths: dict) -> dict:
    values = {"vdd": corner.vdd, "temp": corner.temp, **widths,
              **{f"{d}{p}": corner[f"{d}_{p}"] for d in "np" for p in PROCESS}, **v}
    res = run_print(apply_params(netlist, values), NETLIST.parent)
    # ngspice i(Vx) is the current into the + node of Vx; the current it delivers is -i(Vx)
    delivered = {n: -res[f"i(v{n})"] for n in INPUTS}
    i_vdd = -res["i(vdd)"]
    return {"vector": k, **v, "corner": int(corner.corner), "vdd": corner.vdd, "temp": corner.temp,
            "v_out": res["v(out)"], "out_logic": out_logic,
            "i_vdd": i_vdd, **{f"i_{n}": c for n, c in delivered.items()},
            "leak": i_vdd + sum(delivered[n] for n in INPUTS if v[n])}


def main() -> None:
    cfg = load_config()
    corners = pd.read_csv(DATA_DIR / "gates" / "corners.csv")
    wl = cfg["gates"]["widths_lmin"]
    widths = {k: wl[cell][t] * cfg.lmin for k, (cell, t) in WIDTHS.items()}
    netlist = NETLIST.read_text()
    gates = gate_list()
    jobs = [(k, v, evaluate(gates, v)["out"], c, netlist, widths)
            for k, v in enumerate(vectors()) for _, c in corners.iterrows()]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg["gates"]["n_jobs"]) as pool:
        rows = list(pool.map(lambda j: simulate(*j), jobs))
    dt = time.time() - t0
    df = pd.DataFrame(rows)
    out = DATA_DIR / "circuit"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "circuit_spice.csv", index=False, float_format="%.12e")

    # checks: output at the expected rail, leakage positive
    v_err = np.abs(df.v_out - df.out_logic * df.vdd) / df.vdd
    print(f"{len(df)} simulations in {dt:.0f} s | max |Vout - expected| = {v_err.max():.2e} VDD | "
          f"leak <= 0: {(df.leak <= 0).sum()}")
    meta = {"ngspice": ngspice_version(), "model_file": cfg.model_path.name, "n_corners": len(corners),
            "n_vectors": 256, "n_gates": gates.type.value_counts().to_dict(), "widths_m": widths,
            "seconds": round(dt, 1), "max_vout_error_frac_vdd": float(v_err.max()),
            "n_leak_nonpositive": int((df.leak <= 0).sum())}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    summary = (df.groupby("vector").leak
               .agg(spice_median_A="median", spice_min_A="min", spice_max_A="max").reset_index())
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "circuit_spice_summary.csv", index=False)
    print(f"median leakage over vectors: {summary.spice_median_A.min():.3e} .. {summary.spice_median_A.max():.3e} A")


if __name__ == "__main__":
    main()
