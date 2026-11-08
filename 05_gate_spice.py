"""Phase 4 (Part B step 3): SPICE reference leakage of the five gates, 18 input cases x 200 corners.

Each gate has a hand-written netlist netlists/gates/<gate>.cir (deliverable 2). For every corner of
data/gates/corners.csv and every input combination the script replaces the netlist's `.param`
values (corner, inputs at exactly 0 or VDD, widths from config gates.widths_lmin), runs ngspice
and records the output voltage, the internal node voltages and the current of every source.

Leakage = total static power / VDD = current delivered by the VDD supply plus the currents
delivered by the inputs held at VDD (inputs at 0 V deliver no power). This counts the gate
tunnelling current that the input sources feed into the gates as well as the supply current.
The individual source currents are stored too, so other definitions can be rebuilt.
These results are for evaluation only and are not used for training.

Outputs
  data/gates/gate_spice.csv        one row per (gate, inputs, corner): corner values, node voltages,
                                   i_vdd, i_a, i_b (current delivered by each source) and leak [A]
  data/gates/metadata.json         simulator version, widths, timing
  results/gate_spice_summary.csv   per gate and input: median / min / max SPICE leakage
Run:  python 05_gate_spice.py
"""
from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.config import DATA_DIR, NETLIST_DIR, RESULTS_DIR, load_config
from src.spice import apply_params, ngspice_version, run_print

PROCESS = ["vth0", "toxe", "ndep", "u0", "xj"]
# per gate: number of inputs, logic function, internal nodes, netlist width param -> (cell, n/p)
GATES = {
    "inv":   {"n_in": 1, "logic": lambda a: 1 - a[0], "nodes": [],
              "widths": {"wn": ("inv", "n"), "wp": ("inv", "p")}},
    "nand2": {"n_in": 2, "logic": lambda a: 1 - (a[0] & a[1]), "nodes": ["x"],
              "widths": {"wn": ("nand2", "n"), "wp": ("nand2", "p")}},
    "nor2":  {"n_in": 2, "logic": lambda a: 1 - (a[0] | a[1]), "nodes": ["y"],
              "widths": {"wn": ("nor2", "n"), "wp": ("nor2", "p")}},
    "and2":  {"n_in": 2, "logic": lambda a: a[0] & a[1], "nodes": ["x", "nd"],
              "widths": {"wn_nand": ("nand2", "n"), "wp_nand": ("nand2", "p"),
                         "wn_inv": ("inv", "n"), "wp_inv": ("inv", "p")}},
    "or2":   {"n_in": 2, "logic": lambda a: a[0] | a[1], "nodes": ["y", "nr"],
              "widths": {"wn_nor": ("nor2", "n"), "wp_nor": ("nor2", "p"),
                         "wn_inv": ("inv", "n"), "wp_inv": ("inv", "p")}},
}
INPUT_NAMES = ["a", "b"]


def simulate(gate: str, inputs: tuple[int, ...], corner: pd.Series, netlist: str, widths: dict) -> dict:
    g = GATES[gate]
    values = {"vdd": corner.vdd, "temp": corner.temp, **widths,
              **{f"{d}{p}": corner[f"{d}_{p}"] for d in "np" for p in PROCESS},
              **{f"in{n}": v for n, v in zip(INPUT_NAMES, inputs)}}
    res = run_print(apply_params(netlist, values), NETLIST_DIR / "gates")
    # ngspice i(Vx) is the current into the + node of Vx; the current it delivers is -i(Vx)
    delivered = {n: -res[f"i(v{n})"] for n in INPUT_NAMES[:g["n_in"]]}
    i_vdd = -res["i(vdd)"]
    return {"gate": gate, "inputs": "".join(map(str, inputs)), "corner": int(corner.corner),
            "vdd": corner.vdd, "temp": corner.temp,
            "v_out": res["v(out)"], **{f"v_{n}": res[f"v({n})"] for n in g["nodes"]},
            "out_logic": g["logic"](inputs), "i_vdd": i_vdd, **{f"i_{n}": v for n, v in delivered.items()},
            "leak": i_vdd + sum(delivered[n] for n, v in zip(INPUT_NAMES, inputs) if v)}


def main() -> None:
    cfg = load_config()
    corners = pd.read_csv(DATA_DIR / "gates" / "corners.csv")
    wl = cfg["gates"]["widths_lmin"]
    jobs = []
    for gate, g in GATES.items():
        netlist = (NETLIST_DIR / "gates" / f"{gate}.cir").read_text()
        widths = {k: wl[cell][t] * cfg.lmin for k, (cell, t) in g["widths"].items()}
        for inputs in itertools.product([0, 1], repeat=g["n_in"]):
            jobs += [(gate, inputs, c, netlist, widths) for _, c in corners.iterrows()]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg["gates"]["n_jobs"]) as pool:
        rows = list(pool.map(lambda j: simulate(*j), jobs))
    dt = time.time() - t0
    df = pd.DataFrame(rows)
    out = DATA_DIR / "gates"
    df.to_csv(out / "gate_spice.csv", index=False, float_format="%.12e")

    # checks: output at the expected rail, leakage positive
    v_err = np.abs(df.v_out - df.out_logic * df.vdd) / df.vdd
    print(f"{len(df)} simulations in {dt:.0f} s | max |Vout - expected| = {v_err.max():.2e} VDD | "
          f"leak <= 0: {(df.leak <= 0).sum()}")
    meta = {"ngspice": ngspice_version(), "model_file": cfg.model_path.name, "n_corners": len(corners),
            "widths_m": {gate: {k: wl[c][t] * cfg.lmin for k, (c, t) in g["widths"].items()}
                         for gate, g in GATES.items()},
            "seconds": round(dt, 1), "max_vout_error_frac_vdd": float(v_err.max())}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    summary = (df.groupby(["gate", "inputs"], sort=False).leak
               .agg(spice_median_A="median", spice_min_A="min", spice_max_A="max").reset_index())
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "gate_spice_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
