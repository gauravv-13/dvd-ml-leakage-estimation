"""Phase 1 (Part A step 1): generate the four single-transistor datasets.

For each case (nmos_on, nmos_off, pmos_on, pmos_off) draw N random samples of
VDD, T, W, Vd, Vs and the five process parameters (spec section 4), apply each sample
to the hand-written netlist netlists/transistor/<case>.cir, simulate it in ngspice and
store inputs + terminal currents (current INTO each terminal).

Outputs
  data/transistor/<case>.csv          deliverable 1 (columns in spec order)
  data/transistor/metadata.json       simulator version, seed, timing

Run:  python 01_gen_transistor_data.py            (all cases, N from config.yaml)
      python 01_gen_transistor_data.py --n 500 --cases nmos_off --out data/smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.config import CASES, CSV_COLUMNS, DATA_DIR, load_config
from src.sampling import sample_transistor_inputs
from src.spice import ngspice_version, simulate_single_fet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, help="samples per case (default: config transistor_data.n_samples)")
    ap.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    ap.add_argument("--out", type=Path, default=DATA_DIR / "transistor")
    args = ap.parse_args()

    cfg = load_config()
    n = args.n or cfg["transistor_data"]["n_samples"]
    seed = cfg["transistor_data"]["seed"]
    args.out.mkdir(parents=True, exist_ok=True)

    # One independent, reproducible stream per case (independent of which cases are run).
    streams = dict(zip(CASES, np.random.SeedSequence(seed).spawn(len(CASES))))
    meta = {"ngspice": ngspice_version(), "model_file": cfg.model_path.name,
            "vdd_nominal": cfg.vdd_nominal, "lmin": cfg.lmin, "seed": seed, "n": n, "cases": {}}

    for case in args.cases:
        t0 = time.time()
        samples = sample_transistor_inputs(case, n, cfg, np.random.default_rng(streams[case]))
        df = simulate_single_fet(case, samples, cfg)[CSV_COLUMNS]
        df.to_csv(args.out / f"{case}.csv", index=False, float_format="%.12e")

        I = df[["id", "ig", "is", "ib"]].to_numpy()
        resid = np.abs(I.sum(1))
        dt = time.time() - t0
        meta["cases"][case] = {"seconds": round(dt, 1), "max_kcl_residual_A": float(resid.max()),
                               "id_abs_range_A": [float(np.abs(df.id).min()), float(np.abs(df.id).max())]}
        print(f"{case:9s} {n} samples in {dt:5.1f} s | max |sum I| {resid.max():.1e} A | "
              f"|Id| {np.abs(df.id).min():.1e} .. {np.abs(df.id).max():.1e} A")

    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
