"""Phase 4 (Part B step 3): draw the 200 random corners shared by the gate and test-circuit runs.

A corner is one random draw of VDD, temperature and the five process parameters (spec section 4):
all NMOS in a gate share one draw (n_*) and all PMOS share another (p_*).
  VDD ~ U(0.9, 1.1) * nominal, T ~ U(-40, 125) degC, each process parameter ~ nominal * U(0.9, 1.1)

Output: data/gates/corners.csv  (corner, vdd, temp, n_vth0 .. n_xj, p_vth0 .. p_xj)
Run:    python 04_gen_corners.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATA_DIR, load_config
from src.sampling import sample_process


def main() -> None:
    cfg = load_config()
    g, s = cfg["gates"], cfg["sampling"]
    n = g["n_corners"]
    rng = np.random.default_rng(g["seed"])
    corners = pd.DataFrame({
        "corner": np.arange(n),
        "vdd": cfg.vdd_nominal * rng.uniform(1 - s["vdd_tolerance"], 1 + s["vdd_tolerance"], n),
        "temp": rng.uniform(*s["temp_range"], n),
    })
    for dev in ["nmos", "pmos"]:
        for p, v in sample_process(dev, n, cfg, rng).items():
            corners[f"{dev[0]}_{p}"] = v
    out = DATA_DIR / "gates"
    out.mkdir(parents=True, exist_ok=True)
    corners.to_csv(out / "corners.csv", index=False, float_format="%.12e")
    print(f"{n} corners -> {out / 'corners.csv'}")
    print(corners.describe().loc[["min", "max"]].T.to_string())


if __name__ == "__main__":
    main()
