"""Random sampling of the PVT / geometry / terminal-voltage space (spec section 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CASES, Config


def sample_process(dev: str, n: int, cfg: Config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Each process parameter ~ nominal * U(1 - tol, 1 + tol).
    Multiplying keeps the sign, so PMOS vth0 stays negative."""
    tol = cfg["sampling"]["process_tolerance"]
    return {p: cfg.nominal[dev][p] * rng.uniform(1 - tol, 1 + tol, n)
            for p in cfg["sampling"]["process_params"]}


def sample_transistor_inputs(case: str, n: int, cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n independent samples for one single-FET case.

    VDD is drawn first because Vd/Vs ranges depend on it; Vg/Vb are fixed by the case.
    """
    s = cfg["sampling"]
    spec = CASES[case]
    vdd = cfg.vdd_nominal * rng.uniform(1 - s["vdd_tolerance"], 1 + s["vdd_tolerance"], n)
    df = pd.DataFrame({
        "vdd": vdd,
        "temp": rng.uniform(*s["temp_range"], n),
        "w": cfg.lmin * rng.uniform(*s["width_range_lmin"], n),
        "vd": rng.uniform(0.0, 1.0, n) * vdd,
        "vs": rng.uniform(0.0, 1.0, n) * vdd,
    })
    df["vg"] = vdd if spec["vg"] == "vdd" else float(spec["vg"])
    df["vb"] = vdd if spec["vb"] == "vdd" else float(spec["vb"])
    for p, v in sample_process(spec["dev"], n, cfg, rng).items():
        df[p] = v
    return df

