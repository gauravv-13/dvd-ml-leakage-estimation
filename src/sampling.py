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


def nominal_sweep(case: str, column: str, values, cfg: Config, w_lmin: float = 2.0) -> pd.DataFrame:
    """Rows at nominal conditions (VDD_nom, 27 C, W = w_lmin*Lmin, nominal process, full VDD
    across the channel in the conducting direction) with one column swept over `values`."""
    spec = CASES[case]
    dev, vdd = spec["dev"], cfg.vdd_nominal
    n = len(values)
    df = pd.DataFrame({"vdd": [vdd] * n, "temp": 27.0, "w": w_lmin * cfg.lmin,
                       "vd": vdd if dev == "nmos" else 0.0, "vs": 0.0 if dev == "nmos" else vdd,
                       "vg": vdd if spec["vg"] == "vdd" else 0.0,
                       "vb": vdd if spec["vb"] == "vdd" else 0.0})
    for p in cfg["sampling"]["process_params"]:
        df[p] = cfg.nominal[dev][p]
    df[column] = np.asarray(values, dtype=float)
    return df
