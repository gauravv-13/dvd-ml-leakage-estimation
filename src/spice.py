"""Run the hand-written single-transistor netlists (netlists/transistor/<case>.cir) for each sample.

Each netlist holds one sample in its `.param` lines. For every row of the dataset the
script replaces those `.param` values with the row's values, runs ngspice and reads the
four printed currents.

Sign convention: a terminal current is the current flowing INTO the device at that
terminal. ngspice prints i(Vx) as the current flowing into the + node of source Vx, so
I_terminal = -i(Vx).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import NETLIST_DIR, Config

NGSPICE = shutil.which("ngspice") or "ngspice"
SAMPLE_PARAMS = ["vdd", "temp", "w", "vd", "vs", "vth0", "toxe", "ndep", "u0", "xj"]
_PRINT_RE = re.compile(r"^\s*i\(v[dgsb]\)\s*=\s*(\S+)", re.IGNORECASE)


def ngspice_version() -> str:
    out = subprocess.run([NGSPICE, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"ngspice-(\S+)", out)
    return m.group(0) if m else "unknown"


def apply_sample(netlist: str, row: pd.Series) -> str:
    """Replace each `name=value` of SAMPLE_PARAMS in the `.param` lines with the row's value."""
    for name in SAMPLE_PARAMS:
        netlist, n = re.subn(rf"(?m)^(\.param\b.*?\b{name}=)\S+", rf"\g<1>{row[name]:.12e}", netlist)
        if n != 1:
            raise ValueError(f"expected exactly one .param '{name}=' in the netlist, found {n}")
    return netlist


def run_netlist(netlist: str, cwd: Path) -> np.ndarray:
    """Run one netlist (from stdin, so relative .include paths resolve from `cwd`) and
    return [Id, Ig, Is, Ib] as currents INTO the device."""
    res = subprocess.run([NGSPICE, "-b"], input=netlist, capture_output=True, text=True, cwd=cwd)
    out = res.stdout + res.stderr
    vals = [float(m.group(1)) for line in out.splitlines() if (m := _PRINT_RE.match(line))]
    if res.returncode != 0 or len(vals) != 4:
        raise RuntimeError(f"ngspice failed (rc={res.returncode}):\n{out[-3000:]}")
    return -np.asarray(vals)


def simulate_single_fet(case: str, samples: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply every sample to netlists/transistor/<case>.cir; returns samples + id, ig, is, ib."""
    path = NETLIST_DIR / "transistor" / f"{case}.cir"
    netlist = path.read_text()
    # Threads suffice: each worker just waits on its own ngspice subprocess.
    with ThreadPoolExecutor(max_workers=cfg["transistor_data"]["n_jobs"]) as pool:
        currents = list(pool.map(lambda r: run_netlist(apply_sample(netlist, r), path.parent),
                                 (r for _, r in samples.iterrows())))
    out = samples.copy()
    out[["id", "ig", "is", "ib"]] = np.vstack(currents)
    return out
