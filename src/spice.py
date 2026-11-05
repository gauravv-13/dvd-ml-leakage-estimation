"""ngspice helpers: netlist text generation, batch execution, output parsing.

Sign convention used everywhere in this project: a terminal current is the current
flowing INTO the device at that terminal. ngspice reports i(Vx) as the current
flowing into the + node of source Vx, so I_terminal = -i(Vx) for a source whose
+ node is the terminal and - node is ground.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CASES, Config

NGSPICE = shutil.which("ngspice") or "ngspice"
_PRINT_RE = re.compile(r"^\s*(i\(v\w+\))\s*=\s*(\S+)", re.IGNORECASE)


def ngspice_version() -> str:
    out = subprocess.run([NGSPICE, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"ngspice-(\S+)", out)
    return m.group(0) if m else "unknown"


def run_ngspice(netlist: str, workdir: Path | None = None, timeout: int = 600) -> str:
    """Run ngspice in batch mode on the netlist text and return stdout+stderr."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        cir = Path(tmp) / "run.cir"
        cir.write_text(netlist)
        res = subprocess.run([NGSPICE, "-b", str(cir)], capture_output=True, text=True,
                             timeout=timeout, cwd=tmp)
    out = res.stdout + res.stderr
    if res.returncode != 0 or re.search(r"(?im)^\s*error", out):
        raise RuntimeError(f"ngspice failed (rc={res.returncode}):\n{out[-3000:]}")
    return out


def geometry(w: float, cfg: Config) -> dict[str, float]:
    """Drain/source junction geometry from width (same rule in every netlist)."""
    a = w * cfg.ldiff
    p = 2.0 * (w + cfg.ldiff)
    return {"ad": a, "as": a, "pd": p, "ps": p}


def mos_line(name: str, d: str, g: str, s: str, b: str, model: str, w: float, cfg: Config) -> str:
    g_ = geometry(w, cfg)
    return (f"{name} {d} {g} {s} {b} {model} W={w:.6e} L={cfg.lmin:.6e} "
            f"AD={g_['ad']:.6e} AS={g_['as']:.6e} PD={g_['pd']:.6e} PS={g_['ps']:.6e}")


def altermod_lines(model: str, params: dict[str, float]) -> list[str]:
    """One `altermod` per parameter.

    ngspice-47 applies only the FIRST parameter of a multi-parameter altermod line
    and silently ignores the rest (verified: `altermod nmos vth0=.. toxe=..` leaves
    toxe unchanged). Never put more than one parameter on a line.
    """
    return [f"altermod {model} {p}={v:.9e}" for p, v in params.items()]


def header(title: str, cfg: Config, include: str | None = None) -> str:
    """`include` overrides the model path (e.g. a relative path for exported netlists)."""
    return (f"* {title}\n"
            f".include {include or cfg.model_path}\n"
            f"{cfg['spice']['options']}\n")


# ---------------------------------------------------------------------------
# Single-transistor characterisation (Part A)
# ---------------------------------------------------------------------------

def example_netlist(case: str, row: pd.Series, cfg: Config, include: str) -> str:
    """Deliverable 2: a plain single-transistor netlist for one sample (one .op point)."""
    dev = CASES[case]["dev"]
    lines = [header(f"{case}: single {dev.upper()}, one sample", cfg, include) + f".temp {row.temp:.6g}",
             "",
             mos_line("M1", "d", "g", "s", "b", dev, row.w, cfg),
             f"Vd d 0 dc {row.vd:.6g}",
             f"Vg g 0 dc {row.vg:.6g}",
             f"Vs s 0 dc {row.vs:.6g}",
             f"Vb b 0 dc {row.vb:.6g}",
             "",
             ".control"]
    lines += altermod_lines(dev, {p: row[p] for p in cfg["sampling"]["process_params"]})
    lines += ["op", "print i(Vd) i(Vg) i(Vs) i(Vb)", ".endc", ".end", ""]
    return "\n".join(lines)


def single_fet_netlist(case: str, samples: pd.DataFrame, cfg: Config) -> str:
    """One netlist that simulates every row of `samples` as a separate .op point.

    Every terminal has its own voltage source (spec netlist requirement).
    Per row: process params via altermod, width/geometry via alter, temperature
    via `option temp`, terminal voltages via alter on the sources.
    """
    dev = CASES[case]["dev"]
    first = samples.iloc[0]
    lines = [header(f"single {dev.upper()} characterisation, case {case}", cfg),
             mos_line("M1", "d", "g", "s", "b", dev, first.w, cfg),
             "Vd d 0 0", "Vg g 0 0", "Vs s 0 0", "Vb b 0 0",
             ".control", "set noaskquit", f"set numdgt={cfg['spice']['numdgt']}"]
    for _, r in samples.iterrows():
        g = geometry(r.w, cfg)
        lines += altermod_lines(dev, {p: r[p] for p in cfg["sampling"]["process_params"]})
        lines += [
            f"alter @M1[w]={r.w:.9e}",
            f"alter @M1[ad]={g['ad']:.9e}", f"alter @M1[as]={g['as']:.9e}",
            f"alter @M1[pd]={g['pd']:.9e}", f"alter @M1[ps]={g['ps']:.9e}",
            f"alter Vd dc={r.vd:.9e}", f"alter Vg dc={r.vg:.9e}",
            f"alter Vs dc={r.vs:.9e}", f"alter Vb dc={r.vb:.9e}",
            f"option temp={r.temp:.9e}",
            "op",
            "print i(Vd) i(Vg) i(Vs) i(Vb)",
        ]
    lines += ["quit", ".endc", ".end", ""]
    return "\n".join(lines)


def parse_terminal_currents(output: str, n_expected: int) -> np.ndarray:
    """Parse the `print i(Vd) i(Vg) i(Vs) i(Vb)` blocks -> (n, 4) array of
    currents INTO the device (sign flipped from ngspice convention)."""
    vals = [float(m.group(2)) for line in output.splitlines() if (m := _PRINT_RE.match(line))]
    if len(vals) != 4 * n_expected:
        raise RuntimeError(f"expected {4 * n_expected} current values, parsed {len(vals)}")
    return -np.asarray(vals).reshape(n_expected, 4)


def _simulate_chunk(args) -> np.ndarray:
    case, chunk, cfg = args
    out = run_ngspice(single_fet_netlist(case, chunk, cfg))
    return parse_terminal_currents(out, len(chunk))


def simulate_single_fet(case: str, samples: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Simulate all samples (in parallel chunks); returns samples + id, ig, is, ib."""
    td = cfg["transistor_data"]
    size = td["chunk_size"]
    chunks = [samples.iloc[i:i + size] for i in range(0, len(samples), size)]
    # Threads suffice: each worker just waits on its own ngspice subprocess.
    with ThreadPoolExecutor(max_workers=td["n_jobs"]) as pool:
        results = list(pool.map(_simulate_chunk, [(case, c, cfg) for c in chunks]))
    out = samples.copy()
    out[["id", "ig", "is", "ib"]] = np.vstack(results)
    return out
