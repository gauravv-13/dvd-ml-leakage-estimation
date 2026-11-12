"""Gate leakage from the four transistor models (spec Part B step 4; reused by Phase 6).

Topology: the MOSFET lines of netlists/gates/<gate>.cir (`M<name> d g s b nmos|pmos W={param}`), so the
ML side always uses the simulated circuit. Widths: config gates.widths_lmin, mapped from the netlist
width parameter as 05_gate_spice.py does (wn/wp -> the gate itself, wn_nand/wp_inv/... -> that cell).
Input naming follows the course slides: A drives the transistor next to the rail (ground for the
NAND2 NMOS stack, VDD for the NOR2 PMOS stack), B the transistor at the output.

Terminal voltages (spec step 4): the SPICE node voltages of the same gate, inputs and corner
(output, x, y, nd, nr), the inputs at exactly 0 or VDD, VDD and ground. A transistor is ON when its
gate voltage turns it on (NMOS gate above VDD/2, PMOS gate below VDD/2) and uses that case's model.

OFF transistors with |Vds| below `gates.vds_floor_off` (1 mV) are evaluated at |Vds| = 1 mV, same
direction: in a gate an OFF device with both ends on one rail sits at 0.1-600 uV, where the OFF
training data is almost empty and the model extrapolates to absurd currents. Its channel current
there is a small fraction of the gate leakage, so the floor changes the result very little.

Leakage = static power / VDD = current delivered by the VDD supply + currents delivered by the inputs
held at VDD, the SPICE definition of 05_gate_spice.py. With the current into every terminal, this is
the sum of the predicted currents into the terminals on `vdd` and on inputs at 1 (KCL at the source
nodes). `leak_kcl` additionally replaces the Id/Is of ON transistors by Kirchhoff's current law at
the internal and output nodes (fewest unknowns first; parallel ON devices share by width); it is
reported for comparison only.
"""
from __future__ import annotations

import re
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import NETLIST_DIR, SAVED_MODELS_DIR, load_config

TERMINALS = ["d", "g", "s", "b"]          # predict() columns: Id, Ig, Is, Ib
PROCESS = ["vth0", "toxe", "ndep", "u0", "xj"]
INPUTS = ["a", "b"]
FREE_NODES = {"out", "x", "y", "nd", "nr"}  # nodes not tied to a source
CELL = {"nand": "nand2", "nor": "nor2", "inv": "inv"}
CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]


@lru_cache(maxsize=None)
def models() -> dict:
    """The four chosen (Phase 3) transistor models."""
    return {c: joblib.load(SAVED_MODELS_DIR / "mlp" / f"{c}.joblib") for c in CASES}


@lru_cache(maxsize=None)
def transistors(gate: str) -> tuple[dict, ...]:
    """Transistors of a gate netlist: name, d, g, s, b, type, w [m]."""
    cfg = load_config()
    wl = cfg["gates"]["widths_lmin"]
    out = []
    for line in (NETLIST_DIR / "gates" / f"{gate}.cir").read_text().splitlines():
        if not line.startswith("M"):
            continue
        name, d, g, s, b, kind = line.split()[:6]
        param = re.search(r"W=\{(\w+)\}", line).group(1)          # wn, wp, wn_nand, wp_inv, ...
        t, _, cell = param[1:].partition("_")
        w = wl[CELL[cell] if cell else gate][t] * cfg.lmin
        out.append({"name": name, "d": d, "g": g, "s": s, "b": b, "type": kind, "w": w})
    return tuple(out)


def node_voltages(rows: pd.DataFrame, inputs: str) -> dict[str, np.ndarray]:
    """Voltage of every gate node for rows of one (gate, inputs) case, from the SPICE node voltages."""
    vdd = rows.vdd.to_numpy()
    v = {"vdd": vdd, "0": np.zeros(len(rows))}
    v.update({n: int(bit) * vdd for n, bit in zip(INPUTS, inputs)})
    v.update({n: rows[f"v_{n}"].to_numpy() for n in FREE_NODES if f"v_{n}" in rows})
    return v


def predict_transistor(case: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n, 4) currents into d, g, s, b of one transistor case, with the OFF |Vds| floor; and the
    mask of rows where the floor was applied."""
    floored = np.zeros(len(df), bool)
    if case.endswith("off"):
        floor = load_config()["gates"]["vds_floor_off"]
        vds = (df.vd - df.vs).to_numpy()
        floored = np.abs(vds) < floor
        if floored.any():
            sign = np.where(vds != 0, np.sign(vds), 1.0 if case.startswith("nmos") else -1.0)
            df = df.copy()
            df.loc[floored, "vd"] = df.vs[floored] + sign[floored] * floor
    return models()[case].predict(df), floored


def gate_currents(gate: str, inputs: str, rows: pd.DataFrame) -> list[dict]:
    """Every transistor of the gate with its case and predicted terminal currents, for rows of one
    (gate, inputs) case. rows: vdd, temp, v_<node>, and the corner's n_/p_ process parameters."""
    v = node_voltages(rows, inputs)
    out = []
    for t in transistors(gate):
        V = {k: v[t[k]] for k in TERMINALS}
        on = (V["g"] > rows.vdd.to_numpy() / 2) if t["type"] == "nmos" else (V["g"] < rows.vdd.to_numpy() / 2)
        assert on.all() or not on.any(), f"{gate} {inputs} {t['name']}: ON/OFF differs between corners"
        case = f"{t['type']}_{'on' if on[0] else 'off'}"
        p = "n_" if t["type"] == "nmos" else "p_"
        df = pd.DataFrame({"vdd": rows.vdd.to_numpy(), "temp": rows.temp.to_numpy(), "w": t["w"],
                           "vd": V["d"], "vs": V["s"], "vg": V["g"], "vb": V["b"],
                           **{k: rows[p + k].to_numpy() for k in PROCESS}})
        I, floored = predict_transistor(case, df)
        out.append({**t, "case": case, "I": {k: I[:, j] for j, k in enumerate(TERMINALS)}, "floored": floored})
    return out


def apply_kcl(devs: list[dict]) -> list[dict]:
    """Copy of devs with the Id/Is of ON transistors from KCL at the free nodes."""
    devs = [{**d, "I": dict(d["I"])} for d in devs]
    unknown = {(i, k) for i, d in enumerate(devs) if d["case"].endswith("on") for k in ("d", "s")}
    while unknown:
        options = [(n, [(i, k) for i, k in unknown if devs[i][k] == n]) for n in FREE_NODES]
        options = [o for o in options if o[1]]
        if not options:
            raise ValueError("ON transistor with no channel terminal on a free node")
        node, u = min(options, key=lambda o: len(o[1]))
        known = sum(d["I"][k] for i, d in enumerate(devs) for k in TERMINALS
                    if d[k] == node and (i, k) not in unknown)
        wsum = sum(devs[i]["w"] for i, _ in u)
        for i, k in u:
            d, other = devs[i], ("s" if k == "d" else "d")
            d["I"][k] = -known * d["w"] / wsum
            d["I"][other] = -(d["I"][k] + d["I"]["g"] + d["I"]["b"])
            unknown -= {(i, k), (i, other)}
    return devs


def leakage(devs: list[dict], inputs: str) -> np.ndarray:
    """Sum of the currents into the terminals on vdd and on inputs held at VDD."""
    at_vdd = {"vdd"} | {n for n, bit in zip(INPUTS, inputs) if bit == "1"}
    return sum(d["I"][k] for d in devs for k in TERMINALS if d[k] in at_vdd)
