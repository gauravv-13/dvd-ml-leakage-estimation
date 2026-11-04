"""Phase 0: verify the toolchain and the simulation set-up before generating data.

Checks
  1. ngspice and the Python packages are available.
  2. The model card parses and nominal process parameters are found.
  3. A short run of every single-FET case satisfies Kirchhoff (sum of currents ~ 0).
  4. Physics sanity: OFF current rises with T, falls with vth0; gate current falls with toxe;
     ON current scales with W. If any of these fail the netlist set-up is wrong.
Run:  python 00_check_env.py
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd

from src.config import CASES, load_config
from src.sampling import sample_transistor_inputs
from src.spice import ngspice_version, simulate_single_fet

ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")


print("1. Toolchain")
print(f"  ngspice: {ngspice_version()}   python: {sys.version.split()[0]}")
for m in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "xgboost", "lightgbm", "torch", "yaml"]:
    try:
        importlib.import_module(m)
        check(True, f"import {m}")
    except ImportError as e:
        check(False, f"import {m}: {e}")

cfg = load_config()
print(f"\n2. Model card {cfg.model_path.name}: Lmin={cfg.lmin:.0e} m, Ldiff={cfg.ldiff:.0e} m, "
      f"VDD_nom={cfg.vdd_nominal} V")
for dev, p in cfg.nominal.items():
    print(f"  {dev}: " + ", ".join(f"{k}={v:.4g}" for k, v in p.items()))
check(cfg.nominal["nmos"]["vth0"] > 0 > cfg.nominal["pmos"]["vth0"], "vth0 signs (NMOS > 0 > PMOS)")

print("\n3. Kirchhoff check, 200 random samples per case")
# Pass rule: |sum I| <= max(1e-6 * max|I|, 1e-15 A). The absolute floor exists because the
# PTM cards enable the gate- and substrate-resistance networks (rgatemod=1, rbodymod=1);
# ngspice's BSIM4 loses ~1e-16 A through them (verified: residual -> 1e-31 A with both off).
rng = np.random.default_rng(0)
for case in CASES:
    df = simulate_single_fet(case, sample_transistor_inputs(case, 200, cfg, rng), cfg)
    I = df[["id", "ig", "is", "ib"]].to_numpy()
    resid = np.abs(I.sum(1))
    ok_rows = resid <= np.maximum(1e-6 * np.abs(I).max(1), 1e-15)
    check(ok_rows.all(), f"{case:9s} max |sum I| = {resid.max():.1e} A, rows passing {ok_rows.sum()}/200, "
                         f"|Id| {np.abs(df.id).min():.1e} .. {np.abs(df.id).max():.1e} A")


def one_at_a_time(case: str, col: str, values) -> pd.DataFrame:
    """Hold everything at nominal (Vd=VDD, Vs=0, W=2Lmin, T=27C) and sweep one column."""
    base = sample_transistor_inputs(case, 1, cfg, rng).iloc[[0] * len(values)].reset_index(drop=True)
    dev = CASES[case]["dev"]
    vdd = cfg.vdd_nominal
    base[["vdd", "temp", "w"]] = [vdd, 27.0, 2 * cfg.lmin]
    base["vg"] = vdd if CASES[case]["vg"] == "vdd" else 0.0
    base["vb"] = vdd if CASES[case]["vb"] == "vdd" else 0.0
    # Put the full VDD across the channel, oriented so the device conducts "forward".
    base["vd"], base["vs"] = (vdd, 0.0) if dev == "nmos" else (0.0, vdd)
    for p in cfg["sampling"]["process_params"]:
        base[p] = cfg.nominal[dev][p]
    base[col] = values
    return simulate_single_fet(case, base, cfg)


print("\n4. Physics sanity (one parameter swept, rest nominal)")
t = one_at_a_time("nmos_off", "temp", [-40.0, 27.0, 125.0])
check(np.all(np.diff(np.abs(t.id)) > 0), f"NMOS OFF |Id| rises with T: {np.abs(t.id).round(15).tolist()}")
v = one_at_a_time("nmos_off", "vth0", cfg.nominal["nmos"]["vth0"] * np.array([0.9, 1.0, 1.1]))
check(np.all(np.diff(np.abs(v.id)) < 0), f"NMOS OFF |Id| falls with vth0: {np.abs(v.id).round(15).tolist()}")
x = one_at_a_time("nmos_on", "toxe", cfg.nominal["nmos"]["toxe"] * np.array([0.9, 1.0, 1.1]))
check(np.all(np.diff(np.abs(x.ig)) < 0), f"NMOS ON |Ig| falls with toxe: {np.abs(x.ig).round(15).tolist()}")
w = one_at_a_time("pmos_on", "w", cfg.lmin * np.array([1.0, 4.0, 8.0]))
check(np.all(np.diff(np.abs(w.id)) > 0), f"PMOS ON |Id| rises with W: {np.abs(w.id).round(12).tolist()}")
p = one_at_a_time("pmos_off", "temp", [-40.0, 27.0, 125.0])
check(np.all(np.diff(np.abs(p.id)) > 0), f"PMOS OFF |Id| rises with T: {np.abs(p.id).round(15).tolist()}")

print("\n5. Every process parameter reaches the simulator (ngspice altermod pitfall)")
for case in ["nmos_off", "pmos_on"]:
    dev = CASES[case]["dev"]
    for prm in cfg["sampling"]["process_params"]:
        r = one_at_a_time(case, prm, cfg.nominal[dev][prm] * np.array([0.9, 1.1]))
        change = np.abs(r[["id", "ig", "ib"]].iloc[1].to_numpy() / r[["id", "ig", "ib"]].iloc[0].to_numpy() - 1).max()
        check(change > 1e-3, f"{case:9s} {prm:5s} +/-10% changes currents by up to {100 * change:.1f}%")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
