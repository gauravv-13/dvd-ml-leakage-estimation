# ML-Driven PVT-Variation-Aware Leakage Estimation

Digital VLSI Design course project (Semester 5). Leakage current of CMOS circuits under
process, voltage and temperature (PVT) variation is estimated with machine learning instead of
full SPICE simulation, using a three-level model:

```
Transistor ML models (NMOS/PMOS x ON/OFF)  ──>  Gate leakage (INV, NAND2, NOR2, AND2, OR2)  ──>  Test circuit leakage
      trained on ngspice data                    sum of per-transistor predictions             sum of gate predictions
```

Spec: [`reference/DVD_ML_PROJECT.pdf`](reference/DVD_ML_PROJECT.pdf) · Lecture slides: [`reference/Project-1_leakage_estimation_annotated.pdf`](reference/Project-1_leakage_estimation_annotated.pdf)

## Assignment

| Item | Value |
|---|---|
| Model file | `32nm_HP.pm` (Lmin = 32 nm) |
| Nominal VDD | 0.9 V (sampled 0.81–0.99 V) |
| Test circuit | 8-bit parity checker |
| Simulator | ngspice-47 |

Configured in [`config.yaml`](config.yaml).

## Phases

| # | Phase | Spec part | Scripts | Status |
|---|---|---|---|---|
| 0 | Setup and toolchain validation | n/a | (local only) | ✅ done |
| 1 | Transistor dataset generation (4 × 10k samples) | A.1 | `01_gen_transistor_data.py` | ✅ done (4 × 10,000 samples) |
| 2 | Data checks: Kirchhoff + 3 trend plots | A.1 / deliverable 3 | `02_data_checks.py` | ✅ done |
| 3 | ML models: 4 model types × 4 cases, select, test | A.2 / deliverables 4–5 | `03_train_models.py` | ⬜ |
| 4 | 200 shared corners + gate SPICE reference (18 cases) | B.3 | `04_gen_corners.py`, `05_gate_spice.py` | ⬜ |
| 5 | Gate-level ML prediction and comparison | B.4–5 / deliverables 6–7 | `06_gate_ml.py` | ⬜ |
| 6 | Test circuit: SPICE + hierarchical ML | C.6–8 / deliverable 8 | `07_circuit_spice.py`, `08_circuit_ml.py` | ⬜ |
| 7 | Figures, tables, report | deliverables 9–10 | `09_figures_tables.py`, `report/` | ⬜ |

Scripts run in numeric order. All randomness comes from seeds in `config.yaml`.

## Setup

```bash
brew install ngspice libomp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 01_gen_transistor_data.py   # ~6 s
python 02_data_checks.py           # results/kcl_check.csv, figures/
```

## Layout

```
config.yaml            assignment, sampling ranges, seeds (single source of truth)
src/config.py          config loader + model-card parser (nominal vth0, toxe, ndep, u0, xj)
src/sampling.py        random draws per spec section 4
src/spice.py           netlist generation, parallel ngspice runs, output parsing
models/                PTM BSIM4 model cards
reference/             spec, slides, TA example netlist
data/ netlists/ results/ figures/ saved_models/ report/   generated outputs
```

## Conventions and verified simulator behaviour

- **Current sign:** every stored current is the current flowing *into* the device terminal
  (`I = -i(Vsource)` in ngspice).
- **Junction geometry:** `AD = AS = W·Ldiff`, `PD = PS = 2(W + Ldiff)`, `Ldiff = 2·Lmin`
  (`Ldiff` is defined in the TA example netlist), and the same rule in every netlist. Only Ib depends on it
  (measured on NMOS OFF, W = 64 nm: Ib = 3.3e-17 A with geometry vs 1.0e-14 A without; Id, Ig, Is unchanged).
- **ngspice `altermod` pitfall:** a multi-parameter line such as `altermod nmos vth0=.. toxe=..`
  applies **only the first** parameter and silently ignores the rest. The code emits one
  `altermod` per parameter. A sensitivity test confirmed that each of the five process parameters
  changes the simulated currents.
- **Kirchhoff floor:** the PTM cards enable the gate and substrate resistance networks
  (`rgatemod=1`, `rbodymod=1`). With them on, ngspice's BSIM4 shows a residual `|ΣI|` of about 1e-16 A,
  whatever the bias. With both off, the residual is 1e-31 A. The card is used unmodified, and the
  Kirchhoff check reports the median and max `|Id+Ig+Is+Ib|` per case.
  Currents below about 1e-15 A carry the same ~1e-16 A numerical noise, so they differ slightly
  between equivalent netlists.
- **Netlist options** (spec): `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`.
  The TA example netlist (`reference/22n_NMOS_ON.net`) uses older options and is kept only for reference.
