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
| 3 | ML models: 4 model types × 4 cases, select, test | A.2 / deliverables 4–5 | `03_make_split.py`, `03_train_mlp.py`, `03_train_xgboost.py`, `03_train_poly_ridge.py`, `03_train_rf.py`, `03_select_and_test.py` | ✅ done (MLP for all four cases) |
| 4 | 200 shared corners + gate SPICE reference (18 cases) | B.3 | `04_gen_corners.py`, `05_gate_spice.py` | ✅ done (3,600 simulations) |
| 5 | Gate-level ML prediction and comparison | B.4–5 / deliverables 6–7 | `06_gate_preds.py` | ✅ done |
| 6 | Test circuit: SPICE + hierarchical ML | C.6–8 / deliverable 8 | `07_circuit_spice.py`, `08_circuit_ml.py` | 🟨 SPICE reference done (51,200 simulations) |
| 7 | Figures, tables, report | deliverables 9–10 | `09_figures_tables.py`, `report/` | ⬜ |

Scripts run in numeric order. All randomness comes from seeds in `config.yaml`.

Optional, not a pipeline step: `python experiments/target_transforms.py` (~2.5 h, all cores; 94 min
on a 10-core laptop) then `python experiments/worst_rows.py` regenerate the report appendix
"Target-transform study" (`results/experiments/`), which chose the earlier MLP target recipes.
`python experiments/channel_target.py` (~12 min) regenerates the study behind the present MLP target
(`results/experiments/channel_target/`, report appendix "Channel target").

## Setup

```bash
brew install ngspice libomp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python 01_gen_transistor_data.py   # ~2 min
python 02_data_checks.py           # results/kcl_check.csv, figures/
```

## Layout

```
config.yaml            assignment, sampling ranges, seeds (single source of truth)
src/config.py          config loader + model-card parser (nominal vth0, toxe, ndep, u0, xj)
src/sampling.py        random draws per spec section 4
src/spice.py           applies each sample to the hand-written netlists, runs ngspice, reads currents
src/mlp_model.py       MLP transistor model (MLPFET class, saved by 03_train_mlp.py)
src/target_transform.py  MLP target recipes: drain/source swap, log10|I| targets, sign outputs, channel target (chan)
src/xgb_model.py       XGBoost transistor model (features, XGBFET class, saved by 03_train_xgboost.py)
src/poly_ridge.py      polynomial ridge transistor model (features, PolyRidgeFET class)
src/circuit.py         test-circuit gate list (read from the netlist) and logic; writes data/circuit/gates.csv
models/                PTM BSIM4 model cards
reference/             spec, slides, TA example netlist
netlists/transistor/   hand-written single-transistor netlists, one per case (deliverable 2)
netlists/gates/        hand-written gate netlists: inv, nand2, nor2, and2, or2 (deliverable 2)
netlists/circuit/      hand-written test-circuit netlist: parity8.cir, 28 NAND2, out = 1 for an odd number of 1s (deliverable 2)
experiments/           optional studies (target_transforms.py, worst_rows.py, channel_target.py) -> results/experiments/
data/ results/ figures/ saved_models/ report/   generated outputs
```

## How the transistor data is simulated

Each case has a hand-written netlist in `netlists/transistor/<case>.cir`. Its `.param` lines
hold one sample (`vdd, temp, w, vd, vs, vth0, toxe, ndep, u0, xj`). It runs on its own with
`ngspice -b <case>.cir`. `01_gen_transistor_data.py` replaces those `.param` values with each
of the 10,000 sampled rows in turn and runs ngspice once per row.

## Conventions and verified simulator behaviour

- **Current sign:** every stored current is the current flowing *into* the device terminal
  (`I = -i(Vsource)` in ngspice).
- **Junction geometry:** `AD = AS = W·Ldiff`, `PD = PS = 2(W + Ldiff)`, `Ldiff = 2·Lmin`
  (`Ldiff` is defined in the TA example netlist), written as expressions in each netlist. Only Ib depends on it
  (measured on NMOS OFF, W = 64 nm: Ib = 3.3e-17 A with geometry vs 1.0e-14 A without; Id, Ig, Is unchanged).
- **ngspice `altermod` pitfall:** a multi-parameter line such as `altermod nmos vth0=.. toxe=..`
  applies **only the first** parameter and silently ignores the rest. The netlists use one
  `altermod` line per parameter. A sensitivity test confirmed that each of the five process parameters
  changes the simulated currents.
- **Kirchhoff floor:** the PTM cards enable the gate and substrate resistance networks
  (`rgatemod=1`, `rbodymod=1`). With them on, ngspice's BSIM4 shows a residual `|ΣI|` of about 1e-16 A,
  whatever the bias. With both off, the residual is 1e-31 A. The card is used unmodified, and the
  Kirchhoff check reports the median and max `|Id+Ig+Is+Ib|` per case.
  Currents below about 1e-15 A sit at this ~1e-16 A noise floor and are not reproducible between
  equivalent simulations (most affected: PMOS OFF Ig, PMOS ON/OFF Ib).
- **Netlist options** (spec): `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`.
  The TA example netlist (`reference/22n_NMOS_ON.net`) uses older options and is kept only for reference.

## Gate SPICE reference (Phase 4)

- **Corners:** `04_gen_corners.py` draws 200 corners (VDD, T, one NMOS and one PMOS process draw;
  seed `gates.seed`) into `data/gates/corners.csv`. Phase 6 reuses the same file.
- **Widths** (not set by the spec, `gates.widths_lmin`, Wmin = Lmin = 32 nm): INV n = 1, p = 2;
  NAND2 n = 2, p = 2; NOR2 n = 1, p = 4 (× Lmin). AND2 / OR2 use NAND2 / NOR2 + INV sizes.
  NAND2: B drives the NMOS at the output, A the NMOS at ground. NOR2: A drives the PMOS at VDD.
- **Junction geometry** as in the transistor netlists, so the gates match the training data.
- **Leakage** = static power / VDD = current delivered by the VDD supply + currents delivered by the
  inputs held at VDD (`leak` in `data/gates/gate_spice.csv`); `i_vdd`, `i_a`, `i_b` are stored too.
- **Simulator:** Phase 4 ran with ngspice-42 (Linux). Re-simulating 1,200 dataset rows with it gave
  currents identical to the ngspice-47 dataset except at the ~1e-16 A noise floor.

## Polynomial ridge model (Phase 3)

`03_train_poly_ridge.py` (~12 min on 22 cores) saves `saved_models/poly_ridge/<case>.joblib`.
Load with `joblib.load(...)` from the repo root (the class lives in `src/poly_ridge.py`);
`model.predict(df)` takes the dataset input columns and returns Id, Ig, Is, Ib [A] (n × 4).

## Chosen transistor models (Phase 3)

`03_select_and_test.py` (run after the four model scripts) builds the validation comparison
(`results/model_comparison.csv`), chooses per case the model type with the lowest validation median
error averaged over Id, Ig, Is, Ib (`results/chosen_models.csv`), and only then scores the chosen
models on the test set (`results/test_results.csv`, `figures/test_pred_vs_spice_<case>.png`).

| Case | Chosen | Saved model |
|---|---|---|
| NMOS ON | MLP | `saved_models/mlp/nmos_on.joblib` |
| NMOS OFF | MLP | `saved_models/mlp/nmos_off.joblib` |
| PMOS ON | MLP | `saved_models/mlp/pmos_on.joblib` |
| PMOS OFF | MLP | `saved_models/mlp/pmos_off.joblib` |

All four MLPs use one target recipe, `chan` (`RECIPES` in `03_train_mlp.py`, flags in
`src/target_transform.py`, study in `experiments/channel_target.py`):
- the network predicts the channel current Ich = (Id − Is)/2 as log10(|Ich| / (W·(1 − e^(−|Vds|/Vt)))),
  plus log10(|Ig|/W) and log10(|Ib|/W), in the canonical drain/source frame;
- Id = Ich + Icm and Is = −Ich + Icm with Icm = (Id + Is)/2 = −(Ig + Ib)/2 from Kirchhoff, so every
  prediction satisfies KCL exactly;
- Ich = 0 at Vds = 0 by construction, and swapping drain and source swaps Id and Is exactly.
  This is the standard NN compact-model construction (Ids = factor(Vds) × exp(NN): Guo et al.,
  *Micromachines* 14:1150, 2023, Eq. 3; Novkin et al., arXiv:2611.06127, Eq. 1).

It replaces the per-case recipes of the target-transform study (T3b / T4 on T3a / T5 on T3a). Their
OFF-case input log10(1 − e^(−|Vds|/Vt)) lies ~30 training standard deviations out at the
|Vds| < 0.1 mV that transistors see inside gates, where those networks returned up to 1e265 A.
Validation mean median error, old → new: NMOS ON 0.155 → 0.183 %, NMOS OFF 0.133 → 0.116 %,
PMOS ON 0.152 → 0.164 %, PMOS OFF 8.13 → 8.03 %; gate leakage over all 3,600 rows (Σ V·I / VDD, no
|Vds| floor): median 1.88 → 1.32 %, p95 27 → 8.4 %, max 2.7e72 → 26.5 %.

Every saved model has `model.predict(df)`: `df` holds the dataset input columns
(`vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj`), the result is an (n, 4) array of
Id, Ig, Is, Ib [A] (current into each terminal). Load with `joblib.load(...)` from the repo root.
A script that uses MLP and XGBoost models together (e.g. `03_select_and_test.py`) must `import xgboost` first and call `torch.set_num_threads(1)`; otherwise their two OpenMP libraries
crash the process on macOS.

## Gate ML prediction (Phase 5)

`06_gate_preds.py` predicts the leakage of every row of `data/gates/gate_spice.csv` with the four
MLPs and compares it with SPICE.
- Terminal voltages of every transistor from the SPICE node voltages (`v_out`, `v_x`, `v_y`, `v_nd`,
  `v_nr`), the inputs (exactly 0 or VDD), VDD and ground; each transistor goes through the model of its
  case.
- Leakage = Σ V·I over all terminals of all transistors / VDD (total static power / VDD).
- Outputs: `results/gate_results_table.csv` (deliverable 6) and `results/gate_parity_plot.png`
  (deliverable 7).
- Reads `gate_spice.csv` with `dtype={"inputs": str}` (otherwise "01" becomes 1).
