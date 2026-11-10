"""Optional experiment: target-transform study for the transistor models (validation set only).

Regenerates the report appendix "Target-transform study" (results/experiments/appendix_table.tex)
and the choice of RECIPES in 03_train_mlp.py. Not a pipeline step; takes about 1.5-2.5 h
(94 min on a 10-core laptop, all cores busy). Resumable: finished jobs are skipped.

Transforms (flags in src/target_transform.py; each builds on the previous unless noted):
  asinh  the earlier target, asinh(I / 1e-18) (MLP, seed 42 only, the baseline)
  T0     log10|I| + sign classifier
  T1     T0 + canonical drain/source swap (Id sign from physics; Is, Ig, Ib keep the classifier)
  T2     T1 + per-width target log10(|I| / W)
  T3a    T2 + extra inputs Vds/Vt, Vgs/Vt, Vbs/Vt, log10(1 - exp(-|Vds|/Vt) + 1e-12)
  T3b    T2 + Id/Is target log10(|I| / (W (1 - exp(-|Vds|/Vt)) + 1e-30))
  T4     best of T0-T3b + Is = -(Id + Ig + Ib)
  T5     best of T0-T4 + training labels clipped to log10(max(|I|, 1e-17 A)), scored on raw SPICE
"Best" = lowest mean over the seeds of the mean of the four validation medians, per model and case.
Models: MLP (03_train_mlp.py settings, src/mlp_model.py) and XGBoost (03_train_xgboost.py settings,
features and linear base, src/xgb_model.py) applied to Id, Ig, Is, Ib directly, one booster per
magnitude and one binary:logistic booster per sign. Seeds 42, 43, 44. Split and metrics: load() and
metrics() of the training scripts. The test rows are never loaded.

Each job runs in its own process (torch and xgboost must not share one on macOS).
Outputs (results/experiments/):
  runs/<model>_<transform>_<seed>_<case>.npy   validation predictions (n, 4) = Id, Ig, Is, Ib [A]
  stage2_choices.json                          base transform of T4 and T5 per model and case
  transforms_per_seed.csv, transforms_summary.csv (mean over seeds, sd, min, max)
  transforms_tables.md                         one table per case, winner marked
  appendix_table.tex                           MLP, seed 42: the rows shown in the report
  evidence.txt                                 Is sign after the swap; T3b rows with |Vds| < 1e-4 V
Run:  python experiments/target_transforms.py [--jobs 5]
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import CASES, RESULTS_DIR  # noqa: E402
from src.target_transform import CURRENTS, Plan, canonical, extra_inputs, saturation  # noqa: E402

OUT = RESULTS_DIR / "experiments"
RUNS = OUT / "runs"
SEEDS = [42, 43, 44]
MODELS = ["mlp", "xgb"]
STAGE1 = {"T0": {}, "T1": {"swap": 1}, "T2": {"swap": 1, "perw": 1},
          "T3a": {"swap": 1, "perw": 1, "extra": 1}, "T3b": {"swap": 1, "perw": 1, "sat": 1}}
ORDER = ["asinh", *STAGE1, "T4", "T5"]
APPENDIX_ROWS = ["asinh", "T0", "T1", "T3a", "T3b", "T4", "T5"]


# ---------------------------------------------------------------- one job (own process)
def run_mlp(case: str, seed: int, recipe: dict) -> np.ndarray:
    import torch
    torch.set_num_threads(2)
    from src.mlp_model import MLPFET
    tr = importlib.import_module("03_train_mlp")
    d = tr.load(case)
    return MLPFET.fit(case, tr.SETTINGS[case], recipe, d["train"], seed).predict(d["val"])


def run_xgb(case: str, seed: int, recipe: dict) -> np.ndarray:
    import xgboost as xgb
    from src.xgb_model import NTHREAD, LinearBase, features
    tr = importlib.import_module("03_train_xgboost")
    d = tr.load(case)
    s = dict(tr.SETTINGS[case])
    rounds = s.pop("rounds")
    params = {**s, "seed": seed, "nthread": NTHREAD}

    def inputs(df):
        X = features(df)
        if recipe.get("extra"):
            X = X.assign(**{f"extra{i}": col for i, col in enumerate(extra_inputs(df).T)})
        return X.to_numpy().astype(float)

    dtr, _ = canonical(d["train"], case, recipe)
    dva, swapped = canonical(d["val"], case, recipe)
    plan = Plan(case, recipe, dtr)
    Y, S = plan.targets(dtr, recipe)
    Xtr, Xva = inputs(dtr), inputs(dva)
    y, p = [], []
    for j in range(Y.shape[1]):
        lin = LinearBase().fit(Xtr, Y[:, j])
        b = xgb.train(params, xgb.DMatrix(Xtr, label=Y[:, j], base_margin=lin.predict(Xtr)), num_boost_round=rounds)
        y.append(b.predict(xgb.DMatrix(Xva, base_margin=lin.predict(Xva))).astype(float))
    for j in range(S.shape[1]):
        b = xgb.train({**params, "objective": "binary:logistic"}, xgb.DMatrix(Xtr, label=S[:, j]),
                      num_boost_round=rounds)
        p.append(b.predict(xgb.DMatrix(Xva)))
    p_pos = np.column_stack(p) if p else np.zeros((len(Xva), 0))
    return plan.currents(dva, swapped, recipe, np.column_stack(y), p_pos)


def job_main(model: str, transform: str, seed: str, case: str, recipe: str) -> None:
    t0 = time.time()
    I = (run_mlp if model == "mlp" else run_xgb)(case, int(seed), json.loads(recipe))
    path = RUNS / f"{model}_{transform}_{seed}_{case}"
    np.save(path.with_suffix(".npy"), I)
    path.with_suffix(".json").write_text(json.dumps({"recipe": json.loads(recipe), "seconds": round(time.time() - t0)}))


# ---------------------------------------------------------------- driver
def npy(model, transform, seed, case):
    return RUNS / f"{model}_{transform}_{seed}_{case}.npy"


def run_jobs(jobs: list[tuple], n: int) -> None:
    def one(j):
        model, transform, seed, case, recipe = j
        if npy(model, transform, seed, case).exists():
            return
        r = subprocess.run([sys.executable, __file__, "--job", model, transform, str(seed), case, json.dumps(recipe)],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{model} {transform} {seed} {case} failed:\n{r.stderr[-2000:]}")
        print(time.strftime("%H:%M:%S"), "done", model, transform, seed, case, flush=True)
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(one, jobs))


def val_rows() -> dict[str, pd.DataFrame]:
    tr = importlib.import_module("03_train_xgboost")             # same load() as 03_train_mlp.py (tested)
    return {c: tr.load(c)["val"].reset_index(drop=True) for c in CASES}


def score(case: str, va: pd.DataFrame, I_pred: np.ndarray) -> dict:
    """metrics() of the training scripts plus the study's subsets; relative error as in metrics()."""
    tr = importlib.import_module("03_train_xgboost")
    T = va[CURRENTS].to_numpy()
    out = tr.metrics(T, np.arcsinh(I_pred / tr.I_SCALE))
    E = np.abs(I_pred - T) / np.abs(T) * 100
    vds, vdd = (va.vd - va.vs).abs().to_numpy(), va.vdd.to_numpy()
    subsets = {"gatebias": vds >= 0.5 * vdd if case.endswith("off") else vds <= 0.1 * vdd,
               "smallvds": vds < 0.02}
    cancel = np.abs(T[:, 2]) < 0.01 * np.abs(T[:, 0])            # lower-terminal near-cancellation
    med = lambda e, m: float(np.median(e[m])) if m.any() else np.nan
    for j, c in enumerate(CURRENTS):
        out[f"{c}_median_ge1e-15_pct"] = med(E[:, j], np.abs(T[:, j]) >= 1e-15)
        out[f"{c}_wrong_sign"] = int((np.sign(I_pred[:, j]) != np.sign(T[:, j])).sum())
        for name, m in subsets.items():
            out[f"{c}_{name}_median_pct"] = med(E[:, j], m)
    out |= {f"n_{k}": int(m.sum()) for k, m in subsets.items()}
    out |= {"is_cancel_median_pct": med(E[:, 2], cancel), "is_other_median_pct": med(E[:, 2], ~cancel),
            "n_is_cancel": int(cancel.sum())}
    out["mean_median_pct"] = np.mean([out[f"{c}_median_err_pct"] for c in CURRENTS])
    out["mean_p95_pct"] = np.mean([out[f"{c}_p95_err_pct"] for c in CURRENTS])
    return out


def mean_score(per_seed: pd.DataFrame, model, transform, case) -> float:
    r = per_seed[(per_seed.model == model) & (per_seed.transform == transform) & (per_seed.case == case)]
    return r.mean_median_pct.mean()


def collect(val: dict) -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*.npy")):
        model, transform, seed, case = f.stem.split("_", 3)
        rows.append({"model": model, "transform": transform, "seed": int(seed), "case": case,
                     **score(case, val[case], np.load(f))})
    return pd.DataFrame(rows)


def evidence(val: dict) -> str:
    from src.config import DATA_DIR
    lines = ["Is opposite to the physics sign after the canonical swap (all 10,000 rows | rows with |Is| >= 1e-15 A):"]
    for c in CASES:
        d = pd.read_csv(DATA_DIR / "transistor" / f"{c}.csv")
        dc, _ = canonical(d, c, {"swap": 1})
        n = c.startswith("nmos")
        wrong = np.sign(dc["is"]) != (-1 if n else 1)
        big = dc["is"].abs() >= 1e-15
        wrong_d = np.sign(dc["id"]) != (1 if n else -1)
        lines.append(f"  {c:9s} Is {wrong.mean() * 100:6.2f} % | {wrong[big].mean() * 100:6.2f} %   (Id: {wrong_d.mean() * 100:.2f} %)")
    lines.append("\nT3b: rows with |Vds| < 1e-4 V (any split) and their Id target vs the training range:")
    split = {c: pd.read_csv(DATA_DIR / "transistor" / f"split_{c}.csv").set for c in CASES}
    for c in CASES:
        d = pd.read_csv(DATA_DIR / "transistor" / f"{c}.csv")
        dc, _ = canonical(d, c, {"swap": 1})
        f = saturation(dc)
        y = np.log10(dc["id"].abs() / (dc.w * f + 1e-30))
        trn = (split[c] == "train").to_numpy()
        for i in np.where((dc.vd - dc.vs).abs() < 1e-4)[0]:
            lines.append(f"  {c:9s} row {i} ({split[c][i]}): |Vds| {abs(dc.vd[i] - dc.vs[i]):.2e} V, factor {f[i]:.2e}, "
                         f"target {y[i]:.2f} (train range {y[trn].min():.2f} .. {y[trn].max():.2f})")
    lines.append(f"  validation rows with |Vds| < 1e-4 V: "
                 f"{sum(int(((v.vd - v.vs).abs() < 1e-4).sum()) for v in val.values())}")
    return "\n".join(lines) + "\n"


def fmt(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 1000 else f"{x:.2g}"


def tables(summary: pd.DataFrame, choices: dict) -> str:
    out = []
    for case in CASES:
        a = summary[summary.case == case].copy()
        a["o"] = a["transform"].map(ORDER.index)
        a = a.sort_values(["model", "o"])
        win = a.loc[a.mean_median_pct.idxmin()]
        on = case.endswith("on")
        out.append(f"\n### {case}  (validation; gate-bias rows {int(a.n_gatebias.iloc[0])}"
                   + (f", |Vds| < 20 mV rows {int(a.n_smallvds.iloc[0])})" if on else ")") + "\n")
        head = ["model", "transform", "seeds", "mean of 4 medians (sd; min-max)", "median % Id/Ig/Is/Ib",
                "p95 % Id/Ig/Is/Ib", "median, |I| >= 1e-15", "wrong sign", "gate-bias median"]
        head += ["|Vds| < 20 mV median"] if on else []
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for i, r in a.iterrows():
            t = r["transform"]
            if t in ("T4", "T5"):
                t += f" ({choices[f'{r.model}|{case}|{t}']['base']}+)"
            cells = [r.model, t + (" **winner**" if i == win.name else ""), str(int(r.n_seeds)),
                     f"{r.mean_median_pct:.3f} ({r.sd:.3f}; {r.min:.3f}-{r.max:.3f})",
                     " / ".join(fmt(r[f"{c}_median_err_pct"]) for c in CURRENTS),
                     " / ".join(fmt(r[f"{c}_p95_err_pct"]) for c in CURRENTS),
                     " / ".join(fmt(r[f"{c}_median_ge1e-15_pct"]) for c in CURRENTS),
                     " / ".join(f"{r[f'{c}_wrong_sign']:.0f}" for c in CURRENTS),
                     " / ".join(fmt(r[f"{c}_gatebias_median_pct"]) for c in CURRENTS)]
            if on:
                cells.append(" / ".join(fmt(r[f"{c}_smallvds_median_pct"]) for c in CURRENTS))
            out.append("| " + " | ".join(cells) + " |")
    out.append("\n### T4: Is median error on near-cancellation rows (|Is| < 0.01 |Id|) vs other rows\n")
    out += ["| case | model | rows (cancel / other) | T4 base: cancel / other | T4: cancel / other |", "|---|---|---|---|---|"]
    for case in CASES:
        for m in MODELS:
            b = choices[f"{m}|{case}|T4"]["base"]
            x = summary[(summary.case == case) & (summary.model == m)].set_index("transform")
            n = int(x.n_is_cancel.iloc[0])
            out.append(f"| {case} | {m} | {n} / {1500 - n} | {b}: {fmt(x.loc[b, 'is_cancel_median_pct'])} / "
                       f"{fmt(x.loc[b, 'is_other_median_pct'])} | {fmt(x.loc['T4', 'is_cancel_median_pct'])} / "
                       f"{fmt(x.loc['T4', 'is_other_median_pct'])} |")
    return "\n".join(out) + "\n"


def appendix(per_seed: pd.DataFrame, choices: dict, adopted: dict) -> str:
    """MLP, seed 42: mean of the four medians / mean of the four p95 per case; adopted recipe in bold."""
    s42 = per_seed[(per_seed.model == "mlp") & (per_seed.seed == 42)].set_index(["transform", "case"])
    names = {"asinh": r"asinh (before the study)", "T0": r"T0: $\log_{10}|I|$ + sign output",
             "T1": r"T1: + drain/source swap", "T3a": r"T3a: + per-width target, $V/V_t$ inputs",
             "T3b": r"T3b: + per-width target, saturation target", "T4": r"T4: + $I_s$ from Kirchhoff",
             "T5": r"T5: + labels clipped at \SI{1e-17}{A}"}
    lines = [r"\begin{tabular}{l" + "c" * len(CASES) + "}", r"\toprule",
             "Target & " + " & ".join(c.replace("_", " ").upper() for c in CASES) + r" \\", r"\midrule"]
    for t in APPENDIX_ROWS:
        cells = []
        for c in CASES:
            if (t, c) not in s42.index:
                cells.append("--")
                continue
            r = s42.loc[(t, c)]
            v = f"{r.mean_median_pct:.3f} / {r.mean_p95_pct:.1f}" if r.mean_median_pct < 1 else \
                f"{r.mean_median_pct:.2f} / {r.mean_p95_pct:.0f}"
            if t in ("T4", "T5"):
                v += f" ({choices[f'mlp|{c}|{t}']['base']})"
            cells.append(r"\textbf{" + v + "}" if adopted[c] == t else v)
        lines.append(names[t] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(n_jobs: int) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_jobs([("mlp", "asinh", 42, c, {"asinh": 1}) for c in CASES]
             + [(m, t, s, c, f) for c in CASES for m in MODELS for t, f in STAGE1.items() for s in SEEDS], n_jobs)
    val = val_rows()
    choices, flags = {}, {}
    per_seed = collect(val)
    for m in MODELS:
        for c in CASES:
            b = min(STAGE1, key=lambda t: mean_score(per_seed, m, t, c))
            flags[(m, c, "T4")] = {**STAGE1[b], "kcl": 1}
            choices[f"{m}|{c}|T4"] = {"base": b, "recipe": flags[(m, c, "T4")]}
    run_jobs([(m, "T4", s, c, flags[(m, c, "T4")]) for c in CASES for m in MODELS for s in SEEDS], n_jobs)
    per_seed = collect(val)
    for m in MODELS:
        for c in CASES:
            cand = {**STAGE1, "T4": flags[(m, c, "T4")]}
            b = min(cand, key=lambda t: mean_score(per_seed, m, t, c))
            flags[(m, c, "T5")] = {**cand[b], "clip": 1}
            choices[f"{m}|{c}|T5"] = {"base": b, "recipe": flags[(m, c, "T5")]}
    run_jobs([(m, "T5", s, c, flags[(m, c, "T5")]) for c in CASES for m in MODELS for s in SEEDS], n_jobs)
    print(f"all jobs done in {time.time() - t0:.0f} s", flush=True)

    per_seed = collect(val)
    per_seed.to_csv(OUT / "transforms_per_seed.csv", index=False)
    (OUT / "stage2_choices.json").write_text(json.dumps(choices, indent=1))
    num = [c for c in per_seed.columns if c not in ("model", "transform", "seed", "case")]
    g = per_seed.groupby(["case", "model", "transform"])
    summary = g[num].mean()
    summary["sd"], summary["min"], summary["max"] = g.mean_median_pct.std(), g.mean_median_pct.min(), g.mean_median_pct.max()
    summary["n_seeds"] = g.size()
    summary = summary.reset_index()
    summary.to_csv(OUT / "transforms_summary.csv", index=False)
    mlp = summary[summary.model == "mlp"]
    adopted = {c: mlp[mlp.case == c].set_index("transform").mean_median_pct.idxmin() for c in CASES}
    (OUT / "transforms_tables.md").write_text(tables(summary, choices))
    (OUT / "appendix_table.tex").write_text(appendix(per_seed, choices, adopted))
    (OUT / "evidence.txt").write_text(evidence(val))
    print("MLP choice by the rule (mean over seeds):", adopted)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--job":
        job_main(*sys.argv[2:7])
    else:
        ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        ap.add_argument("--jobs", type=int, default=5, help="jobs at a time (2 threads each)")
        main(ap.parse_args().jobs)
