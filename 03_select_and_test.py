"""Phase 3 (Part A step 2, deliverables 4-5): compare the four model types, choose one per case on the
validation set, and report the chosen models on the test set.

1. Comparison (deliverable 4): the validation results of the four model scripts
   (03_train_mlp.py, 03_train_xgboost.py, 03_train_poly_ridge.py, 03_train_rf.py), one row per
   model type and case: R^2, median and 95th-percentile error of Id, Ig, Is, Ib.
2. Choice: per case, the model type with the lowest validation median error, averaged over the four
   currents (Id, Ig, Is, Ib). The 95th percentile is reported next to it as a check.
   Only validation results are used for the choice.
3. Test set (deliverable 5): the chosen saved model of each case predicts its test rows
   (data/transistor/split_<case>.csv, set = test) once; median and 95th-percentile error per current,
   and a predicted-vs-SPICE plot per case.
   Before that, each saved model is checked against its validation results, so the file scored on
   the test set is the model that was compared.

Metrics: relative error |I_pred - I_SPICE| / |I_SPICE| * 100 [%] on all rows, per current;
         R^2 on asinh(I / 1e-18) (validation table only).

Outputs: results/model_comparison.csv      validation, 4 model types x 4 cases
         results/chosen_models.csv         chosen model type and saved model file per case
         results/test_results.csv          test-set median and 95th-percentile error per case
         figures/test_pred_vs_spice_<case>.png
Run:     python 03_select_and_test.py      (after the four model scripts)
"""
from __future__ import annotations

# The MLP (torch) and XGBoost models each bring their own OpenMP library (libomp). Used in one
# process on macOS they crash (segmentation fault) unless xgboost is imported first and torch runs
# on one thread. Predictions do not depend on this; the saved models are checked below.
import xgboost  # noqa: F401, E402  (keep first)
import torch  # noqa: E402

torch.set_num_threads(1)

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import CASES, DATA_DIR, FIGURES_DIR, RESULTS_DIR, ROOT, SAVED_MODELS_DIR  # noqa: E402

CURRENTS = ["id", "ig", "is", "ib"]
LABELS = {"id": "$I_d$", "ig": "$I_g$", "is": "$I_s$", "ib": "$I_b$"}
# model type -> (validation results of its script, folder of its saved models; None = not saved)
MODEL_TYPES = {
    "MLP":       ("mlp_validation.csv",        "mlp"),
    "XGBoost":   ("xgboost_validation.csv",    "xgboost"),
    "PolyRidge": ("poly_ridge_validation.csv", "poly_ridge"),
    "RF":        ("rf_validation.csv",         None),
}


def rel_err(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true) / np.abs(true) * 100


def rows(case: str, subset: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv")
    split = pd.read_csv(DATA_DIR / "transistor" / f"split_{case}.csv")
    assert (split.row.to_numpy() == np.arange(len(df))).all()
    return df[split.set.to_numpy() == subset]


def comparison() -> pd.DataFrame:
    out = []
    for model, (csv, _) in MODEL_TYPES.items():
        res = pd.read_csv(RESULTS_DIR / csv).set_index("case")
        for case in CASES:
            r = res.loc[case]
            row = {"model": model, "case": case}
            for c in CURRENTS:
                row.update({f"{c}_r2": r[f"{c}_r2"], f"{c}_median_err_pct": r[f"{c}_median_err_pct"],
                            f"{c}_p95_err_pct": r[f"{c}_p95_err_pct"]})
            row["mean_median_err_pct"] = np.mean([r[f"{c}_median_err_pct"] for c in CURRENTS])
            row["mean_p95_err_pct"] = np.mean([r[f"{c}_p95_err_pct"] for c in CURRENTS])
            out.append(row)
    return pd.DataFrame(out)


def choose(cmp: pd.DataFrame) -> pd.DataFrame:
    best = cmp.loc[cmp.groupby("case", sort=False).mean_median_err_pct.idxmin()]
    out = []
    for _, b in best.iterrows():
        folder = MODEL_TYPES[b.model][1]
        if folder is None:
            raise RuntimeError(f"{b.case}: chosen model type {b.model} has no saved model files")
        path = SAVED_MODELS_DIR / folder / f"{b.case}.joblib"
        out.append({"case": b.case, "model": b.model, "file": path.relative_to(ROOT).as_posix(),
                    "val_mean_median_err_pct": b.mean_median_err_pct, "val_mean_p95_err_pct": b.mean_p95_err_pct})
    return pd.DataFrame(out)


def check_saved_model(model, case: str, expected: pd.Series) -> None:
    """The saved model must reproduce the validation errors its script reported."""
    va = rows(case, "val")
    err = rel_err(model.predict(va), va[CURRENTS].to_numpy())
    for j, c in enumerate(CURRENTS):
        for name, value in [("median", np.median(err[:, j])), ("p95", np.quantile(err[:, j], 0.95))]:
            ref = expected[f"{c}_{name}_err_pct"]
            if not np.isclose(value, ref, rtol=1e-6, atol=1e-9):
                raise RuntimeError(f"{case}: saved model gives validation {c} {name} {value:.6g} %, "
                                   f"its script reported {ref:.6g} %")


def plot(case: str, model_name: str, I_true: np.ndarray, I_pred: np.ndarray, test: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    for ax, (j, c) in zip(axes.flat, enumerate(CURRENTS)):
        t, p = np.abs(I_true[:, j]), np.abs(I_pred[:, j])
        flip = np.sign(I_true[:, j]) != np.sign(I_pred[:, j])
        lo, hi = min(t.min(), p[p > 0].min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color="0.4", lw=1, zorder=1)
        ax.scatter(t[~flip], p[~flip], s=4, alpha=0.5, lw=0, zorder=2, label="same sign")
        if flip.any():
            ax.scatter(t[flip], p[flip], s=10, marker="x", color="tab:red", lw=0.8, zorder=3,
                       label=f"opposite sign ({flip.sum()})")
        ax.set(xscale="log", yscale="log", xlabel=f"SPICE |{LABELS[c]}| [A]", ylabel=f"predicted |{LABELS[c]}| [A]",
               title=f"{LABELS[c]}: median {test[f'{c}_median_err_pct']:.2f} %, p95 {test[f'{c}_p95_err_pct']:.1f} %")
        ax.legend(loc="upper left", fontsize=8, markerscale=2)
        ax.grid(True, which="major", alpha=0.3)
    fig.suptitle(f"{case.replace('_', ' ').upper()}: {model_name}, test set ({len(I_true)} samples)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"test_pred_vs_spice_{case}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cmp = comparison()
    cmp.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    chosen = choose(cmp)
    chosen.to_csv(RESULTS_DIR / "chosen_models.csv", index=False)
    pd.set_option("display.width", 250)
    print("Validation, mean over Id/Ig/Is/Ib [%]:")
    print(cmp.pivot(index="case", columns="model", values="mean_median_err_pct").loc[list(CASES)]
          .to_string(float_format=lambda x: f"{x:.3f}"), "\n")
    print(chosen.to_string(index=False), "\n")

    FIGURES_DIR.mkdir(exist_ok=True)
    out = []
    for _, ch in chosen.iterrows():
        model = joblib.load(ROOT / ch.file)
        check_saved_model(model, ch.case, cmp.set_index(["model", "case"]).loc[(ch.model, ch.case)])
        te = rows(ch.case, "test")
        I_true, I_pred = te[CURRENTS].to_numpy(), model.predict(te)
        err = rel_err(I_pred, I_true)
        row = {"case": ch.case, "model": ch.model, "n_test": len(te)}
        for j, c in enumerate(CURRENTS):
            row[f"{c}_median_err_pct"] = np.median(err[:, j])
            row[f"{c}_p95_err_pct"] = np.quantile(err[:, j], 0.95)
        row["mean_median_err_pct"] = np.mean([row[f"{c}_median_err_pct"] for c in CURRENTS])
        row["mean_p95_err_pct"] = np.mean([row[f"{c}_p95_err_pct"] for c in CURRENTS])
        out.append(row)
        plot(ch.case, ch.model, I_true, I_pred, row)
    res = pd.DataFrame(out)
    res.to_csv(RESULTS_DIR / "test_results.csv", index=False)
    print("Test set:")
    print(res.T.to_string())


if __name__ == "__main__":
    main()
