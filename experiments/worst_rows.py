"""Optional experiment: the 30 worst validation rows of the T0 MLP (log10|I| + sign output, seed 42)
per case, and where they cluster. Motivates the transforms of experiments/target_transforms.py
(report appendix "Target-transform study"); run that script first, it saves the T0 predictions.

Outputs (results/experiments/):
  worst30_T0_mlp_<case>.csv     inputs, worst current, SPICE and predicted value, error [%]
  worst30_T0_mlp_summary.csv    share of the 30 vs all validation rows: |Vds| < 20 mV / 1 mV,
                                |I| < 1e-15 A, wrong sign, T > 100 C; medians of T, |vth0|, VDD, W, |Vgs|
Run:  python experiments/worst_rows.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import CASES, RESULTS_DIR  # noqa: E402
from src.target_transform import CURRENTS  # noqa: E402

OUT = RESULTS_DIR / "experiments"
COLS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vds", "vgs", "vth0", "toxe", "worst", "true", "pred", "err_pct"]


def stats(d: pd.DataFrame) -> dict:
    vds = d.vds.abs()
    return {"vds_lt_20mV_pct": (vds < 0.02).mean() * 100, "vds_lt_1mV_pct": (vds < 1e-3).mean() * 100,
            "abs_i_lt_1e-15_pct": (d.true.abs() < 1e-15).mean() * 100,
            "wrong_sign_pct": (np.sign(d.pred) != np.sign(d.true)).mean() * 100,
            "t_gt_100C_pct": (d.temp > 100).mean() * 100, "median_temp": d.temp.median(),
            "median_abs_vth0": d.vth0.abs().median(), "median_vdd": d.vdd.median(),
            "median_w_nm": d.w.median() * 1e9, "median_abs_vgs": d.vgs.abs().median()}


def main() -> None:
    load = importlib.import_module("03_train_xgboost").load        # same load() as 03_train_mlp.py
    summary = []
    for case in CASES:
        va = load(case)["val"].reset_index(drop=True)
        P = np.load(OUT / "runs" / f"mlp_T0_42_{case}.npy")
        T = va[CURRENTS].to_numpy()
        E = np.abs(P - T) / np.abs(T) * 100
        j = E.argmax(1)
        rows = np.arange(len(T))
        va = va.assign(worst=[CURRENTS[k] for k in j], true=T[rows, j], pred=P[rows, j], err_pct=E.max(1))
        worst = va.sort_values("err_pct", ascending=False).head(30)
        worst[COLS].to_csv(OUT / f"worst30_T0_mlp_{case}.csv", index=False)
        summary.append({"case": case, **{f"worst30_{k}": v for k, v in stats(worst).items()},
                        **{f"all_{k}": v for k, v in stats(va).items()},
                        **{f"worst30_n_{c}": int((worst.worst == c).sum()) for c in CURRENTS}})
        print(f"{case}: worst current counts {worst.worst.value_counts().to_dict()}")
    pd.DataFrame(summary).to_csv(OUT / "worst30_T0_mlp_summary.csv", index=False)


if __name__ == "__main__":
    main()
