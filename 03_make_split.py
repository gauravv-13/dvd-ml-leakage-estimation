"""Phase 3 (Part A step 2): fixed train / validation / test split, shared by every model type.

For each case, the rows of data/transistor/<case>.csv are shuffled with the fixed seed
(config split.seed) and divided 70 / 15 / 15 (config split.fractions).

Output: data/transistor/split_<case>.csv with columns `row` (0-based row index of the
dataset CSV) and `set` (train / val / test).
Run:    python 03_make_split.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CASES, DATA_DIR, load_config


def main() -> None:
    cfg = load_config()
    seed = cfg["split"]["seed"]
    f_train, f_val, _ = cfg["split"]["fractions"]
    for case in CASES:
        n = len(pd.read_csv(DATA_DIR / "transistor" / f"{case}.csv"))
        perm = np.random.default_rng(seed).permutation(n)
        n_train, n_val = round(f_train * n), round(f_val * n)
        sets = np.empty(n, dtype=object)
        sets[perm[:n_train]] = "train"
        sets[perm[n_train:n_train + n_val]] = "val"
        sets[perm[n_train + n_val:]] = "test"
        split = pd.DataFrame({"row": np.arange(n), "set": sets})
        split.to_csv(DATA_DIR / "transistor" / f"split_{case}.csv", index=False)
        print(f"{case:9s} " + ", ".join(f"{s}={c}" for s, c in split.set.value_counts().items()))


if __name__ == "__main__":
    main()
