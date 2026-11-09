"""XGBoost transistor model of one single-transistor case (Phase 3, model type: gradient-boosted trees).

Four boosters per case, for Ich = (Id - Is) / 2, Icm = (Id + Is) / 2, Ig and Ib, each fitted on
asinh(I / (1e-18 * W / Lmin)) on top of a least-squares line through the features (base_margin).
Ich is multiplied by sign(Vd - Vs) so it does not jump where Vd - Vs crosses 0.
Then Id = Icm + Ich and Is = Icm - Ich. See 03_train_xgboost.py for the reasoning.

03_train_xgboost.py stores one fitted XGBFET per case in saved_models/xgboost/<case>.joblib.

Usage (Phase 5): model = joblib.load("saved_models/xgboost/<case>.joblib")
                 I = model.predict(df)   # df: dataset input columns; I: (n, 4) = Id, Ig, Is, Ib [A]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

CURRENTS = ["id", "ig", "is", "ib"]
I_SCALE = 1e-18
SEED = 42
NTHREAD = 2                           # fixed thread count, part of the reproducible setup
BOLTZMANN_OVER_Q = 8.617333e-5        # V / K
MIN_GAP = 1e-9                        # V, keeps log() finite when Vd = Vs
PARTS = ["ich", "icm", "ig", "ib"]


def features(df: pd.DataFrame) -> pd.DataFrame:
    vt = BOLTZMANN_OVER_Q * (df.temp + 273.15)
    lo = df[["vd", "vs"]].min(axis=1)
    hi = df[["vd", "vs"]].max(axis=1)
    gap = np.maximum(hi - lo, MIN_GAP)
    tox_nm = df.toxe * 1e9
    return pd.DataFrame({
        "vdd": df.vdd, "temp": df.temp, "inv_t": 1.0 / (df.temp + 273.15), "w": df.w, "vg": df.vg, "vb": df.vb,
        "vth0": df.vth0, "toxe": df.toxe, "ndep": df.ndep, "u0": df.u0, "xj": df.xj,
        "lo": lo, "hi": hi, "gap": gap, "ln_sat": np.log(-np.expm1(-gap / vt)),
        "vg_lo": df.vg - lo, "vg_hi": df.vg - hi, "vb_lo": df.vb - lo, "vb_hi": df.vb - hi,
        "vov_lo": (df.vg - lo - df.vth0) / vt, "vov_hi": (df.vg - hi - df.vth0) / vt,
        "vb_lo_vt": (df.vb - lo) / vt, "vb_hi_vt": (df.vb - hi) / vt,
        "vg_lo_tox": (df.vg - lo) / tox_nm, "vg_hi_tox": (df.vg - hi) / tox_nm,
        "vgb_tox": (df.vg - df.vb) / tox_nm,
        "ln_w": np.log(df.w), "ln_tox": np.log(tox_nm), "ln_ndep": np.log(df.ndep), "ln_u0": np.log(np.abs(df.u0)),
        "vth_vt": df.vth0 / vt,
    }, index=df.index)


class LinearBase:
    """Least-squares line through the standardised features; the boosters start from it."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearBase":
        self.x_mean, x_std = X.mean(0), X.std(0)
        self.x_std = np.where(x_std == 0, 1.0, x_std)      # constant column (e.g. vb = 0 for nmos_on)
        self.coef = np.linalg.lstsq(self._design(X), y, rcond=None)[0]
        return self

    def _design(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), (X - self.x_mean) / self.x_std])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.coef


def linear_baseline(X_tr: np.ndarray, y: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit on X_tr; returns its values on X_tr and X_va."""
    lin = LinearBase().fit(X_tr, y)
    return lin.predict(X_tr), lin.predict(X_va)


def terminal_currents(ich: np.ndarray, icm: np.ndarray, ig: np.ndarray, ib: np.ndarray) -> dict[str, np.ndarray]:
    """Id and Is from Ich = (Id - Is) / 2 and Icm = (Id + Is) / 2."""
    return {"id": icm + ich, "ig": ig, "is": icm - ich, "ib": ib}


class XGBFET:
    """Four boosters (Ich, Icm, Ig, Ib) of one case; predicts Id, Ig, Is, Ib [A]."""

    def __init__(self, case: str, settings: dict, lmin: float):
        self.case, self.settings, self.lmin = case, dict(settings), lmin
        self.lin: dict[str, LinearBase] = {}
        self.boosters: dict[str, xgb.Booster] = {}

    @staticmethod
    def _targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
        Id, Is = df.id.to_numpy(), df["is"].to_numpy()
        return {"ich": (Id - Is) / 2, "icm": (Id + Is) / 2, "ig": df.ig.to_numpy(), "ib": df.ib.to_numpy()}

    @staticmethod
    def _sign(df: pd.DataFrame, part: str) -> np.ndarray | float:
        return np.sign((df.vd - df.vs).to_numpy()) if part == "ich" else 1.0

    def fit(self, df: pd.DataFrame) -> "XGBFET":
        X = features(df).to_numpy()
        w = df.w.to_numpy() / self.lmin
        s = self.settings
        params = {**{k: v for k, v in s.items() if k != "rounds"}, "seed": SEED, "nthread": NTHREAD}
        for part, I in self._targets(df).items():
            y = self._sign(df, part) * np.arcsinh(I / (I_SCALE * w))
            self.lin[part] = LinearBase().fit(X, y)
            dm = xgb.DMatrix(X, label=y, base_margin=self.lin[part].predict(X))
            self.boosters[part] = xgb.train(params, dm, num_boost_round=s["rounds"])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 4) array of Id, Ig, Is, Ib [A], current INTO each terminal."""
        X = features(df).to_numpy()
        w = df.w.to_numpy() / self.lmin
        out = {}
        for part in PARTS:
            raw = self.boosters[part].predict(xgb.DMatrix(X, base_margin=self.lin[part].predict(X))).astype(float)
            out[part] = I_SCALE * w * np.sinh(self._sign(df, part) * raw)
        cur = terminal_currents(out["ich"], out["icm"], out["ig"], out["ib"])
        return np.column_stack([cur[c] for c in CURRENTS])
