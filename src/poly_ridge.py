"""Polynomial ridge regression model of one single-transistor case (Phase 3, model type: polynomial ridge).

Raw Id and Is change sign with Vd - Vs, which a polynomial cannot follow on a log scale, so the
model predicts three sign-consistent quantities and rebuilds the four terminal currents from them:

    D  = Id - Is   = Vds * exp(f_D(x))       D / Vds is even in Vds and > 0 in every sample
    Ig             = s_g * exp(f_g(x))
    Ib             = s_b * exp(f_b(x))
    Id = (D - Ig - Ib) / 2,   Is = (-D - Ig - Ib) / 2        so Id + Ig + Is + Ib = 0 exactly

Each f is a ridge-regularised polynomial of the standardised features x, with its own degree,
alpha and choice of whether the junction feature fjun is used (all chosen on the validation set).
s_g, s_b are the majority signs of Ig, Ib in the training set; only PMOS OFF has mixed signs,
from currents at the ~1e-16 A simulator noise floor.

Features (20) are built in a symmetric NMOS-like frame:
  * PMOS voltages are mirrored, V -> VDD - V, so a PMOS looks like an NMOS (body at 0, gate at
    VDD when ON).
  * Drain and source are sorted into the lower (lo) and higher (hi) terminal, so the features
    are unchanged when they are swapped and Id(vd, vs) = Is(vs, vd) holds exactly.
  Base (10):    vdd, v_lo, v_hi, 1/T, ln W, |vth0|, toxe, ndep, u0, xj
  Physics (10): lsub   ln[(1 - exp(-Vds/VT)) / (Vds/VT)]          subthreshold drain factor
                lq_lo, lq_hi   ln q, q = ln(1 + exp((Vg - V - |vth0|) / (2 n VT)))
                               EKV inversion charge at each channel end (weak to strong inversion)
                lekv   ln[(q_lo^2 - q_hi^2) / (Vds/VT)]           EKV channel current / Vds
                lg_lo, lg_hi   ln ln(1 + exp((Vg - V - |vth0|) / VT))   sharper inversion onset of the
                               gate-to-channel tunnelling current (BSIM4 Vaux with nigc = 1)
                tv_lo, tv_hi   toxe * (Vg - V)                    oxide-field term of the tunnelling exponent
                eg_vt  Eg(T) / VT                                  junction saturation current vs T
                fjun   (1 - exp(-V_lo/VT)) + (1 - exp(-V_hi/VT))  junction reverse-bias factor, in [0, 2]
                       (optional; kept linear: its log has a -15 sigma tail that degree-4 terms blow up)

Usage (Phase 5): model = joblib.load("saved_models/poly_ridge/<case>.joblib")
                 I = model.predict(df)   # df: dataset input columns; I: (n, 4) = Id, Ig, Is, Ib [A]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

K_B = 8.617333262e-5          # Boltzmann constant [eV/K]
N_SLOPE = 1.5                 # subthreshold slope factor used in the EKV features
PARTS = ["d", "ig", "ib"]     # D/Vds, Ig, Ib
FEATURES = ["vdd", "v_lo", "v_hi", "inv_t", "logw", "vth0", "toxe", "ndep", "u0", "xj",
            "lsub", "lq_lo", "lq_hi", "lekv", "lg_lo", "lg_hi", "tv_lo", "tv_hi", "eg_vt", "fjun"]


def features(df: pd.DataFrame, dev: str, junction: bool = True) -> np.ndarray:
    """(n, 20) feature matrix in the symmetric NMOS-like frame (see module docstring);
    (n, 19) without the junction feature fjun (always the last column)."""
    vdd = df.vdd.to_numpy()
    mirror = (lambda v: v) if dev == "nmos" else (lambda v: vdd - v)
    vd, vs, vg = mirror(df.vd.to_numpy()), mirror(df.vs.to_numpy()), mirror(df.vg.to_numpy())
    v_lo, v_hi = np.minimum(vd, vs), np.maximum(vd, vs)
    vds = v_hi - v_lo
    tk = df.temp.to_numpy() + 273.15
    vt = K_B * tk
    vth = np.abs(df.vth0.to_numpy())

    x = np.maximum(vds, 1e-12) / vt
    lsub = np.log(-np.expm1(-x) / x)

    k = 2 * N_SLOPE * vt
    z_lo, z_hi = (vg - v_lo - vth) / k, (vg - v_hi - vth) / k
    q_lo, q_hi = np.logaddexp(0, z_lo), np.logaddexp(0, z_hi)
    # (q_lo - q_hi) * VT / Vds, with its Vds -> 0 limit sigmoid(z_lo) / (2 n)
    with np.errstate(divide="ignore", invalid="ignore"):
        dq = np.where(vds > 1e-9, (q_lo - q_hi) / np.maximum(vds, 1e-30) * vt,
                      1 / (1 + np.exp(-z_lo)) / (2 * N_SLOPE))
    lekv = np.log(dq * (q_lo + q_hi))

    lg_lo = np.log(np.logaddexp(0, (vg - v_lo - vth) / vt))
    lg_hi = np.log(np.logaddexp(0, (vg - v_hi - vth) / vt))
    toxe = df.toxe.to_numpy()

    eg = 1.16 - 7.02e-4 * tk ** 2 / (tk + 1108)
    fjun = -np.expm1(-np.abs(v_lo) / vt) - np.expm1(-np.abs(v_hi) / vt)

    X = np.column_stack([vdd, v_lo, v_hi, 1 / tk, np.log(df.w.to_numpy()), vth,
                         toxe, df.ndep, df.u0, df.xj,
                         lsub, np.log(q_lo), np.log(q_hi), lekv, lg_lo, lg_hi,
                         toxe * (vg - v_lo), toxe * (vg - v_hi), eg / vt, fjun])
    return X if junction else X[:, :-1]


def targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Signed quantities modelled by the three parts (before taking logs)."""
    return {"d": (df.id - df["is"]).to_numpy() / (df.vd - df.vs).to_numpy(),
            "ig": df.ig.to_numpy(), "ib": df.ib.to_numpy()}


def pipeline(degree: int, alpha: float) -> Pipeline:
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), Ridge(alpha=alpha))


class PolyRidgeFET:
    """Three polynomial ridge regressions (D/Vds, Ig, Ib) for one case; predicts Id, Ig, Is, Ib."""

    def __init__(self, case: str, dev: str, settings: dict[str, tuple[int, float, bool]]):
        self.case, self.dev = case, dev
        self.settings = settings              # {part: (degree, alpha, junction)}
        self.sign: dict[str, float] = {}
        self.pipes: dict[str, Pipeline] = {}

    def fit(self, df: pd.DataFrame) -> "PolyRidgeFET":
        X = features(df, self.dev)
        for part, t in targets(df).items():
            degree, alpha, junction = self.settings[part]
            self.sign[part] = 1.0 if np.mean(t > 0) >= 0.5 else -1.0
            y = np.log(np.abs(t))             # magnitude; opposite-sign rows keep their |I|
            self.pipes[part] = pipeline(degree, alpha).fit(X if junction else X[:, :-1], y)
        return self

    def predict_parts(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        X = features(df, self.dev)
        out = {}
        for p in PARTS:
            y = self.pipes[p].predict(X if self.settings[p][2] else X[:, :-1])
            # clip the log-prediction to +/-300 (|I| < 1e130 A) so wild extrapolation cannot overflow
            out[p] = self.sign[p] * np.exp(np.clip(y, -300, 300))
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 4) array of Id, Ig, Is, Ib [A], current INTO each terminal."""
        p = self.predict_parts(df)
        d = p["d"] * (df.vd - df.vs).to_numpy()
        return np.column_stack([(d - p["ig"] - p["ib"]) / 2, p["ig"],
                                (-d - p["ig"] - p["ib"]) / 2, p["ib"]])
