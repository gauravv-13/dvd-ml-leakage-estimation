"""Target transforms of the transistor models (Phase 3): how the four currents are turned into
training targets and back. Shared by src/mlp_model.py and experiments/target_transforms.py; it does
not import torch or xgboost.

A recipe is a dict of flags (missing flag = off):
  asinh   target asinh(I / 1e-18) for Id, Ig, Is, Ib (the signed scale used before the transform
          study); all other flags are ignored
  (default) target log10|I|; the sign comes from a classifier output when the current has both
          signs in the training set, otherwise from the constant training sign
  swap    canonical drain/source: NMOS rows with Vd < Vs and PMOS rows with Vs < Vd get Vd/Vs and
          Id/Is swapped, so the "drain" is always the high-current-into terminal side. The canonical
          Id then has a fixed sign (NMOS +, PMOS -) and needs no classifier. Canonical Is does not:
          in the OFF cases its sign is set by a near-cancellation of channel and junction leakage
          (opposite to the channel direction in 59 % of NMOS OFF and 44 % of PMOS OFF rows), so it
          keeps its classifier. Predictions are swapped back to the original terminals.
  perw    target log10(|I| / W)
  extra   extra inputs Vds/Vt, Vgs/Vt, Vbs/Vt and log10(1 - exp(-|Vds|/Vt) + 1e-12), Vt = kT/q
  sat     Id, Is target log10(|I| / (W (1 - exp(-|Vds|/Vt)) + 1e-30)) (needs perw)
  kcl     Is is not predicted: Is = -(Id + Ig + Ib)
  clip    training labels log10(max(|I|, 1e-17 A)); predictions are still scored against SPICE
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CURRENTS = ["id", "ig", "is", "ib"]
INPUTS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
BOLTZMANN_OVER_Q = 8.617333e-5        # V / K
ASINH_SCALE = 1e-18                   # A, asinh recipe
CLIP = 1e-17                          # A, clip recipe


def canonical(df: pd.DataFrame, case: str, recipe: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Copy of df in the canonical drain/source frame (swap recipe) and the mask of swapped rows."""
    d = df.copy()
    swapped = np.zeros(len(d), bool)
    if recipe.get("swap") and not recipe.get("asinh"):
        swapped = (d.vd < d.vs).to_numpy() if case.startswith("nmos") else (d.vs < d.vd).to_numpy()
        d.loc[swapped, ["vd", "vs"]] = d.loc[swapped, ["vs", "vd"]].to_numpy()
        if "id" in d:
            d.loc[swapped, ["id", "is"]] = d.loc[swapped, ["is", "id"]].to_numpy()
    return d, swapped


def saturation(d: pd.DataFrame) -> np.ndarray:
    """1 - exp(-|Vds| / Vt)."""
    vt = BOLTZMANN_OVER_Q * (d.temp + 273.15)
    return (-np.expm1(-np.abs(d.vd - d.vs) / vt)).to_numpy()


def extra_inputs(d: pd.DataFrame) -> np.ndarray:
    vt = BOLTZMANN_OVER_Q * (d.temp + 273.15)
    return np.column_stack([(d.vd - d.vs) / vt, (d.vg - d.vs) / vt, (d.vb - d.vs) / vt,
                            np.log10(saturation(d) + 1e-12)])


def mlp_inputs(d: pd.DataFrame, recipe: dict) -> np.ndarray:
    """The 12 dataset inputs, Vd-Vs, Vg-Vs, Vb-Vs, 1/T (T in K), and the extra inputs if requested."""
    X = np.column_stack([d[INPUTS].to_numpy(), d.vd - d.vs, d.vg - d.vs, d.vb - d.vs, 1.0 / (d.temp + 273.15)])
    if recipe.get("extra") and not recipe.get("asinh"):
        X = np.column_stack([X, extra_inputs(d)])
    return X.astype(float)


def denominator(d: pd.DataFrame, current: str, recipe: dict) -> np.ndarray:
    den = np.ones(len(d))
    if recipe.get("perw"):
        den = d.w.to_numpy().astype(float)
    if recipe.get("sat") and current in ("id", "is"):
        den = d.w.to_numpy() * saturation(d) + 1e-30
    return den


class Plan:
    """Which currents get a magnitude output and where each sign comes from (fixed on the training set)."""

    def __init__(self, case: str, recipe: dict, train: pd.DataFrame):
        self.asinh = bool(recipe.get("asinh"))
        self.mags = [c for c in CURRENTS if self.asinh or not (recipe.get("kcl") and c == "is")]
        self.phys: dict[str, float] = {}                 # sign fixed by the canonical frame
        self.const: dict[str, float] = {}                # one sign in the whole training set
        self.cls: list[str] = []                         # sign from a classifier output
        if self.asinh:
            return
        if recipe.get("swap"):
            self.phys["id"] = 1.0 if case.startswith("nmos") else -1.0
        for c in self.mags:
            if c in self.phys:
                continue
            pos = train[c].to_numpy() > 0
            if pos.all() or not pos.any():
                self.const[c] = 1.0 if pos.all() else -1.0
            else:
                self.cls.append(c)

    def targets(self, d: pd.DataFrame, recipe: dict) -> tuple[np.ndarray, np.ndarray]:
        """Magnitude targets (n, len(mags)) and sign labels (n, len(cls)) of canonical rows d."""
        if self.asinh:
            return np.arcsinh(d[CURRENTS].to_numpy() / ASINH_SCALE), np.zeros((len(d), 0))
        lo = CLIP if recipe.get("clip") else 0.0
        Y = np.column_stack([np.log10(np.maximum(np.abs(d[c].to_numpy()), lo) / denominator(d, c, recipe))
                             for c in self.mags])
        S = np.column_stack([d[c].to_numpy() > 0 for c in self.cls]) if self.cls else np.zeros((len(d), 0))
        return Y, S.astype(float)

    def currents(self, d: pd.DataFrame, swapped: np.ndarray, recipe: dict,
                 y: np.ndarray, p_pos: np.ndarray) -> np.ndarray:
        """(n, 4) Id, Ig, Is, Ib [A] at the original terminals, from the model outputs on canonical rows d:
        y = magnitude outputs (target scale), p_pos = probability that each classified current is > 0."""
        if self.asinh:
            return ASINH_SCALE * np.sinh(y)
        out = {}
        for j, c in enumerate(self.mags):
            mag = 10.0 ** np.clip(y[:, j], -300, 300) * denominator(d, c, recipe)
            if c in self.phys:
                sign = self.phys[c]
            elif c in self.const:
                sign = self.const[c]
            else:
                sign = np.where(p_pos[:, self.cls.index(c)] > 0.5, 1.0, -1.0)
            out[c] = sign * mag
        if "is" not in out:
            out["is"] = -(out["id"] + out["ig"] + out["ib"])
        I = np.column_stack([out[c] for c in CURRENTS])
        I[swapped] = I[swapped][:, [2, 1, 0, 3]]            # back to the original drain / source
        return I
