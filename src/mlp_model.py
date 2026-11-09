"""Saved MLP transistor model of one single-transistor case (Phase 3, model type: MLP).

03_train_mlp.py trains the network and stores it, together with its input and output scaling,
as one MLPFET object in saved_models/mlp/<case>.joblib.

Usage (Phase 5): model = joblib.load("saved_models/mlp/<case>.joblib")
                 I = model.predict(df)   # df: dataset input columns; I: (n, 4) = Id, Ig, Is, Ib [A]
A script that also uses the XGBoost models must `import xgboost` before torch (or this module)
and call torch.set_num_threads(1): both bring their own OpenMP library, and on macOS using both in
one process otherwise crashes it (see 03_select_and_test.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

INPUTS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
DERIVED = ["vds", "vgs", "vbs", "inv_t"]
N_OUT = 4                             # Id, Ig, Is, Ib
I_SCALE = 1e-18                       # target = asinh(I / I_SCALE)


def make_net(n_in: int, width: int, depth: int) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, N_OUT))


def network_inputs(df: pd.DataFrame) -> np.ndarray:
    """The 12 dataset input columns plus Vd-Vs, Vg-Vs, Vb-Vs and 1/T (T in kelvin)."""
    derived = np.column_stack([df.vd - df.vs, df.vg - df.vs, df.vb - df.vs, 1.0 / (df.temp + 273.15)])
    return np.column_stack([df[INPUTS].to_numpy(), derived])


class MLPFET:
    """Trained MLP of one case with its training-set scaling; predicts Id, Ig, Is, Ib [A]."""

    def __init__(self, case: str, settings: dict, net: nn.Sequential,
                 x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray):
        self.case, self.settings = case, dict(settings)
        self.x_mean, self.x_std, self.y_mean, self.y_std = x_mean, x_std, y_mean, y_std
        # weights stored as numpy arrays, so the file does not depend on torch's pickling
        self.weights = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        self._net = None

    def network(self) -> nn.Sequential:
        if self._net is None:
            net = make_net(len(self.x_mean), self.settings["width"], self.settings["depth"])
            net.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights.items()})
            self._net = net.eval()
        return self._net

    def predict_asinh(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 4) predictions of asinh(I / I_SCALE)."""
        X = torch.tensor((network_inputs(df) - self.x_mean) / self.x_std, dtype=torch.float32)
        with torch.no_grad():
            return self.network()(X).numpy() * self.y_std + self.y_mean

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 4) array of Id, Ig, Is, Ib [A], current INTO each terminal."""
        return I_SCALE * np.sinh(self.predict_asinh(df))

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_net"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._net = None
