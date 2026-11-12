"""Saved MLP transistor model of one single-transistor case (Phase 3, model type: MLP).

One network per case predicts the transformed currents of its recipe (src/target_transform.py):
magnitude outputs (L1 loss, standardised on the training set) plus, for currents whose sign is
not fixed, sign outputs (logistic loss). 03_train_mlp.py trains it and stores it, with its input and
output scaling, as one MLPFET object in saved_models/mlp/<case>.joblib.

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

from .target_transform import Plan, canonical, mlp_inputs


def make_net(n_in: int, width: int, depth: int, n_out: int) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, n_out))


def train_net(X: torch.Tensor, Y: torch.Tensor, S: torch.Tensor, s: dict, seed: int) -> nn.Sequential:
    """L1 loss on the standardised magnitudes Y plus logistic loss on the sign labels S (if any);
    AdamW, one-cycle learning rate, mini-batches in a seeded random order."""
    torch.manual_seed(seed)
    n_mag = Y.shape[1]
    net = make_net(X.shape[1], s["width"], s["depth"], n_mag + S.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    steps = int(np.ceil(len(X) / s["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s["lr"], epochs=s["epochs"], steps_per_epoch=steps)
    l1, bce = nn.L1Loss(), nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(s["epochs"]):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), s["batch"]):
            idx = perm[i:i + s["batch"]]
            opt.zero_grad()
            out = net(X[idx])
            loss = l1(out[:, :n_mag], Y[idx]) + (bce(out[:, n_mag:], S[idx]) if S.shape[1] else 0)
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


class MLPFET:
    """Trained MLP of one case with its recipe and training-set scaling; predicts Id, Ig, Is, Ib [A]."""

    @classmethod
    def fit(cls, case: str, settings: dict, recipe: dict, train: pd.DataFrame, seed: int) -> "MLPFET":
        self = cls.__new__(cls)
        self.case, self.settings, self.recipe, self.seed = case, dict(settings), dict(recipe), seed
        d, _ = canonical(train, case, recipe)
        self.plan = Plan(case, recipe, d)
        X = mlp_inputs(d, recipe)
        Y, S = self.plan.targets(d, recipe)
        self.x_mean, self.x_std = X.mean(0), X.std(0)
        self.x_std[self.x_std == 0] = 1.0                 # constant column (e.g. vb = 0 for nmos_on)
        self.y_mean, self.y_std = Y.mean(0), Y.std(0)
        self.y_std[self.y_std == 0] = 1.0
        f32 = lambda a: torch.tensor(a, dtype=torch.float32)
        net = train_net(f32((X - self.x_mean) / self.x_std), f32((Y - self.y_mean) / self.y_std), f32(S),
                        settings, seed)
        # weights stored as numpy arrays, so the file does not depend on torch's pickling
        self.weights = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        self._net = net
        return self

    def network(self) -> nn.Sequential:
        if self._net is None:
            n_out = len(self.plan.mags) + len(self.plan.cls)
            net = make_net(len(self.x_mean), self.settings["width"], self.settings["depth"], n_out)
            net.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights.items()})
            self._net = net.eval()
        return self._net

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 4) array of Id, Ig, Is, Ib [A], current INTO each terminal."""
        d, swapped = canonical(df, self.case, self.recipe)
        X = torch.tensor((mlp_inputs(d, self.recipe) - self.x_mean) / self.x_std, dtype=torch.float32)
        with torch.no_grad():
            out = self.network()(X).numpy()
        n_mag = len(self.plan.mags)
        y = out[:, :n_mag] * self.y_std + self.y_mean
        p_pos = 1.0 / (1.0 + np.exp(-out[:, n_mag:].astype(float)))
        return self.plan.currents(d, swapped, self.recipe, y, p_pos)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_net"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._net = None
