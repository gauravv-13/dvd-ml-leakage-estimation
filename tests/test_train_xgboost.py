"""The XGBoost row must be comparable with the MLP row: same split and metrics."""
from __future__ import annotations

import ast
import csv
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
xgb_script = importlib.import_module("03_train_xgboost")

FAST = {"objective": "reg:squarederror", "rounds": 100, "eta": 0.3, "max_depth": 3}
METRIC_SUFFIXES = ("_r2", "_median_err_pct", "_p95_err_pct")


def top_level_sources(path: Path, names: set[str]) -> dict[str, str]:
    """AST dumps of the named top-level functions and constants (comments are ignored)."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names:
            found[node.targets[0].id] = ast.dump(node)
    return found


def test_load_metrics_and_constants_are_identical_to_the_mlp_script():
    names = {"load", "metrics", "CURRENTS", "I_SCALE", "SEED"}

    mlp = top_level_sources(ROOT / "03_train_mlp.py", names)
    xgb = top_level_sources(ROOT / "03_train_xgboost.py", names)

    assert set(mlp) == names
    assert xgb == mlp


def test_load_returns_the_shared_train_and_val_rows_and_never_the_test_rows():
    split = pd.read_csv(ROOT / "data" / "transistor" / "split_nmos_off.csv")

    d = xgb_script.load("nmos_off")

    assert set(d) == {"train", "val"}
    assert set(d["train"].index) == set(split.row[split.set == "train"])
    assert set(d["val"].index) == set(split.row[split.set == "val"])


def test_metrics_reports_a_ten_percent_high_prediction_as_ten_percent_error():
    I_true = np.array([[1e-12, -2e-9, 3e-7, -4e-15], [5e-11, -6e-10, 7e-8, -8e-14]])
    y_pred = np.arcsinh(1.1 * I_true / xgb_script.I_SCALE)

    m = xgb_script.metrics(I_true, y_pred)

    for cur in xgb_script.CURRENTS:
        assert abs(m[f"{cur}_median_err_pct"] - 10.0) < 1e-6
        assert abs(m[f"{cur}_p95_err_pct"] - 10.0) < 1e-6


def test_features_do_not_change_when_drain_and_source_are_swapped():
    df = xgb_script.load("nmos_off")["val"].iloc[:50]
    swapped = df.assign(vd=df.vs, vs=df.vd, vds=-df.vds)

    pd.testing.assert_frame_equal(xgb_script.features(df), xgb_script.features(swapped))


def test_linear_baseline_recovers_a_linear_target_on_unseen_rows_with_a_constant_column():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(1e-9, 2e-9, 200), np.zeros(200)])
    y = 5.0 + 2.0 * X[:, 0] - 3e9 * X[:, 1]

    _, base_va = xgb_script.linear_baseline(X[:150], y[:150], X[150:])

    np.testing.assert_allclose(base_va, y[150:], atol=1e-9)


@pytest.mark.parametrize("case", ["nmos_on", "nmos_off", "pmos_on", "pmos_off"])
def test_terminal_currents_rebuilds_true_labels_even_where_they_break_kirchhoff(case):
    va = xgb_script.load(case)["val"]
    Id, Is = va.id.to_numpy(), va["is"].to_numpy()

    cur = xgb_script.terminal_currents((Id - Is) / 2, (Id + Is) / 2, va.ig.to_numpy(), va.ib.to_numpy())

    np.testing.assert_allclose(cur["id"], Id, rtol=1e-9)
    np.testing.assert_allclose(cur["is"], Is, rtol=1e-9)


def test_fit_predict_restores_the_scale_and_sign_of_the_currents():
    d = xgb_script.load("nmos_on")
    I_true = d["val"][xgb_script.CURRENTS].to_numpy()

    y_pred = xgb_script.fit_predict(d["train"], d["val"], FAST)

    assert y_pred.shape == I_true.shape
    I_pred = xgb_script.I_SCALE * np.sinh(y_pred)
    assert np.median(np.abs(I_pred - I_true) / np.abs(I_true)) < 0.25
    channel = (I_pred[:, 0] - I_pred[:, 2]) / 2
    assert (np.sign(channel) == np.sign(d["val"].vds.to_numpy())).all()


def test_only_pmos_off_trains_on_absolute_error():
    objectives = {case: s["objective"] for case, s in xgb_script.SETTINGS.items()}

    assert objectives == {"nmos_on": "reg:squarederror", "nmos_off": "reg:squarederror",
                          "pmos_on": "reg:squarederror", "pmos_off": "reg:absoluteerror"}


def test_run_case_row_has_the_metric_columns_of_the_mlp_results(monkeypatch):
    monkeypatch.setitem(xgb_script.SETTINGS, "nmos_off", FAST)
    with open(ROOT / "results" / "mlp_validation.csv", newline="") as f:
        mlp_columns = next(csv.reader(f))
    expected = {c for c in mlp_columns if c.endswith(METRIC_SUFFIXES)} | {"mean_median_err_pct", "model", "case"}

    row = xgb_script.run_case("nmos_off")

    assert row["model"] == "XGBoost"
    assert row["case"] == "nmos_off"
    assert expected <= set(row)
