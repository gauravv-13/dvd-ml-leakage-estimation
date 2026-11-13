"""08_circuit_ml.py: every NAND2 of the parity checker gets the right input combination for every vector."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))
circuit_ml = importlib.import_module("08_circuit_ml")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """main() on the real Phase 5 and circuit SPICE data, writing into a temporary directory."""
    tmp = tmp_path_factory.mktemp("c08")
    (tmp / "circuit").mkdir()
    (tmp / "gates").symlink_to(DATA / "gates")
    (tmp / "circuit" / "circuit_spice.csv").symlink_to(DATA / "circuit" / "circuit_spice.csv")
    with mock.patch.multiple(circuit_ml, DATA_DIR=tmp, RESULTS_DIR=tmp, FIGURES_DIR=tmp):
        circuit_ml.main()
    return (pd.read_csv(tmp / "circuit" / "circuit_ml.csv"),
            pd.read_csv(tmp / "circuit_results.csv").set_index("estimate"))


def test_vectors_match_the_input_columns_of_the_circuit_spice_data():
    cols = [f"in{i}" for i in range(8)]
    spice = pd.read_csv(DATA / "circuit" / "circuit_spice.csv", usecols=["vector", *cols])

    per_vector = spice.drop_duplicates("vector").set_index("vector").sort_index()[cols]

    assert per_vector.to_dict("index") == dict(enumerate(circuit_ml.vectors()))


def test_sum_of_gate_spice_leakages_matches_circuit_spice_within_1_pct(run):
    # physical oracle: with the right pin order and vector encoding the composition is near exact
    assert run[0].gate_sum_err_pct.abs().max() < 1


def test_ml_sum_of_all_zero_vector_is_7_nand2_at_00_plus_14_at_01_plus_7_at_11(run):
    g = pd.read_csv(DATA / "gates" / "gate_ml.csv", dtype={"inputs": str})
    ml = g[g.gate == "nand2"].pivot(index="corner", columns="inputs", values="leak_preds")
    got = run[0][run[0].vector == 0].set_index("corner").ml_leak

    expected = (7 * ml["00"] + 14 * ml["01"] + 7 * ml["11"]).reindex(got.index)

    pd.testing.assert_series_equal(got, expected, check_names=False, rtol=1e-9, atol=0)


def test_results_p95_is_the_95th_percentile_of_row_errors(run):
    df, table = run

    assert table.loc["ML sum of gate leakages", "p95_err_pct"] == pytest.approx(df.ml_err_pct.abs().quantile(0.95))


def test_every_vector_puts_each_of_the_28_nand2_in_exactly_one_input_case():
    counts = circuit_ml.input_counts()

    assert counts.shape == (256, 4)
    assert (counts.sum(axis=1) == 28).all()


def test_all_zero_vector_gives_each_xor_one_00_two_01_and_one_11_nand2():
    # each XOR is 4 NAND2: with both inputs at 0 the first sees 00, the middle two see (input 0, first output 1)
    # and the last sees both middle outputs at 1; all 7 XORs of the tree see 0 0
    counts = circuit_ml.input_counts()

    assert counts.loc[0].to_dict() == {"00": 7, "01": 14, "10": 0, "11": 7}


def test_circuit_sum_adds_each_gate_leakage_once_per_gate_in_that_input_case():
    counts = pd.DataFrame([[1, 2, 0, 3], [0, 0, 4, 0]], columns=circuit_ml.INPUT_CASES)
    gates = pd.DataFrame({"inputs": circuit_ml.INPUT_CASES * 2, "corner": [0] * 4 + [1] * 4,
                          "leak": [1.0, 10.0, 100.0, 1000.0, 2.0, 20.0, 200.0, 2000.0]})

    total = circuit_ml.circuit_sum(counts, gates, "leak")

    assert total.to_dict() == {(0, 0): 3021.0, (0, 1): 6042.0, (1, 0): 400.0, (1, 1): 800.0}
