"""Gate list and logic of the test circuit (Phase 6), read from its netlist.

The gate list comes straight from the gate instances (lines `X<name> <inputs..> <output> vdd <type>`)
of netlists/circuit/parity8.cir, so it always matches the simulated circuit. Pins are listed in the
netlist's order: for a NAND2, pin a drives the NMOS at the output and pin b the NMOS at ground,
as in netlists/gates/nand2.cir.

Run `python -m src.circuit` to write data/circuit/gates.csv (one row per gate: name, type, a, b, out).
"""
from __future__ import annotations

import itertools

import pandas as pd

from .config import DATA_DIR, NETLIST_DIR

NETLIST = NETLIST_DIR / "circuit" / "parity8.cir"
INPUTS = [f"in{i}" for i in range(8)]
N_IN = {"inv": 1, "nand2": 2}
LOGIC = {"inv": lambda a: 1 - a[0], "nand2": lambda a: 1 - (a[0] & a[1])}


def gate_list(path=NETLIST) -> pd.DataFrame:
    """Gate instances of the top level, in netlist order: name, type, a, b (empty for INV), out."""
    rows, in_subckt = [], False
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok or tok[0].startswith("*"):
            continue
        if tok[0].lower() == ".subckt":
            in_subckt = True
        elif tok[0].lower() == ".ends":
            in_subckt = False
        elif not in_subckt and tok[0][0] in "Xx":
            kind = tok[-1].lower()
            pins = tok[1:-2]                     # inputs.., output; then the vdd node and the type
            assert tok[-2] == "vdd" and len(pins) == N_IN[kind] + 1, line
            rows.append({"gate": tok[0], "type": kind, "a": pins[0],
                         "b": pins[1] if kind == "nand2" else "", "out": pins[-1]})
    return pd.DataFrame(rows)


def evaluate(gates: pd.DataFrame, inputs: dict[str, int]) -> dict[str, int]:
    """Logic value of every net for one input vector (gates may be listed in any order)."""
    val = dict(inputs)
    todo = list(gates.itertuples(index=False))
    while todo:
        rest = []
        for g in todo:
            pins = [g.a] if g.type == "inv" else [g.a, g.b]
            if all(p in val for p in pins):
                val[g.out] = LOGIC[g.type]([val[p] for p in pins])
            else:
                rest.append(g)
        if len(rest) == len(todo):
            raise ValueError(f"unresolved gates: {[g.gate for g in rest]}")
        todo = rest
    return val


def vectors() -> list[dict[str, int]]:
    """All 256 input vectors; vector k has in_i = bit i of k."""
    return [{f"in{i}": (k >> i) & 1 for i in range(8)} for k in range(256)]


def main() -> None:
    gates = gate_list()
    out = DATA_DIR / "circuit"
    out.mkdir(parents=True, exist_ok=True)
    gates.to_csv(out / "gates.csv", index=False)
    # the circuit must be an 8-bit parity checker: out = 1 when the number of 1s is odd
    for v in vectors():
        assert evaluate(gates, v)["out"] == sum(v.values()) % 2
    print(f"{len(gates)} gates ({gates.type.value_counts().to_dict()}) -> {out / 'gates.csv'}; "
          f"logic checked on all 256 vectors")


if __name__ == "__main__":
    main()
