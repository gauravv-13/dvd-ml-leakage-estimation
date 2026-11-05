"""Load config.yaml and the nominal process parameters from the model card."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
NETLIST_DIR = ROOT / "netlists"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SAVED_MODELS_DIR = ROOT / "saved_models"

# The four single-transistor cases. Gate/body voltages per spec section 4:
#   "vdd" means the sample's VDD, 0 means ground.
CASES = {
    "nmos_on":  {"dev": "nmos", "vg": "vdd", "vb": 0},
    "nmos_off": {"dev": "nmos", "vg": 0,     "vb": 0},
    "pmos_on":  {"dev": "pmos", "vg": 0,     "vb": "vdd"},
    "pmos_off": {"dev": "pmos", "vg": "vdd", "vb": "vdd"},
}

# Column order required by deliverable 1.
CSV_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
               "vth0", "toxe", "ndep", "u0", "xj",
               "id", "ig", "is", "ib"]


@dataclass
class Config:
    raw: dict
    model_path: Path
    lmin: float
    vdd_nominal: float
    nominal: dict = field(default_factory=dict)  # {"nmos": {...}, "pmos": {...}}

    def __getitem__(self, key):
        return self.raw[key]


def parse_model_card(path: Path, params: list[str]) -> dict[str, dict[str, float]]:
    """Return {"nmos": {param: value}, "pmos": {...}} for the requested params."""
    text = path.read_text()
    blocks = re.split(r"(?im)^\s*\.model\s+", text)[1:]
    out: dict[str, dict[str, float]] = {}
    for block in blocks:
        name = block.split()[0].lower()
        values = {k.lower(): float(v) for k, v in
                  re.findall(r"(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", block)}
        missing = [p for p in params if p not in values]
        if missing:
            raise ValueError(f"{path.name}: model '{name}' lacks {missing}")
        out[name] = {p: values[p] for p in params}
    if set(out) != {"nmos", "pmos"}:
        raise ValueError(f"{path.name}: expected models 'nmos' and 'pmos', found {list(out)}")
    return out


def load_config(path: Path | None = None) -> Config:
    raw = yaml.safe_load((path or ROOT / "config.yaml").read_text())
    a = raw["assignment"]
    if not a.get("confirmed", False):
        warnings.warn("config.yaml: assignment.confirmed is false -- running with placeholder "
                      f"model={a['model_file']}, VDD={a['vdd_nominal']}, circuit={a['test_circuit']}",
                      stacklevel=2)
    model_path = MODELS_DIR / a["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    lmin = raw["technology"]["lmin"]
    if lmin == "auto":
        m = re.match(r"(\d+)nm", model_path.name)
        if not m:
            raise ValueError(f"cannot infer Lmin from {model_path.name}; set technology.lmin")
        lmin = int(m.group(1)) * 1e-9
    lmin = float(lmin)

    return Config(
        raw=raw,
        model_path=model_path,
        lmin=lmin,
        vdd_nominal=float(a["vdd_nominal"]),
        nominal=parse_model_card(model_path, raw["sampling"]["process_params"]),
    )
