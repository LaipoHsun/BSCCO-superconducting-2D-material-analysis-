"""Clean, config-driven analysis pipeline for the hBN-BSCCO IV dataset.

Modules
-------
io       load the cleaned CSV into a canonical tidy table (+ manifest)
qc       add boolean quality flags (never deletes or overwrites data)
metrics  Ic / retrapping / nonreciprocity / hysteresis / alpha(T) with bootstrap CIs

Design rules
------------
* One config.yaml holds every tunable constant.
* QC marks; it does not mutate. Downstream code filters on flags explicitly.
* Physical leg is derived, not trusted from upstream labels:
      forward sweep ramps  I: +max -> -max   reverse sweep ramps  I: -max -> +max
      => leg(up/down) and branch are a deterministic function of (direction, sign I).
"""
from pathlib import Path
import yaml

__all__ = ["load_config", "PROJECT_ROOT"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "pipeline" / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = PROJECT_ROOT
    return cfg
