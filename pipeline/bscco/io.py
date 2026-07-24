"""Load the cleaned IV CSV into a canonical tidy table.

The upstream ``all_cleaned_iv.csv`` is already sweep-split and symmetry-cleaned.
Here we (a) normalise the schema, (b) *recompute* R = V/I rather than trust the
destructively-forced R column, and (c) derive the physical sweep leg / branch so
that switching and retrapping currents can be told apart downstream.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# Canonical tidy schema (column order is part of the contract).
TIDY_COLS = [
    "device", "T_K", "direction", "scan", "sweep_part",
    "sample_order", "I_mA", "V_V", "R_ohm",
    "polarity", "leg", "branch",
]


def _branch(direction: str, sign: int) -> tuple[str, str]:
    """Return (leg, branch) from sweep direction and current sign.

    forward: I ramps +max -> -max ; reverse: I ramps -max -> +max.
    On the leg where |I| is *rising* the junction switches SC->normal (switching Ic);
    on the leg where |I| is *falling* it retraps normal->SC (retrapping Ir).
    """
    pol = "pos" if sign > 0 else "neg"
    rising = (direction == "reverse" and sign > 0) or (direction == "forward" and sign < 0)
    leg = "up" if rising else "down"
    kind = "switch" if rising else "retrap"
    return leg, f"{kind}_{pol}"


def load_tidy(cfg: dict) -> pd.DataFrame:
    root = cfg["_root"]
    raw = pd.read_csv(root / cfg["paths"]["cleaned_csv"])

    df = pd.DataFrame()
    df["device"] = np.full(len(raw), cfg["device"])
    df["T_K"] = raw["temp"].str.replace("K", "", regex=False).astype(float)
    df["direction"] = raw["direction"].astype(str)
    df["scan"] = raw["scan_idx"].astype(int)
    df["sweep_part"] = raw["sweep_part"].astype(str)
    df["I_mA"] = raw["I"].astype(float)
    df["V_V"] = raw["V"].astype(float)

    # Recompute R from I,V; leave inf/nan where I==0 (flagged later, not forced to 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        df["R_ohm"] = df["V_V"] / (df["I_mA"] * 1e-3)  # mA -> A

    sign = np.sign(df["I_mA"]).astype(int).replace(0, 1)
    df["polarity"] = np.where(sign > 0, "pos", "neg")
    legs, branches = zip(*(_branch(d, s) for d, s in zip(df["direction"], sign)))
    df["leg"] = list(legs)
    df["branch"] = list(branches)

    # measurement order within each (T, direction, scan): preserve CSV row order
    df["sample_order"] = raw.index.astype(int)
    df = df.sort_values(["T_K", "direction", "scan", "sample_order"]).reset_index(drop=True)
    return df[TIDY_COLS]


def build_manifest(df: pd.DataFrame) -> pd.DataFrame:
    m = (df.groupby(["T_K", "direction", "scan"])
           .agg(n_points=("I_mA", "size"),
                I_min_mA=("I_mA", "min"), I_max_mA=("I_mA", "max"),
                V_absmax_V=("V_V", lambda s: s.abs().max()))
           .reset_index()
           .sort_values(["T_K", "direction", "scan"]))
    return m


def write_outputs(df: pd.DataFrame, manifest: pd.DataFrame, cfg: dict) -> None:
    root = cfg["_root"]
    (root / cfg["paths"]["out_dir"]).mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / cfg["paths"]["tidy"], index=False)
    manifest.to_csv(root / cfg["paths"]["manifest"], index=False)
