"""Quality-control flags. These ADD boolean columns; they never drop or edit rows.

Downstream analysis decides which flags to filter on, so every cut is explicit
and reversible instead of being baked destructively into the data file.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FLAG_COLS = ["flag_rail", "flag_noise", "flag_zeroI", "flag_R_implausible"]


def add_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    q = cfg["qc"]
    df = df.copy()
    absV = df["V_V"].abs()
    df["flag_rail"] = absV >= q["rail_tol_V"]                    # hit compliance rail
    df["flag_noise"] = absV < q["noise_floor_V"]                # indistinguishable from baseline
    df["flag_zeroI"] = df["I_mA"].abs() < cfg["physics"]["current_step_mA"] / 2
    R = df["R_ohm"].abs()
    df["flag_R_implausible"] = (~np.isfinite(df["R_ohm"])) | (R < q["r_valid_min"]) | (R > q["r_valid_max"])
    return df


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df[FLAG_COLS].mean()
            .rename("fraction_flagged").to_frame()
            .assign(n_flagged=df[FLAG_COLS].sum().astype(int)))
