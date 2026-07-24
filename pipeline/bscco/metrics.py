"""Physical metrics with honest uncertainties.

Critical currents are split into four branches (switching vs retrapping) x (pos/neg),
which lets us separate the two asymmetries that both live in these sweeps:

    nonreciprocity (true diode):  Ic+  vs  |Ic-|      (same leg, opposite polarity)
    switch/retrap hysteresis:     Ic   vs   Ir        (same polarity, opposite leg)

All aggregate quantities carry a bootstrap CI over repeated scans, plus a
threshold-sensitivity spread as a systematic band.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── low-level: interpolated critical current ──────────────────────────────
def ic_crossing(absI: np.ndarray, absV: np.ndarray, vth: float) -> float:
    """|I| at which |V| first crosses vth, scanning ascending |I| (linear interp)."""
    order = np.argsort(absI)
    x, y = absI[order], absV[order]
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.nan
    above = np.where(y >= vth)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (vth - y0) * (x1 - x0) / (y1 - y0))


# ── per-scan critical currents ────────────────────────────────────────────
def per_scan_ic(df: pd.DataFrame, vth: float) -> pd.DataFrame:
    rows = []
    for (T, direction, scan), g in df.groupby(["T_K", "direction", "scan"]):
        for branch, gb in g.groupby("branch"):
            ic = ic_crossing(gb["I_mA"].abs().to_numpy(), gb["V_V"].abs().to_numpy(), vth)
            rows.append(dict(T_K=T, branch=branch, scan=scan, Ic_mA=ic))
    return pd.DataFrame(rows)


def _boot_ci(vals: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator):
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    if len(vals) == 1:
        return float(vals[0]), np.nan, np.nan
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)


def aggregate_ic(per_scan: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = cfg["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    out = []
    for (T, branch), g in per_scan.groupby(["T_K", "branch"]):
        mean, lo, hi = _boot_ci(g["Ic_mA"].to_numpy(), b["n_resamples"], b["ci"], rng)
        out.append(dict(T_K=T, branch=branch, Ic_mA=mean, Ic_lo=lo, Ic_hi=hi,
                        Ic_std=g["Ic_mA"].std(), n_scan=g["Ic_mA"].notna().sum()))
    return pd.DataFrame(out)


# ── derived asymmetries (bootstrap over scans jointly) ────────────────────
def _pick(g: pd.DataFrame, branch: str, rng, n) -> np.ndarray:
    v = g.loc[g.branch == branch, "Ic_mA"].to_numpy()
    v = v[np.isfinite(v)]
    return rng.choice(v, size=n, replace=True) if len(v) else np.full(n, np.nan)


def derived_asymmetries(per_scan: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    b = cfg["bootstrap"]
    rng = np.random.default_rng(b["seed"] + 1)
    nb = b["n_resamples"]
    lohi = [(1 - b["ci"]) / 2, 1 - (1 - b["ci"]) / 2]
    rows = []
    for T, g in per_scan.groupby("T_K"):
        sp = _pick(g, "switch_pos", rng, nb); sn = _pick(g, "switch_neg", rng, nb)
        rp = _pick(g, "retrap_pos", rng, nb); rn = _pick(g, "retrap_neg", rng, nb)
        eta_sw = (sp - sn) / (sp + sn)                      # diode nonreciprocity (switching)
        hyst_p = (sp - rp) / (sp + rp)                      # switch/retrap hysteresis (+)
        hyst_n = (sn - rn) / (sn + rn)                      # switch/retrap hysteresis (-)
        rec = dict(T_K=T)
        for name, arr in [("eta_switch", eta_sw), ("hyst_pos", hyst_p), ("hyst_neg", hyst_n)]:
            a = arr[np.isfinite(arr)]
            if len(a):
                rec[name] = float(np.mean(a))
                rec[name + "_lo"], rec[name + "_hi"] = map(float, np.quantile(a, lohi))
            else:
                rec[name] = rec[name + "_lo"] = rec[name + "_hi"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("T_K")


def ic_threshold_sensitivity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mean switching Ic+ vs several voltage thresholds -> systematic band."""
    out = []
    for vth in cfg["ic"]["v_threshold_scan"]:
        ps = per_scan_ic(df, vth)
        m = ps[ps.branch == "switch_pos"].groupby("T_K")["Ic_mA"].mean()
        out.append(m.rename(f"vth_{vth:g}"))
    return pd.concat(out, axis=1).reset_index()


# ── fluctuation exponent alpha(T): windowed, gated power law ───────────────
def _runs_z(resid: np.ndarray) -> float:
    s = np.sign(resid - np.median(resid)); s[s == 0] = 1
    runs = 1 + np.sum(s[:-1] != s[1:])
    npos, nneg = np.sum(s > 0), np.sum(s < 0); ntot = npos + nneg
    if ntot < 2:
        return np.nan
    mu = 2 * npos * nneg / ntot + 1
    var = (2 * npos * nneg * (2 * npos * nneg - ntot)) / (ntot ** 2 * (ntot - 1))
    return (runs - mu) / np.sqrt(var) if var > 0 else np.nan


def alpha_of_T(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    lo, hi = cfg["bkt"]["window_V"]
    rows = []
    for T, g in df.groupby("T_K"):
        absI, absV = g["I_mA"].abs().to_numpy(), g["V_V"].abs().to_numpy()
        m = (absV > lo) & (absV < hi) & (absI > 0)
        x, y = np.log10(absI[m]), np.log10(absV[m])
        if len(x) < cfg["bkt"]["min_points"]:
            rows.append(dict(T_K=T, alpha=np.nan, r2=np.nan, z_runs=np.nan, n=len(x), acceptable=False))
            continue
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
        z = _runs_z(resid)
        # bootstrap SE on the slope
        rng = np.random.default_rng(cfg["bootstrap"]["seed"] + int(T))
        idx = rng.integers(0, len(x), size=(cfg["bootstrap"]["n_resamples"], len(x)))
        slopes = np.array([np.polyfit(x[i], y[i], 1)[0] for i in idx])
        # Gate on R^2 only. The runs-test z is reported as a diagnostic but NOT used as a
        # hard cut: its power grows with n, so at ~10^2 points it rejects independence for
        # even negligible curvature. |z| large simply flags "not a *pure* power law".
        acceptable = bool(r2 >= cfg["bkt"]["good_r2"])
        rows.append(dict(T_K=T, alpha=slope, alpha_se=slopes.std(), r2=r2,
                         z_runs=z, n=len(x), acceptable=acceptable))
    return pd.DataFrame(rows).sort_values("T_K")
