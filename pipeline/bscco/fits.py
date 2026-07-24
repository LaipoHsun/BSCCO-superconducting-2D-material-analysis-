"""Model fits with bootstrap uncertainties for Phases 1-2.

  * Ic(T) near Tc:  Ic = Ic0 (1 - T/Tc)^n         (Ginzburg-Landau-like envelope)
  * R(T) BKT tail:  R = R0 exp(-b / sqrt(T - T_BKT))   (Halperin-Nelson)
  * BKT from IV:    alpha(T) power law, alpha=3 crossing, and its window-sensitivity
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ── Ic(T) power-law envelope ──────────────────────────────────────────────
def _ic_model(T, Ic0, Tc, n):
    return Ic0 * np.clip(1 - T / Tc, 0, None) ** n


def fit_ic_power_law(T, Ic, tc_guess=88.0, n_boot=1000, seed=0):
    T, Ic = np.asarray(T, float), np.asarray(Ic, float)
    m = np.isfinite(T) & np.isfinite(Ic) & (Ic > 0)
    T, Ic = T[m], Ic[m]
    p0 = [Ic.max(), tc_guess, 1.5]
    bounds = ([0, T.max() * 0.5, 0.3], [Ic.max() * 5, T.max() * 1.5, 6])
    popt, _ = curve_fit(_ic_model, T, Ic, p0=p0, bounds=bounds, maxfev=20000)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(T), len(T))
        try:
            b, _ = curve_fit(_ic_model, T[idx], Ic[idx], p0=popt, bounds=bounds, maxfev=20000)
            boots.append(b)
        except Exception:
            pass
    boots = np.array(boots)
    names = ["Ic0_mA", "Tc_K", "n"]
    out = {}
    for i, nm in enumerate(names):
        out[nm] = popt[i]
        if len(boots):
            out[nm + "_lo"], out[nm + "_hi"] = np.quantile(boots[:, i], [0.025, 0.975])
    return out


# ── Halperin-Nelson R(T) BKT fit ──────────────────────────────────────────
def _hn_model(T, R0, b, Tbkt):
    return R0 * np.exp(-b / np.sqrt(np.clip(T - Tbkt, 1e-6, None)))


def fit_halperin_nelson(T, R, tbkt_guess=85.0, n_boot=800, seed=0):
    T, R = np.asarray(T, float), np.asarray(R, float)
    m = np.isfinite(T) & np.isfinite(R) & (R > 0)
    T, R = T[m], R[m]
    # fit the rising tail: from just above the foot up to ~normal state
    p0 = [R.max(), 5.0, tbkt_guess]
    bounds = ([R.max() * 1e-3, 0.1, T.min() - 20], [R.max() * 10, 60, T.min() + 5])
    popt, _ = curve_fit(_hn_model, T, R, p0=p0, bounds=bounds, maxfev=40000)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(T), len(T))
        try:
            b, _ = curve_fit(_hn_model, T[idx], R[idx], p0=popt, bounds=bounds, maxfev=40000)
            boots.append(b)
        except Exception:
            pass
    boots = np.array(boots)
    out = dict(R0=popt[0], b=popt[1], Tbkt_K=popt[2], n=len(T))
    if len(boots):
        out["Tbkt_lo"], out["Tbkt_hi"] = np.quantile(boots[:, 2], [0.025, 0.975])
    return out


# ── BKT-from-IV: alpha=3 crossing and its window sensitivity ───────────────
def alpha_table(df: pd.DataFrame, window_V, min_points=8):
    lo, hi = window_V
    rows = []
    for T, g in df.groupby("T_K"):
        aI, aV = g["I_mA"].abs().to_numpy(), g["V_V"].abs().to_numpy()
        sel = (aV > lo) & (aV < hi) & (aI > 0)
        x, y = np.log10(aI[sel]), np.log10(aV[sel])
        if len(x) < min_points:
            rows.append(dict(T_K=T, alpha=np.nan, r2=np.nan, n=len(x)))
            continue
        s, c = np.polyfit(x, y, 1)
        r = y - (s * x + c)
        sstot = np.sum((y - y.mean()) ** 2)
        rows.append(dict(T_K=T, alpha=s, r2=1 - np.sum(r ** 2) / sstot if sstot > 0 else np.nan, n=len(x)))
    return pd.DataFrame(rows).sort_values("T_K")


def alpha3_crossing(tab: pd.DataFrame, r2_min=0.9):
    a = tab[(tab.r2 >= r2_min) & tab.alpha.notna()].sort_values("T_K")
    T, al = a["T_K"].to_numpy(), a["alpha"].to_numpy()
    out = []
    for i in range(len(al) - 1):
        if (al[i] - 3) * (al[i + 1] - 3) < 0:
            out.append(T[i] + (3 - al[i]) * (T[i + 1] - T[i]) / (al[i + 1] - al[i]))
    return out


def crossing_vs_window(df: pd.DataFrame, windows, r2_min=0.9):
    rows = []
    for w in windows:
        cr = alpha3_crossing(alpha_table(df, w), r2_min)
        rows.append(dict(window_V=f"[{w[0]:g}, {w[1]:g}]",
                         crossings_K=", ".join(f"{c:.1f}" for c in cr) or "none"))
    return pd.DataFrame(rows)
