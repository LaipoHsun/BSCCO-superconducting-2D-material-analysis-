"""Threshold-free I-V curve-shape descriptors + unsupervised regime discovery.

Motivation: a single fixed voltage threshold samples different physical features at
different temperatures (e.g. the low-V flux-flow "foot" at ~35-55 K), producing
non-physical jumps in Ic(T). Instead we describe each up-leg (switching branch) by a
handful of SELF-NORMALISED shape features, then let PCA + clustering find the
temperature regimes objectively. No absolute threshold enters the descriptors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COMPLIANCE_V = 0.4
NOISE_V = 1.5e-4


def _upleg(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (|I|, |V|) of one up-leg (rising |I|) branch, sorted, de-duplicated."""
    g = g.reindex(g["I_mA"].abs().sort_values().index)
    x = g["I_mA"].abs().to_numpy()
    y = g["V_V"].abs().to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _i_at_fraction(x, y, frac):
    """|I| where |V| first reaches frac*max(|V|)  (self-normalised, threshold-free)."""
    vmax = y.max()
    if vmax <= NOISE_V:
        return np.nan
    thr = frac * vmax
    idx = np.where(y >= thr)[0]
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return x[0]
    if y[i] == y[i - 1]:
        return x[i]
    return x[i - 1] + (thr - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])


def descriptors(g: pd.DataFrame) -> dict:
    """Shape features for one (T, scan) up-leg branch."""
    x, y = _upleg(g)
    if len(x) < 8 or y.max() <= NOISE_V:
        return {}
    vmax = y.max()
    i10, i50, i90 = (_i_at_fraction(x, y, f) for f in (0.1, 0.5, 0.9))
    imax = x.max()
    # differential resistance on the top (resistive) quarter of the branch
    top = x >= np.quantile(x, 0.75)
    dR = np.polyfit(x[top], y[top], 1)[0] if top.sum() >= 3 else np.nan
    return dict(
        I_switch=i50,                                   # robust switch current (50% of own Vmax)
        Vmax_frac=vmax / COMPLIANCE_V,                  # did it reach the rail? (0..1)
        sharp_width=(i90 - i10) / i50 if i50 else np.nan,  # 10->90% rise width / I_switch (small=sharp)
        sc_fraction=np.mean(y < 5 * NOISE_V),           # dissipationless fraction of the sweep
        foot_extent=np.mean((y > 1e-3) & (y < 0.1 * vmax)),  # low-V flux-flow foot fraction
        dR_top=dR,                                       # differential R near the top of the branch
        reach_frac=imax / (i50 if i50 else np.nan),      # how far past switch the sweep went
    )


FEATURES = ["I_switch", "Vmax_frac", "sharp_width", "sc_fraction", "foot_extent", "dR_top"]


def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-temperature mean feature vector (averaged over scans and up-leg branches)."""
    rows = []
    up = df[df["leg"] == "up"]
    for (T, scan, branch), g in up.groupby(["T_K", "scan", "branch"]):
        d = descriptors(g)
        if d:
            d.update(T_K=T, scan=scan, branch=branch)
            rows.append(d)
    per_scan = pd.DataFrame(rows)
    agg = per_scan.groupby("T_K")[FEATURES].mean().reset_index()
    return per_scan, agg


# ── minimal PCA + k-means (numpy only) ────────────────────────────────────
def zscore(M):
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1
    return (M - mu) / sd, mu, sd


def pca(Z, k=2):
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    scores = (Z - Z.mean(0)) @ Vt[:k].T
    evr = (S ** 2 / np.sum(S ** 2))[:k]
    return scores, Vt[:k], evr


def kmeans(X, k, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = np.linalg.norm(X[:, None] - C[None], axis=2)
        lab = d.argmin(1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    inertia = np.sum((X - C[lab]) ** 2)
    return lab, C, inertia


def change_points_1d(t, v, n_cp=1):
    """Locate n_cp change point(s) in v(t) by minimising within-segment SSE (ordered by t)."""
    order = np.argsort(t)
    t, v = np.asarray(t)[order], np.asarray(v)[order]

    def sse(a):
        return np.sum((a - a.mean()) ** 2) if len(a) else 0.0

    def best_split(lo, hi):
        best, bi = np.inf, None
        for i in range(lo + 2, hi - 1):
            c = sse(v[lo:i]) + sse(v[i:hi])
            if c < best:
                best, bi = c, i
        return bi
    bounds = [0, len(v)]
    for _ in range(n_cp):
        cand = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            bi = best_split(a, b)
            if bi is not None:
                cand.append((sse(v[a:b]) - (sse(v[a:bi]) + sse(v[bi:b])), bi))
        if not cand:
            break
        bounds.append(max(cand)[1])
        bounds = sorted(set(bounds))
    cps = [(t[b - 1] + t[b]) / 2 for b in bounds[1:-1]]
    return cps
