"""Phase 3: threshold-free lineshape analysis.

Describe each up-leg I-V branch by six threshold-free shape features, then use PCA to
find how the device's transport organises with temperature, and a single change-point
search on PC1(T). The change point is a diagnostic only: run_revision_analysis.py
(section F) shows it moves between 27.5 and 77.5 K with the descriptor set, so T* is
taken from the current-based observables instead.

Usage: python pipeline/run_phase3_regimes.py   (after run.py builds tidy_iv.parquet)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bscco import load_config, shape  # noqa: E402


def _gap_statistic(X, kmax=5, seed=0):
    """Pick k by the largest drop in inertia (elbow) with a simple gap heuristic."""
    inertias = []
    for k in range(1, kmax + 1):
        _, _, inertia = shape.kmeans(X, k, seed=seed)
        inertias.append(inertia)
    return np.array(inertias)


def main() -> None:
    cfg = load_config()
    root = cfg["_root"]
    mdir = root / cfg["paths"]["metrics_dir"]
    fdir = root / cfg["paths"]["figures_dir"]
    tidy = pd.read_parquet(root / cfg["paths"]["tidy"])

    per_scan, agg = shape.feature_table(tidy)
    agg = agg.dropna().sort_values("T_K").reset_index(drop=True)
    agg.to_csv(mdir / "shape_features.csv", index=False)

    Z, mu, sd = shape.zscore(agg[shape.FEATURES].to_numpy())
    scores, comps, evr = shape.pca(Z, k=2)
    agg["PC1"] = scores[:, 0]
    agg["PC2"] = scores[:, 1]
    # orient PC1 so it increases with temperature (sign of PCA axes is arbitrary)
    if np.corrcoef(agg["PC1"], agg["T_K"])[0, 1] < 0:
        agg["PC1"] *= -1
        comps[0] *= -1

    inertias = _gap_statistic(Z, kmax=5)   # reported only to show there is NO clean elbow

    # change-points on the ordered PC1(T) trajectory: this is a continuous crossover,
    # not discrete phases, so we locate it by change-point, not by hard clustering.
    cps = shape.change_points_1d(agg["T_K"].to_numpy(), agg["PC1"].to_numpy(), n_cp=2)

    # bootstrap the crossover T* (lower boundary) over scans
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    groups = {T: g.index.to_numpy() for T, g in per_scan.groupby("T_K")}
    boots = []
    for _ in range(300):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in groups.values()])
        samp = per_scan.loc[idx]
        ab = samp.groupby("T_K")[shape.FEATURES].mean().dropna().sort_values("T_K")
        if len(ab) < 6:
            continue
        Zb, *_ = shape.zscore(ab[shape.FEATURES].to_numpy())
        sb, *_ = shape.pca(Zb, k=1)
        cp = shape.change_points_1d(ab.index.to_numpy(), sb[:, 0], n_cp=1)
        if cp:
            boots.append(cp[0])
    boots = np.array(boots)
    tstar = (np.median(boots), *np.quantile(boots, [0.16, 0.84])) if len(boots) else (np.nan,)*3

    agg.to_csv(mdir / "regimes.csv", index=False)

    # ── report ──
    print("=== Phase 3: transport-shape crossover (threshold-free) ===")
    print(f"PCA explained variance: PC1={evr[0]*100:.0f}%  PC2={evr[1]*100:.0f}%  "
          f"(temperatures lie on a 1-D arc -> single latent transport coordinate)")
    print("PC1 loadings (what the shape coordinate means):")
    for f, w in sorted(zip(shape.FEATURES, comps[0]), key=lambda t: -abs(t[1])):
        print(f"    {f:12s} {w:+.2f}")
    print(f"\nPC1(T) change-points: {[round(c,1) for c in cps]} K")
    print(f"PC1 change point, all six descriptors (bootstrap median, 68% CI): "
          f"{tstar[0]:.1f} K  [{tstar[1]:.1f}, {tstar[2]:.1f}]  (n_boot={len(boots)})")
    print("  note: descriptor-dependent (27.5-77.5 K); see run_revision_analysis.py section F")

    _figure(agg, comps, evr, boots, cps, tstar, fdir)
    print(f"\nWrote -> {mdir/'regimes.csv'} and {fdir/'phase3_regimes.png'}")


def _figure(agg, comps, evr, boots, cps, tstar, fdir):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = agg.sort_values("T_K")

    # (a) PCA arc colored by T -> single latent coordinate
    sc = ax[0, 0].scatter(agg["PC1"], agg["PC2"], c=agg["T_K"], cmap="viridis", s=75,
                          edgecolor="k", linewidth=.5)
    for _, r in agg.iterrows():
        ax[0, 0].annotate(f"{r['T_K']:.0f}", (r["PC1"], r["PC2"]), fontsize=6,
                          xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax[0, 0], label="T (K)")
    ax[0, 0].set(xlabel=f"PC1 ({evr[0]*100:.0f}%)", ylabel=f"PC2 ({evr[1]*100:.0f}%)",
                 title="(a) IV-shape PCA: temperatures lie on a single 1-D arc")

    # (b) PC1(T) trajectory + bootstrapped crossover
    ax[0, 1].plot(a["T_K"], a["PC1"], "o-", color="slateblue")
    if np.isfinite(tstar[0]):
        ax[0, 1].axvspan(tstar[1], tstar[2], color="gold", alpha=.3,
                         label=f"change point {tstar[0]:.0f} K  [{tstar[1]:.0f}, {tstar[2]:.0f}] (68%)")
        ax[0, 1].axvline(tstar[0], color="darkorange", lw=1.5)
    ax[0, 1].set(xlabel="T (K)", ylabel="PC1 (transport-shape coordinate)",
                 title="(b) PC1(T) change point, all six descriptors\n"
                       "(descriptor-dependent; not used as T*)")
    ax[0, 1].legend(fontsize=9)

    # (c) concordance: independent raw descriptors all inflect near T*
    def nz(s):
        s = (s - s.min()) / (s.max() - s.min())
        return s
    ax[1, 0].plot(a["T_K"], nz(a["sharp_width"]), "o-", label="switch width (soft→)")
    ax[1, 0].plot(a["T_K"], nz(a["foot_extent"]), "s-", label="flux-flow foot")
    ax[1, 0].plot(a["T_K"], nz(1 - a["Vmax_frac"]), "^-", label="1 − Vmax/rail")
    ax[1, 0].plot(a["T_K"], nz(1 - a["sc_fraction"]), "d-", label="dissipative fraction")
    if np.isfinite(tstar[0]):
        ax[1, 0].axvspan(tstar[1], tstar[2], color="gold", alpha=.3)
    ax[1, 0].set(xlabel="T (K)", ylabel="normalised feature (0–1)",
                 title="(c) raw descriptors vs T")
    ax[1, 0].legend(fontsize=8)

    # (d) bootstrap distribution of T*
    if len(boots):
        ax[1, 1].hist(boots, bins=np.arange(0, 90, 5), color="teal", alpha=.75,
                      edgecolor="k")
        ax[1, 1].axvline(tstar[0], color="darkorange", lw=2, label=f"median {tstar[0]:.0f} K")
    ax[1, 1].set(xlabel="bootstrap change point (K)", ylabel="count",
                 title="(d) bootstrap over sweeps (sampling noise only)")
    ax[1, 1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fdir / "phase3_regimes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
