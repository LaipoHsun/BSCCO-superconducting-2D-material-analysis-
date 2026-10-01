"""Phase 4: assemble the paper's central result -- multiple INDEPENDENT observables
all mark the same crossover T* ~ 0.55 Tc in a single hBN-BSCCO weak link.

Pulls the per-phase metric tables together, builds the master concordance figure,
and writes a concordance table used for the manuscript.

Usage: python pipeline/run_phase4_concordance.py   (after run.py / run_phase12 / phase3 / phase3b)
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
from bscco import load_config, metrics  # noqa: E402

TC = 86.7           # Tc from Halperin-Nelson (Phase 2)
# T* band from the current-based observables (peaks at 45 and 50 K, +/- half grid step).
# The PCA change point is descriptor-dependent and is not used for T*.
TSTAR = (47.5, 42.5, 52.5)   # mid, lo, hi


def _norm(s):
    s = np.asarray(s, float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    return (s - lo) / (hi - lo) if hi > lo else s


def main() -> None:
    cfg = load_config()
    root = cfg["_root"]
    mdir = root / cfg["paths"]["metrics_dir"]
    fdir = root / cfg["paths"]["figures_dir"]
    tidy = pd.read_parquet(root / cfg["paths"]["tidy"])

    # switching / retrapping currents
    ps = metrics.per_scan_ic(tidy, cfg["ic"]["v_threshold_V"])
    piv = ps.groupby(["T_K", "branch"])["Ic_mA"].mean().unstack("branch")
    piv["Ic"] = piv[["switch_pos", "switch_neg"]].mean(axis=1)
    piv["Ir"] = piv[["retrap_pos", "retrap_neg"]].mean(axis=1)
    piv["IrIc"] = piv["Ir"] / piv["Ic"]
    piv = piv.reset_index()

    # other observables
    regimes = pd.read_csv(mdir / "regimes.csv")[["T_K", "PC1"]]
    asym = pd.read_csv(mdir / "asymmetries.csv")[["T_K", "eta_switch"]]
    shape = pd.read_csv(mdir / "shape_features.csv")[["T_K", "foot_extent", "sharp_width"]]

    C = (piv[["T_K", "Ic", "Ir", "IrIc"]]
         .merge(regimes, on="T_K", how="left")
         .merge(asym, on="T_K", how="left")
         .merge(shape, on="T_K", how="left")
         .sort_values("T_K"))
    C["hyst"] = 1 - C["IrIc"]
    C.to_csv(mdir / "concordance_table.csv", index=False)

    # report
    print("=== Phase 4: concordance summary ===")
    def peakT(col, kind="max"):
        s = C.dropna(subset=[col])
        i = s[col].idxmax() if kind == "max" else s[col].idxmin()
        return s.loc[i, "T_K"]
    print(f"Tc (Halperin-Nelson)          : {TC:.1f} K")
    print(f"hysteresis (1-Ir/Ic) maximum  : {peakT('hyst'):.0f} K")
    print(f"Ir/Ic minimum                 : {peakT('IrIc','min'):.0f} K  (={C['IrIc'].min():.2f})")
    print(f"nonreciprocity |eta| maximum  : {peakT('eta_switch') if C['eta_switch'].notna().any() else float('nan'):.0f} K")
    print(f"T* band (observables)         : {TSTAR[1]:.1f}-{TSTAR[2]:.1f} K,  T*/Tc = {TSTAR[0]/TC:.2f}")

    _master_figure(C, fdir)
    print(f"\nWrote -> {mdir/'concordance_table.csv'} and {fdir/'phase4_concordance.png'}")


def _master_figure(C, fdir):
    T = C["T_K"].to_numpy()
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.28], hspace=0.42, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_d = fig.add_subplot(gs[2, :])

    def tstar_band(ax):
        ax.axvspan(TSTAR[1], TSTAR[2], color="gold", alpha=.25, zorder=0)
        ax.axvline(TC, color="firebrick", ls="--", lw=1, zorder=0)

    # (a) Ic and Ir vs T
    tstar_band(ax_a)
    ax_a.plot(T, C["Ic"] * 1e3, "o-", color="C0", ms=4, label="switching $I_c$")
    ax_a.plot(T, C["Ir"] * 1e3, "s--", color="C3", ms=4, label="retrapping $I_r$")
    ax_a.annotate("$T_c$", (TC, ax_a.get_ylim()[1] * 0.9), color="firebrick", fontsize=9)
    ax_a.set(xlabel="T (K)", ylabel="current (µA)", title="(a) supercurrent envelope")
    ax_a.legend(fontsize=8)

    # (b) Ir/Ic ratio with its minimum
    tstar_band(ax_b)
    ax_b.plot(T, C["IrIc"], "o-", color="purple", ms=4)
    imin = C["IrIc"].idxmin()
    ax_b.plot(C.loc[imin, "T_K"], C.loc[imin, "IrIc"], "*", color="red", ms=16,
              label=f"min {C['IrIc'].min():.2f} @ {C.loc[imin,'T_K']:.0f} K")
    ax_b.axhline(1, color="gray", ls=":", lw=1)
    ax_b.set(xlabel="T (K)", ylabel="$I_r / I_c$",
             title="(b) hysteresis strength peaks at T*")
    ax_b.legend(fontsize=8)

    # (c) concordance overlay -- RATE/PEAK observables that genuinely mark T*
    tstar_band(ax_c)
    def smooth(a, w=3):
        k = np.ones(w) / w
        return np.convolve(a, k, mode="same")
    dIcdT = np.abs(np.gradient(C["Ic"].to_numpy(), T))
    dPC1dT = np.abs(np.gradient(smooth(C["PC1"].to_numpy()), T))
    dPC1dT[:1] = dPC1dT[-1:] = np.nan   # drop derivative edge artifacts
    ax_c.plot(T, _norm(dIcdT), "v-", label="supercurrent collapse |d$I_c$/dT|")
    ax_c.plot(T, _norm(C["hyst"]), "s-", label="hysteresis $1-I_r/I_c$")
    if C["eta_switch"].notna().any():
        ax_c.plot(T, _norm(np.abs(C["eta_switch"])), "^-", label="nonreciprocity |η|")
    ax_c.plot(T, _norm(dPC1dT), "o--", alpha=.5, label="lineshape change-rate |dPC1/dT| (ref.)")
    ax_c.text(TSTAR[0], 1.03, "T*", color="darkorange", ha="center", fontsize=9)
    ax_c.text(TC, 1.03, f"$T_c$ = {TC:.0f} K", color="firebrick", ha="center", fontsize=9)
    ax_c.set(xlabel="T (K)", ylabel="normalised (0–1)",
             title="(c) three current-based observables peak together at T*", ylim=(-0.05, 1.15))
    ax_c.legend(fontsize=8, ncol=2, loc="upper right")

    # (d) regime map bar
    ax_d.set_xlim(T.min(), T.max()); ax_d.set_ylim(0, 1)
    ax_d.axvspan(T.min(), TSTAR[1], color="C0", alpha=.18)
    ax_d.axvspan(TSTAR[1], 60, color="gold", alpha=.28)
    ax_d.axvspan(60, TC, color="C2", alpha=.18)
    ax_d.axvspan(TC, T.max(), color="lightgray", alpha=.5)
    for x, lab in [(25, "hysteretic\nself-heated switching"), (49, "crossover\nT*"),
                   (73, "overdamped\nflux-flow"), (100, "normal")]:
        ax_d.text(x, 0.5, lab, ha="center", va="center", fontsize=8)
    ax_d.axvline(TC, color="firebrick", ls="--", lw=1)
    ax_d.set(yticks=[], xlabel="T (K)", title="(d) proposed regime map")

    fig.savefig(fdir / "phase4_concordance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
