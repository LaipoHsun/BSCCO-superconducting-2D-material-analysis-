"""Phase 3b: is the ~50 K switch-retrap hysteresis real, or a dIc/dT drift artifact?

The four currents come from two files (forward, reverse). A genuine switch-retrap
hysteresis is SYMMETRIC in polarity (hyst+ ~ hyst-); an inter-file temperature-drift
artifact is ANTISYMMETRIC (hyst+ ~ -hyst-) and scales with |dIc/dT|. Decomposing into
symmetric/antisymmetric parts therefore separates intrinsic hysteresis from drift.

    S+/-  = switching current (up-leg), R+/- = retrapping current (down-leg)
    h(T)  = 1/2[(S+-R+)+(S--R-)]        intrinsic hysteresis (symmetric)
    d(T)  = 1/2[(S+-R+)-(S--R-)]        drift proxy (antisymmetric) ~ Ic'(T)*delta
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


def main() -> None:
    cfg = load_config()
    root = cfg["_root"]
    fdir = root / cfg["paths"]["figures_dir"]
    mdir = root / cfg["paths"]["metrics_dir"]
    tidy = pd.read_parquet(root / cfg["paths"]["tidy"])

    per_scan = metrics.per_scan_ic(tidy, cfg["ic"]["v_threshold_V"])
    # mean current per (T, branch)
    piv = per_scan.groupby(["T_K", "branch"])["Ic_mA"].mean().unstack("branch")
    for c in ["switch_pos", "switch_neg", "retrap_pos", "retrap_neg"]:
        if c not in piv:
            piv[c] = np.nan
    piv = piv.sort_index()
    T = piv.index.to_numpy()

    Sp, Sn = piv["switch_pos"].to_numpy(), piv["switch_neg"].to_numpy()
    Rp, Rn = piv["retrap_pos"].to_numpy(), piv["retrap_neg"].to_numpy()
    hp, hn = Sp - Rp, Sn - Rn                       # per-polarity hysteresis (abs current)
    h_sym = 0.5 * (hp + hn)                          # intrinsic hysteresis
    d_anti = 0.5 * (hp - hn)                         # drift proxy

    # |dIc/dT| from a smoothed switching current (average both polarities)
    Sc = np.nanmean(np.vstack([Sp, Sn]), axis=0)
    dIcdT = np.gradient(Sc, T)

    # normalise to compare shapes
    def unit(a):
        a = np.abs(a); m = np.nanmax(a)
        return a / m if m > 0 else a

    # correlation of the drift proxy with |dIc/dT| (artifact signature)
    mask = np.isfinite(d_anti) & np.isfinite(dIcdT)
    r_drift = np.corrcoef(np.abs(d_anti[mask]), np.abs(dIcdT[mask]))[0, 1]
    # correlation of the intrinsic (symmetric) part with |dIc/dT|
    mask2 = np.isfinite(h_sym) & np.isfinite(dIcdT)
    r_sym = np.corrcoef(np.abs(h_sym[mask2]), np.abs(dIcdT[mask2]))[0, 1]

    out = pd.DataFrame(dict(T_K=T, S_switch=Sc, hyst_intrinsic=h_sym, drift_proxy=d_anti,
                            dIc_dT=dIcdT))
    out.to_csv(mdir / "hysteresis_confound.csv", index=False)

    # implied inter-file drift delta(T) = d_anti / Ic'(T)   (Kelvin)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = d_anti / dIcdT
    med_delta = np.nanmedian(np.abs(delta[(T > 20) & (T < 70)]))

    print("=== Phase 3b: hysteresis confound test ===")
    print(f"corr(|drift proxy|, |dIc/dT|)      = {r_drift:+.2f}   (high => antisym part IS the drift)")
    print(f"corr(|intrinsic hyst|, |dIc/dT|)   = {r_sym:+.2f}   (high => 'hysteresis' just tracks the slope)")
    peakT = T[np.nanargmax(np.abs(h_sym))]
    print(f"intrinsic (symmetric) hysteresis peaks at T = {peakT:.0f} K, "
          f"magnitude {np.nanmax(np.abs(h_sym))*1e3:.1f} uA")
    frac_sym = np.nansum(np.abs(h_sym)) / (np.nansum(np.abs(h_sym)) + np.nansum(np.abs(d_anti)))
    print(f"share of total hysteresis carried by the SYMMETRIC (intrinsic) part: {frac_sym*100:.0f}%")
    print(f"implied inter-file drift |delta| (20-70 K median): {med_delta:.1f} K")

    _fig(T, Sp, Sn, Rp, Rn, h_sym, d_anti, dIcdT, unit, fdir)
    print(f"\nWrote -> {mdir/'hysteresis_confound.csv'} and {fdir/'phase3b_confound.png'}")


def _fig(T, Sp, Sn, Rp, Rn, h_sym, d_anti, dIcdT, unit, fdir):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    ax[0].plot(T, Sp * 1e3, "o-", color="C0", label="switch +")
    ax[0].plot(T, Sn * 1e3, "o--", color="C0", alpha=.6, label="switch −")
    ax[0].plot(T, Rp * 1e3, "s-", color="C3", label="retrap +")
    ax[0].plot(T, Rn * 1e3, "s--", color="C3", alpha=.6, label="retrap −")
    ax[0].set(xlabel="T (K)", ylabel="|I| (µA)", title="(a) switching vs retrapping, both polarities")
    ax[0].legend(fontsize=8)

    ax[1].plot(T, h_sym * 1e3, "o-", color="darkgreen", label="intrinsic (symmetric)")
    ax[1].plot(T, d_anti * 1e3, "^-", color="crimson", label="drift proxy (antisymmetric)")
    ax[1].axhline(0, color="gray", lw=.8)
    ax[1].set(xlabel="T (K)", ylabel="hysteresis (µA)",
              title="(b) intrinsic hysteresis vs drift artifact")
    ax[1].legend(fontsize=8)

    ax[2].plot(T, unit(h_sym), "o-", color="darkgreen", label="|intrinsic hyst| (norm)")
    ax[2].plot(T, unit(d_anti), "^-", color="crimson", label="|drift proxy| (norm)")
    ax[2].plot(T, unit(dIcdT), "d:", color="black", label="|dIc/dT| (norm)")
    ax[2].set(xlabel="T (K)", ylabel="normalised", title="(c) does each track |dIc/dT| ?")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fdir / "phase3b_confound.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
