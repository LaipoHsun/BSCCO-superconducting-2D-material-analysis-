"""Phase 1-2: Ic(T) scaling, threshold sensitivity, and BKT done honestly.

Produces three independent estimates of the transition temperature and shows how
much they (dis)agree:
    (i)   Tc from Ic(T) = Ic0 (1 - T/Tc)^n envelope fit
    (ii)  T_BKT from R(T) Halperin-Nelson tail fit (per bias current)
    (iii) T where alpha(T)=3 from IV power law -- and its fit-window sensitivity

Usage: python pipeline/run_phase12.py   (run pipeline/run.py first to build tidy_iv.parquet)
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
from bscco import load_config, metrics, fits  # noqa: E402


def main() -> None:
    cfg = load_config()
    root = cfg["_root"]
    mdir = root / cfg["paths"]["metrics_dir"]
    fdir = root / cfg["paths"]["figures_dir"]
    tidy = pd.read_parquet(root / cfg["paths"]["tidy"])

    # ---- Phase 1: Ic(T) scaling + threshold sensitivity ----
    per_scan = metrics.per_scan_ic(tidy, cfg["ic"]["v_threshold_V"])
    agg = metrics.aggregate_ic(per_scan, cfg)
    br = cfg["ic_scaling"]["fit_branch"]
    ic_br = agg[agg.branch == br].sort_values("T_K")
    fit_lo = ic_br[ic_br.T_K <= cfg["ic_scaling"]["tc_guess_K"] + 5]
    ic_fit = fits.fit_ic_power_law(fit_lo["T_K"], fit_lo["Ic_mA"],
                                   tc_guess=cfg["ic_scaling"]["tc_guess_K"])
    sens = metrics.ic_threshold_sensitivity(tidy, cfg)

    # ---- Phase 2a: R(T) Halperin-Nelson per current ----
    rt_rows = []
    rt_data = {}
    w0, w1 = cfg["rt"]["hn_fit_window_K"]
    for lab, fn in cfg["rt"]["files"].items():
        d = pd.read_excel(root / cfg["rt"]["dir"] / fn)
        d = d[["T_K", "R_Ohm"]].dropna()
        # average duplicate T for a clean curve
        curve = d.groupby("T_K", as_index=False)["R_Ohm"].mean().sort_values("T_K")
        rt_data[lab] = curve
        seg = curve[(curve.T_K >= w0) & (curve.T_K <= w1)]
        if len(seg) > 8:
            hn = fits.fit_halperin_nelson(seg["T_K"], seg["R_Ohm"], tbkt_guess=85.0)
            hn["current"] = lab
            rt_rows.append(hn)
    rt_fits = pd.DataFrame(rt_rows)

    # ---- Phase 2b: BKT from IV, window sensitivity ----
    windows = [tuple(w) for w in cfg["bkt"]["windows_scan"]]
    win_tab = fits.crossing_vs_window(tidy, windows, r2_min=cfg["bkt"]["good_r2"])
    narrow = fits.alpha_table(tidy, windows[0])
    narrow_cross = fits.alpha3_crossing(narrow, r2_min=cfg["bkt"]["good_r2"])

    # ---- save + print consolidated summary ----
    sens.to_csv(mdir / "ic_threshold_sensitivity.csv", index=False)
    rt_fits.to_csv(mdir / "hn_fits.csv", index=False)
    win_tab.to_csv(mdir / "bkt_window_sensitivity.csv", index=False)

    print("=== Phase 1: Ic(T) envelope fit (branch %s) ===" % br)
    print(f"  Tc = {ic_fit['Tc_K']:.1f} K  [{ic_fit.get('Tc_K_lo', float('nan')):.1f}, "
          f"{ic_fit.get('Tc_K_hi', float('nan')):.1f}]   n = {ic_fit['n']:.2f}")
    print("\n=== Phase 2a: Halperin-Nelson T_BKT per current ===")
    for _, r in rt_fits.iterrows():
        print(f"  {r['current']:>7}:  T_BKT = {r['Tbkt_K']:.1f} K "
              f"[{r.get('Tbkt_lo', float('nan')):.1f}, {r.get('Tbkt_hi', float('nan')):.1f}]")
    print("\n=== Phase 2b: alpha=3 crossing vs fit window (the honesty check) ===")
    print(win_tab.to_string(index=False))
    print(f"\n  narrow-window {windows[0]} crossings: {[round(c,1) for c in narrow_cross]}")

    _figure(ic_br, ic_fit, sens, rt_data, rt_fits, narrow, narrow_cross, fdir)
    print(f"\nWrote figure -> {fdir/'phase12_summary.png'}")


def _figure(ic_br, ic_fit, sens, rt_data, rt_fits, narrow, narrow_cross, fdir):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Ic(T) + envelope fit
    ax[0, 0].errorbar(ic_br["T_K"], ic_br["Ic_mA"],
                      yerr=[ic_br["Ic_mA"] - ic_br["Ic_lo"], ic_br["Ic_hi"] - ic_br["Ic_mA"]],
                      fmt="o", ms=4, color="C0", capsize=2, label="switching Ic+")
    Tg = np.linspace(ic_br["T_K"].min(), ic_fit["Tc_K"], 200)
    ax[0, 0].plot(Tg, fits._ic_model(Tg, ic_fit["Ic0_mA"], ic_fit["Tc_K"], ic_fit["n"]),
                  "r-", label=f"$I_c\\propto(1-T/T_c)^n$\n$T_c$={ic_fit['Tc_K']:.1f} K, n={ic_fit['n']:.2f}")
    ax[0, 0].set(xlabel="T (K)", ylabel="Ic (mA)", title="(a) Phase 1: Ic(T) envelope fit")
    ax[0, 0].legend(fontsize=8)

    # (b) threshold sensitivity band
    cols = [c for c in sens.columns if c.startswith("vth_")]
    for c in cols:
        ax[0, 1].plot(sens["T_K"], sens[c], "-", lw=1, label=c.replace("vth_", "Vth="))
    ax[0, 1].fill_between(sens["T_K"], sens[cols].min(axis=1), sens[cols].max(axis=1),
                          color="gray", alpha=.2)
    ax[0, 1].set(xlabel="T (K)", ylabel="Ic+ (mA)",
                 title="(b) Phase 1: Ic depends on the V threshold\n(systematic band)")
    ax[0, 1].legend(fontsize=7)

    # (c) R-T + HN fits
    w0, w1 = 86.0, 95.0
    for lab, curve in rt_data.items():
        ax[1, 0].plot(curve["T_K"], curve["R_Ohm"], ".", ms=2, alpha=.5, label=f"{lab}")
        row = rt_fits[rt_fits.current == lab]
        if len(row):
            r = row.iloc[0]
            Tg = np.linspace(r["Tbkt_K"] + 0.5, w1, 200)
            ax[1, 0].plot(Tg, fits._hn_model(Tg, r["R0"], r["b"], r["Tbkt_K"]), "k-", lw=1)
    ax[1, 0].set(xlabel="T (K)", ylabel="R (Ω)", xlim=(80, 100),
                 title="(c) Phase 2a: R–T Halperin–Nelson fits")
    ax[1, 0].legend(fontsize=7, title="bias")

    # (d) alpha(T) narrow window + method comparison
    good = narrow[narrow.r2 >= 0.9]
    ax[1, 1].scatter(good["T_K"], good["alpha"], color="navy", s=25, label="α (R²≥0.9)")
    ax[1, 1].scatter(narrow[narrow.r2 < 0.9]["T_K"], narrow[narrow.r2 < 0.9]["alpha"],
                     color="lightgray", s=18, label="α (R²<0.9)")
    ax[1, 1].axhline(3, color="crimson", ls="--", label="α=3")
    ax[1, 1].axhline(1, color="k", ls=":")
    for c in narrow_cross:
        ax[1, 1].axvline(c, color="green", ls="-.", lw=1)
    if len(rt_fits):
        tb = rt_fits["Tbkt_K"].mean()
        ax[1, 1].axvspan(rt_fits["Tbkt_K"].min(), rt_fits["Tbkt_K"].max(),
                         color="gold", alpha=.25, label=f"HN T_BKT ~{tb:.0f} K")
    ax[1, 1].axvline(ic_fit["Tc_K"], color="purple", ls="-", lw=1, label=f"Ic-fit Tc={ic_fit['Tc_K']:.0f} K")
    ax[1, 1].set(xlabel="T (K)", ylabel="α", ylim=(0, 9), xlim=(0, 115),
                 title="(d) Phase 2b: three transition estimates")
    ax[1, 1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(fdir / "phase12_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
