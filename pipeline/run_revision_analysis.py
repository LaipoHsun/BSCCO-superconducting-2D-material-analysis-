"""Referee-response analysis (major revision).

Addresses, quantitatively:
  R2/R4  joint bootstrap of the four observables -> per-peak uncertainties,
         peak-location covariance/correlation, pairwise differences;
         plus correlation matrix of the observable curves themselves.
  R5     V_th sensitivity of the Ir/Ic minimum (0.3-5 mV).
  R8     Halperin-Nelson systematics: fit-window grid, residuals, (b, T_BKT)
         bootstrap correlation.
  R9     synthetic test of the polarity symmetric/antisymmetric drift
         decomposition, incl. an adversarial intrinsic polarity asymmetry.
  R7/+   thermal (hot-spot) power-balance calibration: implied thermal
         conductance G(T) from measured (Ir, Vr), forward prediction of Ir/Ic;
         quantitative RCSJ comparison (implied capacitance / predicted Ir/Ic).

Outputs -> pipeline/outputs/metrics/*.csv and paths.paper_figures_dir/fig5,6.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bscco import load_config, metrics, shape  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.8, "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

HBAR, E = 1.0546e-34, 1.602e-19
TC = 86.7


def smooth3(x):
    return np.convolve(x, np.ones(3) / 3, mode="same")


# ── A. joint bootstrap of the four peak locations ─────────────────────────
def joint_peak_bootstrap(per_scan_ic, per_scan_feat, n_boot, seed):
    rng = np.random.default_rng(seed)
    ic_groups = {k: g["Ic_mA"].to_numpy()
                 for k, g in per_scan_ic.groupby(["T_K", "branch"])}
    feat_groups = {T: g[shape.FEATURES].to_numpy()
                   for T, g in per_scan_feat.groupby("T_K")}
    temps = np.array(sorted(per_scan_feat["T_K"].unique()))
    mlow = temps <= 80

    peaks = []
    for _ in range(n_boot):
        Ic = np.full(len(temps), np.nan); Ir = np.full(len(temps), np.nan)
        eta = np.full(len(temps), np.nan)
        F = np.full((len(temps), len(shape.FEATURES)), np.nan)
        for i, T in enumerate(temps):
            def draw(branch):
                v = ic_groups.get((T, branch))
                if v is None or not len(v):
                    return np.nan
                return rng.choice(v, len(v), replace=True).mean()
            sp, sn = draw("switch_pos"), draw("switch_neg")
            rp, rn = draw("retrap_pos"), draw("retrap_neg")
            Ic[i] = np.nanmean([sp, sn]); Ir[i] = np.nanmean([rp, rn])
            eta[i] = (sp - sn) / (sp + sn) if np.isfinite(sp + sn) else np.nan
            fg = feat_groups[T]
            F[i] = fg[rng.integers(0, len(fg), len(fg))].mean(axis=0)

        dIc = np.abs(np.gradient(Ic, temps))
        irc = Ir / Ic
        Z, *_ = shape.zscore(F)
        pc1 = shape.pca(Z, k=1)[0][:, 0]
        if np.corrcoef(pc1, temps)[0, 1] < 0:
            pc1 = -pc1
        dpc = np.abs(np.gradient(smooth3(pc1), temps))
        dpc[:2] = dpc[-2:] = np.nan

        def argpeak(v, minimum=False):
            vv = np.where(mlow, v, np.nan)
            if not np.isfinite(vv).any():
                return np.nan
            return temps[np.nanargmin(vv) if minimum else np.nanargmax(vv)]
        peaks.append([argpeak(dIc), argpeak(irc, minimum=True),
                      argpeak(np.abs(eta)), argpeak(dpc)])
    return np.array(peaks), temps


# ── F. descriptor-set sensitivity of the PCA change point ─────────────────
DIMENSIONLESS = ["sharp_width", "sc_fraction", "foot_extent"]   # d3, d4, d5


def tstar_for_features(per_scan_feat, feats, n_boot, seed):
    """Change point of PC1(T) for a descriptor subset, with bootstrap over sweeps."""
    def cp_of(table):
        agg = table.groupby("T_K")[feats].mean().dropna().sort_index()
        Z, *_ = shape.zscore(agg.to_numpy())
        sc, comps, evr = shape.pca(Z, k=1)
        cp = shape.change_points_1d(agg.index.to_numpy(), sc[:, 0], n_cp=1)
        return (cp[0] if cp else np.nan), evr[0]
    t0, evr = cp_of(per_scan_feat)
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in per_scan_feat.groupby("T_K")]
    boots = np.array([cp_of(per_scan_feat.loc[np.concatenate(
        [rng.choice(ix, len(ix), replace=True) for ix in groups])])[0]
        for _ in range(n_boot)])
    return dict(T_star=t0, boot_median=np.nanmedian(boots),
                boot_lo=np.nanquantile(boots, 0.16), boot_hi=np.nanquantile(boots, 0.84),
                pc1_var=evr)


# ── E. thermal power-balance calibration ──────────────────────────────────
def thermal_calibration(tidy, per_scan_ic):
    agg = per_scan_ic.groupby(["T_K", "branch"])["Ic_mA"].mean().unstack("branch")
    agg["Ic"] = agg[["switch_pos", "switch_neg"]].mean(axis=1)
    agg["Ir"] = agg[["retrap_pos", "retrap_neg"]].mean(axis=1)
    mono = agg.loc[agg.index <= 55].sort_index()      # Ic monotone decreasing here
    ic_of_T = PchipInterpolator(mono.index.to_numpy(), mono["Ic"].to_numpy())
    # strictly-decreasing subset for the inverse (low-T plateau has duplicates)
    Tm_, Icm_ = mono.index.to_numpy(), mono["Ic"].to_numpy()
    keep = [0]
    for i in range(1, len(Icm_)):
        if Icm_[i] < Icm_[keep[-1]] - 1e-6:
            keep.append(i)
    T_of_ic = PchipInterpolator(Icm_[keep][::-1], Tm_[keep][::-1])

    rows = []
    for T in mono.index:
        Pr, Vr, Irm = [], [], []
        for br in ["retrap_pos", "retrap_neg"]:
            for scan, g in tidy[(tidy.T_K == T) & (tidy.branch == br)].groupby("scan"):
                g = g.reindex(g.I_mA.abs().sort_values().index)
                aI, aV = g.I_mA.abs().to_numpy(), g.V_V.abs().to_numpy()
                idx = np.where(aV >= 1e-3)[0]
                if not len(idx):
                    continue
                i = idx[0]; j = slice(i, min(i + 3, len(aI)))
                Irm.append(aI[i]); Vr.append(aV[j].mean())
                Pr.append((aV[j] * aI[j] * 1e-3).mean() * 1e6)   # microwatt
        if not Pr:
            continue
        Ir_T = np.mean(Irm)
        Tj = float(T_of_ic(np.clip(Ir_T, mono["Ic"].min(), mono["Ic"].max())))
        dT = Tj - T
        rows.append(dict(T_K=T, Ir_mA=Ir_T, Vr_V=np.mean(Vr), P_uW=np.mean(Pr),
                         Tj_K=Tj, dT_K=dT,
                         G_uW_per_K=np.mean(Pr) / dT if dT > 0.5 else np.nan))
    th = pd.DataFrame(rows)

    # calibrate smooth G(T) = G0 (T/25)^m on valid points, then forward-predict Ir
    val = th.dropna(subset=["G_uW_per_K"])
    lg = np.polyfit(np.log(val.T_K / 25), np.log(val.G_uW_per_K), 1)
    m_exp, G0 = lg[0], float(np.exp(lg[1]))
    G_of_T = lambda T: G0 * (T / 25) ** m_exp

    pred = []
    for T in mono.index:
        best = np.nan
        for br in ["retrap_pos"]:
            g = tidy[(tidy.T_K == T) & (tidy.branch == br) & (tidy.scan == 1)]
            g = g.reindex(g.I_mA.abs().sort_values().index)
            aI, aV = g.I_mA.abs().to_numpy(), g.V_V.abs().to_numpy()
            for i in range(len(aI) - 1, -1, -1):          # descending |I|
                Tj = T + (aV[i] * aI[i] * 1e-3 * 1e6) / G_of_T(T)
                if Tj < 55 and ic_of_T(np.clip(Tj, 2, 55)) >= aI[i]:
                    best = aI[i]
                    break
        pred.append(dict(T_K=T, Ir_pred=best, Ic=float(mono.loc[T, "Ic"])))
    pred = pd.DataFrame(pred)
    pred["IrIc_pred"] = pred.Ir_pred / pred.Ic
    pred["IrIc_meas"] = (mono["Ir"] / mono["Ic"]).to_numpy()
    return th, pred, G0, m_exp


def main():
    cfg = load_config()
    root = cfg["_root"]
    mdir = root / cfg["paths"]["metrics_dir"]
    fdir = root / cfg["paths"]["paper_figures_dir"]
    fdir.mkdir(parents=True, exist_ok=True)
    tidy = pd.read_parquet(root / cfg["paths"]["tidy"])

    ps_ic = metrics.per_scan_ic(tidy, cfg["ic"]["v_threshold_V"])
    ps_feat, _ = shape.feature_table(tidy)

    # ---------- A: joint peak bootstrap (R2, R4) ----------
    peaks, temps = joint_peak_bootstrap(ps_ic, ps_feat, n_boot=400,
                                        seed=cfg["bootstrap"]["seed"])
    names = ["dIc/dT", "Ir/Ic min", "|eta|", "dPC1/dT"]
    pk = pd.DataFrame(peaks, columns=names)
    print("=== A. peak locations (median [16%,84%]) ===")
    summ = {}
    for c in names:
        med, lo, hi = np.nanmedian(pk[c]), *np.nanquantile(pk[c], [0.16, 0.84])
        summ[c] = (med, lo, hi)
        print(f"  {c:10s}: {med:5.1f} K  [{lo:.1f}, {hi:.1f}]")
    print("\npairwise peak differences (median [16,84]):")
    for i in range(4):
        for j in range(i + 1, 4):
            d = pk[names[i]] - pk[names[j]]
            print(f"  {names[i]} - {names[j]}: {np.nanmedian(d):+5.1f} K "
                  f"[{np.nanquantile(d,0.16):+.1f}, {np.nanquantile(d,0.84):+.1f}]")
    if np.all(pk.std() < 1e-9):
        print("\npeak locations identical in ALL resamples: statistical uncertainty is"
              " below the 5 K temperature-grid spacing; quote +/- 2.5 K (grid-limited).")
    else:
        print("\npeak-location correlation matrix (bootstrap):")
        print(pk.corr().round(2).to_string())
    pk.to_csv(mdir / "peak_bootstrap.csv", index=False)

    # correlation of the observable curves themselves (shared-data caveat, R2)
    C = pd.read_csv(mdir / "concordance_table.csv")
    Tm = C.T_K.to_numpy()
    curves = pd.DataFrame({
        "dIc/dT": np.abs(np.gradient(C.Ic.to_numpy(), Tm)),
        "1-Ir/Ic": 1 - C.IrIc.to_numpy(),
        "|eta|": np.abs(C.eta_switch.to_numpy()),
        "dPC1/dT": np.abs(np.gradient(smooth3(C.PC1.to_numpy()), Tm)),
    }, index=Tm).iloc[2:-2]
    curves = curves[curves.index <= 88]
    print("\ncurve-level Pearson correlations (T<=88 K):")
    print(curves.corr().round(2).to_string())
    curves.corr().to_csv(mdir / "observable_curve_correlations.csv")

    # ---------- B: V_th sensitivity of Ir/Ic minimum (R5) ----------
    print("\n=== B. Ir/Ic minimum vs V_th ===")
    vrows, vcurves = [], {}
    for vth in [3e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
        p = metrics.per_scan_ic(tidy, vth)
        a = p.groupby(["T_K", "branch"])["Ic_mA"].mean().unstack("branch")
        r = ((a["retrap_pos"] + a["retrap_neg"]) /
             (a["switch_pos"] + a["switch_neg"])).dropna()
        r = r[r.index <= 80]
        vcurves[vth] = r
        vrows.append(dict(vth_mV=vth * 1e3, T_min=r.idxmin(), IrIc_min=r.min()))
        print(f"  Vth={vth*1e3:4.1f} mV -> min {r.min():.3f} at {r.idxmin():.0f} K")
    pd.DataFrame(vrows).to_csv(mdir / "iric_vth_sensitivity.csv", index=False)

    # ---------- C: Halperin-Nelson systematics (R8) ----------
    print("\n=== C. HN window systematics ===")
    def hn(T, R0, b, Tb):
        return R0 * np.exp(-b / np.sqrt(np.clip(T - Tb, 1e-6, None)))
    hn_rows, resid_store = [], {}
    for lab, fn in cfg["rt"]["files"].items():
        d = pd.read_excel(root / cfg["rt"]["dir"] / fn)[["T_K", "R_Ohm"]].dropna()
        cu = d.groupby("T_K", as_index=False)["R_Ohm"].mean().sort_values("T_K")
        for wlo in [85.5, 86.0, 86.5, 87.0]:
            for whi in [93.0, 95.0, 97.0]:
                seg = cu[(cu.T_K >= wlo) & (cu.T_K <= whi)]
                try:
                    p, _ = curve_fit(hn, seg.T_K, seg.R_Ohm,
                                     p0=[seg.R_Ohm.max(), 3, 85.5],
                                     bounds=([1e-3, .1, 70], [50, 60, 88.5]),
                                     maxfev=40000)
                    rms = np.sqrt(np.mean((seg.R_Ohm - hn(seg.T_K, *p)) ** 2))
                    hn_rows.append(dict(current=lab, wlo=wlo, whi=whi,
                                        Tbkt=p[2], b=p[1], rms_mOhm=rms * 1e3))
                    if (wlo, whi) == (86.0, 95.0):
                        resid_store[lab] = (seg.T_K.to_numpy(),
                                            (seg.R_Ohm - hn(seg.T_K, *p)).to_numpy())
                except Exception:
                    pass
    hn_df = pd.DataFrame(hn_rows)
    hn_df.to_csv(mdir / "hn_systematics.csv", index=False)
    g = hn_df.groupby("current")["Tbkt"]
    print(g.agg(["min", "max"]).round(2).to_string())
    print(f"overall T_BKT range across windows+currents: "
          f"{hn_df.Tbkt.min():.2f} - {hn_df.Tbkt.max():.2f} K")
    # parameter covariance from the fit itself (1 uA, baseline window)
    d = pd.read_excel(root / cfg["rt"]["dir"] / cfg["rt"]["files"]["1uA"])[["T_K", "R_Ohm"]].dropna()
    cu = d.groupby("T_K", as_index=False)["R_Ohm"].mean().sort_values("T_K")
    seg = cu[(cu.T_K >= 86) & (cu.T_K <= 95)].reset_index(drop=True)
    p, pcov = curve_fit(hn, seg.T_K, seg.R_Ohm, p0=[1, 3, 85.5],
                        bounds=([1e-3, .1, 70], [50, 60, 88.5]), maxfev=40000)
    pcorr = pcov / np.sqrt(np.outer(np.diag(pcov), np.diag(pcov)))
    print(f"1uA baseline fit: R0={p[0]:.3f}, b={p[1]:.3f}, Tbkt={p[2]:.3f}")
    print(f"parameter correlations: corr(b,Tbkt)={pcorr[1,2]:+.3f}, "
          f"corr(R0,b)={pcorr[0,1]:+.3f}, corr(R0,Tbkt)={pcorr[0,2]:+.3f}")

    # ---------- D: synthetic drift-decomposition test (R9) ----------
    print("\n=== D. synthetic drift test ===")
    Cc = pd.read_csv(mdir / "hysteresis_confound.csv")
    Tg = Cc.T_K.to_numpy()
    Ic0 = PchipInterpolator(Tg, smooth3(Cc.S_switch.to_numpy()))
    h_true = np.clip(Cc.hyst_intrinsic.to_numpy(), 0, None)
    rng = np.random.default_rng(7)
    res = []
    for asym_frac in [0.0, 0.2]:
        for _ in range(200):
            dlt = 0.3
            hp_t = h_true * (1 + asym_frac)
            hn_t = h_true * (1 - asym_frac)
            noise = lambda: rng.normal(0, 3e-4, len(Tg))     # 0.3 uA scan noise
            sp = Ic0(Tg) + noise()
            sn = Ic0(Tg + dlt) + noise()
            rp = Ic0(Tg + dlt) - hp_t + noise()
            rn = Ic0(Tg) - hn_t + noise()
            h_sym = .5 * ((sp - rp) + (sn - rn))
            h_anti = .5 * ((sp - rp) - (sn - rn))
            dIcdT = np.gradient(Ic0(Tg), Tg)
            band = (Tg >= 20) & (Tg <= 55)
            with np.errstate(divide="ignore", invalid="ignore"):
                delta_rec = np.nanmedian((-h_anti / dIcdT)[band & (np.abs(dIcdT) > 2e-4)])
            hbias = np.nanmedian((h_sym - h_true)[band] / np.maximum(h_true[band], 1e-4))
            res.append(dict(asym=asym_frac, delta_rec=delta_rec, h_bias=hbias))
    rs = pd.DataFrame(res)
    rs.to_csv(mdir / "drift_synthetic_test.csv", index=False)
    for a, gg in rs.groupby("asym"):
        print(f"  intrinsic polarity asym {a*100:.0f}%:  recovered delta = "
              f"{gg.delta_rec.median():.2f} K (true 0.30), "
              f"h_sym bias = {gg.h_bias.median()*100:+.1f}%")

    # ---------- E: thermal calibration + RCSJ (R7) ----------
    print("\n=== E. thermal power-balance calibration ===")
    th, pred, G0, m_exp = thermal_calibration(tidy, ps_ic)
    th.to_csv(mdir / "thermal_model.csv", index=False)
    pred.to_csv(mdir / "thermal_model_prediction.csv", index=False)
    print(th[["T_K", "Ir_mA", "Vr_V", "P_uW", "Tj_K", "dT_K", "G_uW_per_K"]]
          .round(3).to_string(index=False))
    print(f"calibrated G(T) = {G0:.2f} uW/K * (T/25K)^{m_exp:.2f}")

    # RCSJ numbers at low T (dissipative-branch slope above the switch)
    g2 = tidy[(tidy.T_K.isin([2, 3, 5])) & (tidy.branch == "switch_pos")]
    g2 = g2.reindex(g2.I_mA.abs().sort_values().index)
    top = g2[g2.V_V.abs() > 0.05]
    Rd = np.polyfit(top.I_mA.abs() * 1e-3, top.V_V.abs(), 1)[0]  # ohm
    Ic2 = 64.0e-6
    for Cgeo in [1e-13, 1e-12]:
        Q = np.sqrt(2 * E * Ic2 * Rd ** 2 * Cgeo / HBAR)
        print(f"  RCSJ with R_d={Rd/1e3:.1f} kOhm, C={Cgeo*1e12:.1f} pF: "
              f"Q={Q:.0f} -> Ir/Ic ~ {4/(np.pi*Q):.1e}  (measured 0.93)")
    Qneed = 4 / (np.pi * 0.93)
    Cneed = Qneed ** 2 * HBAR / (2 * E * Ic2 * Rd ** 2)
    print(f"  C needed to reproduce Ir/Ic=0.93: {Cneed*1e18:.2g} aF (unphysical)")

    # ---------- F: descriptor-set sensitivity of T* ----------
    print("\n=== F. T* vs descriptor set (bootstrap 68% over sweeps) ===")
    sets = {"all six": shape.FEATURES, "dimensionless d3-d5": DIMENSIONLESS}
    sets.update({f"drop {f}": [g for g in shape.FEATURES if g != f] for f in shape.FEATURES})
    srows = []
    for name, feats in sets.items():
        r = tstar_for_features(ps_feat, feats, n_boot=300, seed=cfg["bootstrap"]["seed"])
        r["descriptor_set"] = name
        srows.append(r)
        print(f"  {name:24s} T*={r['T_star']:5.1f} K  boot [{r['boot_lo']:.1f}, "
              f"{r['boot_hi']:.1f}]  PC1 var={r['pc1_var']*100:.0f}%")
    sdf = pd.DataFrame(srows)[["descriptor_set", "T_star", "boot_median", "boot_lo",
                               "boot_hi", "pc1_var"]]
    sdf.to_csv(mdir / "tstar_descriptor_sensitivity.csv", index=False)
    print(f"  range of T* over descriptor sets: {sdf.T_star.min():.1f}-{sdf.T_star.max():.1f} K")

    # ---------- figures ----------
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.35))
    cols = plt.cm.plasma(np.linspace(0.05, 0.8, len(vcurves)))
    for (vth, r), c in zip(vcurves.items(), cols):
        ax[0].plot(r.index, r.values, "o-", ms=2.5, color=c,
                   label=f"{vth*1e3:g} mV")
    ax[0].axvspan(38, 48, color="#f5c542", alpha=.22, lw=0)
    ax[0].set(xlabel=r"$T$ (K)", ylabel=r"$I_r/I_c$")
    ax[0].legend(title=r"$V_{\mathrm{th}}$", fontsize=6.5, title_fontsize=7)
    ax[0].text(0.03, 0.94, "(a)", transform=ax[0].transAxes, fontweight="bold",
               fontsize=9, va="top")
    for lab, (Ts, rs_) in resid_store.items():
        ax[1].plot(Ts, rs_ * 1e3, ".", ms=2, label=lab)
    ax[1].axhline(0, color="gray", lw=.6)
    ax[1].set(xlabel=r"$T$ (K)", ylabel=r"HN residual (m$\Omega$)", xlim=(86, 95))
    ax[1].legend(fontsize=6.5)
    ax[1].text(0.03, 0.94, "(b)", transform=ax[1].transAxes, fontweight="bold",
               fontsize=9, va="top")
    for a, gg in rs.groupby("asym"):
        ax[2].hist(gg.delta_rec, bins=25, alpha=.65,
                   label=f"{a*100:.0f}% polarity asym.")
    ax[2].axvline(0.3, color="k", lw=1, ls="--", label=r"true $\delta T$")
    ax[2].set(xlabel=r"recovered $\delta T$ (K)", ylabel="count")
    ax[2].legend(fontsize=6.5)
    ax[2].text(0.03, 0.94, "(c)", transform=ax[2].transAxes, fontweight="bold",
               fontsize=9, va="top")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(fdir / "fig5_robustness.pdf"); fig.savefig(fdir / "fig5_robustness.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    val = th.dropna(subset=["G_uW_per_K"])
    ax[0].plot(val.T_K, val.G_uW_per_K, "o", color="#1f5fa6", label="implied $G$ (data)")
    Tgrid = np.linspace(val.T_K.min(), val.T_K.max(), 100)
    ax[0].plot(Tgrid, G0 * (Tgrid / 25) ** m_exp, "-", color="#c23b22",
               label=fr"fit $G_0(T/25\,{{\rm K}})^{{{m_exp:.1f}}}$")
    ax[0].set(xlabel=r"$T$ (K)", ylabel=r"$G$ ($\mu$W/K)", yscale="log")
    ax[0].legend()
    ax[0].text(0.03, 0.94, "(a)", transform=ax[0].transAxes, fontweight="bold",
               fontsize=9, va="top")
    ax[1].plot(pred.T_K, pred.IrIc_meas, "o-", color="#5d3a9b", label="measured")
    ax[1].plot(pred.T_K, pred.IrIc_pred, "s--", color="#c23b22",
               label="thermal model")
    ax[1].axhline(4 / (np.pi * 1957), color="gray", ls=":", lw=1)
    ax[1].text(5, 0.05, r"RCSJ, $C=0.1$ pF", fontsize=7, color="gray")
    ax[1].set(xlabel=r"$T$ (K)", ylabel=r"$I_r/I_c$", ylim=(0, 1.1))
    ax[1].legend(loc="lower right")
    ax[1].text(0.03, 0.94, "(b)", transform=ax[1].transAxes, fontweight="bold",
               fontsize=9, va="top")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(fdir / "fig6_thermal.pdf"); fig.savefig(fdir / "fig6_thermal.png", dpi=300)
    plt.close(fig)
    print(f"\nwrote fig5_robustness / fig6_thermal to {fdir}")


if __name__ == "__main__":
    main()
