"""Publication-quality figures for the manuscript (PRB / arXiv style).

Regenerates the four main-text figures as vector PDFs (+PNG previews) in
Report/paper/figures/. Style: serif/STIX, inward ticks, no titles (captions
carry the information), panel labels inside the axes.

Usage: python pipeline/run_paper_figures.py   (after the analysis phases have run)
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

TC = 86.7
TSTAR = (48.0, 38.0, 48.0)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

SINGLE = 3.375          # PR single-column width (in)
DOUBLE = 7.0


def panel_label(ax, s, x=0.03, y=0.94):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def tstar_band(ax, label=False):
    ax.axvspan(TSTAR[1], TSTAR[2], color="#f5c542", alpha=0.22, lw=0, zorder=0)
    if label:
        ax.axvline(TC, color="#8b1a1a", ls=(0, (4, 2)), lw=0.9, zorder=0)


def save(fig, name, fdir):
    fig.savefig(fdir / f"{name}.pdf")
    fig.savefig(fdir / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", fdir / f"{name}.pdf")


# ── Fig 1: IV phenomenology + R-T ─────────────────────────────────────────
def fig1(tidy, cfg, root, fdir):
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 2.35))

    # (a) up-leg (switching branch) |V| vs |I| at representative temperatures
    temps = [5, 35, 45, 55, 68, 85]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(temps)))
    for T, c in zip(temps, cmap):
        g = tidy[(tidy.T_K == T) & (tidy.branch == "switch_pos") & (tidy.scan == 1)]
        g = g.reindex(g.I_mA.abs().sort_values().index)
        ax[0].plot(g.I_mA.abs() * 1e3, g.V_V.abs() * 1e3, "-", color=c,
                   label=f"{T} K", lw=1.0)
    ax[0].set(xlabel=r"$|I|$ ($\mu$A)", ylabel=r"$|V|$ (mV)", xlim=(0, 80), ylim=(-12, 410))
    ax[0].legend(ncol=2, handlelength=1.2, columnspacing=0.8, loc="upper right",
                 bbox_to_anchor=(0.98, 0.97))
    panel_label(ax[0], "(a)")

    # (b) switching vs retrapping hysteresis at 45 K (log-V: dissipation is gradual here)
    up = tidy[(tidy.T_K == 45) & (tidy.branch == "switch_pos") & (tidy.scan == 1)]
    dn = tidy[(tidy.T_K == 45) & (tidy.branch == "retrap_pos") & (tidy.scan == 1)]
    up = up.reindex(up.I_mA.abs().sort_values().index)
    dn = dn.reindex(dn.I_mA.abs().sort_values().index)
    ax[1].semilogy(up.I_mA * 1e3, np.clip(up.V_V.abs() * 1e3, 3e-2, None), "-o",
                   color="#1f5fa6", ms=2.5, label=r"$|I|$ increasing (switch)")
    ax[1].semilogy(dn.I_mA * 1e3, np.clip(dn.V_V.abs() * 1e3, 3e-2, None), "-s",
                   color="#c23b22", ms=2.5, label=r"$|I|$ decreasing (retrap)")
    ax[1].axhline(1.0, color="gray", ls=":", lw=0.8)
    ax[1].text(16, 1.25, r"$V_{\mathrm{th}}$", fontsize=7, color="gray")
    ax[1].annotate(r"$I_c$", (34.3, 2.6), fontsize=9, color="#1f5fa6", ha="left")
    ax[1].annotate(r"$I_r$", (24.6, 2.6), fontsize=9, color="#c23b22", ha="right")
    ax[1].set(xlabel=r"$I$ ($\mu$A)", ylabel=r"$|V|$ (mV)", xlim=(15, 60), ylim=(3e-2, 3e2))
    ax[1].legend(loc="upper left", bbox_to_anchor=(0.13, 0.97))
    panel_label(ax[1], "(b)", x=0.03, y=0.94)
    ax[1].text(0.97, 0.05, r"$T=45$ K", transform=ax[1].transAxes, ha="right", fontsize=8)

    # (c) R-T with Halperin-Nelson fits
    hn = pd.read_csv(root / "pipeline/outputs/metrics/hn_fits.csv")
    colors = {"316nA": "#4c72b0", "1uA": "#55a868", "3.16uA": "#c44e52", "10uA": "#8172b2"}
    lab_map = {"316nA": "316 nA", "1uA": r"1 $\mu$A", "3.16uA": r"3.16 $\mu$A", "10uA": r"10 $\mu$A"}
    for lab, fn in cfg["rt"]["files"].items():
        d = pd.read_excel(root / cfg["rt"]["dir"] / fn)[["T_K", "R_Ohm"]].dropna()
        curve = d.groupby("T_K", as_index=False)["R_Ohm"].mean().sort_values("T_K")
        ax[2].plot(curve.T_K, curve.R_Ohm, ".", ms=1.6, color=colors[lab],
                   label=lab_map[lab], rasterized=True)
        row = hn[hn.current == lab]
        if len(row):
            r = row.iloc[0]
            Tg = np.linspace(r.Tbkt_K + 0.15, 95, 300)
            ax[2].plot(Tg, r.R0 * np.exp(-r.b / np.sqrt(Tg - r.Tbkt_K)), "k-", lw=0.8)
    ax[2].axvline(TC, color="#8b1a1a", ls=(0, (4, 2)), lw=0.9)
    ax[2].text(TC - 0.45, 0.83, r"$T_{\mathrm{BKT}}$", rotation=90, fontsize=8,
               color="#8b1a1a", ha="right", va="top")
    ax[2].set(xlabel=r"$T$ (K)", ylabel=r"$R$ ($\Omega$)", xlim=(83, 97), ylim=(-0.04, 1.05))
    ax[2].legend(loc="lower right", bbox_to_anchor=(0.98, 0.05), handletextpad=0.1,
                 markerscale=4)
    panel_label(ax[2], "(c)")

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig1_iv_overview", fdir)


# ── Fig 2: supercurrent envelope + hysteresis ratio ───────────────────────
def fig2(C, fdir):
    fig, ax = plt.subplots(2, 1, figsize=(SINGLE, 3.9), sharex=True,
                           gridspec_kw=dict(hspace=0.08))
    T = C.T_K.to_numpy()

    tstar_band(ax[0], label=True)
    ax[0].plot(T, C.Ic * 1e3, "o-", color="#1f5fa6", label=r"switching $I_c$")
    ax[0].plot(T, C.Ir * 1e3, "s--", color="#c23b22", label=r"retrapping $I_r$")
    ax[0].set(ylabel=r"current ($\mu$A)", ylim=(-2, 70))
    ax[0].legend(loc="upper right")
    panel_label(ax[0], "(a)")

    tstar_band(ax[1], label=True)
    m = T <= 88
    ax[1].plot(T[m], C.IrIc[m], "o-", color="#5d3a9b")
    imin = C.IrIc.idxmin()
    ax[1].plot(C.loc[imin, "T_K"], C.loc[imin, "IrIc"], "*", color="#d1495b", ms=11, zorder=5)
    ax[1].axhline(1, color="gray", ls=":", lw=0.8)
    ax[1].annotate(f"min {C.IrIc.min():.2f}", (C.loc[imin, 'T_K'] + 3, C.IrIc.min()),
                   fontsize=8, va="center")
    ax[1].text(TSTAR[0] - 5, 0.99, r"$T^{*}$", color="#b8860b", fontsize=9, ha="right")
    ax[1].text(TC + 1, 0.78, r"$T_{\mathrm{BKT}}$", color="#8b1a1a", fontsize=8, rotation=90)
    ax[1].set(xlabel=r"$T$ (K)", ylabel=r"$I_r/I_c$", xlim=(0, 92), ylim=(0.70, 1.04))
    panel_label(ax[1], "(b)")

    save(fig, "fig2_envelope", fdir)


# ── Fig 3: lineshape PCA + T* ─────────────────────────────────────────────
def fig3(reg, boots, fdir):
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.6))

    sc = ax[0].scatter(reg.PC1, reg.PC2, c=reg.T_K, cmap="viridis", s=34,
                       edgecolor="k", linewidth=0.4)
    for _, r in reg.iterrows():
        if r.T_K in (2, 10, 30, 45, 55, 68, 85):
            ax[0].annotate(f"{r.T_K:.0f} K", (r.PC1, r.PC2), fontsize=7,
                           xytext=(4, 3), textcoords="offset points")
    ax[0].annotate("110 K", (reg.loc[reg.T_K == 110, "PC1"].iloc[0],
                             reg.loc[reg.T_K == 110, "PC2"].iloc[0]),
                   fontsize=7, xytext=(-26, -9), textcoords="offset points")
    cb = plt.colorbar(sc, ax=ax[0], pad=0.02)
    cb.set_label(r"$T$ (K)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax[0].set(xlabel="PC1 (43%)", ylabel="PC2 (36%)")
    panel_label(ax[0], "(a)")

    a = reg.sort_values("T_K")
    tstar_band(ax[1], label=True)
    ax[1].plot(a.T_K, a.PC1, "o-", color="#3c5488")
    ax[1].text(TSTAR[0], 2.35, r"$T^{*}=48$ K", color="#b8860b", ha="center", fontsize=8.5)
    ax[1].text(TC + 1, -1.9, r"$T_{\mathrm{BKT}}$", color="#8b1a1a", fontsize=8, rotation=90)
    ax[1].set(xlabel=r"$T$ (K)", ylabel="PC1 (lineshape coordinate)", ylim=(-2.6, 2.75))
    panel_label(ax[1], "(b)")

    # inset: bootstrap distribution of T* (upper-left region is data-free)
    ins = ax[1].inset_axes([0.075, 0.63, 0.32, 0.30])
    ins.hist(boots, bins=np.arange(25, 60, 3.4), color="#6aa1c8", edgecolor="k", lw=0.4)
    ins.axvline(np.median(boots), color="#b8860b", lw=1.2)
    ins.set_xlabel(r"bootstrap $T^{*}$ (K)", fontsize=6.5, labelpad=1.5)
    ins.set_yticks([])
    ins.tick_params(labelsize=6)
    ins.patch.set_alpha(1.0)

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig3_pca", fdir)


# ── Fig 4: intrinsic hysteresis + concordance ─────────────────────────────
def fig4(C, conf, asym, fdir):
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    T = C.T_K.to_numpy()

    # (a) symmetric vs antisymmetric hysteresis
    tstar_band(ax[0], label=True)
    ax[0].plot(conf.T_K, conf.hyst_intrinsic * 1e3, "o-", color="#1b7837",
               label="intrinsic (polarity-symmetric)")
    ax[0].plot(conf.T_K, conf.drift_proxy * 1e3, "^--", color="#b2182b", ms=3.2,
               label="drift proxy (antisymmetric)")
    ax[0].axhline(0, color="gray", lw=0.6)
    ax[0].set(xlabel=r"$T$ (K)", ylabel=r"$I_c-I_r$ ($\mu$A)", xlim=(0, 112))
    ax[0].legend(loc="upper right")
    panel_label(ax[0], "(a)")

    # (b) concordance of normalised observables
    def norm(s):
        s = np.asarray(s, float)
        lo, hi = np.nanmin(s), np.nanmax(s)
        return (s - lo) / (hi - lo)

    def smooth(x, w=3):
        return np.convolve(x, np.ones(w) / w, mode="same")

    dIcdT = np.abs(np.gradient(C.Ic.to_numpy(), T))
    dPC1 = np.abs(np.gradient(smooth(C.PC1.to_numpy()), T))
    dPC1[:2] = dPC1[-2:] = np.nan   # mask smoothing/derivative edge points
    eta = np.abs(C.eta_switch.to_numpy())

    tstar_band(ax[1], label=True)
    ax[1].plot(T, norm(dIcdT), "v-", color="#4477aa", label=r"$|dI_c/dT|$")
    ax[1].plot(T, norm(1 - C.IrIc), "s-", color="#ee8833", label=r"$1-I_r/I_c$")
    ax[1].plot(T, norm(eta), "^-", color="#228833", label=r"$|\eta|$")
    ax[1].plot(T, norm(dPC1), "o-", color="#aa3377", label=r"$|d\,\mathrm{PC1}/dT|$")
    ax[1].text(TSTAR[0], 1.05, r"$T^{*}$", color="#b8860b", ha="center", fontsize=9)
    ax[1].text(TC + 1.5, 1.02, r"$T_{\mathrm{BKT}}$", color="#8b1a1a", fontsize=8,
               rotation=90, va="top")
    ax[1].set(xlabel=r"$T$ (K)", ylabel="normalised", xlim=(0, 112), ylim=(-0.05, 1.18))
    ax[1].legend(loc="upper left", ncol=1, handlelength=1.4,
                 bbox_to_anchor=(0.015, 0.97))
    panel_label(ax[1], "(b)", x=0.30)

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig4_concordance", fdir)


def bootstrap_tstar(tidy, cfg, n_boot=300):
    per_scan, _ = shape.feature_table(tidy)
    rng = np.random.default_rng(cfg["bootstrap"]["seed"])
    groups = {T: g.index.to_numpy() for T, g in per_scan.groupby("T_K")}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in groups.values()])
        ab = per_scan.loc[idx].groupby("T_K")[shape.FEATURES].mean().dropna().sort_values("T_K")
        if len(ab) < 6:
            continue
        Zb, *_ = shape.zscore(ab[shape.FEATURES].to_numpy())
        sb, *_ = shape.pca(Zb, k=1)
        cp = shape.change_points_1d(ab.index.to_numpy(), sb[:, 0], n_cp=1)
        if cp:
            boots.append(cp[0])
    return np.array(boots)


def main():
    cfg = load_config()
    root = cfg["_root"]
    fdir = root / "Report/paper/figures"
    fdir.mkdir(parents=True, exist_ok=True)
    tidy = pd.read_parquet(root / cfg["paths"]["tidy"])
    mdir = root / cfg["paths"]["metrics_dir"]
    C = pd.read_csv(mdir / "concordance_table.csv")
    reg = pd.read_csv(mdir / "regimes.csv")
    conf = pd.read_csv(mdir / "hysteresis_confound.csv")
    asym = pd.read_csv(mdir / "asymmetries.csv")

    boots = bootstrap_tstar(tidy, cfg)
    print(f"T* bootstrap: median={np.median(boots):.1f}  "
          f"68% CI=[{np.quantile(boots,0.16):.1f},{np.quantile(boots,0.84):.1f}]  n={len(boots)}")

    fig1(tidy, cfg, root, fdir)
    fig2(C, fdir)
    fig3(reg, boots, fdir)
    fig4(C, conf, asym, fdir)


if __name__ == "__main__":
    main()
