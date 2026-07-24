"""
Exploratory re-analysis of the cleaned BSCCO IV data.

Two goals:
  1. Show that the Phase-1 BKT alpha(T) extraction (single power-law fit over the
     WHOLE log V - log I curve) is dominated by the switching jump at low T and
     yields unphysical exponents. Contrast with a physically-motivated windowed
     fit restricted to the dissipative branch.
  2. Cleanly separate the two asymmetries that both live in these sweeps:
       (a) within-sweep nonreciprocity  eta = (Ic+ - |Ic-|)/(Ic+ + |Ic-|)   [true SDE]
       (b) between-sweep hysteresis     forward vs reverse Ic (switch vs retrap)
Outputs: figures + a metrics table in analysis_code/exploratory/.
Uses numpy/pandas/matplotlib only (no scipy).
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "Processed_data/cleaned_iv_data/all_cleaned_iv.csv"
OUT = ROOT / "analysis_code/exploratory"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df["Tk"] = df["temp"].str.replace("K", "", regex=False).astype(float)
df["absI"] = df["I"].abs()
df["absV"] = df["V"].abs()
temps = np.array(sorted(df["Tk"].unique()))

VTH = 1e-3          # Ic voltage criterion (V); well above ~1e-4 noise floor
WIN_LO, WIN_HI = 3e-4, 0.30   # dissipative window for BKT slope (V)
RAIL = 0.39         # compliance rail (V) to exclude


def linfit(x, y):
    """OLS slope/intercept + R^2 + runs-test z on residual signs."""
    if len(x) < 6:
        return dict(slope=np.nan, intercept=np.nan, r2=np.nan, z_runs=np.nan, n=len(x))
    b, a = np.polyfit(x, y, 1)
    yhat = b * x + a
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    s = np.sign((y - yhat) - np.median(y - yhat))
    s[s == 0] = 1
    runs = 1 + np.sum(s[:-1] != s[1:])
    npos, nneg = np.sum(s > 0), np.sum(s < 0)
    ntot = npos + nneg
    mu = 2 * npos * nneg / ntot + 1
    var = (2 * npos * nneg * (2 * npos * nneg - ntot)) / (ntot ** 2 * (ntot - 1)) if ntot > 1 else np.nan
    z = (runs - mu) / np.sqrt(var) if var and var > 0 else np.nan
    return dict(slope=b, intercept=a, r2=r2, z_runs=z, n=len(x))


# ---- 1. alpha(T): full-range vs windowed ---------------------------------
rows = []
for T in temps:
    s = df[df.Tk == T]
    full = s[(s.absI >= 1e-2) & (s.absV > 1e-7)]
    fr = linfit(np.log10(full.absI.values), np.log10(full.absV.values))
    win = s[(s.absV > WIN_LO) & (s.absV < WIN_HI)]
    wr = linfit(np.log10(win.absI.values), np.log10(win.absV.values))
    rows.append(dict(T=T,
                     a_full=fr["slope"], r2_full=fr["r2"], z_full=fr["z_runs"], n_full=fr["n"],
                     a_win=wr["slope"], r2_win=wr["r2"], z_win=wr["z_runs"], n_win=wr["n"]))
alpha = pd.DataFrame(rows)
alpha.to_csv(OUT / "alpha_full_vs_windowed.csv", index=False)


def cross3(Tv, av):
    """last upward/downward crossing of a=3."""
    out = []
    for i in range(len(av) - 1):
        if np.isfinite(av[i]) and np.isfinite(av[i + 1]) and (av[i] - 3) * (av[i + 1] - 3) < 0:
            out.append(Tv[i] + (3 - av[i]) * (Tv[i + 1] - Tv[i]) / (av[i + 1] - av[i]))
    return out


# ---- 2. Ic per branch + asymmetry decomposition --------------------------
def ic_of(sub):
    """smallest |I| at which |V| first crosses VTH (ascending |I|)."""
    sub = sub.sort_values("absI")
    hit = sub[sub.absV >= VTH]
    return hit.absI.iloc[0] if len(hit) else np.nan

recs = []
for T in temps:
    for direction in ["forward", "reverse"]:
        sd = df[(df.Tk == T) & (df.direction == direction)]
        if sd.empty:
            continue
        for scan, sc in sd.groupby("scan_idx"):
            icp = ic_of(sc[sc.I > 0])
            icn = ic_of(sc[sc.I < 0])
            recs.append(dict(T=T, direction=direction, scan=scan, Icp=icp, Icn=icn))
ic = pd.DataFrame(recs)
ic["eta_within"] = (ic.Icp - ic.Icn) / (ic.Icp + ic.Icn)   # true SDE metric

# aggregate across scans (mean +/- std)
g = ic.groupby(["T", "direction"]).agg(
    Icp=("Icp", "mean"), Icn=("Icn", "mean"),
    eta=("eta_within", "mean"), eta_sd=("eta_within", "std"), nsc=("Icp", "size")).reset_index()

# between-sweep hysteresis: |Ic(forward) - Ic(reverse)| on the positive side
piv = g.pivot(index="T", columns="direction", values="Icp")
hyst = ((piv["forward"] - piv["reverse"]).abs() /
        (0.5 * (piv["forward"] + piv["reverse"]))).rename("hyst_rel")
g.to_csv(OUT / "ic_asymmetry.csv", index=False)

# ---- FIGURE ---------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))

# (a) representative log-log curves
for T, c in [(5, "navy"), (50, "seagreen"), (85, "darkorange")]:
    s = df[(df.Tk == T) & (df.absI > 1e-3) & (df.absV > 1e-7)].sort_values("absI")
    ax[0, 0].scatter(s.absI, s.absV, s=8, color=c, alpha=.5, label=f"{T} K")
ax[0, 0].axhspan(WIN_LO, WIN_HI, color="gold", alpha=.15, label="dissipative window")
ax[0, 0].axhline(1e-4, color="gray", ls=":", label="noise floor ~1e-4 V")
ax[0, 0].set(xscale="log", yscale="log", xlabel="|I| (mA)", ylabel="|V| (V)",
             title="(a) log-log IV: low-T curve is flat then jumps\n(single power law is meaningless there)")
ax[0, 0].legend(fontsize=8)

# (b) alpha(T) full vs windowed
ax[0, 1].plot(alpha["T"], alpha.a_full, "o-", color="crimson", label="full-range fit (Phase-1)")
ok = alpha.r2_win > 0.9
ax[0, 1].plot(alpha["T"], alpha.a_win, "s--", color="steelblue", alpha=.4, label="windowed fit (all)")
ax[0, 1].scatter(alpha["T"][ok], alpha.a_win[ok], color="steelblue", zorder=5,
                 label="windowed fit (R2>0.9, trustworthy)")
ax[0, 1].axhline(3, color="k", ls="--", lw=1); ax[0, 1].axhline(1, color="k", ls=":", lw=1)
cw = cross3(alpha["T"].values, alpha.a_win.values)
for tc in cw:
    ax[0, 1].axvline(tc, color="green", ls="-.", label=f"windowed a=3 @ {tc:.1f} K")
ax[0, 1].set(xlabel="T (K)", ylabel=r"power-law exponent $\alpha$",
             title="(b) alpha(T): full-range fit inflates low-T alpha\nto ~4.3 (artifact of the switch)")
ax[0, 1].legend(fontsize=8)

# (c) Ic(T) four branches
for direction, mk in [("forward", "o"), ("reverse", "s")]:
    gg = g[g.direction == direction].sort_values("T")
    ax[1, 0].plot(gg["T"], gg.Icp, mk + "-", label=f"{direction} Ic+")
    ax[1, 0].plot(gg["T"], gg.Icn, mk + "--", label=f"{direction} |Ic-|", alpha=.6)
ax[1, 0].set(xlabel="T (K)", ylabel="Ic (mA)",
             title=f"(c) Ic(T), Vth={VTH*1e3:.0f} mV, four branches")
ax[1, 0].legend(fontsize=8)

# (d) within-sweep eta vs between-sweep hysteresis
gf = g[g.direction == "forward"].set_index("T")
ax[1, 1].errorbar(gf.index, 100 * gf.eta, yerr=100 * gf.eta_sd, fmt="o-", color="purple",
                  capsize=2, label="within-sweep SDE  eta (forward)")
ax[1, 1].plot(hyst.index, 100 * hyst.values, "^-", color="teal",
              label="between-sweep hysteresis (fwd vs rev Ic+)")
ax[1, 1].axhline(0, color="gray", lw=.8)
ax[1, 1].set(xlabel="T (K)", ylabel="asymmetry (%)",
             title="(d) TWO different asymmetries, decomposed")
ax[1, 1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "eda_overview.png", dpi=160)
print("saved:", OUT / "eda_overview.png")

# ---- console summary ------------------------------------------------------
print("\n=== alpha(T): full-range vs windowed ===")
print(alpha.round(2).to_string(index=False))
print("\nwindowed a=3 crossings:", [round(c, 1) for c in cw])
print("\n=== within-sweep SDE eta (forward), by T ===")
print(gf[["Icp", "Icn", "eta", "eta_sd", "nsc"]].round(4).to_string())
print("\n=== between-sweep hysteresis (rel) ===")
print(hyst.round(4).to_string())
