"""End-to-end Phase-0 pipeline: load -> QC -> metrics -> figures.

Usage:  python pipeline/run.py
Everything is driven by pipeline/config.yaml. Outputs land in pipeline/outputs/.
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
from bscco import load_config, io, qc, metrics  # noqa: E402


def main() -> None:
    cfg = load_config()
    root = cfg["_root"]
    mdir = root / cfg["paths"]["metrics_dir"]; mdir.mkdir(parents=True, exist_ok=True)
    fdir = root / cfg["paths"]["figures_dir"]; fdir.mkdir(parents=True, exist_ok=True)

    # 1. tidy + manifest
    tidy = io.load_tidy(cfg)
    manifest = io.build_manifest(tidy)
    io.write_outputs(tidy, manifest, cfg)
    print(f"tidy: {tidy.shape[0]} rows, {tidy['T_K'].nunique()} temperatures, "
          f"{manifest.shape[0]} (T,dir,scan) groups")

    # 2. QC flags
    tidy = qc.add_flags(tidy, cfg)
    print("\nQC flag fractions:")
    print(qc.qc_summary(tidy).round(3).to_string())

    # 3. metrics
    vth = cfg["ic"]["v_threshold_V"]
    per_scan = metrics.per_scan_ic(tidy, vth)
    agg = metrics.aggregate_ic(per_scan, cfg)
    asym = metrics.derived_asymmetries(per_scan, cfg)
    alpha = metrics.alpha_of_T(tidy, cfg)
    sens = metrics.ic_threshold_sensitivity(tidy, cfg)

    per_scan.to_csv(mdir / "ic_per_scan.csv", index=False)
    agg.to_csv(mdir / "ic_aggregate.csv", index=False)
    asym.to_csv(mdir / "asymmetries.csv", index=False)
    alpha.to_csv(mdir / "alpha_of_T.csv", index=False)
    sens.to_csv(mdir / "ic_threshold_sensitivity.csv", index=False)

    cross = _alpha3_crossings(alpha[alpha.acceptable])
    print(f"\nalpha(T) acceptable at T = {sorted(alpha.loc[alpha.acceptable,'T_K'])}")
    print(f"alpha=3 crossings (acceptable pts only): {[round(c,1) for c in cross]}")

    # 4. figures
    _fig_summary(agg, asym, alpha, fdir)
    print(f"\nWrote metrics to {mdir}\nWrote figures to {fdir}")


def _alpha3_crossings(a: pd.DataFrame) -> list[float]:
    a = a.sort_values("T_K"); T = a["T_K"].to_numpy(); al = a["alpha"].to_numpy()
    out = []
    for i in range(len(al) - 1):
        if (al[i] - 3) * (al[i + 1] - 3) < 0:
            out.append(T[i] + (3 - al[i]) * (T[i + 1] - T[i]) / (al[i + 1] - al[i]))
    return out


def _fig_summary(agg, asym, alpha, fdir: Path) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) Ic(T): switching vs retrapping, both polarities
    styles = {"switch_pos": ("o-", "C0"), "switch_neg": ("o--", "C0"),
              "retrap_pos": ("s-", "C3"), "retrap_neg": ("s--", "C3")}
    for br, (st, col) in styles.items():
        g = agg[agg.branch == br].sort_values("T_K")
        ax[0].plot(g["T_K"], g["Ic_mA"], st, color=col, ms=4, label=br)
        ax[0].fill_between(g["T_K"], g["Ic_lo"], g["Ic_hi"], color=col, alpha=.12)
    ax[0].set(xlabel="T (K)", ylabel="|Ic| (mA)", title="(a) switching vs retrapping current")
    ax[0].legend(fontsize=8)

    # (b) asymmetries with bootstrap CIs
    for col, c, lab in [("eta_switch", "purple", "nonreciprocity η (switch)"),
                        ("hyst_pos", "teal", "hysteresis (+)"),
                        ("hyst_neg", "darkorange", "hysteresis (−)")]:
        a = asym.dropna(subset=[col])
        ax[1].plot(a["T_K"], 100 * a[col], "o-", color=c, ms=4, label=lab)
        ax[1].fill_between(a["T_K"], 100 * a[col + "_lo"], 100 * a[col + "_hi"], color=c, alpha=.15)
    ax[1].axhline(0, color="gray", lw=.8)
    ax[1].set(xlabel="T (K)", ylabel="asymmetry (%)", title="(b) diode vs hysteresis, with 95% CI")
    ax[1].legend(fontsize=8)

    # (c) alpha(T), acceptable points highlighted
    ax[2].errorbar(alpha["T_K"], alpha["alpha"], yerr=alpha["alpha_se"], fmt="o",
                   color="lightsteelblue", ms=4, label="all windowed fits")
    ok = alpha[alpha.acceptable]
    ax[2].scatter(ok["T_K"], ok["alpha"], color="navy", zorder=5, label="statistically acceptable")
    ax[2].axhline(3, color="crimson", ls="--", label="α=3 (BKT)")
    ax[2].axhline(1, color="k", ls=":", label="α=1 (ohmic)")
    ax[2].set(xlabel="T (K)", ylabel="α", title="(c) fluctuation exponent α(T)", ylim=(0, 9))
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fdir / "phase0_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
