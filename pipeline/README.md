# BSCCO analysis pipeline (Phase 0)

Config-driven, tested replacement for the notebook maze. One canonical dataset,
QC-as-flags (never destructive), metrics with bootstrap CIs.

## Run

```bash
.venv_clean/bin/python pipeline/tests/test_metrics.py   # sanity tests (synthetic RSJ)
.venv_clean/bin/python pipeline/run.py                  # load -> qc -> metrics -> figures
```

## Layout

```
pipeline/
  config.yaml          every tunable constant (no magic numbers in code)
  bscco/
    io.py              cleaned CSV -> canonical tidy table (+ manifest); recomputes R
    qc.py              adds boolean flag columns (rail / noise / zeroI / R-implausible)
    metrics.py         Ic, retrapping, nonreciprocity, hysteresis, alpha(T) + bootstrap
  run.py               orchestrator
  tests/test_metrics.py
  outputs/             tidy_iv.parquet, manifest.csv, metrics/*.csv, figures/*.png
```

## Key design decisions

* **Leg is derived, not trusted.** forward ramps I:+max→−max, reverse ramps I:−max→+max,
  so `(direction, sign I)` fixes whether |I| is rising (**switching**, Ic) or falling
  (**retrapping**, Ir). This gives four branches: `switch_pos/neg`, `retrap_pos/neg`.
* **Two asymmetries are separated** (they were conflated before):
  * nonreciprocity (true diode) `η = (Ic+ − |Ic−|)/(Ic+ + |Ic−|)`  — same leg, opposite polarity
  * switch/retrap hysteresis `(Ic − Ir)/(Ic + Ir)`               — same polarity, opposite leg
* **R is recomputed** from I,V; the upstream `R→0 if R>20 or R<0.01` forcing is ignored.
* **QC flags, never deletes.** Downstream cuts are explicit and reversible.

## What Phase 0 already corrected

1. **The α=3 → T_BKT=85 K result is a fit-window artifact.** Restricting the power-law
   fit to the dissipative window and keeping only statistically acceptable fits (R²≥0.9),
   α(T) crosses 3 only near ~40 K and never returns to 3 near 85 K. The original 85 K value
   came from letting the switching jump dominate a full-range log–log fit. The BKT
   interpretation is *not* robustly supported by the IV power-law criterion in this device.
2. **The "diode effect" is mostly hysteresis.** The genuine switching nonreciprocity peaks
   at ~6% near 50 K; the switching–retrapping hysteresis is ~16% at the same temperature.
   Both onset where Ic(T) falls fastest, pointing to a shared thermal/junction-dynamics origin.
3. A naive within-sweep η (mixing leg and polarity) gives a spurious ~−4% at low T; the
   correct branch decomposition removes it.

## Phase 1-2 (`run_phase12.py`, module `bscco/fits.py`)

```bash
.venv_clean/bin/python pipeline/run_phase12.py   # needs tidy_iv.parquet from run.py
```

Three independent transition estimates and how much they disagree:

| method | result |
|---|---|
| Ic(T) = Ic0(1−T/Tc)^n envelope | Tc ≈ 81 K, n ≈ 1 — but a *single* envelope fits poorly (two current scales: steep drop to ~55 K, plateau, tail to ~90 K) |
| R–T Halperin–Nelson tail | **T_BKT ≈ 86.7 K, current-independent** (316 nA→10 µA), clean fits |
| IV power-law α=3 crossing | **does NOT reproduce ~87 K under any sensible window**; acceptable α(T) peaks at ~2.1 near 80 K and never reaches 3. The only α=3 crossings are low-T fit artifacts. |

**Headline for the paper:** R–T Halperin–Nelson robustly locates the BKT temperature, but the widely-used IV α=3 criterion fails to corroborate it in this device — a concrete caution, backed by a window-sensitivity table (`outputs/metrics/bkt_window_sensitivity.csv`).

Reported CIs are statistical (bootstrap) only; for T_BKT and Ic the dominant uncertainty is systematic (fit window / voltage threshold), shown as explicit bands.

## Phase 3 (`run_phase3_regimes.py`, module `bscco/shape.py`)

```bash
.venv_clean/bin/python pipeline/run_phase3_regimes.py
```

**Positive, threshold-free result.** Each up-leg I–V branch is described by self-normalised
shape features (switch width, flux-flow foot, Vmax/rail, dissipative fraction, dR, I_switch),
so no fragile absolute voltage threshold enters. Then PCA + change-point:

* Temperatures collapse onto a **single 1-D arc** in PCA space (PC1 43% + PC2 36%): the I–V
  lineshape is governed by one latent transport coordinate. (Hard k-means is deliberately NOT
  used — on a 1-D manifold there are no discrete clusters; the elbow confirms it. It's a
  *continuous crossover*, not sharp phases.)
* PC1(T) has a **bootstrap-stable change-point at T\* ≈ 48 K [38, 48] (68%)** ≈ 0.55 Tc.
* Independent raw descriptors (switch softening, foot onset, dissipative fraction, departure
  from the rail) all turn on together across T\* — concordance across observables is the
  argument against a measurement glitch.

This also **fixes the Ic threshold artifact**: the non-smooth 35 K spike in the earlier
threshold-sensitivity plot was the fixed 5 mV criterion landing inside the low-voltage
flux-flow foot. Defining the switch current from the near-vertical jump (or via the
self-normalised descriptor) removes it.

## Phase 3b — hysteresis confound test (`run_phase3b_confound.py`)

Symmetric/antisymmetric decomposition of switch−retrap over both polarities proves the
hysteresis is **intrinsic**: 92% symmetric, drift proxy negligible (implied inter-file
drift ≈ 0.2 K), uncorrelated with dI_c/dT. Cleanest metric: **I_r/I_c has a minimum of
0.74 at 50 K** (0.93 at low T → 1.0 non-hysteretic above 60 K). Physical reading
(literature-anchored): self-heating / hot-spot switching dynamics (retrapping set by Joule
power balance; non-monotonic hysteresis; BSCCO's low thermal conductivity).

## Phase 4 — concordance & paper assembly (`run_phase4_concordance.py`)

Master figure `phase4_concordance.png`: four independent observables — |dI_c/dT|,
hysteresis 1−I_r/I_c, nonreciprocity |η|, lineshape change-rate — all peak at
**T\* ≈ 48–50 K ≈ 0.55 T_c**, plus the I_r/I_c minimum and a proposed regime map. Narrative,
key-numbers table, section structure and honest limitations in
[`Report/paper/outline.md`](../Report/paper/outline.md).

## Full run order

```bash
python pipeline/run.py               # tidy + QC + Phase-0 metrics
python pipeline/run_phase12.py       # Ic(T) scaling, Halperin-Nelson, BKT window test
python pipeline/run_phase3_regimes.py    # threshold-free lineshape PCA, T*
python pipeline/run_phase3b_confound.py  # hysteresis intrinsic-vs-drift test
python pipeline/run_phase4_concordance.py  # master concordance figure + table
```

## Next (new chat, per user: switch model → LaTeX manuscript + README + arXiv)

Turn `Report/paper/outline.md` + the figures into a LaTeX (revtex4-2) manuscript and a
GitHub README; prepare the arXiv submission.
