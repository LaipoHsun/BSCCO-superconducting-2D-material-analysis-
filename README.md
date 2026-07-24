# Dynamical crossover in an hBN-encapsulated BSCCO-2212 weak link

Transport-data analysis of a single hBN-protected, mechanically exfoliated
Bi₂Sr₂CaCu₂O₈₊δ (BSCCO-2212) device: 27 temperatures (2–120 K), zero magnetic field,
current–voltage sweeps in both ramp directions plus R–T curves at four bias currents.

**Main result.** Four statistically independent observables — a threshold-free
I–V-lineshape coordinate (PCA), the supercurrent collapse rate |dI_c/dT|, the
switching–retrapping hysteresis 1 − I_r/I_c, and the supercurrent nonreciprocity |η| —
all peak at the same crossover temperature

> **T\* = 48 K (68% CI 38–48 K) ≈ 0.55 T_BKT**,  with T_BKT = 86.7 ± 0.2 K
> (bias-independent Halperin–Nelson fits).

The non-monotonic hysteresis (I_r/I_c minimum 0.74 at 50 K, → 1 above 60 K) together
with Josephson energies E_J/k_B ~ 10³ K identifies the crossover as **self-heating
(hot-spot) governed switching dynamics** giving way to overdamped flux flow — not
capacitive or fluctuation-dominated behavior.

![Concordance of four independent observables at T*](figures/fig4_concordance.png)

A polarity-symmetric/antisymmetric decomposition proves the hysteresis is intrinsic
(92% symmetric; implied inter-sweep thermal drift ≈ 0.2 K), and every quantity carries
bootstrap confidence intervals — the analysis is designed to be defensible in the
single-device limit that air-sensitive 2D superconductors impose.

## Manuscript

The paper (REVTeX 4.2, PRB style) lives in [`main.pdf`]:

All four figures are regenerated from the data by the pipeline (below); nothing in the
paper is hand-drawn.

## Reproducing the analysis

Environment: Python ≥ 3.11 with `numpy pandas scipy matplotlib pyyaml pyarrow openpyxl`.

```bash
git clone https://github.com/LaipoHsun/BSCCO-superconducting-2D-material-analysis-.git
```

```bash
python pipeline/run.py                    # tidy dataset + QC flags + core metrics
python pipeline/run_phase12.py            # Ic(T) scaling, Halperin–Nelson, BKT-window test
python pipeline/run_phase3_regimes.py     # threshold-free lineshape PCA -> T*
python pipeline/run_phase3b_confound.py   # intrinsic-vs-drift hysteresis test
python pipeline/run_phase4_concordance.py # concordance table + overview figure
python pipeline/run_revision_analysis.py  # robustness: Vth/window systematics, synthetic
                                          # drift test, thermal-model calibration, RCSJ
python pipeline/run_paper_figures.py      # publication figures (PDF+PNG)
python pipeline/tests/test_metrics.py     # sanity tests on synthetic RSJ data
```

Every tunable constant (thresholds, fit windows, bootstrap settings) is in
[`pipeline/config.yaml`](pipeline/config.yaml) — there are no magic numbers in the code.
Outputs land in `pipeline/outputs/` (`metrics/*.csv`, `figures/*.png`) and
`Report/paper/figures/`.

### Pipeline design

| module | role |
|---|---|
| `pipeline/bscco/io.py` | raw sweeps → canonical tidy table; derives switching/retrapping branch from (ramp direction × current sign); recomputes R |
| `pipeline/bscco/qc.py` | quality flags (voltage rail, noise floor, …) — flags only, never deletes |
| `pipeline/bscco/metrics.py` | I_c, I_r, nonreciprocity η, hysteresis, α(T) — all with bootstrap CIs |
| `pipeline/bscco/shape.py` | six self-normalized I–V lineshape descriptors, PCA, change-point detection |
| `pipeline/bscco/fits.py` | Halperin–Nelson R(T), I_c(T) envelope, BKT window-sensitivity |

Key methodological points:

- **Threshold-free crossover detection.** A fixed voltage criterion samples different
  physical features at different temperatures (e.g. the flux-flow "foot" at 35–55 K).
  The lineshape descriptors are normalized to each curve's own scales, so T\* does not
  depend on any absolute threshold.
- **Confound testing.** Switch–retrap hysteresis is decomposed into polarity-even
  (intrinsic) and polarity-odd (thermal-drift artifact) parts; the drift bound also caps
  the instrumental share of the nonreciprocity at < 0.3 % (measured: 6.2 % at 50 K).
- **Honest BKT treatment.** T_BKT comes from Halperin–Nelson tails, reproducible across
  a 30× range of bias current; the popular I–V α = 3 criterion is shown to be
  fit-window-fragile here and is reported only as a caveat.

## Repository layout

```
data/                      raw measurements only (never modified by code)
  by_temperature/<T>/...     I–V sweeps, forward & reverse
  Pure_RT/                   R–T at fixed bias
data_process_code/         legacy cleaning notebooks (produce Processed_data/)
Processed_data/            cleaned inputs read by the pipeline
  cleaned_iv_data/all_cleaned_iv.csv   ← canonical cleaned dataset
pipeline/                  the analysis (see table above)
  config.yaml, run*.py, bscco/, tests/, outputs/
analysis_code/             exploratory notebooks (superseded by pipeline/)
Report/paper/              REVTeX manuscript + publication figures
```

Workflow rule inherited from the lab: `data/` is raw-only; code reads
`Processed_data/`, writes to `pipeline/outputs/` — raw files are never overwritten.

## Status / caveats

- Single device, zero applied field: the nonreciprocity is reported as a bounded
  observable, **not** as a superconducting-diode claim (no field-reversal test).
- The self-heating interpretation is model-consistent but not thermally verified;
  pulsed I–V or on-chip thermometry is the decisive follow-up.
- Before arXiv submission, fill the `%% TODO` items in `Report/paper/main.tex`
  (author romanization, affiliations, device geometry, repo URL).
