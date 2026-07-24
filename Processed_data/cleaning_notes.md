# Cleaned IV Data Notes

Generated time: 2026-05-16 02:07:55
Notebook: `New_code/cleaning_data.ipynb`

## Data Source
- `data/by_temperature/<TempK>/<direction>/*.xlsx`

## Cleaning Steps
1. Sweep split (`MXTMN` / `MNTMX`) by direction and zero-crossing logic.
2. Monotonic filter on `|I|` with tolerance (`MONOTONIC_CFG`).
3. Symmetry + continuity cleaning (improved method), including spike logic and case-specific overrides.
4. Export final in-memory cleaned scans to CSV (no overwrite to raw xlsx).

## Monotonic Config
- {'x_tol': 0.0002, 'use_abs_x': True}

## Baseline Symmetry Config
- {'n_grid': 220, 'x_tol_abs': None, 'x_tol_step_factor': 0.8, 'x_tol_min': 1e-05, 'min_points_scan': 8, 'min_scan_support_dir': 1, 'sigma_floor': 0.0006, 'z_thresh': 4.5, 'z_hard': 7.0, 'spike_k': 4.8, 'require_spike_for_drop': False}

## Case-specific Special Cleaning (important)
- 10K | MXTMN: z_thresh=3.4, z_hard=5.5, spike_k=4.0, min_scan_support_dir=2, drop_scan_if_bad_frac_gt=0.38, min_points_for_scan_drop=12\n- 25K | MNTMX: tail_cut_after_first_bad=True, z_thresh=4.0, z_hard=6.3\n- 30K | MNTMX: prefer_upper=True, z_lower=2.3, z_thresh=4.0, z_hard=6.2\n- 50K | MXTMN: absI_max=0.07, z_thresh=3.8, z_hard=6.0\n- 55K | MXTMN: absI_max=0.07, z_thresh=3.8, z_hard=6.0\n- 85K | MXTMN: skip_clean=True\n- 88K | MXTMN: skip_clean=True\n- 90K | MNTMX: skip_clean=True\n- 90K | MXTMN: skip_clean=True\n- 95K | MNTMX: skip_clean=True

## Output Files
- Folder: `Processed_data/cleaned_iv_data`
- `cleaned_manifest.csv`: each output csv file and row counts
- `all_cleaned_iv.csv`: concatenated cleaned data
- Per-scan CSVs: one file per `(temp, direction, scan_idx)`

## Notes
- Raw files in `data/` are not overwritten.
- Export transform order is `V -> R -> I`.
- `V` is clipped during export: `V > 0.4 -> 0.4`, `V < -0.4 -> -0.4`.
- `R` is computed as `V/I` using clipped `V` and original `I`, then `R > 20` or `R < 0.01` is forced to `0`.
- After `R` is finalized, `I` is clamped: if `|I| < 1e-5`, set `I = 0`.
- This export reflects the currently executed notebook state.
