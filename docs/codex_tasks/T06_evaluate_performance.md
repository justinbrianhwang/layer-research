# T06 — make `scripts/evaluate.py` and `scripts/select_sites.py` scale to the real sweep

## Problem

The real E1 sweep produces `results/raw/patching_{val,test}.parquet` with **24.5 million rows** each
(2 000 images × 12 layers × 4 fractions × 20 mask seeds × 12 corruption conditions, plus controls;
~775 MB Parquet, >100 GB as a full pandas frame with string columns).

- `select_sites.py` loads every column with `read_table` and takes >30 min and >100 GB RAM.
- `evaluate.py` calls `paired_bootstrap` / `bootstrap_regret` with 1 000 resamples on multi-million-row
  frames, re-running pandas groupbys per resample. Extrapolated cost is tens of hours per split.

## Required changes (scripts only; library modules unchanged)

### 1. Column projection and dtype hygiene

Read only the needed columns with `pyarrow.parquet.read_table(path, columns=[...])`, and cast
`corruption`, `intervention_type`, `mask_policy` to `category` and `image_id` to `category`
before any pandas work. Never load `donor_image_id`, margins, losses, or `model_fingerprint` for
the selection/evaluation paths (read the fingerprint once from the Parquet metadata / first row group
via `pq.ParquetFile(path).schema` or a 1-row read, and keep the existing fingerprint check).

Add `read_table_columns(path, run, columns)` to `_common.py` for this and use it in both scripts.

### 2. Per-image pre-aggregation before bootstrapping

For the partial-channel rows, every original image contributes the same number of rows to every
layer (seeds × conditions). The statistics used (`U` = mean correctness difference, regret over
layer-wise `U`) are means over rows, so they are **exactly** equal to means over per-image averages.
Build once per (experiment, fraction, alpha, norm_cap, domain) a dense array

```
D[image, layer] = mean over (seed, condition) of (post_correct - baseline_correct)
```

(plus the image id order), then:

- `U_layer = 100 * D[:, layer].mean()`
- paired bootstrap of `U` for the chosen layer: resample image indices with `np.random.default_rng(seed)`,
  `100 * D[idx, layer].mean()`; percentile CI as in `statistics.paired_bootstrap`.
- bootstrap regret: same resampled `idx`, compute `100 * D[idx][:, admissible].mean(0)`, take
  `max(...) - value[chosen]` with `none = 0`.

Implement this as `fast_image_bootstrap(D, ...)` in `scripts/_common.py` (numpy only) and keep the
library `paired_bootstrap` untouched. Add a test in `tests/test_scripts_smoke.py` that on a small
synthetic frame the fast estimate of `U` equals `evaluation.accuracy_gain_pp` exactly and the CI
from the fast path with the same seed is within the CI width of the library path (the resample
streams differ, so do not require equality).

Per-image aggregation must also be used for the per-condition `layer_effect_table` inputs where
possible: compute recovery / new-error / margin summaries with vectorised groupby on the projected
frame (they are cheap once strings are categorical); do not loop in Python over groups.

### 3. `select_sites.py`

Use the projected columns (`label, layer_id, intervention_type, fraction, alpha, norm_cap,
experiment, is_observed, baseline_prediction, clean_prediction, post_intervention_prediction`).
Everything else in its logic stays identical (same selectors, directions, admissibility, outputs).

### 4. Progress and cost

Print per-domain/per-selection progress and total wall time. Target: `select_sites` < 3 min and
`evaluate` < 15 min on the real 24.5 M-row files with 1 000 resamples, on a 64-core box.

### 5. Do not change results semantics

`E1_effects`, `E1_seed_variability`, `E3_observed`, `E3_unseen`, `E1_selectors`, `selections`,
`summary.md` keep their schemas. Note in `docs/modules_T05.md` that bootstrap CIs are computed on
per-image aggregates (mathematically identical statistic, different resample stream than the library
function).

Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q` and report the summary line and the
files changed. Do not commit.
