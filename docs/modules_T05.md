# T05: experiment scripts

The scripts call the tested T01–T04 library APIs; no library changes are needed.
Install the project and runtime dependencies before running:

```bash
pip install -e . timm imagecorruptions scikit-learn scipy pandas pyyaml tqdm pyarrow
```

Every entry point accepts `--config`, `--device auto|cpu|cuda`, and `--limit`.
Paths in YAML are relative to the working directory; run commands at the repository
root. `output_root` defaults to `.`; it contains `cache/`, `results/raw/`, and
`results/tables/`. Small tables have both CSV and Parquet copies; raw patching
results are Parquet only. Cache tensors and table
rows contain the SHA-256 model-state/timm-version fingerprint. Analysis checks
fingerprints and cache consumers check original-image order. Each stage writes
`results/manifests/<stage>[_<experiment>][_<split>]/run_manifest.json` with effective
configuration, arguments, commit, fingerprint, timestamps, and elapsed seconds.
Download preparation is model-independent and records a null fingerprint.

## Full GPU sequence

```bash
python scripts/download_data.py --config configs/deit_small.yaml --device cuda
python scripts/cache_features.py --config configs/deit_small.yaml --device cuda
python scripts/check_tokens_precision.py --config configs/deit_small.yaml --device cuda --limit 64
python scripts/compute_metrics.py --config configs/deit_small.yaml --device cuda
python scripts/run_patching.py --config configs/deit_small.yaml --device cuda --experiment E1 --split val
python scripts/select_sites.py --config configs/deit_small.yaml --device cuda
python scripts/run_patching.py --config configs/deit_small.yaml --device cuda --experiment E1 --split test
python scripts/evaluate.py --config configs/deit_small.yaml --device cuda

# Preserve E1 before replacing the active raw tables and selections with E2.
cp results/tables/selections.parquet results/tables/E1_selections.parquet
cp results/tables/selections.csv results/tables/E1_selections.csv
cp results/raw/patching_val.parquet results/raw/E1_patching_val.parquet
cp results/raw/patching_test.parquet results/raw/E1_patching_test.parquet
cp results/tables/E3_observed.parquet results/tables/E1_observed.parquet
cp results/tables/E3_observed.csv results/tables/E1_observed.csv
cp results/tables/E3_unseen.parquet results/tables/E1_unseen.parquet
cp results/tables/E3_unseen.csv results/tables/E1_unseen.csv
cp results/summary.md results/E1_summary.md
python scripts/run_patching.py --config configs/deit_small.yaml --device cuda --experiment E2 --split val
python scripts/select_sites.py --config configs/deit_small.yaml --device cuda
python scripts/run_patching.py --config configs/deit_small.yaml --device cuda --experiment E2 --split test
python scripts/evaluate.py --config configs/deit_small.yaml --device cuda
python scripts/train_adapters.py --config configs/deit_small.yaml --device cuda
# Realistic run: substitute the externally selected layer and use a separate
# YAML output_root to preserve the exhaustive adapter reference artifacts.
# python scripts/train_adapters.py --config configs/selected.yaml --device cuda --site 6
```

E2 includes the full alpha grid; set `patching.norm_cap_rho` to a nonnegative
number to add a capped grid alongside the uncapped grid. Controls use the first
configured fraction, default alpha, first seed, and no cap. Full-state control
always overrides fraction and alpha through the library API. The
`random_fixed_nested` configuration maps to the library's seeded permutation
prefix masks, so budgets share nested masks. Channel scores average token/channel
absolute deltas over observed score conditions only; RMS is computed from fp32
clean token activations before cache quantization.

All 12 block summaries, logits, labels, and RMS values are cached for every
condition even when donor layers are restricted. Full-token donor files are
written only for clean conditions, which skip corruption generation. Donor
channel absolute deltas are reduced during observed score caching and saved in
the summaries for `compute_metrics.py`; corrupted token files are unnecessary.
Donor
tensors default to fp16; `--tokens-dtype float32` changes storage. The precision
check reports actual cached-fp16 versus recomputed-fp32 patched-logit maximum
absolute errors at alpha one for each layer/fraction. Review that report before
accepting fp16 for a full study; rerun caching in fp32 if its discrepancy is
unacceptable. Cache generation holds one condition's selected token tensors in
host memory. Patching retains donors in their cached dtype on the host and casts
only the current batch to fp32 on the device. It bounds result memory to one image
batch's sweep and writes Arrow row groups incrementally, logging elapsed time,
rows written, and estimated remaining time after each condition with stdout flushed.
Shuffled-donor controls require batches of at
least two images; choose a batch size that does not leave a singleton tail.

Both caching and patching accept `--batch-size B`. Defaults come from
`runtime.batch_size_cuda` (128) on CUDA and `runtime.batch_size` (32) on CPU.
Patching also accepts `--mask-seeds N`, `--layers a,b,c`, and
`--fractions q1,q2`; requested layers must already have clean donor caches.
Effective overrides are recorded in the manifest configuration. Loaders use
`runtime.num_workers` (default 4, base config 8), `runtime.pin_memory` (default
true on CUDA, explicitly true in the base config), and nonpersistent workers.
Per-image corruption seeds preserve results across worker counts.

Selection is separate per nominal fraction, alpha and cap. It uses observed score
metrics and observed validation utilities only. Metric directions are frozen by
validation Spearman sign; undefined/zero correlation deterministically uses max.
All selectors receive the validation admissibility check, and metric direction
search costs are recorded separately. Small search uses first/middle/last unique
candidate layers. Clean constraints use the clean-to-clean diagnostic relative
to cached unpatched clean predictions. Evaluation never refits directions or
filters choices using test outcomes. It reports per-condition recovery/new-error,
seed variability, and observed/unseen balanced-mixture gain and regret CIs using
paired original-image bootstrap. Equivalence sets include no intervention.

Selection and evaluation project only analysis columns from Parquet and convert
repeated identifiers to categoricals before pandas aggregation. Fingerprints are
checked across row-group statistics, with bounded Arrow reads when statistics
are absent. Evaluation reads margin columns separately in bounded Arrow batches
and retains only differences to preserve the existing `margin_change` output;
raw margins, losses, donor IDs, and fingerprints are excluded from analysis frames.
Condition effects use vectorized groupby reductions. Bootstrap CIs are computed
on per-image aggregates (the mathematically identical statistic for the balanced
sweep, with a different resample stream than the library function). Evaluation
checks balanced image/layer counts, aggregates each experiment/fraction/alpha/cap/
domain once, and reuses paired NumPy layer replicates across frozen selectors.
No-intervention utility remains zero and each regret replicate recomputes the
best admissible layer including no intervention. Progress is printed per domain
and selection, and manifests/console output retain total wall time. The full
24.5-million-row performance targets require a real-data benchmark; the offline
tests verify correctness rather than those wall-time targets.

Adapter fitting uses only observed corruptions on fit, with matching initialization
and loader seeds across sites. Evaluation includes observed, unseen, and one copy
of clean rows. Checkpoints, loss curves, per-site costs and exhaustive reference
total compute are recorded. `--site` uses the configured default width. Training
steps, optimizer settings and runtime batch size are explicit in the base YAML.

## Smoke and tests

```bash
bash scripts/smoke_test.sh
# Portable equivalent, including Windows:
python scripts/smoke_test.py --device cpu --limit 20
C:/anaconda/envs/layer-research/python.exe -m pytest -q
```

The real smoke uses pretrained DeiT-S, layers 0/11, fraction 0.1, one mask seed,
severity 1, Gaussian noise (observed) and contrast (unseen). It runs all controls,
uses 100 bootstrap replicates, stores artifacts under `outputs/smoke`, and checks
40 metric rows, 480 rows per patching split, 16 selections, 24 effect rows, and 16
selector rows per observed/unseen table. The elapsed-time file explicitly marks
failed chains. The immutable split is generated from the full dataset before
applying the per-split image limit; limited images use deterministic shuffled order.
An existing extracted ImageNetV2 folder skips the entire archive download path.
Existing split files must exactly match both split settings and image membership.

The offline integration test uses a random two-block ViT and 12 PIL-generated
images, exercises E1/E2, all controls, precision checking and one-step adapter
training, verifies row counts, cache identity, fingerprints, and no unseen-data
selection leakage. It takes well under 60 seconds in the development environment.
The kNN library metric requires at least 11 score images; smaller synthetic
configurations must omit that metric rather than silently alter its definition.

On Windows, prepend the conda environment and `Library/bin` to process PATH and
set `MKL_THREADING_LAYER=SEQUENTIAL`, as documented for T01.

## Validation in this workspace

- Requested command: `60 passed, 7 warnings, 5 errors in 8.16s`. The five errors
  occur at fixture setup because the existing `.pytest_tmp` directory is denied
  by the environment; there are no assertion failures in that run.
- Same suite with `--basetemp` under the system temporary directory, outside the
  repository: `65 passed, 2 warnings in 12.10s`. No alternative temporary directory
  was created in the repository.
- Real-data CPU command was actually attempted with `--limit 20`, pretrained
  `deit_small_patch16_224`, and the smoke grid above. Download preparation skipped
  the existing extracted data and verified 4/2/2/2 images per class. The attempt
  stopped loading pretrained weights because the shell's network policy rejects
  Hugging Face connections with `WinError 10013`. Wall time was **62.31 seconds,
  failed/incomplete**; real pretrained tables could not be verified.
- Logs are under `outputs/validation/`; timing and effective smoke YAML are under
  `outputs/smoke/`. No library modules or Vast infrastructure scripts changed, and
  no commits were created.
