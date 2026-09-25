# T05 — experiment scripts (`scripts/`)

Wire the modules from T01–T04 into runnable scripts driven by `configs/deit_small.yaml`.
Read `docs/data_plan.md`, `docs/vast_workflow.md`, and the `docs/modules_T0*.md` API notes.
Scripts must work on CPU with `--limit N` (tiny smoke run) and on CUDA for the real run.

## Common

- `scripts/_common.py`: `load_config(path)`, `get_device(arg)`, `set_all_seeds(seed)`,
  `model_fingerprint(model)` (sha256 of state_dict bytes + timm version) written into every output,
  `OutputPaths(cfg)` giving `cache/`, `results/raw/`, `results/tables/`.
- Every script takes `--config`, `--device {auto,cpu,cuda}`, `--limit` (max images per split, for smoke runs), and writes a `run_manifest.json` (config, git commit hash, fingerprint, timestamps, elapsed seconds).
- Use `tqdm`. Save tables as `.parquet` (pyarrow) **and** `.csv`.

## `scripts/download_data.py`

- Downloads `imagenetv2-matched-frequency.tar.gz` from
  `https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz`
  into `data/` (skip if present, verify size == 1264079360 bytes). NOTE: despite the `.gz` name the file is a
  plain POSIX/pax tar (magic `PaxH`); open with `tarfile.open(path, 'r:*')`. Extract into `data/`, which yields
  `data/imagenetv2-matched-frequency-format-val/<class_idx>/<file>.jpeg` (1000 folders x 10 images). Set
  `data.root` in the config to that folder. A local copy already exists on the dev machine; do not re-download if present.
- Builds `records = [(path, label, image_id)]` where label = int(folder name), image_id = `f"{label}_{filename_stem}"`.
- Calls `data_protocol.make_splits` with the config's split settings and writes `configs/splits_seed{seed}.json`
  (only if it does not exist; refuse to overwrite silently).
- Prints per-split counts per class (should be 4/2/2/2).

## `scripts/cache_features.py`

For each split in `{score, val, test}` (and `fit` only if `--include-fit`), for clean and for each
(corruption, severity) in observed+unseen, with the model in eval mode on `device`:

- One forward per batch through `BlockOutputRecorder` for all 12 layers.
- Save **summaries** (`cls`, `patch_mean`) for all layers as fp32 to
  `cache/{split}/{corruption}_{severity}/summaries.pt` with `image_ids`, `labels`, `logits`.
- Save **full token tensors** only for the layers listed in `cfg.representation.layers`, in fp16, to
  `cache/{split}/{corruption}_{severity}/tokens_layer{l}.pt` (needed by patching). Add a `--tokens-dtype`
  flag and a check script that compares fp16-cached vs fp32-recomputed patched logits on 64 images
  and reports max abs diff (proposal §18.1 last paragraph).
- `clean` is stored once per split (`cache/{split}/clean/...`).

## `scripts/compute_metrics.py`

Loads summaries for `score` split, computes `layerwise_scores` for every metric, corruption,
severity, summary mode. Also computes `channel_scores` (mean |Δ| per channel per layer, for the
`score_topk` mask policy) and `r_l` (clean per-element RMS per layer, for the norm cap). Writes
`results/tables/metrics_score.parquet` and `cache/score/channel_stats.pt`.

## `scripts/run_patching.py`

Experiment B sweep on a given split (`--split val|test`) using cached tokens:
layers × fractions × alphas × mask_seeds × corruptions × severities, plus the control conditions
listed in `cfg.patching.controls` (controls run with 1 seed, default budget). Streams rows to
`results/raw/patching_{split}.parquet` in chunks (do not hold everything in RAM). `--experiment E1|E2`
selects the budget grid (E1: default alpha only; E2: full grid + norm cap if `rho` set).

## `scripts/select_sites.py`

Applies every selector from `site_selection` using **only** `score` metrics and `val` patching
results; writes `results/tables/selections.parquet` with candidate, label_access, selection_cost.
Direction for each metric is chosen on val by rank correlation sign and recorded (never re-chosen later).

## `scripts/evaluate.py`

Loads `results/raw/patching_test.parquet` and `selections.parquet`, computes per-condition U,
recovery / new-error, regret per selector, equivalence sets, and paired bootstrap CIs; separate
tables for observed vs unseen corruptions (E3). Writes `results/tables/E1_*.csv`, `E2_*.csv`, `E3_*.csv`
and a `results/summary.md` with the main tables in markdown (this is pasted into README).

## `scripts/train_adapters.py`

Experiment D: trains adapters (exhaustive reference over layers × widths × seeds, or `--site L`
for the realistic single-site run) on `fit`, evaluates on `val` and `test` (observed + unseen +
clean), writes `results/raw/adapter_{split}.parquet` and cost tables.

## `scripts/smoke_test.sh`

Runs the whole chain on CPU with `--limit 20` and a 2-layer subset to prove the plumbing before
renting a GPU. Must finish in a few minutes.

## Tests

`tests/test_scripts_smoke.py`: with a tiny random ViT and a synthetic 2-class folder of 12 images
(generate with PIL in a tmpdir), run cache → metrics → patching (1 layer, 1 q, 1 seed) → evaluate,
assert outputs exist and row counts are as expected. Mark it `slow` if it exceeds ~60 s.

## Deliverables

Scripts above, tests, `docs/modules_T05.md` with the exact command sequence for a full GPU run.
Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q` and report the summary line.
