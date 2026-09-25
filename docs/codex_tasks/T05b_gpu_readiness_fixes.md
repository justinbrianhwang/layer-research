# T05b — fixes required before the full GPU run (review findings on T05)

All changes are in `scripts/` (and `configs/deit_small.yaml`). Library modules stay untouched.
Keep `tests/test_scripts_smoke.py` green and extend it where noted. Do not commit.

## 1. Cache full-token tensors for the clean condition only  (disk blow-up)

`scripts/cache_features.py` currently saves `tokens_layer{l}.pt` for every (corruption, severity)
condition. Only **clean** tokens are ever read as donors (`run_patching.py` reads
`cache/{split}/clean/tokens_layer*.pt`); corrupted receivers are recomputed by the forward pass.
At 2000 images × 197 × 384 × fp16 × 12 layers × 13 conditions × 3 splits this is ~140 GB.

Fix: write token files only when `name == "clean"`. Summaries, logits, labels and `r_l` stay as
they are for every condition. Also skip the wasted corruption computation for the clean condition:
give `loader()` a `paired: bool = True` argument; when `False` build a plain dataset that returns
`(clean_tensor, clean_tensor, label, image_id)` without calling `corrupt_image` (add a tiny
`CleanImageDataset` in `_common.py` that reuses `PairedImageDataset.pre_transform/post_transform`).

## 2. Parallel corruption generation  (CPU bottleneck on the GPU box)

`loader()` builds `DataLoader(..., shuffle=False)` with the default `num_workers=0`, so ImageNet-C
corruptions (defocus blur is slow) are generated single-threaded. Add `runtime.num_workers`
(default 4; config sets 8) and `runtime.pin_memory` (default true when device is cuda), pass
`persistent_workers=False`. `corrupt_image` is process-safe (per-image sha256 seed), so results
are unchanged. Add a test that a 2-worker loader yields the same tensors as a 0-worker loader.

## 3. Raw patching output: Parquet only

`run_patching.py` appends every batch to a CSV as well as Parquet. At ~25 M rows per split the CSV
is several GB and useless. Remove the CSV path for `results/raw/patching_*.`; keep CSV for the
small tables in `results/tables/`. Keep incremental Parquet row groups.

## 4. Donor memory

`run_patching.py` loads every donor layer as float32 up front (≈7.3 GB for 12 layers on the full
split). Keep donors in their cached dtype (fp16) in host memory and cast the batch slice to float32
on the device inside the loop (`donors[l][offset:stop].to(device, torch.float32)`).

## 5. CLI overrides for scale

Add to `run_patching.py`: `--mask-seeds N` (overrides `patching.mask_seeds`), `--layers a,b,c`,
`--fractions q1,q2`, `--batch-size B`; record the effective values in the run manifest. Add
`--batch-size` to `cache_features.py` as well. Default batch size for cuda should come from
`runtime.batch_size_cuda` (set 128 in config) and `runtime.batch_size` (32) for cpu.

## 6. Progress and ETA

`run_patching.py`: print one line per condition with elapsed time and rows written, so a remote
run can be monitored from the log. Flush stdout.

## 7. Config

In `configs/deit_small.yaml` add under `runtime`: `num_workers: 8`, `batch_size_cuda: 128`,
`pin_memory: true`. Nothing else in the config changes.

## Deliverables

Modified scripts, updated `docs/modules_T05.md` (note the clean-only token cache and the flags),
extended `tests/test_scripts_smoke.py`. Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q`
and report the summary line, plus the list of files changed.
