# T08 — ResNet-50 support for cross-architecture re-validation (E5)

Proposal §6.1, §6.5, §10.3 and §14.2 (E5). The library currently assumes a timm ViT
(`model.blocks[i]` returning `[B, N, d]`, CLS token at index 0, `norm1`, `patch_embed`, ...).
Add a **CNN path** so the same experiments run on `timm` `resnet50.a1_in1k` (ImageNet-1k pretrained,
no extra data) with **16 candidate sites = the 16 bottleneck residual blocks** (`layer1[0..2]`,
`layer2[0..3]`, `layer3[0..5]`, `layer4[0..2]`), block output = the tensor after the residual add + ReLU.

## Design (keep the ViT path byte-for-byte unchanged; all existing tests must still pass)

1. `feature_extractor.py`
   - `resolve_blocks(model, blocks_attr)`: returns a **list of modules**. `blocks_attr="blocks"` →
     `list(model.blocks)`; `blocks_attr="resnet_stages"` → the 16 bottleneck modules in order. Every
     function that indexes `model.blocks[i]` must go through this helper (`BlockOutputRecorder`,
     `shape_report`, `num_blocks`, `iter_paired_block_outputs`, `extract_paired_features`).
   - `summarize(tokens, mode)` must accept 4-D `[B, C, H, W]`: `"gap"` → global average pool `[B, C]`;
     `"cls"` and `"patch_mean"` remain ViT-only and raise a clear error on 4-D input. Add
     `"gap"` to the allowed modes.
2. `patching_engine.py`
   - `Patcher` / `run_patched_forward` / `sweep_layers_budgets`: use `resolve_blocks` (add a
     `blocks_attr` argument, default `"blocks"`). Channel masks for 4-D outputs apply to dim 1
     (`[B, C, H, W]`, mask shape `[C]`, broadcast over H, W). `exclude_cls` is a no-op for 4-D.
     All norm/statistics code must use `flatten(1)` semantics that work for both layouts (they
     mostly already do). `FULL_STATE`, `CLEAN_TO_CLEAN`, `RANDOM_DIRECTION`, `SHUFFLED_DONOR` must all
     work for 4-D. **BatchNorm**: the model is in eval mode, so running stats are frozen; assert
     `not model.training` inside the hook and document.
3. `adapter_training.py`
   - `BottleneckAdapter` gets a `layout` argument: `"tokens"` (current) or `"channels_last_4d"`. For
     4-D input the adapter is applied per spatial position: permute `[B,C,H,W]` → `[B,H,W,C]`, apply
     `LN → down → act → up`, permute back, residual add. Same parameter count formula.
   - `attach_adapter` and the training loop use `resolve_blocks`. Replace the ViT-specific FLOP
     estimator with a generic one: measure forward FLOPs per image with a hook-based count of
     `nn.Linear` and `nn.Conv2d` multiply-adds (2 FLOPs each) over the whole model, and per-block
     FLOPs by the same counter restricted to each block module; backward FLOPs = 2 × forward of the
     blocks after the site + head (document). Keep the ViT numbers within 10 % of the old estimator
     (add a test).
   - The adapter width `d` for a site is inferred from a dummy forward (`shape_report`), not from
     `norm1`.
4. Config + scripts
   - `configs/resnet50.yaml`: copy of `deit_small.yaml` with `model.name: resnet50.a1_in1k`,
     `model.blocks_attr: resnet_stages`, `model.num_blocks: 16`, `representation.layers: [0..15]`,
     `representation.summary_modes: [gap]`, `output_root: outputs_resnet50`, `patching.mask_seeds: 10`,
     `adapter.widths: [32]`, `adapter.training_seeds: 1`. Everything else identical.
   - Scripts must read `model.blocks_attr` from the config and pass it through (`_common.create_model`
     stays; `cache_features.py`, `run_patching.py`, `check_tokens_precision.py`, `train_adapters.py`,
     `compute_metrics.py`). `cache_features.py` summary modes come from the config (currently
     hard-coded `("cls","patch_mean")`). Token caches for ResNet are `[B, C, H, W]` fp32; sizes are
     larger for early stages (256×56×56 ≈ 3.2 MB/image) — cache only the layers in
     `representation.layers` and note the disk need (≈ 2000 images × 16 sites ≈ 20 GB per split) in
     `docs/modules_T08.md`; add a `--layers` flag to `cache_features.py` so a run can be split.
   - `docs/modules_T08.md`: what changed, what "block output" means for ResNet, the measured/estimated
     FLOP numbers for both models, and the exact command sequence for the ResNet E1 + E4 run
     (`scripts/gpu_run_E1.sh`-style driver `scripts/gpu_run_resnet.sh`: download → cache → metrics →
     patch val (10 seeds) → select → patch test → evaluate → adapters width 32 seed 0).
5. Tests (CPU, no downloads): a tiny random `timm.create_model("resnet18", pretrained=False,
   num_classes=10)` with `blocks_attr="resnet_stages"` (8 basic blocks) — shape report, recorder hook
   cleanup, `NONE`/`FULL_STATE`/`CLEAN_TO_CLEAN` controls exact, nested masks over channels,
   adapter identity at init and gradient reaching the adapter at site 0 and the last site, and a
   2-image `sweep_layers_budgets` row count. Plus the FLOP-estimator agreement test on the tiny ViT.

Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q` and report the summary and files changed.
Do not commit.
