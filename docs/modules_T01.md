# T01 modules

These modules implement proposal sections 5–7 and the applicable checks in section 18.
All exports below are available from `layer_research`.

## Data protocol

- `SplitConfig`: immutable split fractions, explicit seed and stratification switch.
- `SplitResult`: dict-compatible split result retaining generation metadata as `.cfg`.
- `make_splits`: split unique original IDs with seeded shuffling and per-class largest-remainder allocation.
- `verify_disjoint`: reject repeated IDs within or across splits.
- `save_splits`: save sorted IDs, configuration and seed to JSON.
- `load_splits`: load split IDs and configuration, validating metadata and disjointness.
- `CorruptionSpec`: immutable supported corruption name, severity (1–5) and explicit seed.
- `corrupt_image`: generate an RGB PIL corruption using imagecorruptions and a stable per-ID seed.
- `PairedImageDataset`: load `(path, label, original_image_id)` records and return clean/corrupted tensors, label and ID.
- `OBSERVED_CORRUPTIONS`, `UNSEEN_CORRUPTIONS`, `DEFAULT_SEVERITIES`: fixed protocol constants.

`PairedImageDataset(records, spec, model, data_config=None)` resolves the model's timm
evaluation transform; `data_config` can explicitly override input size and other transform
settings. `pre_transform` applies resize/crop to the RGB PIL image, then corruption happens
at model input resolution before `post_transform` converts to a tensor and normalizes,
matching the ImageNet-C protocol. `input_resolution` exposes the configured (H, W) for
logging. Pipelines without `ToTensor` or timm's `MaybeToTensor` raise `ValueError`.
The resized/cropped images must meet imagecorruptions' minimum size (32 by 32).
Generated corruptions are not official
ImageNet-C benchmark files. Original identity is supplied by the caller, never inferred
from filenames. Duplicate original IDs are rejected rather than silently deduplicated.

## Feature extraction

- `BlockOutputRecorder`: context manager recording detached block-output snapshots, with guaranteed hook removal.
- `num_blocks`: count blocks under the configurable module attribute.
- `summarize`: compute CLS, patch mean excluding CLS, or concatenated summaries.
- `iter_paired_block_outputs`: yield one CPU batch of full-token pairs, logits, labels and IDs.
- `extract_paired_features`: concatenate image-level summaries and aligned prediction metadata.
- `shape_report`: report each block's output shape using one zero-valued input.

All model operations force eval mode and leave it enabled. Extraction moves the model to
the requested device and disables gradients; the recorder itself only detaches snapshots.
All model-taking feature APIs accept `blocks_attr="blocks"` (including dotted module paths).
Summaries assume exactly one leading CLS token, as in non-distilled DeiT; alternative
prefix-token layouts and CNN pooling require a separate summary implementation.

The extraction dictionary has `clean[layer][summary_mode]` and
`corrupted[layer][summary_mode]` tensors of shape `[n, d]` (or `[n, 2d]`), plus
`clean_logits`, `corrupted_logits`, `labels`, and `image_ids`. Everything is on CPU and
keeps loader order. Streaming uses `clean[layer]`/`corrupted[layer]` full tensors instead;
no hooks or gradient-mode context remain active across a yield. Empty aggregate loaders
raise ValueError. Only summaries accumulate in memory during aggregate extraction.

## Representation metrics

- `relative_distance`: mean per-image L2 difference divided by clean L2 norm plus epsilon.
- `raw_distance`: mean per-image L2 difference.
- `activation_norm`: mean clean per-image L2 norm.
- `cosine_distance`: mean one minus cosine similarity.
- `linear_cka`: column-centered linear CKA, returning NaN and logging for a denominator below epsilon.
- `one_minus_cka`: one minus linear CKA.
- `amplification_ratio`: consecutive nonnegative score ratios with the first ratio fixed to 1.
- `knn_preservation`: mean Jaccard overlap of Euclidean neighbor sets, excluding self.
- `layerwise_scores`: convert the extraction dictionary into the requested six-column metric table.

Metrics accept finite, nonempty, aligned `[n, d]` torch or NumPy matrices, compute on CPU
in float64, and never inspect labels. Corresponding rows must refer to the same original
IDs; metric functions cannot infer identity from matrix values. Cosine norms are floored
at epsilon; a zero vector has cosine zero. kNN requires `1 <= k < n` (default 10), breaks
distance ties by input row order, and uses row-wise distances to avoid an n-by-n cache.
Layerwise kNN therefore requires at least 11 samples.

## Explicit choices and deviations

- The specification's two-argument `save_splits` cannot recover configuration from a plain
  dictionary. `make_splits` and `load_splits` return `SplitResult`, so that form works;
  callers supplying ordinary dictionaries must pass the optional `cfg` argument.
- Corruption seeds use SHA-256 of seed and original ID instead of Python's process-randomized
  `hash`. NumPy's legacy RNG is temporarily seeded and restored under a lock, because
  imagecorruptions uses it. The lock serializes this helper's calls, not unrelated NumPy RNG
  consumers in other threads; use dataset worker processes for concurrent generation.
- Amplification stabilizes division with `max(previous_score, eps)`; first-layer ratio is
  1.0, including when its score is zero. In `layerwise_scores`, amplification is explicitly
  based on relative distance, ordered by ascending requested layer. Missing layers therefore
  mean consecutive *selected* layers, not necessarily adjacent backbone blocks.
- Supported corruption names are the four preregistered conditions; all reference severities
  1–5 are allowed, with 1/3/5 exposed as defaults.

## Validation

Tests use a seeded random three-block ViT with 32-dimensional embeddings on CPU, temporary
images and no network access. They verify split provenance and stratification, all four
corruptions, NumPy RNG restoration, hook cleanup on normal and exceptional exits, shapes,
summaries, streamed and aggregated extraction, direct-logit reproduction, metric-table
alignment, distance/neighbor calculations, and CKA against centered-Gram HSIC.

Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q`. On Windows, activate the
conda environment first or prepend `C:\anaconda\envs\layer-research\Library\bin` and
`C:\anaconda\envs\layer-research` to the test process PATH so NumPy can load its BLAS DLLs.
`MKL_THREADING_LAYER=SEQUENTIAL` was also required in this environment to avoid a native runtime abort when NumPy BLAS and PyTorch run in the same process. These settings were applied only to the test process; no system settings or installed packages were changed.
