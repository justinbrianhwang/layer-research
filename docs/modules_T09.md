# T09: extended selection metrics

`margin_gradient_scores` accepts the paired loader format or mapping batches with
`x_corr`, `y`, `image_id`, and aligned `clean_outputs[layer]`. Without cached
outputs it also needs `x_clean`. It evaluates the true-label margin gradient for
all requested blocks in one autograd call per batch and reports full-token dot
products, cosine scores, gradient norms and delta norms per image. It works with
frozen backbone parameters, leaves parameter gradients untouched, and removes
hooks on errors. The model remains in eval mode. DataFrame attributes record
forward/backward counts and synchronized wall time.

`compute_task_sensitivity.py --config ... --split score` uses clean token caches
with matching model fingerprints and image order. It scores observed configured
corruptions at severities 1/3/5, writes `task_sensitivity_score.csv/.parquet`, and
records backward counts and wall time in its manifest. Cached fp16 donors retain
their quantization; cache float32 tokens when testing precise local derivatives.

`ph_diagrams` lazily imports ripser, returns finite H0/H1 diagrams, and records
infinite-bar removal counts and deterministic sample indices in `.metadata`.
`ph_distance` accepts aligned point clouds or diagram dictionaries, uses persim,
and supports `bottleneck`, `wasserstein`, or `both`. A supplied `pca` must already
be fitted on clean data; otherwise the pair API fits it on the clean matrix once.
Set `pca_dim=None` to disable PCA. Isometry invariance holds before projection;
a fixed dimension-reducing projection need not preserve a rotated cloud's
distances. The default pair subsample is 2,000 points, shared by row index.

`compute_topology.py --config ... --split score` fits one PCA per layer/summary
on the full clean score matrix, reuses it for every corruption, and computes both
distances for H0/H1. It writes `topology_score.csv/.parquet` for configured summary
modes. `--n-points`, `--pca-dim`, and `--seed` expose the defaults 2000/32/0.
Per-call wall time and peak process RSS sampled every 10 ms are recorded in the
manifest on Windows/Linux, including the resident baseline and native ripser
allocations. Short-lived peaks between samples can be missed. Traced allocation
peaks are also recorded; platforms without an RSS reader use that explicitly
labelled fallback, which excludes untraced native allocations.

`select_sites.py` loads optional extra tables by default (Parquet preferred, CSV
supported), validates provenance, and applies the existing observed-only metric
direction/admissibility logic. When an extra table exists it writes only
`selections_extended.csv/.parquet`; otherwise normal output is unchanged.
`--no-extra-metrics` explicitly selects the original workflow.

Evaluate the extended selections without replacing original reports:

```bash
python scripts/evaluate.py --config configs/deit_small.yaml --selections results/tables/selections_extended.parquet --output-suffix _extended
```

The suffix is appended verbatim to table/report names and the evaluation manifest
directory. Empty suffix and omitted selections retain the original behavior.
The optional topology tests skip when ripser/persim is unavailable; sensitivity
and selector preservation tests run without them. All test temporary files must
be placed under the system temporary directory using pytest `--basetemp`.
