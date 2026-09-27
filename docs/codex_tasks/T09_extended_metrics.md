# T09 — Extended selection metrics (E6): task sensitivity and topological (PH) distances

Proposal §7.3 (task-sensitivity and geometry rows), §11.2 and §12 (TDA). These are *additional
selectors* evaluated against the diagnostic sweep that already exists; no new patching runs are needed.

## 1. Task-sensitivity score (label-using, backward-using) — `representation_metrics.py` + script

For block $l$ and image $i$, the local first-order effect of moving the corrupted representation toward
the clean one on the true-label margin is $\langle \nabla_{h_l} m(g_l(h_l(\tilde x)), y),\; h_l(x)-h_l(\tilde x)\rangle$
(§11.2). Implement in a new module `src/layer_research/task_sensitivity.py`:

```python
def margin_gradient_scores(model, loader, layers, device, blocks_attr="blocks") -> pandas.DataFrame
    # one forward on corrupted input with hooks that keep block outputs requiring grad
    # (retain_graph across layers: run one forward, compute margin m = z_y - max_{k!=y} z_k, call
    #  torch.autograd.grad(m.sum(), [h_l for l in layers])), then per image and layer:
    #   dot_full   = <grad, delta>                 (delta = h_clean - h_corr, full tensor)
    #   dot_norm   = dot_full / (||grad|| ||delta|| + eps)   (cosine between gradient and delta)
    #   grad_norm  = ||grad||
    # Return per-image rows: image_id, label, layer_id, dot_full, dot_norm, grad_norm, delta_norm.
```

`scripts/compute_task_sensitivity.py --config ... --split score` computes this on the **score split** for
the observed corruptions (severities 1/3/5) using the cached clean tokens as `h_clean` and writes
`results/tables/task_sensitivity_score.csv` with the per-layer means of `dot_full` and `dot_norm`
(columns: layer, metric_name ∈ {`task_sens_dot`, `task_sens_cos`}, score, n_samples, summary_mode="full",
label_access="labels", corruption, severity, split, is_observed, model_fingerprint). Backward count and
wall time go into the run manifest (this is the cost of the metric, §7.3).

## 2. Persistent-homology distance — `src/layer_research/topology.py`

Add `ripser` to the pip deps (`environment.yml`, install into the conda env with
`C:/anaconda/envs/layer-research/python.exe -m pip install ripser persim`). Implement:

```python
def ph_diagrams(X, maxdim=1, pca_dim=32, seed=0, n_points=None) -> dict[int, ndarray]
    # X: [n, d] (numpy). Optional fixed PCA (fit on the *clean* matrix passed via `pca=` argument,
    # so clean and corrupted use the same projection, §12.3), optional fixed random subsample of rows.
def ph_distance(H, Ht, *, dims=(0, 1), metric="bottleneck"|"wasserstein", **kw) -> dict
    # returns {"ph_bottleneck_H0":..., "ph_bottleneck_H1":..., "ph_wasserstein_H0":..., "ph_wasserstein_H1":...}
    # (persim.bottleneck / persim.wasserstein). Infinite bars: drop, and record how many were dropped.
```

Fix the PCA on the clean representation of the score split for that layer/summary; use the same
projection for the corrupted set. Default 2 000 points, H0 and H1 (ripser on 2 000 points in 32-D is
fine). Record wall time and peak memory per call.

`scripts/compute_topology.py --config ... --split score` writes `results/tables/topology_score.csv`
with the same six-column layout as `metrics_score.csv` (metric names `ph_bottleneck_H0/H1`,
`ph_wasserstein_H0/H1`; `label_access="none"`), for both summary modes.

## 3. Plug into selection and evaluation

`scripts/select_sites.py`: after reading `metrics_score`, also read (if present)
`task_sensitivity_score` and `topology_score` and concatenate them so the same metric-based selector
loop covers them (direction frozen on val exactly as for the others). `scripts/evaluate.py` needs no
change (it iterates over the frozen selections). Add `--extra-metrics` flag defaulting to on.

Because the frozen `selections.parquet` from the main run must not be silently overwritten, write the
combined selections to `results/tables/selections_extended.parquet` and let `evaluate.py` accept
`--selections <path>` (default the original file) and an `--output-suffix` (default empty) for the
`E3_*`, `E1_selectors` and `summary.md` outputs.

## 4. Tests

- Task sensitivity: on the tiny random ViT, for a tiny α the change in margin under partial patching
  equals `alpha * dot_full` to first order (relative error < 5 % for α = 1e-3 on a full-channel mask).
- Topology: identical point clouds give zero distances; a rotated clean cloud (orthogonal Q) gives
  zero bottleneck distance for H0 and H1 (PH is isometry-invariant); subsampling with the same seed is
  deterministic.
- `select_sites` on the smoke fixture with the extra tables present adds the new selectors and leaves
  the original ones unchanged.

Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q` and report the summary and files changed.
Do not commit.
