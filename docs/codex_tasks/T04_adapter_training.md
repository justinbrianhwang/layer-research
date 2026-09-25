# T04 — `adapter_training`

Implements proposal §10 (Experiment D: deployable learned repair). Read §10.1–10.6, §5.3 and
§18.3. Reuse `feature_extractor`, `data_protocol`, `evaluation` APIs (see `docs/modules_T0*.md`).

## Constraints

- CPU tests on a tiny random ViT; a real run uses `configs/deit_small.yaml` on GPU. `device` is an argument everywhere.
- Backbone and classifier parameters frozen (`requires_grad=False`) but the forward **after** the adapter must keep a gradient path to the adapter (do not wrap the whole forward in `no_grad`). Test this.
- Model in `eval()` during both training and evaluation of the adapter (no dropout / drop-path; §18.3). For a CNN variant, BatchNorm stats stay frozen; document.
- Deterministic given `training_seed`.
- Do not commit; no Co-Authored-By trailers.

## Adapter (§10.1)

```python
class BottleneckAdapter(nn.Module):
    """h + W_up( act( W_down( LN(h) ) ) ), token-wise. W_up zero-initialised so the initial
    output equals the original model exactly. LN affine learnable (document). Bias on both linears."""
    def __init__(self, d: int, r: int, act="gelu")
    def extra_params(self) -> int
    def extra_flops_per_token(self) -> int   # approximate, 2*d*r*2 + LN

def attach_adapter(model, layer: int, adapter) -> handle   # forward hook on model.blocks[layer] that returns adapter(output); removable
```

Provide `count_trainable(model)` and a check that only adapter params are trainable.

## Training (§10.2–10.3)

```python
@dataclass
class AdaptTrainConfig:
    layer: int; width: int; lr: float; weight_decay: float; steps: int; batch_size: int
    lambda_clean: float; training_seed: int; amp: bool=False; log_every: int=50

def train_adapter(model, adapter, fit_loader, cfg, device) -> TrainResult
    # loss = CE(F(corrupted), y) + lambda_clean * CE(F(clean), y)   (same batch: paired dataset)
    # fixed number of optimizer updates = cfg.steps for EVERY layer (§10.3 main setting).
    # Record: wall time, number of updates, images seen, estimated training FLOPs
    #   (forward FLOPs of the full model + backward FLOPs only for blocks >= layer; estimate with a
    #    simple per-block FLOP count from timm's shapes; document the formula), peak memory if cuda.
    # AdamW. Cosine or constant LR (configurable). Return the trained adapter + cost dict + loss curve.

def evaluate_adapter(model, adapter, layer, loader, device) -> DataFrame
    # per-image rows: image_id, label, corruption, severity, baseline_prediction, post_intervention_prediction,
    # baseline_margin, post_intervention_margin (baseline = no adapter). Also evaluate on CLEAN inputs
    # to produce clean-drop rows (§10.6). Adapter is removed after evaluation (hook removed; test it).

def train_all_sites(model, layers, widths, seeds, fit_loader, val_loader, cfg_base, device, out_dir) -> DataFrame
    # exhaustive reference (§10.5): loops sites × widths × seeds, saves adapter state_dicts and per-image val results.
    # Also exposes `train_selected_site(...)` for the realistic procedure (train ONE site chosen by a selector).
```

Cost accounting must separate (§10.5): `selection_cost` (passed in), `training_compute` for the
chosen site only, and `reference_total_compute` when all sites were trained.

## Tests

1. Fresh adapter: output equals original model output exactly (zero init).
2. Only adapter params have `requires_grad`; after one step, adapter params changed and backbone params did not.
3. Gradient reaches the adapter placed at layer 0 and at the last layer (non-zero grad norm).
4. After `evaluate_adapter`, the model has no hooks and reproduces the original output.
5. Training on a tiny synthetic paired dataset for a few steps reduces the training loss (smoke).
6. `extra_params` equals the actual parameter count.

## Deliverables

`src/layer_research/adapter_training.py`, `tests/test_adapter_training.py`, `docs/modules_T04.md`.
Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q` and report the summary line.
