# T02 — `patching_engine`

Implements proposal §8 (Experiment B: budget-limited diagnostic partial patching) and the control
conditions of §8.5. Read §5.1, §8.1–8.7 and §18.3 of
`vision_intervention_site_selection_proposal.md` before coding. Build on the existing modules in
`src/layer_research/` (`feature_extractor.BlockOutputRecorder`, `data_protocol`), do not duplicate them.
Read `docs/modules_T01.md` for the public API of T01.

## Constraints

- CPU-runnable tests (tiny random ViT via `timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32, patch_size=8, embed_dim=32, depth=3, num_classes=10)` or similar). No downloads.
- Model always `eval()`. All randomness via explicit seeds.
- Hooks must be removed on exit, including on exception. After any patched forward, a plain forward of the same model must reproduce the un-patched output bit-for-bit (this is tested).
- Do not commit. Do not add Co-Authored-By trailers.

## Core operation (§8.1)

At block `l`, replace the block output with

```
h_hat = h_corr + alpha * M ⊙ (h_clean - h_corr)
```

`h_clean` and `h_corr` are `[B, N, d]` tensors for the **same original images in the same order**.
Then the rest of the model (`g_l`) runs unchanged. Implement as a forward hook on
`model.blocks[l]` that returns the modified output.

## Masks (§8.2)

Channel masks over the `d` dimension, shared across all tokens and all images:

```python
@dataclass(frozen=True)
class MaskSpec:
    policy: Literal["random_fixed", "score_topk"]
    fraction: float          # nominal q
    seed: int                # for random_fixed
    exclude_cls: bool = False  # if True, CLS token row is never modified (see §8.4)

def make_channel_mask(d: int, spec: MaskSpec, channel_scores: Optional[Tensor]=None) -> tuple[Tensor(bool,[d]), int]
    # returns (mask, k) with k = floor(q*d); q=0 -> all False and k=0.
    # random_fixed: permutation from torch.Generator(seed); take first k. NESTED: same seed and larger q must be a superset.
    # score_topk: top-k channels by channel_scores (computed by caller on D_score; e.g. mean |h_clean-h_corr| per channel).
```

Also `nested_masks(d, fractions, seed)` returning a dict {q: mask} guaranteed nested.

Record for every patch: `channel_count k`, `effective_channel_ratio k/d`, `alpha`, `norm_cap`,
`actual_delta_norm` (Frobenius norm of the applied delta, per image), `n_elements_modified`.

## Norm cap (§8.3, auxiliary)

```
delta = h_hat - h_corr
delta_cap = delta * min(1, rho*sqrt(n_l)*r_l / (||delta||_F + eps))
```
per image, where `n_l = N*d` and `r_l` is the per-element RMS of clean activations at block `l`
estimated on D_score (passed in as a float). Never scale a delta up.

## Interventions to support (one enum, one code path)

```python
class InterventionType(str, Enum):
    NONE = "none"                       # q=0 or alpha=0; output must equal un-patched corrupted output
    PARTIAL_CHANNEL = "partial_channel" # main experiment
    FULL_STATE = "full_state"           # positive control: replace the whole [B,N,d] block output with clean
    CLEAN_TO_CLEAN = "clean_to_clean"   # donor == receiver; output must equal clean forward
    RANDOM_DIRECTION = "random_direction"   # same per-image ||delta|| as PARTIAL_CHANNEL, random direction on the same mask
    SHUFFLED_DONOR = "shuffled_donor"   # clean tensors permuted across the batch (donor mismatch control); record same_class flag per pair
```

Provide:

```python
class Patcher:
    """Context manager that installs one hook at `layer` and applies `spec` using a provided
    clean tensor. The clean tensor for the current batch is set via `set_donor(h_clean)` before
    each forward. Exposes `.last_stats` (dict with the recorded fields above)."""

def run_patched_forward(model, layer, x_corr, h_clean, alpha, mask, norm_cap=None, r_l=None, intervention=InterventionType.PARTIAL_CHANNEL, generator=None) -> tuple[logits, stats]

def sweep_layers_budgets(model, batches, layers, fractions, alphas, mask_seeds, device, ...) -> pandas.DataFrame
    # batches: iterable of dicts with keys x_clean, x_corr, y, image_id and, for efficiency, optional pre-recorded clean block outputs
    # For each (layer, q, alpha, seed) run a patched forward, collect per-image: baseline_pred (corrupted, no patch),
    # clean_pred, post_pred, baseline_margin, post_margin, loss before/after, plus stats fields.
    # Must reuse the clean block outputs captured by a single BlockOutputRecorder pass per batch
    # (do not re-run the clean forward per layer).
```

Return **per-image long-format rows** (one row per image × condition) so `evaluation` can compute
accuracy, recovery sets and bootstrap later. Column names must follow proposal §18.2:
`image_id, label, layer_id, intervention_type, mask_policy, mask_seed, channel_count,
effective_channel_ratio, alpha, norm_cap, actual_delta_norm, baseline_prediction, clean_prediction,
post_intervention_prediction, baseline_margin, post_intervention_margin, baseline_loss, post_loss`.

Margin is `z_y - max_{k≠y} z_k` (§8.6).

## Required tests (§8.5 checks, all on the tiny model)

1. `NONE` (q=0 or alpha=0): logits equal un-patched corrupted logits within 1e-6.
2. `CLEAN_TO_CLEAN`: equals clean logits within 1e-6.
3. `FULL_STATE` at any layer: equals clean logits within 1e-5 (deterministic model, eval mode).
4. After exiting `Patcher`, a plain forward equals the original un-patched forward exactly, and `_forward_hooks` is empty on every block; same after an exception inside the context.
5. Nested masks: for seed s, mask(q1) ⊂ mask(q2) whenever q1 < q2; `k == floor(q*d)`.
6. Norm cap never increases `||delta||`, and equals the cap when the raw delta exceeds it.
7. `RANDOM_DIRECTION` produces the same per-image `||delta||` as `PARTIAL_CHANNEL` (rel tol 1e-4) and touches only masked channels.
8. `sweep_layers_budgets` on 2 batches × 3 layers × 2 q × 1 alpha × 2 seeds returns the expected number of rows with no NaNs in prediction columns.
9. Clean block outputs are captured once per batch (assert via a counter/mock that the clean forward is called once per batch, not once per layer).

## Deliverables

`src/layer_research/patching_engine.py`, tests in `tests/test_patching_engine.py`, and
`docs/modules_T02.md` (public API + any deviation with reason). Run
`C:/anaconda/envs/layer-research/python.exe -m pytest -q` and report the summary line.
