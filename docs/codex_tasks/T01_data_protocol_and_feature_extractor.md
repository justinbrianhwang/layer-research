# T01 — `data_protocol` + `feature_extractor` + `representation_metrics`

You are implementing the first three modules of a research codebase. The full research
proposal is in `vision_intervention_site_selection_proposal.md` (Korean). Read sections
5, 6, 7 and 18 before writing code. Follow the layout in `README.md`.

## Constraints (do not violate)

- Package root: `src/layer_research/`. Tests in `tests/`. Use `pytest`.
- Python 3.11, PyTorch >= 2.3, `timm` >= 1.0. No other heavy deps beyond `environment.yml`.
- Everything must run on **CPU** for tests (this machine has no GPU). GPU is optional via a `device` arg.
- Model is always in `eval()` mode for extraction. No dropout/stochastic depth.
- No secrets, no network calls in tests. Tests must not download ImageNet. Use `timm.create_model(..., pretrained=False)` with a tiny config, or random weights, in tests.
- Deterministic: every random choice takes an explicit `seed`.
- Do not add Co-Authored-By trailers. Do not commit; the PM commits.

## Module 1: `data_protocol.py`

Purpose: proposal §6.2–6.4. Splits are by **original image id**, never by derived file.

Implement:

```python
@dataclass(frozen=True)
class SplitConfig:
    fit_frac: float; score_frac: float; val_frac: float; test_frac: float
    seed: int
    class_balanced: bool = True

def make_splits(image_ids: list[str], labels: list[int], cfg: SplitConfig) -> dict[str, list[str]]
    # returns {"fit": [...], "score": [...], "val": [...], "test": [...]}, disjoint, covering all ids
    # class_balanced=True -> stratified by label

def save_splits(splits, path) / load_splits(path)   # JSON, sorted ids, includes cfg + seed

OBSERVED_CORRUPTIONS = ("gaussian_noise", "defocus_blur")
UNSEEN_CORRUPTIONS   = ("contrast", "jpeg_compression")
DEFAULT_SEVERITIES   = (1, 3, 5)

@dataclass(frozen=True)
class CorruptionSpec:
    name: str; severity: int; seed: int

def corrupt_image(img: PIL.Image, spec: CorruptionSpec) -> PIL.Image
    # use the `imagecorruptions` package; seed numpy RNG from spec.seed + hash(image_id) style determinism
    # (accept an `image_id: str` argument and derive a per-image seed so results are reproducible per file)
```

Also implement a `PairedImageDataset` (torch Dataset) that yields
`(clean_tensor, corrupted_tensor, label, image_id)` for one `CorruptionSpec`, using a
standard timm transform for the model (resolve via `timm.data.resolve_data_config` +
`create_transform`). Corruption is applied to the **PIL image before** the model transform.

Add a helper `verify_disjoint(splits)` that raises if any id appears in two splits.

## Module 2: `feature_extractor.py`

Purpose: proposal §5.1, §6.5. Block boundary = residual stream **after** the full block
(attention + MLP residual adds). For timm ViT (`deit_small_patch16_224`) this is the
output of `model.blocks[i]`.

Implement:

```python
class BlockOutputRecorder:
    """Context manager. Registers forward hooks on model.blocks[i] for the requested
    layer indices, stores outputs (detached, optionally moved to CPU / cast to dtype).
    MUST remove all hooks on __exit__, even on exception."""
    def __init__(self, model, layers: Sequence[int], to_cpu=True, dtype=None)
    def __enter__(self) -> Self
    def __exit__(...)
    outputs: dict[int, torch.Tensor]   # layer -> [B, N, d]

def num_blocks(model) -> int
def summarize(tokens: torch.Tensor, mode: Literal["cls","patch_mean","cls+patch_mean"]) -> torch.Tensor  # [B, N, d] -> [B, d] or [B, 2d]

@torch.no_grad()
def extract_paired_features(model, loader, layers, summary_modes, device) -> dict
    # returns per-layer summaries for clean and corrupted + logits + labels + image_ids
    # for "full token" tensors used by patching, expose a separate function that
    # streams batches rather than holding everything in memory:
def iter_paired_block_outputs(model, loader, layers, device) -> Iterator[dict]
```

Also a `shape_report(model, input_size=(3,224,224)) -> dict[int, tuple]` that runs one
dummy forward and returns the tensor shape at every block boundary (proposal §18.3 check).

Prefer `timm` generic attribute names; make the "blocks" attribute name configurable
(`blocks_attr="blocks"`) so a ResNet variant can be added later without rewriting.

## Module 3: `representation_metrics.py`

Purpose: proposal §7.1–7.3. All functions take **already-summarized** representations
`H: [n, d]` (clean) and `Ht: [n, d]` (corrupted) as torch or numpy, for the same image ids
in the same order.

Implement:

- `relative_distance(H, Ht, eps=1e-8)` → mean over images of ‖h−h̃‖ / (‖h‖+eps)   (eq. in §7.1)
- `raw_distance(H, Ht)` → mean ‖h−h̃‖
- `activation_norm(H)` → mean ‖h‖
- `cosine_distance(H, Ht)` → mean (1 − cos)
- `linear_cka(H, Ht)` → scalar, column-centered, exactly the formula in §7.2.
  Must return `nan` and log a warning (not raise, not substitute) if a denominator is < eps.
- `one_minus_cka(H, Ht)`
- `amplification_ratio(scores: Sequence[float], eps)` → per-layer ratio score[l]/score[l-1]; define layer 0 as ratio 1.0 (document this).
- `knn_preservation(H, Ht, k=10)` → mean Jaccard overlap of k-NN sets between clean and corrupted (§7.3 geometry row).
- `layerwise_scores(features: dict, metric_names: list[str]) -> pandas.DataFrame`
  with columns `layer, metric_name, score, n_samples, summary_mode, label_access` (label_access is "none" for all of the above).

Verify linear CKA against a brute-force implementation (HSIC-based) in a test.

## Tests (must all pass on CPU in < 60s)

- splits are disjoint, cover all ids, stratified counts within ±1 per class, reproducible with same seed, different with different seed.
- corrupt_image is deterministic for same (image_id, spec).
- BlockOutputRecorder removes hooks on exit (check `len(model.blocks[i]._forward_hooks) == 0`) and on exception.
- shape_report on a tiny random ViT (`timm.create_model("vit_tiny_patch16_224", pretrained=False)` or smaller custom config, e.g. `img_size=32, patch_size=8, embed_dim=32, depth=3`) returns `[1, 17, 32]`-style shapes for every block.
- CKA: identical inputs → 1.0; orthogonal-rotation invariance; matches brute force.
- relative_distance is 0 for identical inputs.

## Deliverables

- The three modules, `__init__.py` exports, tests, and a short `docs/modules_T01.md`
  listing every public function with a one-line description and any deviation from this spec with a reason.
- Run `pytest -q` and make sure it is green before you finish. Print the final pytest summary in your last message.
