# T02: patching engine

`layer_research.patching_engine` implements proposal sections 5.1, 8.1–8.7 and
18.3 at `model.blocks[layer]` residual-block outputs. Models remain in eval mode.
No weights or block outputs are mutated in place. Contexts remove their own hook
on both normal and exceptional exits, preserving any pre-existing hooks.

## Public API

- `MaskSpec(policy, fraction, seed, exclude_cls=False)` is immutable.
- `make_channel_mask(d, spec, channel_scores=None)` returns a CPU boolean `[d]`
  mask and `floor(fraction*d)`. Fractions must be finite and in `[0,1]`.
  `random_fixed` uses a private seeded CPU generator. `score_topk` requires a
  finite `[d]` score tensor supplied from D_score; ties prefer lower channel indices.
- `nested_masks(d, fractions, seed)` returns `{fraction: mask}`. Every mask uses
  the same permutation, so increasing budgets form supersets.
- `InterventionType` provides `NONE`, `PARTIAL_CHANNEL`, `FULL_STATE`,
  `CLEAN_TO_CLEAN`, `RANDOM_DIRECTION`, and `SHUFFLED_DONOR`.
- `Patcher(model, layer, spec=None, alpha=1.0, mask=None, norm_cap=None, r_l=None,
  intervention=InterventionType.PARTIAL_CHANNEL, generator=None, *,
  exclude_cls=False, channel_scores=None, labels=None)` installs one hook inside
  its context. Supply exactly one of `spec` or a boolean `[d]` `mask`.
  Call `set_donor(h_clean)` before each batch; donors are detached snapshots.
  The current donor remains available for repeated forwards until replaced.
- `run_patched_forward(model, layer, x_corr, h_clean, alpha, mask, norm_cap=None,
  r_l=None, intervention=InterventionType.PARTIAL_CHANNEL, generator=None, *,
  exclude_cls=False, labels=None)` returns `(logits, stats)` without gradients.
- `sweep_layers_budgets(model, batches, layers, fractions, alphas, mask_seeds,
  device="cpu", *, mask_policy="random_fixed", channel_scores=None,
  exclude_cls=False, norm_cap=None, r_l=None,
  intervention=InterventionType.PARTIAL_CHANNEL)` returns per-image pandas rows.
  `channel_scores` and `r_l` are layer-keyed mappings estimated on D_score.
  Iterables for the condition grid are supported, including generators.

## Interventions and statistics

Partial patching uses `h_corr + alpha * M * (h_clean - h_corr)`, with the same
channel mask across images and tokens. `exclude_cls` leaves token zero unchanged.
Alpha must be finite and nonnegative; extrapolation above one is allowed.
`NONE` applies zero delta. For `CLEAN_TO_CLEAN`, the low-level caller must supply
the clean input as receiver; the sweep selects it automatically.

`FULL_STATE` copies the complete donor exactly, including CLS. It overrides the
mask, alpha and norm cap, recording all channels, alpha one and no cap. This
ensures the positive control tests exact state replacement even when it is run
with a partial-budget grid. It must not be used as a main budget comparison.

`norm_cap` is rho, not an absolute norm: the per-image bound is
`rho * sqrt(N*d) * r_l`. Scaling follows the proposal with epsilon `1e-12`, never
scaling a delta up. `r_l` must be a finite nonnegative clean RMS from D_score.
Random directions are Gaussian on the permitted coordinates and rescaled to the
corresponding partial delta norm before applying the same cap. Random controls
require an explicit seeded `torch.Generator`; the sweep uses each mask seed.

`last_stats` contains `channel_count`, `effective_channel_ratio`, `alpha`,
`norm_cap`, `actual_delta_norm`, `n_elements_modified`, `token_count`, and
`exclude_cls`. Norms and modified-element counts are per-image CPU tensors,
measured from the actual rounded block output minus its original value.
Counts therefore mean numerically changed elements, not merely mask capacity.
Floating-point addition can slightly change the intended delta norm.
When `r_l` is provided, `normalized_delta_norm` divides by `sqrt(N*d)*r_l`
with the denominator floored at `1e-12`.

Shuffled donors require at least two images and labels. A seeded random cycle
permutes donors without self-pairs. This deliberately avoids accidental matched
donors from an unrestricted random permutation. Statistics include per-image
`donor_index` and `same_class`; sweep rows also contain `donor_image_id`.
The labels only annotate donor pairs; they never select masks or permutations.

## Sweep data and results

Each batch is a dictionary with `x_clean`, `x_corr`, `y`, and `image_id`, aligned
by original-image order. The caller must guarantee donor identity and D_score
provenance; shapes can be validated, but identity cannot be inferred from tensors.
T01's paired dataset supplies the corresponding aligned items; default tuple
collations need conversion to these named batch fields.

Optional `clean_outputs` maps layer indices to full `[B,N,d]` tensors, and
`clean_logits` caches clean predictions. Without outputs, one
`BlockOutputRecorder` forward captures every requested layer per batch. With
outputs but no logits, one clean forward obtains logits; supplying both avoids
clean forwards entirely. Caches must use the same model, images, order, eval mode
and precision. There is one unpatched corrupted baseline forward per batch.

Rows include all T02-required column names:
`image_id`, `label`, `layer_id`, `intervention_type`, `mask_policy`, `mask_seed`,
`channel_count`, `effective_channel_ratio`, `alpha`, `norm_cap`,
`actual_delta_norm`, `baseline_prediction`, `clean_prediction`,
`post_intervention_prediction`, `baseline_margin`, `post_intervention_margin`,
`baseline_loss`, and `post_loss`. Additional columns include nominal `fraction`
and the statistics described above. An absent cap is represented by `None`.
Margins are true-label logits minus the largest competing logit; losses are
unreduced cross entropy. Baseline columns always describe the corrupted input,
including for the clean-to-clean diagnostic. No recovery sets or aggregates are
selected here. Empty input produces an empty DataFrame.

The sweep and forward convenience function disable gradients. The low-level
Patcher preserves the receiver's autograd path while detaching donors; it is
intended for diagnostics, not donor-side training.

## Validation

Tests use a seeded, randomly initialized three-block ViT on CPU without downloads.
They cover all controls, the patch equation, nested and scored masks, CLS exclusion,
per-image norm caps, random-direction support and reproducibility, shuffled pair
metadata, exact post-context baseline reproduction, exceptional cleanup, sweep
row counts, margin/loss calculations, and clean-forward/cache reuse.

Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q tests/test_patching_engine.py`.
As documented for T01, this Windows environment needs the conda environment and
its `Library/bin` on the process PATH and `MKL_THREADING_LAYER=SEQUENTIAL`.
These are process-local test settings, not machine configuration changes.
