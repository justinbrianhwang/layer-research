# T03: evaluation, statistics, and site selection

Implements proposal sections 5.2, 5.4, 8.6–8.7, 9, 13, and 18.2 using pandas,
NumPy, and SciPy. No model calls, downloads, or test-driven selection occur here.

## Input contracts

Evaluation uses T02's `image_id`, `label`, `layer_id`, `baseline_prediction`,
`clean_prediction`, `post_intervention_prediction`, `baseline_margin`, and
`post_intervention_margin`. Short prediction names from the T03 pseudocode
(`baseline_pred`, `clean_pred`, `post_pred`, `post_margin`) are also accepted.
Each evaluation group must contain one row per image within a fixed run/condition.
Missing prediction values raise rather than count as errors. Inputs are not mutated.

Metric selection accepts T01's `layer`, `metric_name`, `score`, `summary_mode`,
`n_samples`, and `label_access`, or `layer_id` in place of `layer`. Add
`corruption_type` (or `corruption`) when concatenating tables across corruptions.
Filter to one representation summary, model, budget, and intended observed mixture
before selection. Selectors reject test/train values in supplied `split` or
`score_data_split` columns and false values in supplied `is_observed` metadata.
Without provenance metadata, callers must enforce observed corruption and split
boundaries; functions cannot infer provenance from numeric scores.

## Evaluation API

- `accuracy_gain_pp(df, group_cols)` returns grouping columns, `U` (percentage
  points), and `n_images` (input row count). Use `[]` for a single whole-table group.
- `recovery_and_new_error` returns `recovery_rate`, `new_error_rate`,
  `corruption_failure_set_size`, and `still_correct_set_size`. Rates are fractions;
  empty conditioning sets produce NaN and zero size. Sets depend only on clean,
  baseline, and label columns.
- `margin_change` returns mean post minus baseline margin as `margin_change`.
- `clean_accuracy_change_pp` has the same result schema as `accuracy_gain_pp`;
  supply clean-input rows with **unpatched clean baseline predictions**.
- `layer_effect_table(df, budget_cols=("effective_channel_ratio", "alpha",
  "norm_cap"), group_cols=None)` joins these effects. Defaults retain available
  experiment/model/split/corruption/severity/corruption-seed/intervention/policy
  columns, all requested budget columns, layer ID, and mask seed. Null norm caps
  remain valid groups. Explicit `group_cols` replaces this grouping.
- `aggregate_over_seeds(df, group_cols=None, seed_col="mask_seed")` summarizes
  run-level effects via `seed_variability`; predictions are never ensembled.
- `regret(u_test, chosen, admissible)` uses candidate-indexed utilities and the
  fixed validation admissible set. Always includes `none` at exactly zero, even
  if omitted or supplied with another value. A choice outside that set raises.
- `equivalence_set(u, tau)` returns supplied candidates within nonnegative `tau`
  of their maximum. Include `none` explicitly if desired in this diagnostic set.

## Selection API and conventions

Every selector returns `Selection(candidate, label_access, selection_cost)`.
Candidate layer order breaks ties; validation ties at zero prefer `none`.
Layers must be unique and exclude the reserved candidate `none`.

- `select_fixed(position, layers)` takes first, index `len(layers)//2`, or last
  for front/middle/back. For even counts, middle is the later central candidate.
- `select_random(layers, seed, include_none=False)` samples uniformly with NumPy's
  local generator, preserving integer/string candidate types.
- `select_by_metric(metric_table, metric_name, direction, layers,
  aggregate="mean_rank")` requires explicit min/max direction. Repeated rows
  (e.g. severities) first average within each corruption/layer, then average
  tie-aware ranks across equally weighted corruptions. `mean_score` instead
  averages those corruption scores. Every corruption must cover every candidate.
  Label access is inherited from metric rows; validation used to determine the
  direction externally must also be recorded by the calling experiment.
- `select_val_sweep(u_val, layers, clean_drop_val=None, eps_clean=None)` accepts
  a DataFrame with `layer_id`/`layer` and `U`, or a candidate-indexed Series.
  Repeated utility rows are equally weighted. Drops are candidate-indexed Series,
  expressed as positive accuracy losses in pp; negative drops are improvements.
  Choose the greatest positive admissible utility, otherwise `none`.
- `select_small_search(u_val_partial, tried_layers, ...)` reads only tried sites
  and applies the same rule. Both validation selectors record `n_tried_sites`.
- `apply_admissibility(selection, u_val_for_chosen, clean_drop_for_chosen,
  eps_clean)` reads scalar validation results for only the chosen site. Rejects
  a clean drop above epsilon; additionally rejects nonpositive validation gain
  to implement the proposal's no-intervention option. `eps_clean=None` disables
  only the clean constraint. A check marks label access as `labels` and increments
  `n_admissibility_checks`; an existing `none` is returned without checking.
- `rank_correlation(s, u)` aligns unique candidate indices and reports `spearman`,
  `spearman_pvalue`, `kendall`, `kendall_pvalue`, and `n_layers`. Missing pairs are
  excluded; constant vectors or fewer than two pairs give NaN correlations.

Fixed/random methods record zero tried sites. Metrics also record scored sites
and metric rows. These are operation counts, not measured timing or FLOPs;
calling experiments must record feature-extraction and training costs separately.

## Statistical API

- `paired_bootstrap(df, stat_fn, unit_col="image_id", n_boot=1000, seed=0,
  stratify_col=None, confidence=0.95)` resamples unique original IDs with
  replacement, carrying **all** their rows together. Strata must be constant
  within each ID. Returns `mean` (original-data statistic), percentile `ci_low`
  and `ci_high`, and the NumPy `samples` array. The callback must return a finite
  scalar and preserve repeated draws: do not deduplicate IDs. Original IDs stay
  unchanged; the sampled DataFrame receives a fresh positional index.
- `bootstrap_layer_gain(df, layer_col="layer_id", group_cols=None, **kwargs)`
  returns one bootstrap result per layer plus explicit condition grouping.
  Filter to one condition/run, or pass the desired condition and seed columns.
  If pooling a mixture, ensure the row multiplicities implement the prespecified
  weights. Seeds are held fixed, not resampled.
- `bootstrap_regret(df, chosen, admissible, layer_col="layer_id", **kwargs)`
  fixes the validation choice and admissible set, resamples paired image rows
  jointly across layers, and recomputes the test maximum in each replicate.
  Supply one fixed condition or balanced prespecified mixture; matched image
  multiplicities across admissible layers are required. This quantifies image
  uncertainty, not a correction for optimism of the maximum.
- `seed_variability(df_agg, seed_col, group_cols=None, value_cols=None)` requires
  one row per condition/seed and reports `n_seeds` and `<metric>_mean/std/min/max`.
  Defaults recognize the evaluation effect/count columns as values, treating all
  others as grouping columns. Specify `value_cols` for custom metrics. Standard
  deviation uses ddof=1 (NaN for one run); undefined conditional rates are skipped
  by pandas summaries. Grouping retains null values.
- `selection_stability(selections_by_bootstrap)` accepts candidates or Selection
  objects and returns `candidate`, `count`, and `frequency` for observed choices.
- `holm_correction(pvals)` returns monotone Holm-adjusted p-values in original
  order. Missing/nonfinite or out-of-range p-values raise.

Image uncertainty and mask/training seed variability are separate. No hierarchical
resampling or bootstrap refitting of selectors is implicit; for selection stability,
resample observed data and invoke the selector in the calling experiment.

## Verification

`C:/anaconda/envs/layer-research/python.exe -m pytest -q tests/test_T03.py`

Tests use synthetic tables only: hand-computed effects and fixed recovery sets,
null budgets and separate runs, regret and equivalence, cross-corruption ranks,
admissibility and provenance, rank alignment, reproducible stratified cluster
sampling, paired regret uncertainty, seed summaries, stability, and Holm values.
The Windows process uses the T01-documented conda DLL PATH and
`MKL_THREADING_LAYER=SEQUENTIAL` settings when running tests.
