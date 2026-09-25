# T03 — `evaluation`, `statistics`, `site_selection`

Implements proposal §8.6–8.7, §9 (Experiment C), §13. Read those sections plus §5.2, §5.4 and
§18.2 of `vision_intervention_site_selection_proposal.md`. Consumes the per-image long-format
DataFrame produced by `patching_engine.sweep_layers_budgets` (see `docs/modules_T02.md`) and the
metric table from `representation_metrics.layerwise_scores` (see `docs/modules_T01.md`).

All functions are pure pandas / numpy / scipy; no model code here. Tests use synthetic
DataFrames. No downloads.

## `evaluation.py`

```python
def accuracy_gain_pp(df, group_cols) -> DataFrame
    # U = 100 * (mean(post_pred==label) - mean(baseline_pred==label)) per group (§8.6). Include n_images.

def recovery_and_new_error(df, group_cols) -> DataFrame
    # Fix the sets BEFORE intervention (§8.7):
    #   corruption_failure_set: clean_pred==label & baseline_pred!=label  -> recovery_rate = P(post_pred==label | set)
    #   still_correct_set:      baseline_pred==label                       -> new_error_rate = P(post_pred!=label | set)
    # Return both rates + set sizes per group.

def margin_change(df, group_cols) -> DataFrame   # mean(post_margin - baseline_margin)

def clean_accuracy_change_pp(df_clean_patched, group_cols) -> DataFrame
    # for adapter/clean-constraint evaluation later; same shape as accuracy_gain_pp.

def layer_effect_table(df, budget_cols=("effective_channel_ratio","alpha","norm_cap"), ...) -> DataFrame
    # one row per (corruption, severity, budget, layer, mask_seed) with U, recovery, new_error, margin change.
    # Then `aggregate_over_seeds` -> mean and std over mask_seed (report as independent runs, §8.6).

def regret(u_test: Series indexed by candidate (layers + "none"), chosen: candidate, admissible: set) -> float
    # max over admissible of U_test minus U_test[chosen]   (§9.2). "none" has U=0.

def equivalence_set(u: Series, tau: float) -> list   # §13.4: {l : max_j U_j - U_l <= tau}
```

## `site_selection.py`

Every selector takes only **observed** data (score/val splits, observed corruptions) and returns a
`Selection(candidate, label_access: Literal["none","labels"], selection_cost: dict)`.
Candidate is a layer id or `"none"`.

```python
def select_fixed(position: Literal["front","middle","back"], layers) -> Selection
def select_random(layers, seed, include_none=False) -> Selection
def select_by_metric(metric_table, metric_name, direction: Literal["max","min"], layers) -> Selection
    # direction is decided on val (§7.5) and passed in explicitly; never inferred from test.
    # If metric_table has several corruptions, average the per-layer RANK across corruptions first (§5.2, §9.5) unless `aggregate="mean_score"`.
def select_val_sweep(u_val: DataFrame, layers, clean_drop_val: Series|None, eps_clean: float|None) -> Selection
    # picks argmax U_val among candidates whose clean drop <= eps_clean; falls back to "none".
def select_small_search(u_val_partial: DataFrame, tried_layers: list, ...) -> Selection
    # evaluates only `tried_layers` (subset), picks best among them or "none"; records number of tried sites.
def apply_admissibility(selection, u_val_for_chosen, clean_drop_for_chosen, eps_clean) -> Selection
    # §9.2: the cheap selector checks ONLY its chosen site on val; if it violates eps_clean, return "none".
def rank_correlation(s: Series, u: Series) -> dict   # spearman & kendall with p-values (scipy)
```

## `statistics.py`

```python
def paired_bootstrap(df, stat_fn, unit_col="image_id", n_boot=1000, seed=0, stratify_col=None) -> dict(mean, ci_low, ci_high, samples)
    # resample ORIGINAL IMAGE IDS with replacement; all rows sharing an id move together (§13.2).
    # stratify_col (e.g. label) -> resample within strata.
def bootstrap_layer_gain(df, layer_col="layer_id", ...) -> DataFrame   # CI of U per layer
def bootstrap_regret(...) -> dict
def seed_variability(df_agg, seed_col) -> DataFrame    # mean/std/min/max across mask seeds or training seeds
def selection_stability(selections_by_bootstrap) -> DataFrame   # how often each candidate is chosen across resamples
def holm_correction(pvals: Sequence[float]) -> list[float]   # §13.3
```

## Tests

- `accuracy_gain_pp` on a hand-built df with known accuracies gives exact pp values.
- Recovery/new-error sets are computed from pre-intervention columns only (mutating post columns must not change set sizes).
- `regret` = 0 when chosen is the argmax; equals gap otherwise; "none" handled.
- `select_by_metric` respects direction and averages ranks across corruptions.
- `apply_admissibility` returns "none" when clean drop exceeds eps.
- `paired_bootstrap` keeps all rows of an id together (check by constructing ids with 3 rows each and asserting every resample has row counts divisible by 3); reproducible with seed.
- `holm_correction` matches a known example.

## Deliverables

The three modules, tests, `docs/modules_T03.md`. Run
`C:/anaconda/envs/layer-research/python.exe -m pytest -q` and report the summary line.
