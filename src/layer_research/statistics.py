"""Original-image cluster bootstrap and separate run variability summaries."""
import numpy as np
import pandas as pd
from .evaluation import _groups, accuracy_gain_pp, regret


def paired_bootstrap(df, stat_fn, unit_col="image_id", n_boot=1000, seed=0,
                     stratify_col=None, confidence=0.95):
    """Percentile CI of a scalar statistic; mean is its original-data estimate.

    All copies of a sampled ID retain all rows and the original ID. Statistics
    must preserve multiplicities (do not deduplicate IDs in stat_fn).
    """
    if df.empty or df[unit_col].isna().any():
        raise ValueError("Bootstrap requires nonempty data with nonmissing image IDs")
    if not isinstance(n_boot, (int, np.integer)) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    clusters = list(df.groupby(unit_col, sort=False, observed=True).indices.values())
    strata = [np.arange(len(clusters))]
    if stratify_col is not None:
        if df[stratify_col].isna().any():
            raise ValueError("Strata must not be missing")
        labels = []
        for positions in clusters:
            unique = df.iloc[positions][stratify_col].unique()
            if len(unique) != 1:
                raise ValueError("Each original image must belong to exactly one stratum")
            labels.append(unique[0])
        codes, _ = pd.factorize(pd.Series(labels), sort=False)
        strata = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    rng = np.random.default_rng(seed)
    estimate = float(stat_fn(df.copy()))
    samples = np.empty(n_boot)
    for i in range(n_boot):
        chosen = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in strata])
        positions = np.concatenate([clusters[j] for j in chosen])
        samples[i] = float(stat_fn(df.iloc[positions].reset_index(drop=True)))
    if not np.isfinite(estimate) or not np.isfinite(samples).all():
        raise ValueError("Statistic must be finite on original data and every resample")
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return dict(mean=estimate, ci_low=float(low), ci_high=float(high), samples=samples)


def bootstrap_layer_gain(df, layer_col="layer_id", group_cols=None, **bootstrap_kwargs):
    """CI per layer/condition. Seeds stay fixed; only original images resample."""
    group_cols = [] if group_cols is None else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    cols = list(dict.fromkeys(group_cols + [layer_col]))
    rows = []
    for keys, part in _groups(df, cols):
        result = paired_bootstrap(part, lambda data: accuracy_gain_pp(data, []).U.iloc[0], **bootstrap_kwargs)
        rows.append({**keys, **result})
    return pd.DataFrame(rows)


def bootstrap_regret(df, chosen, admissible, layer_col="layer_id", **bootstrap_kwargs):
    """Fixed validation choice/set; recompute the test reference in each resample.

    Pass one prespecified condition or mixture, containing matched image IDs
    across every admissible layer. No selector is fitted on this test table.
    """
    candidates = list(dict.fromkeys(list(admissible) + ["none"]))
    if chosen not in candidates:
        raise ValueError("Chosen candidate is not admissible")
    required = [c for c in candidates if c != "none"]
    unit_col = bootstrap_kwargs.get("unit_col", "image_id")
    part = df.loc[df[layer_col].isin(required)] if required else df
    if required:
        counts = part.groupby([unit_col, layer_col], dropna=False).size().unstack(layer_col).reindex(columns=required)
        if counts.empty or counts.isna().any().any() or counts.nunique(axis=1).gt(1).any():
            raise ValueError("Regret requires matched image rows across admissible layers")
    def statistic(data):
        utilities = accuracy_gain_pp(data, [layer_col]).set_index(layer_col).U
        return regret(utilities, chosen, candidates)
    return paired_bootstrap(part, statistic, **bootstrap_kwargs)


def seed_variability(df_agg, seed_col, group_cols=None, value_cols=None):
    """One row per condition/seed required; sample std (ddof=1)."""
    if value_cols is None:
        known = {"U", "n_images", "recovery_rate", "new_error_rate", "margin_change",
                 "corruption_failure_set_size", "still_correct_set_size"}
        value_cols = [c for c in df_agg if c in known]
        if not value_cols:
            raise ValueError("Specify value_cols for custom run metrics")
    value_cols = list(value_cols)
    if group_cols is None:
        group_cols = [c for c in df_agg if c not in value_cols + [seed_col]]
    if df_agg[seed_col].isna().any():
        raise ValueError("Seed identifiers must not be missing")
    rows = []
    for keys, part in _groups(df_agg, group_cols):
        if part[seed_col].duplicated().any():
            raise ValueError("Expected one aggregate row per condition and seed")
        row = {**keys, "n_seeds": len(part)}
        for col in value_cols:
            for name in ("mean", "std", "min", "max"):
                row[f"{col}_{name}"] = getattr(part[col], name)()
        rows.append(row)
    return pd.DataFrame(rows)


def selection_stability(selections_by_bootstrap):
    """Frequency among supplied resamples; accepts candidates or Selections."""
    candidates = [getattr(s, "candidate", s) for s in selections_by_bootstrap]
    counts = pd.Series(candidates, dtype=object).value_counts(sort=False)
    return pd.DataFrame({"candidate": counts.index, "count": counts.to_numpy(),
                         "frequency": counts.to_numpy() / len(candidates) if candidates else []})


def holm_correction(pvals):
    p = np.asarray(list(pvals), dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite and between zero and one")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()
