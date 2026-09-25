"""Prediction effects in percentage points; conditional rates are in [0, 1]."""
import numpy as np
import pandas as pd


def _groups(df, cols):
    cols = [cols] if isinstance(cols, str) else list(cols)
    if not cols:
        yield {}, df
    else:
        for key, part in df.groupby(cols, sort=False, dropna=False, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            yield dict(zip(cols, key)), part


def _column(df, name):
    aliases = {"baseline_prediction": "baseline_pred", "clean_prediction": "clean_pred",
               "post_intervention_prediction": "post_pred", "post_intervention_margin": "post_margin"}
    name = name if name in df else aliases.get(name, name)
    if df[name].isna().any():
        raise ValueError(f"Missing values in {name}")
    return df[name]


def accuracy_gain_pp(df, group_cols):
    """Return U and n_images per group (one input row per image/run)."""
    rows = []
    for keys, part in _groups(df, group_cols):
        y = _column(part, "label")
        before = _column(part, "baseline_prediction").eq(y).mean()
        after = _column(part, "post_intervention_prediction").eq(y).mean()
        rows.append({**keys, "U": 100 * (after - before), "n_images": len(part)})
    return pd.DataFrame(rows)


def recovery_and_new_error(df, group_cols):
    rows = []
    for keys, part in _groups(df, group_cols):
        y = _column(part, "label")
        before = _column(part, "baseline_prediction").eq(y)
        failure = _column(part, "clean_prediction").eq(y) & ~before
        after = _column(part, "post_intervention_prediction").eq(y)
        rows.append({**keys, "recovery_rate": after[failure].mean(),
                     "new_error_rate": (~after[before]).mean(),
                     "corruption_failure_set_size": int(failure.sum()),
                     "still_correct_set_size": int(before.sum())})
    return pd.DataFrame(rows)


def margin_change(df, group_cols):
    return pd.DataFrame([{**keys, "margin_change": float((
        _column(part, "post_intervention_margin") - _column(part, "baseline_margin")).mean())}
        for keys, part in _groups(df, group_cols)])


def clean_accuracy_change_pp(df_clean_patched, group_cols):
    """Baseline predictions must be the unpatched *clean* predictions."""
    return accuracy_gain_pp(df_clean_patched, group_cols)


def layer_effect_table(df, budget_cols=("effective_channel_ratio", "alpha", "norm_cap"),
                       group_cols=None):
    """Keep intervention policies and recorded conditions separate by default."""
    if group_cols is None:
        context = [c for c in ("experiment_id", "model_id", "split", "corruption_type",
                   "corruption", "severity", "corruption_seed", "intervention_type", "mask_policy") if c in df]
        group_cols = list(dict.fromkeys(context + list(budget_cols) + ["layer_id", "mask_seed"]))
    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    tables = [fn(df, group_cols) for fn in (accuracy_gain_pp, recovery_and_new_error, margin_change)]
    if not group_cols:
        return pd.concat(tables, axis=1)
    result = tables[0]
    for table in tables[1:]:
        result = result.merge(table, on=group_cols, validate="one_to_one")
    return result


def aggregate_over_seeds(df, group_cols=None, seed_col="mask_seed"):
    """Summarize run-level effects, never pool predictions across seeds."""
    from .statistics import seed_variability
    return seed_variability(df, seed_col, group_cols=group_cols)


def _utilities(u):
    u = u.astype(float).copy()
    if not u.index.is_unique or not np.isfinite(u.to_numpy()).all():
        raise ValueError("Utilities require unique candidates and finite values")
    u.loc["none"] = 0.0
    return u


def regret(u_test, chosen, admissible):
    u = _utilities(u_test)
    candidates = list(dict.fromkeys(list(admissible) + ["none"]))
    if chosen not in candidates:
        raise ValueError("Chosen candidate is not admissible")
    return float(u.loc[candidates].max() - u.loc[chosen])


def equivalence_set(u, tau):
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and nonnegative")
    if not u.index.is_unique or not np.isfinite(u.to_numpy(dtype=float)).all():
        raise ValueError("Utilities require unique candidates and finite values")
    return u.index[u.max() - u <= tau].tolist()
