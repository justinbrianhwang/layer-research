"""Selectors consume observed score/validation data only, never test utilities."""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Selection:
    candidate: object
    label_access: Literal["none", "labels"]
    selection_cost: dict


def _layers(layers):
    layers = list(layers)
    if len(set(layers)) != len(layers) or "none" in layers:
        raise ValueError("layers must be unique and exclude 'none'")
    return layers


def _observed(table, allowed):
    for col in ("split", "score_data_split"):
        if col in table and not table[col].isin(allowed).all():
            raise ValueError(f"{col} must contain only {sorted(allowed)}")
    if "is_observed" in table and not table["is_observed"].eq(True).all():
        raise ValueError("Unobserved corruption data cannot drive selection")


def select_fixed(position, layers):
    layers = _layers(layers)
    if position not in ("front", "middle", "back") or not layers:
        raise ValueError("A valid position and nonempty layers are required")
    index = {"front": 0, "middle": len(layers) // 2, "back": -1}[position]
    return Selection(layers[index], "none", {"n_tried_sites": 0})


def select_random(layers, seed, include_none=False):
    candidates = _layers(layers) + (["none"] if include_none else [])
    if not candidates:
        raise ValueError("No candidates")
    return Selection(candidates[int(np.random.default_rng(seed).integers(len(candidates)))],
                     "none", {"n_tried_sites": 0})


def select_by_metric(metric_table, metric_name, direction, layers, aggregate="mean_rank"):
    layers = _layers(layers)
    if not layers or direction not in ("min", "max") or aggregate not in ("mean_rank", "mean_score"):
        raise ValueError("Invalid layers, direction, or aggregation")
    table = metric_table.loc[metric_table.metric_name.eq(metric_name)].copy()
    layer_col = "layer_id" if "layer_id" in table else "layer"
    table = table.loc[table[layer_col].isin(layers)]
    _observed(table, {"score", "val"})
    if "summary_mode" in table and table.summary_mode.nunique() > 1:
        raise ValueError("Choose one representation summary before selection")
    if table.empty or not np.isfinite(table.score.to_numpy(dtype=float)).all():
        raise ValueError("Metric scores must be present and finite")
    corruption = next((c for c in ("corruption_type", "corruption") if c in table), None)
    if corruption:
        scores = table.groupby([corruption, layer_col], dropna=False).score.mean().unstack(layer_col).reindex(columns=layers)
    else:
        scores = table.groupby(layer_col).score.mean().reindex(layers).to_frame().T
    if scores.isna().any().any():
        raise ValueError("Every corruption must have scores for every candidate")
    if aggregate == "mean_rank":
        ranking = scores.rank(axis=1, ascending=direction == "min", method="average").mean()
        candidate = ranking.idxmin()
    else:
        means = scores.mean()
        candidate = means.idxmax() if direction == "max" else means.idxmin()
    access = "none"
    if "label_access" in table:
        if not table.label_access.isin(["none", "labels"]).all():
            raise ValueError("Unknown label_access")
        if table.label_access.eq("labels").any():
            access = "labels"
    return Selection(candidate, access, {"n_tried_sites": 0, "n_scored_sites": len(layers),
                                        "n_metric_rows": len(table)})


def _val_utilities(table, layers):
    if isinstance(table, pd.Series):
        values = table.reindex(layers)
    else:
        col = "layer_id" if "layer_id" in table else "layer"
        table = table.loc[table[col].isin(layers)]
        _observed(table, {"val"})
        if not np.isfinite(table.U.to_numpy(dtype=float)).all():
            raise ValueError("Validation utilities must be finite")
        values = table.groupby(col).U.mean().reindex(layers)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Each tried layer needs a finite validation utility")
    return values


def select_val_sweep(u_val, layers, clean_drop_val=None, eps_clean=None):
    layers = _layers(layers)
    values = _val_utilities(u_val, layers)
    if eps_clean is not None:
        if not np.isfinite(eps_clean) or eps_clean < 0 or clean_drop_val is None:
            raise ValueError("Clean constraint needs nonnegative epsilon and validation drops")
        drops = clean_drop_val.reindex(layers)
        if not np.isfinite(drops.to_numpy(dtype=float)).all():
            raise ValueError("Each tried layer needs a finite clean drop")
        values = values[drops <= eps_clean]
    candidate = values.idxmax() if len(values) and values.max() > 0 else "none"
    return Selection(candidate, "labels", {"n_tried_sites": len(layers)})


def select_small_search(u_val_partial, tried_layers, clean_drop_val=None, eps_clean=None):
    return select_val_sweep(u_val_partial, tried_layers, clean_drop_val, eps_clean)


def apply_admissibility(selection, u_val_for_chosen, clean_drop_for_chosen, eps_clean):
    """Check only the selected site; nonpositive validation gain also yields none."""
    if selection.candidate == "none":
        return selection
    if not np.isfinite(u_val_for_chosen):
        raise ValueError("Validation utility must be finite")
    acceptable = u_val_for_chosen > 0
    if eps_clean is not None:
        if not np.isfinite(eps_clean) or eps_clean < 0 or clean_drop_for_chosen is None or not np.isfinite(clean_drop_for_chosen):
            raise ValueError("Clean constraint requires finite drop and nonnegative epsilon")
        acceptable = acceptable and clean_drop_for_chosen <= eps_clean
    cost = dict(selection.selection_cost)
    cost["n_admissibility_checks"] = cost.get("n_admissibility_checks", 0) + 1
    return Selection(selection.candidate if acceptable else "none", "labels", cost)


def rank_correlation(s, u):
    if not s.index.is_unique or not u.index.is_unique:
        raise ValueError("Candidate indices must be unique")
    pairs = pd.concat([s.rename("s"), u.rename("u")], axis=1, join="inner").dropna()
    if not np.isfinite(pairs.to_numpy(dtype=float)).all():
        raise ValueError("Scores must be finite")
    if len(pairs) < 2 or pairs.s.nunique() < 2 or pairs.u.nunique() < 2:
        return dict(spearman=np.nan, spearman_pvalue=np.nan, kendall=np.nan, kendall_pvalue=np.nan, n_layers=len(pairs))
    sp, kp = stats.spearmanr(pairs.s, pairs.u), stats.kendalltau(pairs.s, pairs.u)
    return dict(spearman=float(sp.statistic), spearman_pvalue=float(sp.pvalue),
                kendall=float(kp.statistic), kendall_pvalue=float(kp.pvalue), n_layers=len(pairs))
