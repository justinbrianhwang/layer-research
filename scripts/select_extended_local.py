"""Freeze selections for the extended metrics (task sensitivity, topology) from aggregated validation utilities.

The original E1 validation raw table was lost with its host; only the per-condition aggregate
`results/tables/val_patch_U_by_layer.csv` (mean over 20 mask seeds) survives. Because the validation
utility used by `select_sites.py` is the mean of U over balanced observed conditions, the aggregate
reproduces it exactly, so the same selection rule (direction frozen by validation Spearman sign,
mean-rank aggregation across corruptions, admissibility = all sites, as in the frozen E1 selections)
can be applied to any metric table computed on the score split. Output: selections_extended.parquet
with the original selections plus the new metric-based selectors, ready for `evaluate.py --selections`.
"""
import argparse, json
import numpy as np
import pandas as pd
from layer_research import site_selection as ss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-agg", default="results/tables/val_patch_U_by_layer.csv")
    p.add_argument("--base-selections", default="results/tables/selections.parquet")
    p.add_argument("--metric-tables", nargs="+", required=True)
    p.add_argument("--out", default="results/tables/selections_extended.parquet")
    a = p.parse_args()
    base = pd.read_parquet(a.base_selections)
    fp = base.model_fingerprint.iloc[0]
    val = pd.read_csv(a.val_agg)
    val = val[val.corruption.isin(["gaussian_noise", "defocus_blur"])]
    metrics = pd.concat([pd.read_csv(t) for t in a.metric_tables], ignore_index=True)
    metrics = metrics[metrics.is_observed] if "is_observed" in metrics else metrics
    if "model_fingerprint" in metrics and not metrics.model_fingerprint.eq(fp).all():
        raise ValueError("metric table fingerprint differs from the frozen selections")
    layers = list(range(12))
    rows = []
    for fraction, vg in val.groupby("fraction"):
        u = vg.groupby("layer_id")["mean"].mean().reindex(layers)
        u.index.name = "layer_id"
        basef = base[base.fraction.eq(fraction)]
        admissible = json.loads(basef.admissible.iloc[0])
        alpha, cap = basef.alpha.iloc[0], basef.norm_cap.iloc[0]
        for (metric, mode), mt in metrics.groupby(["metric_name", "summary_mode"]):
            corr = ss.rank_correlation(mt.groupby("layer").score.mean(), u)["spearman"]
            direction = "min" if corr < 0 else "max"
            sel = ss.select_by_metric(mt, metric, direction, layers)
            chosen = sel.candidate
            if chosen != "none":
                sel = ss.apply_admissibility(sel, u[chosen], 0.0, 0.5)
            cost = dict(sel.selection_cost); cost["n_direction_validation_sites"] = len(layers)
            rows.append(dict(selector=f"{metric}/{mode}", candidate=str(sel.candidate), label_access=sel.label_access,
                             selection_cost=json.dumps(cost), direction=direction, val_spearman=corr,
                             admissible=json.dumps(admissible), experiment="E1", fraction=fraction, alpha=alpha,
                             norm_cap=cap, model_fingerprint=fp))
    ext = pd.DataFrame(rows)
    out = pd.concat([base, ext], ignore_index=True)
    out.to_parquet(a.out, index=False); out.to_csv(a.out.replace(".parquet", ".csv"), index=False)
    print(ext[["selector", "fraction", "candidate", "direction", "val_spearman"]].to_string(index=False))
    print(f"wrote {a.out}: {len(base)} base + {len(ext)} extended rows")


if __name__ == "__main__":
    main()
