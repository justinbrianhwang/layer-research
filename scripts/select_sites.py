"""Freeze selectors and metric directions using observed score/validation only."""
import json
import pandas as pd
from layer_research import site_selection as ss
from layer_research.evaluation import accuracy_gain_pp
from _common import Run, parser, read_table, table


def main():
    run = Run(parser(__doc__).parse_args(), "select_sites", model=False)
    metrics = read_table(run.paths.tables / "metrics_score.parquet", run)
    val = read_table(run.paths.raw / "patching_val.parquet", run)
    metrics = metrics[metrics.is_observed]
    val = val[val.is_observed]
    layers = run.cfg["representation"]["layers"]
    rows = []
    main = val[val.intervention_type.eq("partial_channel")]
    for (experiment, fraction, alpha, cap), group in main.groupby(["experiment", "fraction", "alpha", "norm_cap"], dropna=False):
        u = accuracy_gain_pp(group, ["layer_id"]).set_index("layer_id").U
        # Clean-to-clean donor replacement is identity up to cache rounding.
        clean = val[val.intervention_type.eq("clean_to_clean")].copy()
        if clean.empty:
            raise ValueError("clean_to_clean control required for validation admissibility")
        clean["baseline_prediction"] = clean.clean_prediction
        drops = -accuracy_gain_pp(clean, ["layer_id"]).set_index("layer_id").U
        eps = run.cfg["evaluation"]["clean_drop_tolerance_pp"]
        admissible = [l for l in layers if drops[l] <= eps]
        selections = [(f"fixed_{pos}", ss.select_fixed(pos, layers), None, None) for pos in ("front", "middle", "back")]
        selections += [("random", ss.select_random(layers, 0), None, None),
                       ("val_sweep", ss.select_val_sweep(u, layers, drops, eps), None, None)]
        tried = list(dict.fromkeys([layers[0], layers[len(layers)//2], layers[-1]]))
        selections.append(("small_search", ss.select_small_search(u.reindex(tried), tried, drops, eps), None, None))
        for (metric, mode), mt in metrics.groupby(["metric_name", "summary_mode"]):
            correlation = ss.rank_correlation(mt.groupby("layer").score.mean(), u)["spearman"]
            direction = "min" if correlation < 0 else "max"
            selection = ss.select_by_metric(mt, metric, direction, layers)
            selections.append((f"{metric}/{mode}", selection, direction, correlation))
        for name, selection, direction, correlation in selections:
            chosen = selection.candidate
            if chosen != "none":
                selection = ss.apply_admissibility(selection, u[chosen], drops[chosen], eps)
            cost = dict(selection.selection_cost)
            if direction:
                cost["n_direction_validation_sites"] = len(layers)
            rows.append(dict(selector=name, candidate=str(selection.candidate), label_access=selection.label_access,
                selection_cost=json.dumps(cost), direction=direction, val_spearman=correlation,
                admissible=json.dumps(admissible), experiment=experiment, fraction=fraction, alpha=alpha, norm_cap=cap))
    table(pd.DataFrame(rows), run.paths.tables / "selections", run)
    run.finish()


if __name__ == "__main__":
    main()
