"""Compute label-using first-order margin sensitivity on observed score data."""
import pandas as pd

from layer_research.task_sensitivity import margin_gradient_scores
from _common import Run, parser, loader, conditions, condition_dir, read_cache, table


def cached_batches(run, split, name, severity, layers):
    source = loader(run, split, name, severity)
    ids = [record[2] for record in source.dataset.records]
    base = condition_dir(run.paths, split, "clean", 0)
    donors = {l: read_cache(base / f"tokens_layer{l}.pt", run, ids) for l in layers}
    offset = 0
    for clean, corr, labels, batch_ids in source:
        stop = offset + len(batch_ids)
        if ids[offset:stop] != list(batch_ids):
            raise ValueError("Clean donor IDs do not match receiver IDs")
        yield dict(x_corr=corr, y=labels, image_id=batch_ids,
                   clean_outputs={l: donors[l]["tokens"][offset:stop] for l in layers})
        offset = stop


def main():
    p = parser(__doc__)
    p.add_argument("--split", choices=["score"], default="score")
    p.add_argument("--batch-size", type=int)
    run = Run(p.parse_args(), "compute_task_sensitivity")
    layers = run.cfg["representation"]["layers"]
    frames, costs = [], []
    for name, severity in conditions(run.cfg):
        if name not in run.cfg["corruptions"]["observed"] or severity not in (1, 3, 5):
            continue
        scores = margin_gradient_scores(run.model,
            cached_batches(run, run.args.split, name, severity, layers), layers, run.device)
        costs.append(dict(corruption=name, severity=severity, **scores.attrs))
        means = scores.groupby("layer_id")[["dot_full", "dot_norm"]].mean()
        counts = scores.groupby("layer_id").size()
        for column, metric in (("dot_full", "task_sens_dot"), ("dot_norm", "task_sens_cos")):
            frames.append(pd.DataFrame(dict(layer=means.index, metric_name=metric,
                score=means[column].to_numpy(), n_samples=counts.to_numpy(), summary_mode="full",
                label_access="labels", corruption=name, severity=severity, split=run.args.split, is_observed=True)))
    if not frames:
        raise ValueError("No observed score conditions at severities 1, 3 or 5")
    table(pd.concat(frames, ignore_index=True), run.paths.tables / "task_sensitivity_score", run)
    run.metric_cost = dict(backward_count=sum(c["backward_count"] for c in costs),
                           wall_time_seconds=sum(c["wall_time_seconds"] for c in costs), calls=costs)
    run.finish()


if __name__ == "__main__":
    main()
