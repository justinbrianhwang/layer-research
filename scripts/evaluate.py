"""Held-out effects, paired image uncertainty and frozen-selector regret."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from layer_research.evaluation import equivalence_set, aggregate_over_seeds
from _common import Run, parser, read_table_columns, table, fast_image_bootstrap, bootstrap_summary


BUDGET = ["experiment", "fraction", "alpha", "norm_cap"]
GROUPS = ["experiment", "corruption", "severity", "intervention_type", "fraction", "alpha",
          "norm_cap", "layer_id", "mask_seed"]


def effect_table(raw, path):
    """Vectorized library-equivalent summaries, without raw margins in pandas.

    Margin differences are necessary for the existing output contract. Read the
    two source columns only in bounded Arrow batches and discard each batch.
    """
    before = raw.baseline_prediction.eq(raw.label)
    after = raw.post_intervention_prediction.eq(raw.label)
    failure = raw.clean_prediction.eq(raw.label) & ~before
    work = raw[GROUPS].copy()
    work["before"] = before
    work["after"] = after
    work["failure"] = failure
    work["recovered"] = failure & after
    work["new_error"] = before & ~after
    margins = np.empty(len(raw), dtype=np.float64)
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=65536, columns=["baseline_margin", "post_intervention_margin"]):
        baseline, post = (col.to_numpy(zero_copy_only=False) for col in batch.columns)
        if np.isnan(baseline).any() or np.isnan(post).any():
            raise ValueError("Missing values in margin columns")
        margins[offset:offset + len(batch)] = post - baseline
        offset += len(batch)
    if offset != len(raw):
        raise ValueError("Margin rows are not aligned with predictions")
    work["margin_change"] = margins
    del margins
    result = work.groupby(GROUPS, sort=False, observed=True, dropna=False).agg(
        before=("before", "mean"), after=("after", "mean"), n_images=("before", "size"),
        recovered=("recovered", "sum"), new_error=("new_error", "sum"),
        corruption_failure_set_size=("failure", "sum"), still_correct_set_size=("before", "sum"),
        margin_change=("margin_change", "mean")).reset_index()
    result["U"] = 100 * (result.after - result.before)
    result["recovery_rate"] = result.recovered / result.corruption_failure_set_size
    result["new_error_rate"] = result.new_error / result.still_correct_set_size
    return result[GROUPS + ["U", "n_images", "recovery_rate", "new_error_rate",
                            "corruption_failure_set_size", "still_correct_set_size", "margin_change"]]


def image_aggregates(raw):
    """Reduce the sweep once; budget/domain slicing subsequently touches only means."""
    cols = ["is_observed"] + BUDGET + ["image_id", "layer_id"]
    part = raw.loc[raw.intervention_type.eq("partial_channel"), cols].copy()
    if part.image_id.isna().any():
        raise ValueError("Bootstrap requires nonmissing image IDs")
    mask = raw.intervention_type.eq("partial_channel")
    part["difference"] = (raw.loc[mask, "post_intervention_prediction"].eq(raw.loc[mask, "label"]).astype(np.int8)
                          - raw.loc[mask, "baseline_prediction"].eq(raw.loc[mask, "label"]).astype(np.int8))
    return part.groupby(cols, sort=False, observed=True, dropna=False).difference.agg(["mean", "size"]).reset_index()


def image_matrix(aggregates):
    """Return a dense, aligned image/layer matrix and its explicit ID order.

    Equal weights are essential to the equivalence with row means. Fail rather
    than silently change the estimand for an incomplete/unbalanced sweep.
    """
    means = aggregates.pivot(index="image_id", columns="layer_id", values="mean")
    counts = aggregates.pivot(index="image_id", columns="layer_id", values="size")
    if counts.empty or counts.isna().any().any() or np.unique(counts.to_numpy()).size != 1:
        raise ValueError("Per-image bootstrap requires balanced, matched image rows across layers")
    return means.to_numpy(), means.index.to_numpy(), means.columns.tolist()


def gain(df):
    return 100 * ((df.post_intervention_prediction == df.label).mean() - (df.baseline_prediction == df.label).mean())


def main():
    run = Run(parser(__doc__).parse_args(), "evaluate", model=False)
    path = run.paths.raw / "patching_test.parquet"
    raw = read_table_columns(path, run, GROUPS + ["image_id", "is_observed", "label",
                            "baseline_prediction", "clean_prediction", "post_intervention_prediction"])
    selections = read_table_columns(run.paths.tables / "selections.parquet", run,
                                   BUDGET + ["selector", "candidate", "admissible"])
    for column in ("label", "baseline_prediction", "clean_prediction", "post_intervention_prediction"):
        if raw[column].isna().any():
            raise ValueError(f"Missing values in {column}")
    print("Computing condition effects and per-image aggregates", flush=True)
    effects = effect_table(raw, path)
    aggregates = image_aggregates(raw)
    del raw
    for exp, frame in effects.groupby("experiment"):
        table(frame, run.paths.tables / f"{exp}_effects", run)
        table(aggregate_over_seeds(frame), run.paths.tables / f"{exp}_seed_variability", run)
    rows = []
    n_boot = run.cfg["evaluation"]["bootstrap_resamples"]
    for observed, subset in aggregates.groupby("is_observed", observed=True):
        domain = "observed" if observed else "unseen"
        print(f"{domain}: evaluating {len(selections)} frozen selections", flush=True)
        for (experiment, fraction, alpha, norm_cap), choices in selections.groupby(BUDGET, dropna=False, observed=True):
            cap = subset.norm_cap.isna() if pd.isna(norm_cap) else subset.norm_cap.eq(norm_cap)
            data = subset[subset.experiment.eq(experiment) & subset.fraction.eq(fraction) & subset.alpha.eq(alpha) & cap]
            if data.empty:
                raise ValueError("Test budget missing for frozen selection")
            D, image_ids, layers = image_matrix(data)
            boot = fast_image_bootstrap(D, n_boot=n_boot, seed=0)
            utilities = np.append(boot["mean"], 0.)
            samples = np.column_stack((boot["samples"], np.zeros(n_boot)))
            positions = {layer: i for i, layer in enumerate(layers + ["none"])}
            for selection in choices.itertuples(index=False):
                chosen = "none" if selection.candidate == "none" else int(selection.candidate)
                admissible = list(dict.fromkeys(json.loads(selection.admissible) + ["none"]))
                if chosen not in admissible:
                    raise ValueError("Chosen candidate is not admissible")
                indices = [positions[layer] for layer in admissible]
                index = positions[chosen]
                eq = equivalence_set(pd.Series(utilities[indices], index=admissible), run.cfg["evaluation"]["tolerance_tau_pp"])
                ci = bootstrap_summary(utilities[index], samples[:, index])
                regret = bootstrap_summary(utilities[indices].max() - utilities[index],
                                           samples[:, indices].max(axis=1) - samples[:, index])
                rows.append(dict(selector=selection.selector, candidate=str(chosen), domain=domain, experiment=selection.experiment,
                    fraction=selection.fraction, alpha=selection.alpha, norm_cap=selection.norm_cap, U=ci["mean"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                    regret=regret["mean"], regret_ci_low=regret["ci_low"], regret_ci_high=regret["ci_high"], equivalence_set=json.dumps(list(eq))))
                print(f"  {domain}/{experiment} fraction={fraction} alpha={alpha} norm_cap={norm_cap}: "
                      f"{selection.selector} -> {chosen} ({len(image_ids)} images)", flush=True)
    selected = pd.DataFrame(rows)
    for domain, frame in selected.groupby("domain"):
        table(frame, run.paths.tables / f"E3_{domain}", run)
    for exp, frame in selected.groupby("experiment"):
        table(frame, run.paths.tables / f"{exp}_selectors", run)
    text = "# Generated-corruption evaluation on ImageNetV2\n\nModel fingerprint: " + run.fingerprint
    text += "\n\nImage bootstrap uncertainty is separate from mask-seed variability.\n\n"
    text += "| " + " | ".join(selected.columns) + " |\n| " + " | ".join(["---"] * len(selected.columns)) + " |\n"
    text += "\n".join("| " + " | ".join(map(str, row)) + " |" for row in selected.itertuples(index=False, name=None))
    (run.paths.results / "summary.md").write_text(text, encoding="utf-8")
    run.finish()


if __name__ == "__main__":
    main()
