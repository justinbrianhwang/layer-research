"""Held-out effects, paired image uncertainty and frozen-selector regret."""
import json
import pandas as pd
from tqdm import tqdm
from layer_research.evaluation import layer_effect_table, equivalence_set, aggregate_over_seeds
from layer_research.statistics import paired_bootstrap, bootstrap_regret
from _common import Run, parser, read_table, table


def gain(df):
    return 100 * ((df.post_intervention_prediction == df.label).mean() - (df.baseline_prediction == df.label).mean())


def main():
    run = Run(parser(__doc__).parse_args(), "evaluate", model=False)
    raw = read_table(run.paths.raw / "patching_test.parquet", run)
    selections = read_table(run.paths.tables / "selections.parquet", run)
    groups = ["experiment", "corruption", "severity", "intervention_type", "fraction", "alpha", "norm_cap", "layer_id", "mask_seed"]
    effects = layer_effect_table(raw, group_cols=groups)
    for exp, frame in effects.groupby("experiment"):
        table(frame, run.paths.tables / f"{exp}_effects", run)
        table(aggregate_over_seeds(frame), run.paths.tables / f"{exp}_seed_variability", run)
    rows = []
    n_boot = run.cfg["evaluation"]["bootstrap_resamples"]
    for observed, subset in raw[raw.intervention_type.eq("partial_channel")].groupby("is_observed"):
        domain = "observed" if observed else "unseen"
        for _, selection in tqdm(selections.iterrows(), total=len(selections), desc=domain):
            cap = subset.norm_cap.isna() if pd.isna(selection.norm_cap) else subset.norm_cap.eq(selection.norm_cap)
            data = subset[subset.experiment.eq(selection.experiment) & subset.fraction.eq(selection.fraction) & subset.alpha.eq(selection.alpha) & cap]
            if data.empty:
                raise ValueError("Test budget missing for frozen selection")
            chosen = "none" if selection.candidate == "none" else int(selection.candidate)
            admissible = json.loads(selection.admissible)
            utilities = data.groupby("layer_id").apply(gain, include_groups=False).to_dict()
            utilities["none"] = 0.
            eq = equivalence_set(pd.Series({l: utilities[l] for l in admissible + ["none"]}), run.cfg["evaluation"]["tolerance_tau_pp"])
            ci = dict(mean=0., ci_low=0., ci_high=0.) if chosen == "none" else paired_bootstrap(data[data.layer_id.eq(chosen)], gain, n_boot=n_boot, seed=0)
            regret = bootstrap_regret(data, chosen, admissible, n_boot=n_boot, seed=0)
            rows.append(dict(selector=selection.selector, candidate=str(chosen), domain=domain, experiment=selection.experiment,
                fraction=selection.fraction, alpha=selection.alpha, norm_cap=selection.norm_cap, U=ci["mean"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                regret=regret["mean"], regret_ci_low=regret["ci_low"], regret_ci_high=regret["ci_high"], equivalence_set=json.dumps(list(eq))))
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
