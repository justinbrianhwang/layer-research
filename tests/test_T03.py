import numpy as np
import pandas as pd
import pytest

from layer_research.evaluation import (
    accuracy_gain_pp, recovery_and_new_error, margin_change, clean_accuracy_change_pp,
    layer_effect_table, aggregate_over_seeds, regret, equivalence_set,
)
from layer_research.site_selection import (
    select_fixed, select_random, select_by_metric, select_val_sweep,
    select_small_search, apply_admissibility, rank_correlation,
)
from layer_research.statistics import (
    paired_bootstrap, bootstrap_layer_gain, bootstrap_regret, seed_variability,
    selection_stability, holm_correction,
)


@pytest.fixture
def predictions():
    return pd.DataFrame(dict(image_id=[0, 1, 2, 3], label=[0]*4,
        baseline_prediction=[1, 1, 0, 0], clean_prediction=[0, 1, 0, 0],
        post_intervention_prediction=[0, 0, 1, 0], baseline_margin=[-2, -1, 1, 2],
        post_intervention_margin=[1, 1, -1, 3]))


def test_effects_and_frozen_sets(predictions):
    assert accuracy_gain_pp(predictions, []).iloc[0].to_dict() == {"U": 25., "n_images": 4.}
    pd.testing.assert_frame_equal(accuracy_gain_pp(predictions, []), clean_accuracy_change_pp(predictions, []))
    rates = recovery_and_new_error(predictions, []).iloc[0]
    assert rates.to_dict() == dict(recovery_rate=1., new_error_rate=.5,
        corruption_failure_set_size=1., still_correct_set_size=2.)
    changed = predictions.assign(post_intervention_prediction=1)
    sizes = ["corruption_failure_set_size", "still_correct_set_size"]
    pd.testing.assert_frame_equal(recovery_and_new_error(predictions, [])[sizes],
                                  recovery_and_new_error(changed, [])[sizes])
    assert margin_change(predictions, []).margin_change.iloc[0] == 1.
    empty_sets = predictions.assign(baseline_prediction=1, clean_prediction=1)
    assert recovery_and_new_error(empty_sets, []).iloc[0][["recovery_rate", "new_error_rate"]].isna().all()


def test_layer_table_and_seeds(predictions):
    parts = [predictions.assign(layer_id=layer, mask_seed=seed, norm_cap=None,
              effective_channel_ratio=.25, alpha=.5, corruption_type="noise", severity=1,
              intervention_type=policy, mask_policy="random_fixed")
             for layer in [0, 1] for seed in [2, 3] for policy in ["partial_channel", "none"]]
    df = pd.concat(parts, ignore_index=True)
    table = layer_effect_table(df)
    assert len(table) == 8
    assert table.norm_cap.isna().all()
    aggregate = aggregate_over_seeds(table)
    assert len(aggregate) == 4
    assert aggregate.U_mean.eq(25).all() and aggregate.U_std.eq(0).all()
    assert aggregate.n_seeds.eq(2).all()
    with pytest.raises(ValueError):
        aggregate_over_seeds(pd.concat([table, table]))
    custom = pd.DataFrame({"training_seed": [0, 1], "loss": [1., 3.]})
    assert seed_variability(custom, "training_seed", value_cols=["loss"]).loss_mean.iloc[0] == 2


def test_regret_and_equivalence():
    u = pd.Series([3., -1., 2.], index=[0, 1, 2])
    assert regret(u, 0, {0, 1, 2}) == 0
    assert regret(u, 2, {0, 1, 2}) == 1
    assert regret(u, "none", {0, 1, 2}) == 3
    assert regret(u, "none", {1}) == 0
    assert regret(u, 1, {1}) == 1
    assert equivalence_set(u, 1) == [0, 2]
    with pytest.raises(ValueError):
        regret(u, 0, {1})
    with pytest.raises(ValueError):
        equivalence_set(u, -1)


def test_metric_rank_aggregation_and_direction():
    # A huge outlier wins the raw mean but loses two out of three ranks.
    table = pd.DataFrame({"layer": [0, 1]*3, "metric_name": ["distance"]*6,
        "score": [1000, 0, 0, 2, 0, 2], "corruption_type": ["a", "a", "b", "b", "c", "c"],
        "label_access": ["none"]*6, "score_data_split": ["score"]*6})
    assert select_by_metric(table, "distance", "max", [0, 1]).candidate == 1
    assert select_by_metric(table, "distance", "min", [0, 1]).candidate == 0
    assert select_by_metric(table, "distance", "max", [0, 1], aggregate="mean_score").candidate == 0
    assert select_by_metric(table, "distance", "max", [0, 1]).label_access == "none"
    with pytest.raises(ValueError):
        select_by_metric(table.assign(score_data_split="test"), "distance", "max", [0, 1])
    with pytest.raises(ValueError):
        select_by_metric(table.iloc[:-1], "distance", "max", [0, 1])


def test_selectors_and_admissibility():
    assert [select_fixed(p, [0, 1, 2]).candidate for p in ("front", "middle", "back")] == [0, 1, 2]
    assert select_random([0, 1], 42) == select_random([0, 1], 42)
    assert select_random([], 1, include_none=True).candidate == "none"
    u = pd.DataFrame({"layer_id": [0, 1, 2], "U": [3, 2, -1], "split": ["val"]*3})
    drops = pd.Series([1, .1, 0], index=[0, 1, 2])
    assert select_val_sweep(u, [0, 1, 2], drops, .2).candidate == 1
    assert select_small_search(u, [2]).candidate == "none"
    selected = select_small_search(u, [1])
    assert selected.candidate == 1 and selected.selection_cost["n_tried_sites"] == 1
    fixed = select_fixed("front", [0, 1])
    fallback = apply_admissibility(fixed, 3, 1, .2)
    assert fallback.candidate == "none" and fallback.label_access == "labels"
    assert fallback.selection_cost["n_admissibility_checks"] == 1
    assert "n_admissibility_checks" not in fixed.selection_cost
    assert apply_admissibility(fixed, 1, .2, .2).candidate == 0
    assert apply_admissibility(fixed, -1, 0, .2).candidate == "none"
    with pytest.raises(ValueError):
        select_val_sweep(u.assign(split="test"), [0])


def test_rank_correlation_alignment():
    result = rank_correlation(pd.Series([1, 2, 3], index=[0, 1, 2]),
                              pd.Series([30, 10, 20], index=[2, 0, 1]))
    assert result["spearman"] == 1 and result["kendall"] == 1
    assert result["n_layers"] == 3
    assert np.isnan(rank_correlation(pd.Series([1, 1]), pd.Series([1, 2]))["spearman"])


def test_bootstrap_clusters_reproducible_stratified():
    df = pd.DataFrame({"image_id": np.repeat(np.arange(6), 3),
                       "label": np.repeat([0, 0, 0, 1, 1, 1], 3), "value": np.repeat(np.arange(6), 3)})
    def statistic(data):
        assert data.groupby("image_id").size().mod(3).eq(0).all()
        assert data.groupby("label").size().tolist() == [9, 9]
        return data.value.mean()
    a = paired_bootstrap(df, statistic, n_boot=50, seed=3, stratify_col="label")
    b = paired_bootstrap(df, statistic, n_boot=50, seed=3, stratify_col="label")
    np.testing.assert_array_equal(a["samples"], b["samples"])
    assert a["mean"] == 2.5 and a["ci_low"] <= 2.5 <= a["ci_high"]
    assert np.unique(a["samples"]).size > 1
    bad = df.copy()
    bad.loc[0, "label"] = 1
    with pytest.raises(ValueError):
        paired_bootstrap(bad, statistic, stratify_col="label")


def test_gain_and_regret_bootstrap(predictions):
    df = pd.concat([predictions.assign(layer_id=0),
                    predictions.assign(layer_id=1, post_intervention_prediction=predictions.baseline_prediction)], ignore_index=True)
    gain = bootstrap_layer_gain(df, n_boot=30, seed=4)
    assert gain["mean"].tolist() == [25., 0.]
    result = bootstrap_regret(df, "none", {0, 1}, n_boot=30, seed=4)
    assert result["mean"] == 25
    assert (result["samples"] >= 0).all()
    # Both layers move together: identical gains imply zero regret on every resample.
    identical = pd.concat([predictions.assign(layer_id=i) for i in [0, 1]], ignore_index=True)
    identical.post_intervention_prediction = 0
    result = bootstrap_regret(identical, 0, {0, 1}, n_boot=30)
    np.testing.assert_array_equal(result["samples"], np.zeros(30))
    with pytest.raises(ValueError):
        bootstrap_regret(df.iloc[:-1], 0, {0, 1}, n_boot=2)


def test_holm_and_stability():
    np.testing.assert_allclose(holm_correction([.01, .04, .03, .002]), [.03, .06, .06, .008])
    assert holm_correction([]) == []
    assert holm_correction([1, 1]) == [1, 1]
    with pytest.raises(ValueError):
        holm_correction([np.nan])
    result = selection_stability([select_fixed("front", [0]), 0, "none"])
    assert result["count"].tolist() == [2, 1]
    np.testing.assert_allclose(result.frequency, [2/3, 1/3])
    assert selection_stability([]).empty
