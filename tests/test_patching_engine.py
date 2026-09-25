import math

import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from layer_research.feature_extractor import BlockOutputRecorder
from layer_research.patching_engine import (
    InterventionType as IT, MaskSpec, Patcher, make_channel_mask,
    nested_masks, run_patched_forward, sweep_layers_budgets,
)


@pytest.fixture
def setup():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = VisionTransformer(img_size=32, patch_size=8, embed_dim=32,
                                  depth=3, num_heads=4, num_classes=10, drop_rate=.3)
    gen = torch.Generator().manual_seed(19)
    clean = torch.randn(3, 3, 32, 32, generator=gen)
    corr = clean + .6 * torch.randn(clean.shape, generator=gen)
    with torch.no_grad(), BlockOutputRecorder(model, [0, 1, 2]) as rec:
        clean_logits = model(clean)
    with torch.no_grad():
        baseline = model(corr)
    yield model, clean, corr, rec.outputs, clean_logits, baseline
    torch.set_num_threads(threads)


@pytest.mark.parametrize("layer", [0, 1, 2])
@pytest.mark.parametrize("kind,q,alpha", [
    (IT.NONE, .5, 1), (IT.PARTIAL_CHANNEL, 0, 1),
    (IT.PARTIAL_CHANNEL, .5, 0), (IT.CLEAN_TO_CLEAN, .5, 1),
    (IT.FULL_STATE, 0, 0),
])
def test_controls(setup, layer, kind, q, alpha):
    model, clean, corr, outputs, clean_logits, baseline = setup
    mask, _ = make_channel_mask(32, MaskSpec("random_fixed", q, 4))
    receiver = clean if kind == IT.CLEAN_TO_CLEAN else corr
    result, stats = run_patched_forward(model, layer, receiver, outputs[layer], alpha,
                                       mask, intervention=kind)
    expected = clean_logits if kind in (IT.CLEAN_TO_CLEAN, IT.FULL_STATE) else baseline
    torch.testing.assert_close(result, expected, atol=1e-5 if kind == IT.FULL_STATE else 1e-6, rtol=0)
    if kind != IT.FULL_STATE:
        assert not stats["actual_delta_norm"].any()
    with torch.no_grad():
        assert torch.equal(model(corr), baseline)
    assert all(not block._forward_hooks for block in model.blocks)
    assert not model.training


def test_hook_cleanup_and_donor_updates(setup):
    model, clean, corr, outputs, clean_logits, baseline = setup
    patcher = Patcher(model, 1, MaskSpec("random_fixed", 1, 3))
    with torch.no_grad(), patcher:
        patcher.set_donor(outputs[1])
        torch.testing.assert_close(model(corr), clean_logits)
        patcher.set_donor(outputs[1])
        torch.testing.assert_close(model(clean), clean_logits)
        with pytest.raises(RuntimeError, match="already active"):
            patcher.__enter__()
    for from_hook in (False, True):
        with pytest.raises(ValueError), patcher:
            if from_hook:
                patcher.set_donor(outputs[1][:1])
                model(corr)
            else:
                raise ValueError("body exception")
        assert all(not block._forward_hooks for block in model.blocks)
        with torch.no_grad():
            assert torch.equal(model(corr), baseline)


def test_masks():
    state = torch.random.get_rng_state().clone()
    fractions = [0, .01, .1, .3, .8, 1]
    masks = nested_masks(37, fractions, 9)
    previous = torch.zeros(37, dtype=torch.bool)
    for q, mask in masks.items():
        assert int(mask.sum()) == math.floor(q * 37)
        assert torch.all(~previous | mask)
        assert torch.equal(mask, make_channel_mask(37, MaskSpec("random_fixed", q, 9))[0])
        previous = mask
    top, k = make_channel_mask(4, MaskSpec("score_topk", .5, 0), torch.tensor([1., 4., 4., 0.]))
    assert k == 2 and top.tolist() == [False, True, True, False]
    assert torch.equal(state, torch.random.get_rng_state())
    for q in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            make_channel_mask(4, MaskSpec("random_fixed", q, 0))
    with pytest.raises(ValueError):
        make_channel_mask(4, MaskSpec("score_topk", .5, 0))


def test_patch_equation(setup):
    model, _, corr, outputs, _, _ = setup
    spec = MaskSpec("random_fixed", .3, 5, exclude_cls=True)
    mask, _ = make_channel_mask(32, spec)
    with torch.no_grad(), BlockOutputRecorder(model, [1]) as before:
        model(corr)
    with torch.no_grad(), Patcher(model, 1, spec, alpha=.25) as patcher:
        patcher.set_donor(outputs[1])
        with BlockOutputRecorder(model, [1]) as after:
            model(corr)
    receiver = before.outputs[1]
    expected = receiver.clone()
    expected[:, 1:, mask] += .25 * (outputs[1][:, 1:, mask] - receiver[:, 1:, mask])
    assert torch.equal(after.outputs[1], expected)


def captured_patch(model, corr, donor, mask, **kwargs):
    with torch.no_grad(), BlockOutputRecorder(model, [1]) as before:
        model(corr)
    with torch.no_grad(), Patcher(model, 1, mask=mask, **kwargs) as patcher:
        patcher.set_donor(donor)
        with BlockOutputRecorder(model, [1]) as after:
            model(corr)
    return after.outputs[1] - before.outputs[1], patcher.last_stats


@pytest.mark.parametrize("rho", [0., .001, 100.])
def test_norm_cap(setup, rho):
    model, _, corr, outputs, _, _ = setup
    mask, _ = make_channel_mask(32, MaskSpec("random_fixed", .5, 4))
    raw, _ = captured_patch(model, corr, outputs[1], mask)
    capped, stats = captured_patch(model, corr, outputs[1], mask, norm_cap=rho, r_l=.7)
    raw_norm = raw.flatten(1).norm(dim=1)
    cap = rho * math.sqrt(17 * 32) * .7
    expected = raw_norm.clamp(max=cap)
    torch.testing.assert_close(stats["actual_delta_norm"], expected, atol=1e-6, rtol=1e-4)
    assert torch.all(capped.flatten(1).norm(dim=1) <= raw_norm + 1e-6)
    assert torch.equal(stats["n_elements_modified"], torch.count_nonzero(capped.flatten(1), dim=1))


@pytest.mark.parametrize("q", [0, .5])
@pytest.mark.parametrize("rho", [None, .001])
def test_random_direction_and_cls(setup, q, rho):
    model, _, corr, outputs, _, _ = setup
    mask, _ = make_channel_mask(32, MaskSpec("random_fixed", q, 4))
    options = dict(exclude_cls=True, norm_cap=rho, r_l=.7)
    partial, stats = captured_patch(model, corr, outputs[1], mask, **options)
    random, rand_stats = captured_patch(model, corr, outputs[1], mask,
        intervention=IT.RANDOM_DIRECTION, generator=torch.Generator().manual_seed(71), **options)
    repeat, _ = captured_patch(model, corr, outputs[1], mask,
        intervention=IT.RANDOM_DIRECTION, generator=torch.Generator().manual_seed(71), **options)
    assert torch.equal(random, repeat)
    assert not random[:, :, ~mask].any() and not random[:, 0].any()
    assert not partial[:, :, ~mask].any() and not partial[:, 0].any()
    torch.testing.assert_close(rand_stats["actual_delta_norm"], stats["actual_delta_norm"], rtol=1e-4, atol=1e-6)
    if q:
        assert not torch.equal(random, partial)


def test_shuffled_donor(setup):
    model, _, corr, outputs, _, _ = setup
    mask = torch.ones(32, dtype=torch.bool)
    labels = torch.tensor([1, 1, 2])
    _, stats = captured_patch(model, corr, outputs[1], mask,
        intervention=IT.SHUFFLED_DONOR, generator=torch.Generator().manual_seed(9), labels=labels)
    p = stats["donor_index"]
    assert sorted(p.tolist()) == [0, 1, 2] and torch.all(p != torch.arange(3))
    assert torch.equal(stats["same_class"], labels == labels[p])
    with pytest.raises(ValueError, match="explicit generator"):
        Patcher(model, 1, mask=mask, intervention=IT.RANDOM_DIRECTION)


def test_sweep_and_clean_forward_count(setup):
    model, clean, corr, outputs, clean_logits, baseline = setup
    batches = [dict(x_clean=clean, x_corr=corr, y=torch.tensor([0, 2, 1]),
                    image_id=[f"b{b}-i{i}" for i in range(3)]) for b in range(2)]
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args: calls.append(args[0].data_ptr()))
    try:
        table = sweep_layers_budgets(model, iter(batches), iter([0, 1, 2]),
                                     iter([0, .5]), iter([1]), iter([3, 7]), "cpu")
    finally:
        handle.remove()
    assert calls.count(clean.data_ptr()) == 2
    assert len(table) == 2 * 3 * 3 * 2 * 1 * 2
    prediction_columns = ["baseline_prediction", "clean_prediction", "post_intervention_prediction"]
    assert not table[prediction_columns].isna().any().any()
    zero = table[table.fraction == 0]
    assert (zero.baseline_prediction == zero.post_intervention_prediction).all()
    assert (zero.actual_delta_norm == 0).all()
    first = table.iloc[0]
    expected_margin = baseline[0, 0] - baseline[0, 1:].max()
    assert first.baseline_margin == pytest.approx(expected_margin.item())
    assert first.baseline_loss == pytest.approx(torch.nn.functional.cross_entropy(baseline[:1], torch.tensor([0])).item())
    assert all(not block._forward_hooks for block in model.blocks)
    cached = dict(batches[0], clean_outputs=outputs, clean_logits=clean_logits)
    calls.clear()
    handle = model.register_forward_pre_hook(lambda module, args: calls.append(args[0].data_ptr()))
    try:
        sweep_layers_budgets(model, [cached], [0, 1, 2], [.5], [1], [3], "cpu")
    finally:
        handle.remove()
    assert clean.data_ptr() not in calls


@pytest.mark.parametrize("kind", [IT.CLEAN_TO_CLEAN, IT.FULL_STATE, IT.SHUFFLED_DONOR])
def test_sweep_controls(setup, kind):
    model, clean, corr, _, _, _ = setup
    batch = dict(x_clean=clean, x_corr=corr, y=[0, 0, 1], image_id=["a", "b", "c"])
    table = sweep_layers_budgets(model, [batch], [1], [.5], [1], [9], intervention=kind)
    if kind == IT.SHUFFLED_DONOR:
        assert (table.image_id != table.donor_image_id).all()
        assert table.same_class.tolist() == [batch["y"][i] == batch["y"][j]
                                            for i, j in enumerate(table.donor_index)]
    else:
        assert (table.clean_prediction == table.post_intervention_prediction).all()
