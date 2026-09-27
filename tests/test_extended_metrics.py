"""First-order intervention agreement and optional persistent-homology checks."""
import numpy as np
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from layer_research.feature_extractor import BlockOutputRecorder
from layer_research.patching_engine import run_patched_forward
from layer_research.task_sensitivity import margin_gradient_scores
from layer_research.topology import ph_diagrams, ph_distance


def margin(z, labels):
    return z.gather(1, labels[:, None]).squeeze(1) - z.scatter(1, labels[:, None], -torch.inf).amax(1)


def test_topology_cost_accounting():
    from layer_research.topology import _cost
    metadata = {}
    with _cost(metadata):
        values = np.ones((100, 100))
    assert values.sum() == 10000
    assert metadata["peak_memory_bytes"] > 0
    assert metadata["wall_time_seconds"] > 0


def test_margin_first_order_and_cached_frozen_backbone():
    torch.set_num_threads(1)
    torch.manual_seed(14)
    model = VisionTransformer(img_size=16, patch_size=8, embed_dim=16, depth=3,
                              num_heads=2, num_classes=3).double().eval()
    clean = torch.randn(3, 3, 16, 16, dtype=torch.float64)
    corr = clean + .2 * torch.randn_like(clean)
    labels = torch.tensor([0, 1, 2])
    layers = [0, 1, 2]
    batch = (clean, corr, labels, ["a", "b", "c"])
    with torch.no_grad():
        scores = margin_gradient_scores(model, [batch], layers, "cpu")
    with torch.no_grad(), BlockOutputRecorder(model, layers) as recorder:
        model(clean)
        donors = dict(recorder.outputs)
        baseline = model(corr)
    for layer in layers:
        patched, _ = run_patched_forward(model, layer, corr, donors[layer], 1e-3,
                                         torch.ones(16, dtype=torch.bool))
        actual = margin(patched, labels) - margin(baseline, labels)
        expected = torch.tensor(scores[scores.layer_id.eq(layer)].dot_full.to_numpy()) * 1e-3
        torch.testing.assert_close(actual, expected, rtol=.05, atol=1e-10)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cached = dict(x_corr=corr, y=labels, image_id=batch[3], clean_outputs=donors)
    result = margin_gradient_scores(model, [cached], layers, "cpu")
    np.testing.assert_allclose(result.dot_full, scores.dot_full)
    assert result.attrs["backward_count"] == 1
    assert result.attrs["forward_count"] == 1
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert all(not block._forward_hooks for block in model.blocks)
    cached["clean_outputs"] = {l: torch.zeros(1) for l in layers}
    with pytest.raises(ValueError, match="shapes"):
        margin_gradient_scores(model, [cached], layers, "cpu")
    assert all(not block._forward_hooks for block in model.blocks)


@pytest.mark.parametrize("rotated", [False, True])
def test_topology_isometry(rotated):
    pytest.importorskip("ripser")
    pytest.importorskip("persim")
    rng = np.random.default_rng(4)
    H = rng.normal(size=(30, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    result = ph_distance(H, H @ Q if rotated else H, metric="both", pca_dim=None)
    assert all(v == pytest.approx(0, abs=1e-6) for v in result.values())
    assert result.metadata["infinite_bars_dropped"][0] == dict(clean=1, corrupted=1)
    assert result.metadata["wall_time_seconds"] > 0
    assert result.metadata["peak_memory_bytes"] > 0


def test_topology_subsampling_and_shared_pca():
    pytest.importorskip("ripser")
    pytest.importorskip("persim")
    from layer_research.topology import fit_clean_pca
    H = np.random.default_rng(9).normal(size=(40, 5))
    pca = fit_clean_pca(H, 2)
    first = ph_diagrams(H, pca=pca, n_points=15, seed=3)
    second = ph_diagrams(H, pca=pca, n_points=15, seed=3)
    assert first.metadata["row_indices"] == second.metadata["row_indices"]
    for dim in first:
        np.testing.assert_array_equal(first[dim], second[dim])
    corr = H * np.array([1, 2, 3, 4, 5])
    shared = ph_distance(H, corr, pca=pca, seed=3, n_points=15)
    explicit = ph_distance(pca.transform(H), pca.transform(corr), pca_dim=None, seed=3, n_points=15)
    assert shared == explicit
