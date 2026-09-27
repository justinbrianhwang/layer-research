"""Offline ResNet feature, intervention, adapter and compute contracts."""
import pytest
import timm
import torch
from timm.models.vision_transformer import VisionTransformer
from layer_research.feature_extractor import (
    resolve_blocks, shape_report, BlockOutputRecorder, summarize, extract_paired_features,
)
from layer_research.patching_engine import (
    Patcher, MaskSpec, nested_masks, run_patched_forward, sweep_layers_budgets,
)
from layer_research.adapter_training import BottleneckAdapter, AdaptTrainConfig, train_adapter, _flops


@pytest.fixture
def cnn():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(8)
    yield timm.create_model("resnet18", pretrained=False, num_classes=10).eval()
    torch.set_num_threads(old)


def test_features(cnn):
    blocks = resolve_blocks(cnn, "resnet_stages")
    assert len(blocks) == 8
    shapes = shape_report(cnn, (3, 32, 32), "resnet_stages")
    assert shapes[0] == (1, 64, 8, 8)
    assert shapes[7] == (1, 512, 1, 1)
    x = torch.randn(2, 3, 32, 32)
    features = extract_paired_features(cnn, [(x, x, [0, 1], ["a", "b"])],
                                      [0, 7], ["gap"], blocks_attr="resnet_stages")
    assert features["clean"][0]["gap"].shape == (2, 64)
    with pytest.raises(RuntimeError):
        with BlockOutputRecorder(cnn, [0], blocks_attr="resnet_stages"):
            raise RuntimeError("cleanup")
    assert all(not b._forward_hooks for b in blocks)
    h = torch.randn(2, 3, 4, 5)
    torch.testing.assert_close(summarize(h, "gap"), h.mean((2, 3)))
    for mode in ("cls", "patch_mean", "cls+patch_mean"):
        with pytest.raises(ValueError, match="ViT-only"):
            summarize(h, mode)


@pytest.mark.parametrize("layer", [0, 7])
def test_controls_and_channel_support(cnn, layer):
    x, corr = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    with torch.no_grad(), BlockOutputRecorder(cnn, [layer], blocks_attr="resnet_stages") as rec:
        clean_logits = cnn(x)
        donor = rec.outputs[layer]
        baseline = cnn(corr)
        receiver = rec.outputs[layer]
    masks = nested_masks(donor.shape[1], [.25, .5], 3)
    assert (masks[.25] <= masks[.5]).all()
    for kind, inputs, expected in (("none", corr, baseline), ("full_state", corr, clean_logits),
                                   ("clean_to_clean", x, clean_logits)):
        logits, _ = run_patched_forward(cnn, layer, inputs, donor, 1., masks[.25],
                                       intervention=kind, exclude_cls=True, blocks_attr="resnet_stages")
        assert torch.equal(logits, expected)
    for kind in ("partial_channel", "random_direction", "shuffled_donor"):
        with Patcher(cnn, layer, mask=masks[.5], intervention=kind,
                     generator=torch.Generator().manual_seed(4), labels=torch.tensor([0, 1]),
                     exclude_cls=True, norm_cap=.2, r_l=1., blocks_attr="resnet_stages") as patch:
            patch.set_donor(donor)
            result = patch._hook(None, (), receiver)
            assert torch.equal(result[:, ~masks[.5]], receiver[:, ~masks[.5]])
            assert not patch.last_stats["exclude_cls"]
            assert (patch.last_stats["actual_delta_norm"] <= .2 * receiver[0].numel()**.5 + 1e-5).all()
            if kind == "shuffled_donor":
                assert patch.last_stats["donor_index"].tolist() == [1, 0]
            cnn.train()
            with pytest.raises(AssertionError, match="eval mode"):
                patch._hook(None, (), receiver)
            cnn.eval()
    assert all(not b._forward_hooks for b in resolve_blocks(cnn, "resnet_stages"))


@pytest.mark.parametrize("layer", [0, 7])
def test_adapter_training(cnn, layer):
    shapes = shape_report(cnn, (3, 32, 32), "resnet_stages")
    adapter = BottleneckAdapter(shapes[layer][1], 4, layout="channels_last_4d")
    h = torch.randn(shapes[layer])
    assert torch.equal(adapter(h), h)
    assert adapter.extra_params() == 2*adapter.d*4 + 3*adapter.d + 4
    x = torch.randn(2, 3, 32, 32)
    before = {k: v.clone() for k, v in cnn.state_dict().items()}
    cfg = AdaptTrainConfig(layer, 4, .01, 0., 1, 2, 1., 0)
    result = train_adapter(cnn, adapter, [(x, x + .1, [0, 1], ["a", "b"])], cfg,
                           "cpu", blocks_attr="resnet_stages")
    assert adapter.up.weight.grad.abs().sum() > 0
    assert adapter.up.weight.abs().sum() > 0
    assert result.cost["training_compute"] > 0
    assert all(torch.equal(v, before[k]) for k, v in cnn.state_dict().items())
    assert all(not b._forward_hooks for b in resolve_blocks(cnn, "resnet_stages"))


def test_sweep(cnn):
    x = torch.randn(2, 3, 32, 32)
    batch = dict(x_clean=x, x_corr=x+.1, y=[0, 1], image_id=["a", "b"])
    rows = sweep_layers_budgets(cnn, [batch], [0, 7], [.25, .5], [1.], [0, 1],
                                blocks_attr="resnet_stages")
    assert len(rows) == 16
    assert set(rows.channel_count) == {16, 32, 128, 256}


def test_vit_flops_agreement():
    model = VisionTransformer(img_size=16, patch_size=8, embed_dim=16, depth=3,
                              num_heads=2, num_classes=3).eval()
    adapter = BottleneckAdapter(16, 4)
    result = _flops(model, torch.zeros(1, 3, 16, 16), 0, adapter)
    n, d, m = 5, 16, 64
    old_block = 8*n*d*d + 4*n*n*d + 4*n*d*m + 10*n*d
    old = 2*4*16*3*64 + 3*old_block + 2*d*3 + 5*n*d + n*adapter.extra_flops_per_token()
    assert abs(result["forward_flops_per_image"] / old - 1) < .1
    assert all(abs(value / old_block - 1) < .1 for value in result["block_forward_flops"])
    assert result["backward_flops_per_image"] == 2 * (sum(result["block_forward_flops"][1:]) + 2*d*3 + result["adapter_forward_flops_per_image"])


def test_resnet_scripts_split_cache(tmp_path, monkeypatch, cnn):
    import importlib
    import sys
    from pathlib import Path
    import pandas as pd
    import yaml
    from torch.utils.data import DataLoader

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    common = importlib.import_module("_common")
    monkeypatch.setattr(common, "create_model", lambda cfg, device: cnn.to(device))
    x = torch.randn(2, 3, 32, 32)
    def loader(run, split, name, severity, paired=True):
        corr = x if name == "clean" else x + .1
        return DataLoader([(x[i], corr[i], i, f"id{i}") for i in range(2)], batch_size=2)
    monkeypatch.setattr(common, "loader", loader)
    # Modules may already be imported by the existing integration suite.
    for name in ("cache_features", "run_patching", "check_tokens_precision", "train_adapters"):
        monkeypatch.setattr(importlib.import_module(name), "loader", loader)
    cfg = common.load_config(Path(__file__).resolve().parents[1] / "configs/resnet50.yaml")
    cfg["output_root"] = str(tmp_path)
    cfg["model"].update(input_size=[3, 32, 32], pretrained=False)
    cfg["representation"].update(layers=[0, 7], metrics=["relative_distance"])
    cfg["corruptions"].update(observed=["gaussian_noise"], unseen=[], severities=[1])
    cfg["runtime"].update(batch_size=2, num_workers=0)
    cfg["patching"].update(mask_seeds=1, channel_fractions=[.5])
    cfg["adapter"].update(steps=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    def invoke(name, *args):
        monkeypatch.setattr(sys, "argv", [name, "--config", str(path), "--device", "cpu", *args])
        importlib.import_module(name).main()
    invoke("cache_features", "--layers", "0")
    invoke("cache_features", "--layers", "7")
    base = tmp_path / "cache/score"
    clean = torch.load(base / "clean/summaries.pt", weights_only=False)
    assert set(clean["summaries"]) == {0, 7}
    assert set(clean["r_l"]) == {0, 7}
    for layer in (0, 7):
        tokens = torch.load(base / f"clean/tokens_layer{layer}.pt", weights_only=False)["tokens"]
        assert tokens.ndim == 4 and tokens.dtype == torch.float32
    corr = torch.load(base / "gaussian_noise_1/summaries.pt", weights_only=False)
    assert corr["channel_abs_delta"][0].shape == (64,)
    assert corr["channel_abs_delta"][7].shape == (512,)
    invoke("compute_metrics")
    assert len(pd.read_parquet(tmp_path / "results/tables/metrics_score.parquet")) == 2
    invoke("run_patching", "--layers", "0,7")
    assert len(pd.read_parquet(tmp_path / "results/raw/patching_val.parquet")) == 24
    invoke("check_tokens_precision", "--limit", "2")
    assert pd.read_parquet(tmp_path / "results/tables/tokens_precision.parquet").max_abs_diff.max() == 0
    invoke("train_adapters", "--site", "7")
    assert (tmp_path / "results/raw/adapters/layer_7_width_32_seed_0/adapter.pt").exists()
