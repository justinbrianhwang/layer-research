"""Offline script integration with an actual tiny ViT and paired image files."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml
from timm.models.vision_transformer import VisionTransformer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_script_chain(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    common = importlib.import_module("_common")

    def tiny(cfg, device):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        return VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=2,
                                 num_heads=2, num_classes=2).to(device).eval()

    monkeypatch.setattr(common, "create_model", tiny)
    root = tmp_path / "images"
    for label in range(2):
        (root / str(label)).mkdir(parents=True)
        for i in range(6):
            pixels = np.random.default_rng(label*6+i).integers(0, 256, (40, 40, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / str(label) / f"{i}.jpeg")
    cfg = common.load_config(SCRIPTS.parent / "configs/deit_small.yaml")
    cfg["data"].update(root=str(root), split_file=str(tmp_path / "splits.json"))
    cfg["model"].update(pretrained=False, input_size=[3, 32, 32], embed_dim=16)
    cfg["representation"].update(layers=[0], metrics=["relative_distance", "cosine_distance"])
    cfg["corruptions"].update(observed=["gaussian_noise"], unseen=["contrast"], severities=[1])
    cfg["patching"].update(channel_fractions=[.5], mask_seeds=1, alphas=[.5, 1.], norm_cap_rho=.5)
    cfg["evaluation"]["bootstrap_resamples"] = 5
    cfg["runtime"] = dict(batch_size=2, threads=1, num_workers=0)
    cfg["adapter"].update(widths=[4], default_width=4, training_seeds=1, steps=1)
    cfg["output_root"] = str(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))

    def invoke(name, *args):
        monkeypatch.setattr(sys, "argv", [name, "--config", str(path), "--device", "cpu", *args])
        importlib.import_module(name).main()

    invoke("download_data")
    original = (tmp_path / "splits.json").read_bytes()
    invoke("download_data")
    assert (tmp_path / "splits.json").read_bytes() == original
    invoke("cache_features", "--batch-size", "2")
    for split in ("score", "val", "test"):
        cache = tmp_path / "cache" / split
        assert torch.load(cache / "clean/tokens_layer0.pt", weights_only=False)["tokens"].dtype == torch.float16
        for condition in ("gaussian_noise_1", "contrast_1"):
            assert not list((cache / condition).glob("tokens_layer*.pt"))
            assert torch.load(cache / condition / "summaries.pt", weights_only=False)["r_l"][0] > 0

    from types import SimpleNamespace
    run = SimpleNamespace(cfg=cfg, args=SimpleNamespace(limit=None), model=tiny(cfg, "cpu"), device=torch.device("cpu"))
    serial = list(common.loader(run, "score", "gaussian_noise", 1))
    cfg["runtime"]["num_workers"] = 2
    parallel = list(common.loader(run, "score", "gaussian_noise", 1))
    for expected, actual in zip(serial, parallel, strict=True):
        for a, b in zip(expected[:3], actual[:3]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert expected[3] == actual[3]
    cfg["runtime"]["num_workers"] = 0
    import layer_research.data_protocol as protocol
    with monkeypatch.context() as patch:
        def no_corruption(*args):
            raise AssertionError("clean loader must not generate corruption")
        patch.setattr(protocol, "corrupt_image", no_corruption)
        for clean, corr, _, _ in common.loader(run, "score", "clean", 0, paired=False):
            torch.testing.assert_close(clean, corr)
    invoke("compute_metrics")
    assert len(pd.read_parquet(tmp_path / "results/tables/metrics_score.parquet")) == 8
    for split in ("val", "test"):
        invoke("run_patching", "--split", split)
        frame = pd.read_parquet(tmp_path / f"results/raw/patching_{split}.parquet")
        assert not (tmp_path / f"results/raw/patching_{split}.csv").exists()
        assert len(frame) == 2 * 2 * 6
        assert frame.groupby(["corruption", "intervention_type"]).image_id.nunique().eq(2).all()
    invoke("select_sites")
    selections = pd.read_parquet(tmp_path / "results/tables/selections.parquet")
    assert len(selections) == 10
    # Changing unseen val outcomes must not change frozen selectors.
    val_path = tmp_path / "results/raw/patching_val.parquet"
    val = pd.read_parquet(val_path)
    val.loc[~val.is_observed, "post_intervention_prediction"] = 999
    val.to_parquet(val_path, index=False)
    invoke("select_sites")
    pd.testing.assert_frame_equal(selections, pd.read_parquet(tmp_path / "results/tables/selections.parquet"))
    invoke("evaluate")
    assert len(pd.read_parquet(tmp_path / "results/tables/E1_effects.parquet")) == 12
    assert len(pd.read_parquet(tmp_path / "results/tables/E3_unseen.parquet")) == 10
    assert (tmp_path / "results/summary.md").exists()
    invoke("check_tokens_precision", "--limit", "2")
    assert pd.read_parquet(tmp_path / "results/tables/tokens_precision.parquet").max_abs_diff.max() < .01
    for split in ("val", "test"):
        invoke("run_patching", "--split", split, "--experiment", "E2")
        assert len(pd.read_parquet(tmp_path / f"results/raw/patching_{split}.parquet")) == 2 * 2 * (4 + 5)
    invoke("select_sites")
    invoke("evaluate")
    assert (tmp_path / "results/tables/E2_selectors.csv").exists()
    invoke("train_adapters", "--site", "0")
    assert len(pd.read_parquet(tmp_path / "results/raw/adapter_test.parquet")) == 6
    manifest = json.loads((tmp_path / "results/manifests/evaluate/run_manifest.json").read_text())
    assert len(manifest["model_fingerprint"]) == 64
    assert manifest["elapsed_seconds"] > 0
    invoke("run_patching", "--mask-seeds", "2", "--layers", "0", "--fractions", "0.25,0.5", "--batch-size", "2")
    frame = pd.read_parquet(tmp_path / "results/raw/patching_val.parquet")
    assert len(frame) == 2 * 2 * (4 + 5)
    manifest = json.loads((tmp_path / "results/manifests/run_patching_E1_val/run_manifest.json").read_text())
    assert manifest["config"]["patching"]["mask_seeds"] == 2
    assert manifest["config"]["patching"]["channel_fractions"] == [.25, .5]
    assert manifest["config"]["representation"]["layers"] == [0]
    assert manifest["config"]["runtime"]["batch_size"] == 2


def test_fingerprint_and_alignment(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    common = importlib.import_module("_common")
    model = torch.nn.Linear(2, 2)
    fingerprint = common.model_fingerprint(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert common.model_fingerprint(model) != fingerprint
    from types import SimpleNamespace
    run = SimpleNamespace(fingerprint=fingerprint)
    path = tmp_path / "cache.pt"
    torch.save(dict(model_fingerprint=fingerprint, image_ids=["a"]), path)
    with pytest.raises(ValueError, match="alignment"):
        common.read_cache(path, run, ["b"])
    run.fingerprint = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        common.read_cache(path, run)
