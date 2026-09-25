from dataclasses import replace
import json
import io
from pathlib import Path

import pandas as pd
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader

from layer_research.adapter_training import (
    AdaptTrainConfig, BottleneckAdapter, attach_adapter, check_only_adapter_trainable,
    count_trainable, evaluate_adapter, train_adapter, train_all_sites, train_selected_site,
)


@pytest.fixture
def model():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(12)
        model = VisionTransformer(img_size=16, patch_size=8, embed_dim=16,
                                  depth=3, num_heads=2, num_classes=3,
                                  drop_rate=.4, drop_path_rate=.3).eval()
    yield model
    torch.set_num_threads(old)


@pytest.fixture
def loader():
    gen = torch.Generator().manual_seed(33)
    clean = torch.randn(6, 3, 16, 16, generator=gen)
    corr = clean + .3 * torch.randn(clean.shape, generator=gen)
    return DataLoader([(clean[i], corr[i], 0, f"id{i}") for i in range(6)],
                      batch_size=3, shuffle=True, generator=torch.Generator().manual_seed(8))


def config(layer=0, steps=1, **kwargs):
    return AdaptTrainConfig(layer, 4, .03, 0., steps, 3, .5, 17, log_every=1, **kwargs)


def test_identity_and_parameter_budget(model, loader):
    adapter = BottleneckAdapter(16, 4)
    assert adapter.extra_params() == sum(p.numel() for p in adapter.parameters()) == 180
    assert adapter.extra_flops_per_token() == 4*16*4 + 5*16
    x = next(iter(loader))[0]
    before = model(x)
    h = attach_adapter(model, 1, adapter)
    try:
        assert torch.equal(model(x), before)
    finally:
        h.remove()


@pytest.mark.parametrize("layer", [0, 2])
def test_frozen_backbone_and_gradient_path(model, loader, layer):
    backbone = {k: v.clone() for k, v in model.state_dict().items()}
    adapter = BottleneckAdapter(16, 4)
    initial = {k: v.clone() for k, v in adapter.state_dict().items()}
    modes = []
    handle = model.register_forward_pre_hook(lambda m, args: modes.append(m.training))
    try:
        result = train_adapter(model, adapter, loader, config(layer), "cpu")
    finally:
        handle.remove()
    assert check_only_adapter_trainable(model, adapter)
    assert count_trainable(model) == 0
    assert count_trainable(adapter) == adapter.extra_params()
    assert sum(p.grad.norm().item() for p in adapter.parameters() if p.grad is not None) > 0
    assert any(not torch.equal(initial[k], v) for k, v in adapter.state_dict().items())
    assert all(torch.equal(backbone[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    assert modes and not any(modes)
    assert result.cost["updates"] == 1 and result.cost["images_seen"] == 3
    assert result.cost["input_views_seen"] == 6
    assert all(not b._forward_hooks for b in model.blocks)


def test_evaluation_clean_rows_and_removal(model, loader):
    adapter = BottleneckAdapter(16, 4)
    train_adapter(model, adapter, loader, config(steps=3), "cpu")
    x = next(iter(loader))[0]
    before = model(x).detach()
    rows = evaluate_adapter(model, adapter, 0, loader, "cpu")
    assert len(rows) == 12
    assert set(rows.input_type) == {"clean", "corrupted"}
    assert (rows[rows.input_type == "clean"].baseline_prediction ==
            rows[rows.input_type == "clean"].clean_prediction).all()
    assert set(rows[rows.input_type == "clean"].severity) == {0}
    assert all(not b._forward_hooks for b in model.blocks)
    assert torch.equal(before, model(x))
    for batch in loader:
        clean, corr, y, ids = batch
        logits = model(corr)
        other = logits.clone().scatter(1, y[:, None], -torch.inf)
        margins = logits.gather(1, y[:, None]).squeeze(1)-other.max(1).values
        for i, image_id in enumerate(ids):
            row = rows[(rows.image_id == image_id) & (rows.input_type == "corrupted")].iloc[0]
            assert row.baseline_margin == pytest.approx(margins[i].item())


def test_loss_decreases_and_reproducibility(model, loader):
    torch.manual_seed(71)
    first = BottleneckAdapter(16, 4)
    second = BottleneckAdapter(16, 4)
    second.load_state_dict(first.state_dict())
    cfg = config(steps=20, lr_schedule="cosine")
    a = train_adapter(model, first, loader, cfg, "cpu")
    b = train_adapter(model, second, loader, cfg, "cpu")
    assert a.loss_curve[-1]["loss"] < a.loss_curve[0]["loss"]
    assert a.loss_curve == b.loss_curve
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    assert a.cost["updates"] == 20 and a.cost["images_seen"] == 60
    assert a.loss_curve[-1]["lr"] < a.loss_curve[0]["lr"]


def test_exception_cleanup_and_existing_hooks(model, loader):
    adapter = BottleneckAdapter(16, 4)
    observer = model.blocks[0].register_forward_hook(lambda *args: None)
    def fail(*args):
        raise RuntimeError("intentional")
    broken = adapter.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="intentional"):
            evaluate_adapter(model, adapter, 0, loader, "cpu")
        assert len(model.blocks[0]._forward_hooks) == 1
        with pytest.raises(RuntimeError, match="intentional"):
            train_adapter(model, adapter, loader, config(), "cpu")
        assert len(model.blocks[0]._forward_hooks) == 1
    finally:
        observer.remove()
        broken.remove()


def test_sweeps_and_cost_separation(model, loader, monkeypatch):
    # Exercise serialization in memory: the shared Windows basetemp cannot be
    # removed by pytest in this sandbox, and another agent may be using it.
    saved = {}
    original_save = torch.save
    original_csv = pd.DataFrame.to_csv
    def save(state, path):
        buffer = io.BytesIO()
        original_save(state, buffer)
        saved[str(path)] = buffer.getvalue()
    def csv(frame, path, **kwargs):
        saved[str(path)] = original_csv(frame, None, **kwargs)
    monkeypatch.setattr(torch, "save", save)
    monkeypatch.setattr(pd.DataFrame, "to_csv", csv)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(Path, "write_text", lambda path, data, **kwargs: saved.update({str(path): data}))
    tmp_path = Path(".pytest_tmp") / "T04_serialization"
    table = train_all_sites(model, (i for i in [0, 2]), [4], [11, 12], loader, loader,
                            config(), "cpu", tmp_path, selection_cost=123)
    assert len(table) == 4
    assert set(table.selection_cost) == {123}
    assert (table.reference_total_compute == table.training_compute.sum()).all()
    assert table.iloc[0].training_compute > table.iloc[2].training_compute
    assert set(table.updates) == {1}
    for row in table.itertuples():
        state = torch.load(io.BytesIO(saved[row.checkpoint_path]), weights_only=True)
        assert set(state) == set(BottleneckAdapter(16, 4).state_dict())
        assert len(pd.read_csv(io.StringIO(saved[row.validation_path]))) == 12
    selected = train_selected_site(model, 2, 4, 11, loader, loader, config(), "cpu",
                                   tmp_path / "selected", selection_cost=21)
    assert len(selected) == 1 and selected.iloc[0].selection_cost == 21
    assert selected.iloc[0].reference_total_compute is None
    selected_files = {k: v for k, v in saved.items() if str(tmp_path / "selected") in k}
    assert sum(k.endswith("adapter.pt") for k in selected_files) == 1
    record = json.loads(next(v for k, v in selected_files.items() if k.endswith("training.json")))
    assert record["config"]["layer"] == 2
    assert record["cost"]["training_compute"] == selected.iloc[0].training_compute


def test_mapping_metadata_and_no_clean_training(model, loader):
    batches = [{"x_clean": c, "x_corr": x, "y": y, "image_id": ids,
                "corruption": ["contrast"] * len(y), "severity": torch.ones(len(y), dtype=torch.int)}
               for c, x, y, ids in loader]
    adapter = BottleneckAdapter(16, 4)
    result = train_adapter(model, adapter, batches, replace(config(), lambda_clean=0), "cpu")
    assert result.cost["input_views_seen"] == 3
    rows = evaluate_adapter(model, adapter, 0, batches, "cpu")
    assert set(rows[rows.input_type == "corrupted"].corruption) == {"contrast"}
    assert set(rows[rows.input_type == "corrupted"].severity) == {1}


def test_invalid_inputs(model, loader):
    adapter = BottleneckAdapter(16, 4)
    with pytest.raises(ValueError, match="frozen"):
        check_only_adapter_trainable(model, adapter)
    with pytest.raises(ValueError, match="Layer"):
        attach_adapter(model, -1, adapter)
    with pytest.raises(ValueError, match="nonempty"):
        train_adapter(model, adapter, [], config(), "cpu")
    with pytest.raises(ValueError, match="width"):
        train_adapter(model, adapter, loader, replace(config(), width=8), "cpu")
    with pytest.raises(ValueError, match="batch_size"):
        train_adapter(model, adapter, loader, replace(config(), batch_size=9), "cpu")
