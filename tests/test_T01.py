import json
import numpy as np
import pytest
import timm
import torch
from PIL import Image
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.data.transforms import MaybeToTensor

from layer_research import *
from layer_research import data_protocol


@pytest.fixture
def model():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(12)
        net = VisionTransformer(img_size=32, patch_size=8, embed_dim=32,
                                depth=3, num_heads=4, num_classes=5, drop_rate=0.4)
    yield net
    torch.set_num_threads(old_threads)


def test_splits(tmp_path):
    ids = [f"image_{i}" for i in range(103)]
    labels = [i % 3 for i in range(103)]
    cfg = SplitConfig(.4, .2, .2, .2, 7)
    splits = make_splits(ids, labels, cfg)
    verify_disjoint(splits)
    assert set(sum(splits.values(), [])) == set(ids)
    assert splits == make_splits(ids, labels, cfg)
    assert splits == make_splits(ids[::-1], labels[::-1], cfg)
    assert splits != make_splits(ids, labels, SplitConfig(.4, .2, .2, .2, 8))
    for name, fraction in zip(splits, [.4, .2, .2, .2]):
        for label in range(3):
            count = sum(labels[ids.index(i)] == label for i in splits[name])
            assert abs(count - labels.count(label) * fraction) <= 1
    path = tmp_path / "splits.json"
    save_splits(splits, path)
    assert load_splits(path) == splits
    assert load_splits(path).cfg == cfg
    assert json.loads(path.read_text())["seed"] == 7
    with pytest.raises(ValueError):
        verify_disjoint({"fit": ["a"], "test": ["a"]})
    with pytest.raises(ValueError):
        make_splits(["a", "a"], [0, 0], cfg)
    with pytest.raises(ValueError):
        SplitConfig(.5, .5, .5, .5, 0)
    unbalanced = make_splits(ids, labels, SplitConfig(1, 0, 0, 0, 7, False))
    assert unbalanced["fit"] == sorted(ids)


@pytest.mark.parametrize("name", OBSERVED_CORRUPTIONS + UNSEEN_CORRUPTIONS)
def test_corruption(name):
    img = Image.fromarray(np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    state = np.random.get_state()
    spec = CorruptionSpec(name, 1, 19)
    first = np.asarray(corrupt_image(img, spec, "img1"))
    second = np.asarray(corrupt_image(img, spec, "img1"))
    assert np.array_equal(first, second)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    if name == "gaussian_noise":
        assert not np.array_equal(first, np.asarray(corrupt_image(img, spec, "img2")))


def test_hooks_and_shapes(model):
    assert shape_report(model, (3, 32, 32)) == {i: (1, 17, 32) for i in range(3)}
    assert not model.training
    with BlockOutputRecorder(model, [0, 2], dtype=torch.float64) as recorder:
        model(torch.zeros(2, 3, 32, 32))
        assert recorder.outputs[0].dtype == torch.float64
        assert not recorder.outputs[0].requires_grad
    assert all(not b._forward_hooks for b in model.blocks)
    with pytest.raises(RuntimeError):
        with BlockOutputRecorder(model, [1]):
            raise RuntimeError("intentional")
    assert all(not b._forward_hooks for b in model.blocks)
    with pytest.raises(ValueError):
        BlockOutputRecorder(model, [0, 4])
    model.stages = model.blocks
    assert num_blocks(model, "stages") == 3
    assert shape_report(model, (3, 32, 32), "stages")[0] == (1, 17, 32)


def test_summaries():
    tokens = torch.arange(24.).reshape(2, 3, 4)
    assert torch.equal(summarize(tokens, "cls"), tokens[:, 0])
    assert torch.equal(summarize(tokens, "patch_mean"), tokens[:, 1:].mean(1))
    assert summarize(tokens, "cls+patch_mean").shape == (2, 8)
    with pytest.raises(ValueError):
        summarize(tokens, "invalid")


def test_dataset_and_extraction(model, tmp_path):
    records = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((40, 40, 3), 70 + i, dtype=np.uint8)).save(path)
        records.append((path, i, str(i)))
    dataset = PairedImageDataset(records, CorruptionSpec("gaussian_noise", 1, 9), model,
                                 {"input_size": (3, 32, 32)})
    loader = DataLoader(dataset, batch_size=2)
    clean, corrupted, label, image_id = dataset[0]
    with Image.open(records[0][0]) as original:
        prepared = dataset.pre_transform(original.convert("RGB"))
        expected = dataset.post_transform(corrupt_image(prepared, dataset.spec, image_id))
        assert torch.equal(clean, dataset.transform(original.convert("RGB")))
    assert torch.equal(corrupted, expected)
    assert dataset.input_resolution == (32, 32)
    assert clean.shape == corrupted.shape == (3, 32, 32)
    features = extract_paired_features(model, loader, [0, 2], ["cls", "patch_mean"], "cpu")
    assert features["image_ids"] == ["0", "1", "2"]
    assert features["labels"].tolist() == [0, 1, 2]
    assert features["clean"][2]["cls"].shape == (3, 32)
    with torch.no_grad():
        assert torch.allclose(features["clean_logits"], model(torch.stack([dataset[i][0] for i in range(3)])), atol=1e-6)
    stream = iter_paired_block_outputs(model, loader, [0], "cpu")
    batch = next(stream)
    assert torch.is_grad_enabled()
    assert batch["clean"][0].shape == (2, 17, 32)
    assert all(not b._forward_hooks for b in model.blocks)
    stream.close()
    scores = layerwise_scores(features, ["relative_distance", "linear_cka", "activation_norm", "amplification_ratio"])
    assert len(scores) == 16 and set(scores.label_access) == {"none"}
    assert set(scores.n_samples) == {3}


def test_corruption_receives_input_resolution(model, tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    Image.new("RGB", (40, 40), (70, 80, 90)).save(path)
    spec = CorruptionSpec("gaussian_noise", 1, 9)
    received = []

    def spy(image, actual_spec, image_id):
        received.append((image.size, image.mode, np.asarray(image).dtype, actual_spec, image_id))
        return image.copy()

    monkeypatch.setattr(data_protocol, "corrupt_image", spy)
    dataset = PairedImageDataset([(path, 2, "source")], spec, model,
                                 {"input_size": (3, 32, 32)})
    clean, corrupted, label, image_id = dataset[0]
    assert received == [((32, 32), "RGB", np.dtype("uint8"), spec, "source")]
    assert torch.equal(clean, corrupted)
    assert (label, image_id) == (2, "source")


def test_deit_preprocessing(model):
    # The model fixture limits CPU threads and restores them after the test.
    deit = timm.create_model("deit_small_patch16_224", pretrained=False)
    config = timm.data.resolve_data_config({}, model=deit)
    dataset = PairedImageDataset([], CorruptionSpec("contrast", 1, 9), deit, config)
    prepared = dataset.pre_transform(Image.new("RGB", (320, 280)))
    assert isinstance(prepared, Image.Image)
    assert prepared.size == dataset.input_resolution == (224, 224)
    assert dataset.post_transform(prepared).shape == (3, 224, 224)


@pytest.mark.parametrize("conversion", [transforms.ToTensor, MaybeToTensor, None])
def test_transform_conversion_boundary(model, monkeypatch, conversion):
    steps = [transforms.Resize((32, 40))]
    if conversion is not None:
        steps.extend([conversion(), transforms.Normalize((.5,) * 3, (.5,) * 3)])
    monkeypatch.setattr(timm.data, "create_transform", lambda **kwargs: transforms.Compose(steps))
    spec = CorruptionSpec("contrast", 1, 9)
    if conversion is None:
        with pytest.raises(ValueError, match="ToTensor or MaybeToTensor"):
            PairedImageDataset([], spec, model)
    else:
        dataset = PairedImageDataset([], spec, model, {"input_size": (3, 32, 40)})
        prepared = dataset.pre_transform(Image.new("RGB", (50, 50)))
        assert prepared.size == (40, 32)
        assert dataset.input_resolution == (32, 40)
        assert dataset.post_transform(prepared).shape == (3, 32, 40)


def test_cka(caplog):
    rng = np.random.default_rng(1)
    H, Ht = rng.normal(size=(23, 7)), rng.normal(size=(23, 7))
    assert linear_cka(H, H) == pytest.approx(1)
    Q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    assert linear_cka(H, H @ Q) == pytest.approx(1)
    C = np.eye(len(H)) - np.ones((len(H), len(H))) / len(H)
    K, L = C @ (H @ H.T) @ C, C @ (Ht @ Ht.T) @ C
    hsic = lambda A, B: np.trace(A @ B) / (len(H) - 1) ** 2
    brute = hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L))
    assert linear_cka(torch.tensor(H), Ht) == pytest.approx(brute)
    assert linear_cka(H + 100, Ht - 22) == pytest.approx(brute)
    assert np.isnan(linear_cka(np.zeros_like(H), Ht))
    assert "denominator" in caplog.text
    assert one_minus_cka(H, H) == pytest.approx(0, abs=1e-14)


def test_distances_and_neighbors():
    H = np.random.default_rng(2).normal(size=(15, 4))
    assert relative_distance(H, H) == raw_distance(H, H) == 0
    assert relative_distance(H, 2 * H) == pytest.approx(1)
    assert activation_norm(H) == pytest.approx(np.linalg.norm(H, axis=1).mean())
    assert cosine_distance(H, 2 * H) == pytest.approx(0, abs=1e-14)
    assert knn_preservation(H, H, 3) == 1
    Ht = H[::-1].copy()
    def neighbors(X, i):
        return set(sorted((j for j in range(len(X)) if i != j), key=lambda j: np.linalg.norm(X[i] - X[j]))[:3])
    expected = np.mean([len(neighbors(H, i) & neighbors(Ht, i)) / len(neighbors(H, i) | neighbors(Ht, i)) for i in range(len(H))])
    assert knn_preservation(H, Ht, 3) == pytest.approx(expected)
    np.testing.assert_allclose(amplification_ratio([0, 2, 4], .1), [1, 20, 2])
    with pytest.raises(ValueError):
        knn_preservation(H, H, 15)
    with pytest.raises(ValueError):
        relative_distance(H, H[:2])

