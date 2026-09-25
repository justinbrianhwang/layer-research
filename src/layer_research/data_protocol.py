"""Original-image splits and reproducible generated-corruption pairs."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

OBSERVED_CORRUPTIONS = ("gaussian_noise", "defocus_blur")
UNSEEN_CORRUPTIONS = ("contrast", "jpeg_compression")
DEFAULT_SEVERITIES = (1, 3, 5)
_NAMES = ("fit", "score", "val", "test")
_RNG_LOCK = threading.Lock()


@dataclass(frozen=True)
class SplitConfig:
    fit_frac: float
    score_frac: float
    val_frac: float
    test_frac: float
    seed: int
    class_balanced: bool = True

    def __post_init__(self):
        fractions = [self.fit_frac, self.score_frac, self.val_frac, self.test_frac]
        if not all(np.isfinite(f) and 0 <= f <= 1 for f in fractions) or not np.isclose(sum(fractions), 1, rtol=0, atol=1e-10):
            raise ValueError("Split fractions must be finite, nonnegative, and sum to one")


class SplitResult(dict):
    """A split dictionary carrying its immutable generation configuration."""
    def __init__(self, splits, cfg):
        super().__init__(splits)
        self.cfg = cfg


def verify_disjoint(splits):
    """Raise ValueError for duplicate IDs within or across splits."""
    seen = set()
    for ids in splits.values():
        for image_id in ids:
            if image_id in seen:
                raise ValueError(f"Duplicate original image ID: {image_id}")
            seen.add(image_id)


def make_splits(image_ids: list[str], labels: list[int], cfg: SplitConfig) -> dict[str, list[str]]:
    if len(image_ids) != len(labels) or len(set(image_ids)) != len(image_ids):
        raise ValueError("IDs must be unique and have one label each")
    rng = np.random.default_rng(cfg.seed)
    groups = {}
    for image_id, label in zip(image_ids, labels):
        groups.setdefault(label if cfg.class_balanced else 0, []).append(image_id)
    splits = {name: [] for name in _NAMES}
    fractions = np.array([cfg.fit_frac, cfg.score_frac, cfg.val_frac, cfg.test_frac])
    for label in sorted(groups):
        ids = rng.permutation(sorted(groups[label])).tolist()
        quotas = fractions * len(ids)
        counts = np.floor(quotas).astype(int)
        # Largest remainders keep each class allocation within one of its quota.
        order = np.argsort(-(quotas - counts), kind="stable")
        counts[order[:len(ids) - counts.sum()]] += 1
        start = 0
        for name, count in zip(_NAMES, counts):
            splits[name].extend(ids[start:start + count])
            start += count
    return SplitResult({name: sorted(ids) for name, ids in splits.items()}, cfg)


def save_splits(splits, path, cfg=None):
    """Save sorted IDs plus config and seed; plain dictionaries require cfg."""
    cfg = cfg if cfg is not None else getattr(splits, "cfg", None)
    if cfg is None:
        raise ValueError("A SplitConfig is required for provenance")
    verify_disjoint(splits)
    if set(splits) != set(_NAMES):
        raise ValueError("Expected fit, score, val, test splits")
    Path(path).write_text(json.dumps({"cfg": asdict(cfg), "seed": cfg.seed,
        "splits": {k: sorted(splits[k]) for k in _NAMES}}, indent=2), encoding="utf-8")


def load_splits(path):
    """Load a split dictionary with .cfg provenance and check disjointness."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = SplitConfig(**payload["cfg"])
    if payload["seed"] != cfg.seed or set(payload["splits"]) != set(_NAMES):
        raise ValueError("Invalid split metadata")
    verify_disjoint(payload["splits"])
    return SplitResult(payload["splits"], cfg)


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    severity: int
    seed: int

    def __post_init__(self):
        if self.name not in OBSERVED_CORRUPTIONS + UNSEEN_CORRUPTIONS:
            raise ValueError(f"Unsupported corruption: {self.name}")
        if isinstance(self.severity, bool) or not isinstance(self.severity, int) or not 1 <= self.severity <= 5:
            raise ValueError("Severity must be an integer from 1 to 5")


def corrupt_image(img: Image.Image, spec: CorruptionSpec, image_id: str) -> Image.Image:
    """Corrupt RGB pixels before preprocessing, preserving NumPy RNG state."""
    from imagecorruptions import corrupt
    digest = hashlib.sha256(f"{spec.seed}\0{image_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    with _RNG_LOCK:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            pixels = corrupt(np.asarray(img.convert("RGB")).copy(),
                             corruption_name=spec.name, severity=spec.severity)
        finally:
            np.random.set_state(state)
    return Image.fromarray(pixels)


class PairedImageDataset(Dataset):
    """Records are (path, integer label, original image ID); model defines eval transform."""
    def __init__(self, records, spec: CorruptionSpec, model, data_config=None):
        from timm.data import create_transform, resolve_data_config
        self.records = list(records)
        self.spec = spec
        ids = [record[2] for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Records must have unique original image IDs")
        self.data_config = resolve_data_config(data_config or {}, model=model)
        self.transform = create_transform(**self.data_config, is_training=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, image_id = self.records[index]
        with Image.open(path) as source:
            clean = source.convert("RGB")
        corrupted = corrupt_image(clean, self.spec, image_id)
        return self.transform(clean), self.transform(corrupted), label, image_id
