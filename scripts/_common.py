"""Shared CLI, provenance and storage contracts for experiment entry points."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import timm
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from layer_research.data_protocol import CorruptionSpec, PairedImageDataset, load_splits


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(arg):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if arg == "auto" else torch.device(arg)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_fingerprint(model):
    h = hashlib.sha256(timm.__version__.encode())
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class OutputPaths:
    def __init__(self, cfg):
        base = Path(cfg.get("output_root", "."))
        self.cache = base / "cache"
        self.results = base / "results"
        self.raw = self.results / "raw"
        self.tables = self.results / "tables"
        for path in (self.cache, self.raw, self.tables):
            path.mkdir(parents=True, exist_ok=True)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/deit_small.yaml")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--limit", type=int)
    return p


def create_model(cfg, device):
    set_all_seeds(cfg.get("model_seed", 0))
    torch.set_num_threads(cfg.get("runtime", {}).get("threads", 4))
    m = cfg["model"]
    return timm.create_model(m["name"], pretrained=m.get("pretrained", True), **m.get("kwargs", {})).to(device).eval()


class Run:
    def __init__(self, args, name, model=True):
        self.args, self.name = args, name
        self.cfg = load_config(args.config)
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        self.paths = OutputPaths(self.cfg)
        self.device = get_device(args.device)
        runtime = self.cfg.setdefault("runtime", {})
        batch_size = getattr(args, "batch_size", None)
        if batch_size is None:
            batch_size = runtime.get("batch_size_cuda", 128) if self.device.type == "cuda" else runtime.get("batch_size", 32)
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        runtime["batch_size"] = batch_size
        runtime.setdefault("num_workers", 4)
        runtime.setdefault("pin_memory", self.device.type == "cuda")
        self.start = time.perf_counter()
        self.started = datetime.now(timezone.utc).isoformat()
        self.model = create_model(self.cfg, self.device) if model else None
        self.fingerprint = model_fingerprint(self.model) if model else None

    def finish(self):
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        data = dict(config=self.cfg, arguments=vars(self.args), git_commit=commit,
                    model_fingerprint=self.fingerprint, started_at=self.started,
                    finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-self.start)
        suffix = "_".join(str(getattr(self.args, key)) for key in ("experiment", "split") if hasattr(self.args, key))
        path = self.paths.results / "manifests" / (self.name + ("_" + suffix if suffix else ""))
        path.mkdir(parents=True, exist_ok=True)
        (path / "run_manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"{self.name}: {data['elapsed_seconds']:.2f}s")


def records(cfg):
    root = Path(cfg["data"]["root"])
    return [(str(p), int(p.parent.name), f"{int(p.parent.name)}_{p.stem}")
            for p in sorted(root.glob("*/*")) if p.suffix.lower() in (".jpeg", ".jpg", ".png")]


def conditions(cfg):
    c = cfg["corruptions"]
    return [(name, severity) for name in c["observed"] + c["unseen"] for severity in c["severities"]]


def condition_dir(paths, split, name, severity):
    return paths.cache / split / ("clean" if name == "clean" else f"{name}_{severity}")


class CleanImageDataset(PairedImageDataset):
    """Reuse paired evaluation transforms without generating a corruption."""

    def __getitem__(self, index):
        path, label, image_id = self.records[index]
        with Image.open(path) as source:
            clean = self.post_transform(self.pre_transform(source.convert("RGB")))
        return clean, clean, label, image_id


def loader(run, split, name, severity, paired: bool = True):
    ids = set(load_splits(run.cfg["data"]["split_file"])[split])
    rec = [r for r in records(run.cfg) if r[2] in ids]
    # Seeded order avoids a class-sorted prefix in limited smoke runs.
    random.Random(run.cfg["data"]["split"]["seed"]).shuffle(rec)
    rec = rec[:run.args.limit]
    if not rec:
        raise ValueError(f"Empty split: {split}")
    spec = CorruptionSpec(name if name != "clean" else "contrast", severity or 1,
                          run.cfg["corruptions"]["corruption_seed"])
    dataset = PairedImageDataset if paired else CleanImageDataset
    ds = dataset(rec, spec, run.model, {"input_size": run.cfg["model"]["input_size"]})
    runtime = run.cfg.get("runtime", {})
    return DataLoader(ds, batch_size=runtime.get("batch_size", 32), shuffle=False,
                      num_workers=runtime.get("num_workers", 4),
                      pin_memory=runtime.get("pin_memory", run.device.type == "cuda"),
                      persistent_workers=False)


def read_cache(path, run, ids=None):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if run.fingerprint is None:
        run.fingerprint = data["model_fingerprint"]
    if data["model_fingerprint"] != run.fingerprint:
        raise ValueError(f"Model fingerprint mismatch: {path}")
    if ids is not None and data["image_ids"] != list(ids):
        raise ValueError(f"Image alignment mismatch: {path}")
    return data


def table(df, path, run):
    df = df.copy()
    df["model_fingerprint"] = run.fingerprint
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path.with_suffix(".parquet"), index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)


def read_table(path, run):
    df = pd.read_parquet(path)
    fps = df.model_fingerprint.unique()
    if len(fps) != 1 or (run.fingerprint is not None and fps[0] != run.fingerprint):
        raise ValueError("Incompatible table fingerprints")
    run.fingerprint = fps[0]
    return df


def read_table_columns(path, run, columns):
    """Project analysis columns, checking provenance without a repeated string column.

    Row-group min/max statistics certify the fingerprint for all rows. Older
    files without statistics are checked in bounded Arrow batches instead.
    """
    parquet = pq.ParquetFile(path)
    fingerprint_index = parquet.schema.names.index("model_fingerprint")
    fingerprints = set()
    for i in range(parquet.num_row_groups):
        group = parquet.metadata.row_group(i)
        if not group.num_rows:
            continue
        stats = group.column(fingerprint_index).statistics
        if stats is not None and stats.has_min_max and stats.null_count == 0:
            fingerprints.update((stats.min, stats.max))
        else:
            for batch in parquet.iter_batches(row_groups=[i], columns=["model_fingerprint"]):
                fingerprints.update(batch.column(0).unique().to_pylist())
    if (len(fingerprints) != 1 or None in fingerprints or
            (run.fingerprint is not None and run.fingerprint not in fingerprints)):
        raise ValueError("Incompatible table fingerprints")
    run.fingerprint = next(iter(fingerprints))
    columns = list(columns)
    forbidden = {"model_fingerprint", "donor_image_id", "baseline_margin",
                 "post_intervention_margin", "baseline_loss", "post_intervention_loss"}
    if forbidden.intersection(columns):
        raise ValueError("Raw provenance, donor, margin and loss columns are not analysis columns")
    arrow = pq.read_table(path, columns=columns)
    categories = [c for c in ("corruption", "intervention_type", "mask_policy", "image_id", "experiment") if c in columns]
    return arrow.to_pandas(categories=categories)


def bootstrap_summary(estimate, samples, confidence=0.95):
    """Percentile summary for scalar or column-wise NumPy bootstrap samples."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail], axis=0)
    return dict(mean=estimate, ci_low=low, ci_high=high, samples=samples)


def fast_image_bootstrap(D, n_boot=1000, seed=0, confidence=0.95):
    """Bootstrap all layer gains together from balanced per-image differences.

    D has shape (original images, layers). Every replicate uses the same image
    indices for every layer; returned samples can be reused for frozen choices
    and regret against any admissible subset (including a zero-valued none).
    Memory is bounded by one image resample plus the small replicate/layer array.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or 0 in D.shape or not np.isfinite(D).all():
        raise ValueError("D requires nonempty finite image/layer data")
    if not isinstance(n_boot, (int, np.integer)) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, D.shape[1]))
    for i in range(n_boot):
        idx = rng.integers(len(D), size=len(D))
        samples[i] = 100 * D[idx].mean(axis=0)
    return bootstrap_summary(100 * D.mean(axis=0), samples, confidence)
