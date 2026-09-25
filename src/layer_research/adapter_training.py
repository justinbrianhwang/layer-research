"""Single-site learned repair with frozen, eval-mode ViT backbones."""
from dataclasses import asdict, dataclass, replace
from contextlib import contextmanager
from pathlib import Path
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .feature_extractor import BlockOutputRecorder, num_blocks
from .evaluation import accuracy_gain_pp, clean_accuracy_change_pp


class BottleneckAdapter(nn.Module):
    """Token-wise residual MLP; affine LN and both biases are learnable."""
    def __init__(self, d: int, r: int, act="gelu"):
        super().__init__()
        if d <= 0 or r <= 0:
            raise ValueError("d and r must be positive")
        if act not in ("gelu", "relu"):
            raise ValueError("act must be gelu or relu")
        self.d, self.r = d, r
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, r)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()
        self.up = nn.Linear(r, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(self.act(self.down(self.norm(h))))

    def extra_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_flops_per_token(self) -> int:
        return 4 * self.d * self.r + 5 * self.d


def attach_adapter(model, layer: int, adapter):
    """Attach at a block output, retaining autograd; caller removes the handle."""
    if not 0 <= layer < num_blocks(model):
        raise ValueError("Layer index out of range")
    return model.blocks[layer].register_forward_hook(lambda module, inputs, output: adapter(output))


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_only_adapter_trainable(model, adapter):
    """The hook-owned adapter is separate from model.parameters()."""
    allowed = {id(p) for p in adapter.parameters()}
    if any(p.requires_grad and id(p) not in allowed for p in model.parameters()):
        raise ValueError("Backbone/classifier parameters must be frozen")
    if not all(p.requires_grad for p in adapter.parameters()):
        raise ValueError("All adapter parameters must be trainable")
    return True


@dataclass
class AdaptTrainConfig:
    layer: int
    width: int
    lr: float
    weight_decay: float
    steps: int
    batch_size: int
    lambda_clean: float
    training_seed: int
    amp: bool = False
    log_every: int = 50
    lr_schedule: str = "constant"

    def __post_init__(self):
        if min(self.width, self.steps, self.batch_size, self.log_every) <= 0 or self.layer < 0:
            raise ValueError("Invalid layer, width, steps, batch_size or log_every")
        if not all(math.isfinite(v) and v >= 0 for v in (self.lr, self.weight_decay, self.lambda_clean)):
            raise ValueError("Optimizer settings must be finite and nonnegative")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be constant or cosine")


@dataclass
class TrainResult:
    adapter: BottleneckAdapter
    cost: dict
    loss_curve: list[dict]


@contextmanager
def _seeded(seed, loader):
    py_state, np_state = random.getstate(), np.random.get_state()
    generators = []
    for obj in (loader, getattr(loader, "sampler", None)):
        gen = getattr(obj, "generator", None)
        if gen is not None and all(gen is not g for g, _ in generators):
            generators.append((gen, gen.get_state()))
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng():
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            for gen, _ in generators:
                gen.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        for gen, state in generators:
            gen.set_state(state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _batch(batch, device):
    if isinstance(batch, dict):
        clean, corr, y, ids = (batch[k] for k in ("x_clean", "x_corr", "y", "image_id"))
    else:
        clean, corr, y, ids = batch
    y = torch.as_tensor(y, device=device, dtype=torch.long)
    if clean.shape != corr.shape or len(clean) != len(y) or len(ids) != len(y) or not len(y):
        raise ValueError("Expected nonempty aligned paired batch")
    return clean.to(device), corr.to(device), y, list(ids)


@torch.no_grad()
def _flops(model, x, layer, adapter):
    """Approximate standard timm ViT FLOPs per input (multiply-add = two)."""
    with BlockOutputRecorder(model, range(num_blocks(model))) as rec:
        model(x[:1])
    blocks = []
    for i, block in enumerate(model.blocks):
        _, n, d = rec.outputs[i].shape
        m = block.mlp.fc1.out_features
        blocks.append(8*n*d*d + 4*n*n*d + 4*n*d*m + 10*n*d)
    proj = model.patch_embed.proj
    patches = rec.outputs[0].shape[1] - model.num_prefix_tokens
    stem = 2 * patches * proj.out_channels * proj.in_channels * math.prod(proj.kernel_size)
    n, d = rec.outputs[layer].shape[1:]
    head = 2 * d * model.num_classes + 5*n*d
    extra = n * adapter.extra_flops_per_token()
    forward = stem + sum(blocks) + head + extra
    backward = sum(blocks[layer+1:]) + head + 2*extra
    return {"forward_flops_per_image": forward, "backward_flops_per_image": backward,
            "adapter_forward_flops_per_image": extra, "block_forward_flops": blocks}


def train_adapter(model, adapter, fit_loader, cfg, device):
    """Train existing adapter weights; seed initialization before constructing it."""
    device = torch.device(device)
    if cfg.width != adapter.r:
        raise ValueError("Config width differs from adapter width")
    if not 0 <= cfg.layer < num_blocks(model):
        raise ValueError("Layer index out of range")
    if getattr(fit_loader, "batch_size", cfg.batch_size) != cfg.batch_size:
        raise ValueError("Loader batch_size must match config")
    if getattr(fit_loader, "persistent_workers", False):
        raise ValueError("Use nonpersistent workers for reproducible loader restarts")
    model.to(device).eval().requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    adapter.to(device).eval().requires_grad_(True)
    check_only_adapter_trainable(model, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    curve, images, views, estimate = [], 0, 0, None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with _seeded(cfg.training_seed, fit_loader):
        iterator = iter(fit_loader)
        handle = None
        try:
            for step in range(cfg.steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(fit_loader)
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        raise ValueError("fit_loader must be nonempty and reiterable") from None
                clean, corr, y, _ = _batch(batch, device)
                if len(y) > cfg.batch_size:
                    raise ValueError("Batch exceeds config batch_size")
                if estimate is None:
                    estimate = _flops(model, corr, cfg.layer, adapter)
                    handle = attach_adapter(model, cfg.layer, adapter)
                lr = cfg.lr * (0.5*(1+math.cos(math.pi*step/cfg.steps)) if cfg.lr_schedule == "cosine" else 1)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=cfg.amp,
                                    dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    loss_corr = F.cross_entropy(model(corr), y)
                    loss_clean = F.cross_entropy(model(clean), y) if cfg.lambda_clean else loss_corr.new_zeros(())
                    loss = loss_corr + cfg.lambda_clean * loss_clean
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()):
                    raise FloatingPointError("Nonfinite adapter gradients; optimizer update aborted")
                scaler.step(optimizer)
                scaler.update()
                images += len(y)
                views += len(y) * (1 + int(cfg.lambda_clean != 0))
                if step == 0 or (step+1) % cfg.log_every == 0 or step+1 == cfg.steps:
                    curve.append({"step": step+1, "loss": loss.item(), "corrupted_loss": loss_corr.item(),
                                  "clean_loss": loss_clean.item(), "lr": lr})
        finally:
            if handle is not None:
                handle.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    compute = views * (estimate["forward_flops_per_image"] + estimate["backward_flops_per_image"])
    cost = {**estimate, "wall_time_seconds": time.perf_counter()-start, "updates": cfg.steps,
            "images_seen": images, "input_views_seen": views, "estimated_training_flops": compute,
            "training_compute": compute, "extra_params": adapter.extra_params(),
            "profiling_forward_flops": estimate["forward_flops_per_image"] - estimate["adapter_forward_flops_per_image"],
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "precision": "amp" if cfg.amp else "float32"}
    return TrainResult(adapter, cost, curve)


def _metadata(batch, loader, name, n):
    spec = getattr(getattr(loader, "dataset", None), "spec", None)
    value = batch.get(name) if isinstance(batch, dict) else None
    if value is None:
        value = getattr(spec, "name" if name == "corruption" else name, None)
    if isinstance(value, torch.Tensor):
        value = value.cpu().tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError("Metadata must align with images")
        return value
    return [value if value is not None else ("unknown" if name == "corruption" else None)] * n


def _prediction(logits, y):
    competitors = logits.clone()
    competitors.scatter_(1, y[:, None], -torch.inf)
    return (logits.argmax(1).tolist(),
            (logits.gather(1, y[:, None]).squeeze(1)-competitors.max(1).values).tolist(),
            F.cross_entropy(logits, y, reduction="none").tolist())


@torch.no_grad()
def evaluate_adapter(model, adapter, layer, loader, device):
    """Per-image corrupted and clean effects; removes only its own hook."""
    model.to(device).eval()
    adapter.to(device).eval()
    if not 0 <= layer < num_blocks(model):
        raise ValueError("Layer index out of range")
    rows = []
    for batch in loader:
        clean, corr, y, ids = _batch(batch, device)
        before = [model(corr), model(clean)]
        handle = attach_adapter(model, layer, adapter)
        try:
            after = [model(corr), model(clean)]
        finally:
            handle.remove()
        corruptions = _metadata(batch, loader, "corruption", len(y))
        severities = _metadata(batch, loader, "severity", len(y))
        clean_predictions = before[1].argmax(1).tolist()
        for side, baseline, post in zip(("corrupted", "clean"), before, after):
            bp, bm, bl = _prediction(baseline, y)
            pp, pm, pl = _prediction(post, y)
            for i, image_id in enumerate(ids):
                rows.append({"image_id": image_id, "label": y[i].item(), "layer_id": layer,
                    "input_type": side, "corruption": corruptions[i] if side == "corrupted" else "clean",
                    "severity": severities[i] if side == "corrupted" else 0,
                    "baseline_prediction": bp[i], "post_intervention_prediction": pp[i],
                    "clean_prediction": clean_predictions[i], "baseline_margin": bm[i],
                    "post_intervention_margin": pm[i], "baseline_loss": bl[i], "post_loss": pl[i]})
    return pd.DataFrame(rows)


def train_selected_site(model, layer, width, seed, fit_loader, val_loader, cfg_base,
                        device, out_dir, *, selection_cost=0):
    """Train exactly one caller-selected site; selection uses no validation here."""
    cfg = replace(cfg_base, layer=layer, width=width, training_seed=seed)
    if not 0 <= layer < num_blocks(model):
        raise ValueError("Layer index out of range")
    with _seeded(seed, fit_loader):
        adapter = BottleneckAdapter(model.blocks[layer].norm1.normalized_shape[-1], width)
    result = train_adapter(model, adapter, fit_loader, cfg, device)
    rows = evaluate_adapter(model, adapter, layer, val_loader, device)
    if rows.empty:
        raise ValueError("Validation loader must be nonempty")
    result.cost.update(selection_cost=selection_cost, reference_total_compute=None)
    folder = Path(out_dir) / f"layer_{layer}_width_{width}_seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, folder / "adapter.pt")
    rows.to_csv(folder / "validation.csv", index=False)
    (folder / "training.json").write_text(json.dumps({"config": asdict(cfg), "cost": result.cost,
        "loss_curve": result.loss_curve}, indent=2), encoding="utf-8")
    gain = accuracy_gain_pp(rows[rows.input_type == "corrupted"], []).iloc[0].U
    clean_gain = clean_accuracy_change_pp(rows[rows.input_type == "clean"], []).iloc[0].U
    return pd.DataFrame([{ "layer_id": layer, "width": width, "training_seed": seed,
        **result.cost, "U": gain, "clean_accuracy_change_pp": clean_gain,
        "clean_drop_pp": -clean_gain, "checkpoint_path": str(folder / "adapter.pt"),
        "validation_path": str(folder / "validation.csv")}])


def train_all_sites(model, layers, widths, seeds, fit_loader, val_loader, cfg_base,
                    device, out_dir, *, selection_cost=0):
    """Exhaustive reference; each row's training_compute belongs to that site only."""
    layers, widths, seeds = list(layers), list(widths), list(seeds)
    if any(not values or len(set(values)) != len(values) for values in (layers, widths, seeds)):
        raise ValueError("Provide nonempty unique layers, widths and seeds")
    runs = [train_selected_site(model, l, w, s, fit_loader, val_loader, cfg_base, device,
                               out_dir, selection_cost=selection_cost)
            for l in layers for w in widths for s in seeds]
    table = pd.concat(runs, ignore_index=True)
    table["reference_total_compute"] = table.training_compute.sum()
    table.to_csv(Path(out_dir) / "reference.csv", index=False)
    return table
