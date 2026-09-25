"""Budget-limited interventions at full residual-block output boundaries."""
from dataclasses import dataclass
from enum import Enum
from itertools import product
import math
from typing import Literal

import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F

from .feature_extractor import BlockOutputRecorder


@dataclass(frozen=True)
class MaskSpec:
    policy: Literal["random_fixed", "score_topk"]
    fraction: float
    seed: int
    exclude_cls: bool = False


def make_channel_mask(d: int, spec: MaskSpec, channel_scores: Tensor | None = None):
    """Return a CPU bool mask and floor(q*d); score ties use channel order."""
    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer")
    if not math.isfinite(spec.fraction) or not 0 <= spec.fraction <= 1:
        raise ValueError("fraction must be finite and in [0, 1]")
    k = math.floor(spec.fraction * d)
    if spec.policy == "random_fixed":
        order = torch.randperm(d, generator=torch.Generator().manual_seed(spec.seed))
    elif spec.policy == "score_topk":
        if channel_scores is None or channel_scores.shape != (d,):
            raise ValueError("score_topk requires D_score channel_scores of shape [d]")
        scores = channel_scores.detach().cpu()
        if not torch.isfinite(scores).all():
            raise ValueError("channel_scores must be finite")
        order = torch.argsort(scores, descending=True, stable=True)
    else:
        raise ValueError(f"Unknown mask policy: {spec.policy}")
    mask = torch.zeros(d, dtype=torch.bool)
    mask[order[:k]] = True
    return mask, k


def nested_masks(d, fractions, seed):
    return {q: make_channel_mask(d, MaskSpec("random_fixed", q, seed))[0]
            for q in fractions}


class InterventionType(str, Enum):
    NONE = "none"
    PARTIAL_CHANNEL = "partial_channel"
    FULL_STATE = "full_state"
    CLEAN_TO_CLEAN = "clean_to_clean"
    RANDOM_DIRECTION = "random_direction"
    SHUFFLED_DONOR = "shuffled_donor"


class Patcher:
    """Install one hook. Set an aligned donor before each batch forward.

    Supply either a MaskSpec (resolved at the block) or a bool channel mask.
    Stochastic controls require an explicitly seeded generator. Shuffling also
    requires batch labels so every donor pair has a recorded same_class flag.
    """

    def __init__(self, model, layer, spec=None, alpha=1.0, mask=None,
                 norm_cap=None, r_l=None,
                 intervention=InterventionType.PARTIAL_CHANNEL, generator=None,
                 *, exclude_cls=False, channel_scores=None, labels=None):
        if not isinstance(layer, int) or not 0 <= layer < len(model.blocks):
            raise ValueError("layer must index model.blocks")
        if (spec is None) == (mask is None):
            raise ValueError("Provide exactly one of spec or mask")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if r_l is not None and (not math.isfinite(r_l) or r_l < 0):
            raise ValueError("D_score r_l must be finite and nonnegative")
        if norm_cap is not None and (not math.isfinite(norm_cap) or norm_cap < 0
                or r_l is None or not math.isfinite(r_l) or r_l < 0):
            raise ValueError("norm_cap requires finite nonnegative rho and D_score r_l")
        self.intervention = InterventionType(intervention)
        if self.intervention in (InterventionType.RANDOM_DIRECTION,
                                  InterventionType.SHUFFLED_DONOR) and generator is None:
            raise ValueError("Stochastic controls require an explicit generator")
        self.model, self.layer, self.spec = model, layer, spec
        self.alpha, self.mask = alpha, mask
        self.norm_cap, self.r_l = norm_cap, r_l
        self.generator, self.labels = generator, labels
        self.exclude_cls = spec.exclude_cls if spec is not None else exclude_cls
        self.channel_scores = channel_scores
        self.last_stats = {}
        self._donor = None
        self._handle = None

    def set_donor(self, h_clean):
        if not isinstance(h_clean, Tensor) or h_clean.ndim != 3:
            raise ValueError("Donor must be a [B, N, d] tensor")
        self._donor = h_clean.detach().clone()
        self.last_stats = {}

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("Patcher context is already active")
        self.model.eval()
        self._handle = self.model.blocks[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inputs, output):
        self.last_stats = {}
        if not isinstance(output, Tensor) or output.ndim != 3:
            raise ValueError("Block output must have shape [B, N, d]")
        if self._donor is None or self._donor.shape != output.shape:
            raise ValueError("Set an aligned donor with the same [B, N, d] shape")
        donor = self._donor.to(device=output.device, dtype=output.dtype)
        b, n, d = output.shape
        mask = (make_channel_mask(d, self.spec, self.channel_scores)[0]
                if self.spec is not None else self.mask)
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != (d,):
            raise ValueError("mask must be a bool tensor of shape [d]")
        mask = mask.to(output.device)
        kind = self.intervention
        full = kind == InterventionType.FULL_STATE
        if full:
            mask = torch.ones_like(mask)
        support = mask[None, None, :].expand(b, n, d).clone()
        if self.exclude_cls and not full:
            support[:, 0] = False
        extra = {}
        if kind == InterventionType.SHUFFLED_DONOR:
            if b < 2:
                raise ValueError("Shuffled donors require at least two images")
            labels = torch.as_tensor(self.labels) if self.labels is not None else None
            if labels is None or labels.shape != (b,):
                raise ValueError("Shuffled donors require labels of shape [B]")
            # A random cycle is a permutation with no self donors, even for B=2.
            order = torch.randperm(b, generator=self.generator, device=self.generator.device)
            permutation = torch.empty_like(order)
            permutation[order] = order.roll(1)
            donor = donor[permutation.to(output.device)]
            extra = {"donor_index": permutation.cpu(),
                     "same_class": (labels == labels[permutation.to(labels.device)]).cpu()}
        alpha = 1.0 if full else self.alpha
        rho = None if full else self.norm_cap
        delta = torch.where(support, alpha * (donor - output), 0)
        if kind == InterventionType.NONE:
            delta = torch.zeros_like(output)
        if kind == InterventionType.RANDOM_DIRECTION:
            noise = torch.randn(output.shape, generator=self.generator,
                                device=self.generator.device, dtype=output.dtype).to(output.device)
            noise = torch.where(support, noise, 0)
            target = torch.linalg.vector_norm(delta.flatten(1), dim=1)
            length = torch.linalg.vector_norm(noise.flatten(1), dim=1)
            delta = noise * (target / length.clamp_min(torch.finfo(output.dtype).tiny))[:, None, None]
        if rho is not None:
            length = torch.linalg.vector_norm(delta.flatten(1), dim=1)
            cap = rho * math.sqrt(n * d) * self.r_l
            scale = (cap / (length + 1e-12)).clamp(max=1)
            delta = delta * scale[:, None, None]
        # Direct replacement avoids cancellation in the full-state positive control.
        patched = donor.clone() if full else output + delta
        actual = patched - output
        length = torch.linalg.vector_norm(actual.flatten(1), dim=1)
        k = int(mask.sum())
        self.last_stats = {
            "channel_count": k, "effective_channel_ratio": k / d,
            "alpha": alpha, "norm_cap": rho,
            "actual_delta_norm": length.detach().cpu(),
            "n_elements_modified": torch.count_nonzero(actual.flatten(1), dim=1).detach().cpu(),
            "token_count": n, "exclude_cls": self.exclude_cls and not full,
            **extra,
        }
        if self.r_l is not None:
            reference = math.sqrt(n * d) * self.r_l
            self.last_stats["normalized_delta_norm"] = length.detach().cpu() / max(reference, 1e-12)
        return patched


@torch.no_grad()
def run_patched_forward(model, layer, x_corr, h_clean, alpha, mask,
                        norm_cap=None, r_l=None,
                        intervention=InterventionType.PARTIAL_CHANNEL, generator=None,
                        *, exclude_cls=False, labels=None):
    with Patcher(model, layer, alpha=alpha, mask=mask, norm_cap=norm_cap,
                 r_l=r_l, intervention=intervention, generator=generator,
                 exclude_cls=exclude_cls, labels=labels) as patcher:
        patcher.set_donor(h_clean)
        logits = model(x_corr)
        return logits, patcher.last_stats


def _metrics(logits, labels):
    alternatives = logits.clone()
    alternatives.scatter_(1, labels[:, None], -torch.inf)
    margin = logits.gather(1, labels[:, None]).squeeze(1) - alternatives.max(1).values
    return logits.argmax(1), margin, F.cross_entropy(logits, labels, reduction="none")


@torch.no_grad()
def sweep_layers_budgets(model, batches, layers, fractions, alphas, mask_seeds,
                         device="cpu", *, mask_policy="random_fixed", channel_scores=None,
                         exclude_cls=False, norm_cap=None, r_l=None,
                         intervention=InterventionType.PARTIAL_CHANNEL):
    """Return per-image rows. Optional caches: clean_outputs[layer], clean_logits.

    channel_scores and r_l are layer-keyed D_score estimates. Batch data follows
    PairedImageDataset alignment; callers must preserve original image order.
    """
    layers, fractions, alphas, mask_seeds = map(tuple, (layers, fractions, alphas, mask_seeds))
    if not layers or len(set(layers)) != len(layers) or any(
            not isinstance(l, int) or l < 0 or l >= len(model.blocks) for l in layers):
        raise ValueError("Provide unique valid layers")
    model.to(device).eval()
    intervention = InterventionType(intervention)
    rows = []
    for batch in batches:
        clean, corr = batch["x_clean"].to(device), batch["x_corr"].to(device)
        labels = torch.as_tensor(batch["y"], device=device, dtype=torch.long)
        ids = list(batch["image_id"])
        if clean.shape != corr.shape or labels.shape != (len(clean),) or len(ids) != len(clean):
            raise ValueError("Clean/corrupted images, labels and IDs must be aligned")
        outputs = batch.get("clean_outputs")
        clean_logits = batch.get("clean_logits")
        if outputs is None:
            with BlockOutputRecorder(model, layers, to_cpu=False) as recorder:
                clean_logits = model(clean)
            outputs = recorder.outputs
        elif clean_logits is None:
            clean_logits = model(clean)
        clean_logits = clean_logits.to(device)
        baseline = model(corr)
        base_pred, base_margin, base_loss = _metrics(baseline, labels)
        clean_pred = clean_logits.argmax(1)
        for layer, q, alpha, seed in product(layers, fractions, alphas, mask_seeds):
            spec = MaskSpec(mask_policy, q, seed, exclude_cls)
            scores = None if channel_scores is None else channel_scores[layer]
            mask, _ = make_channel_mask(outputs[layer].shape[-1], spec, scores)
            logits, stats = run_patched_forward(
                model, layer, clean if intervention == InterventionType.CLEAN_TO_CLEAN else corr,
                outputs[layer], alpha, mask, norm_cap,
                None if r_l is None else r_l[layer], intervention,
                torch.Generator().manual_seed(seed), exclude_cls=exclude_cls, labels=labels)
            pred, margin, loss = _metrics(logits, labels)
            for i, image_id in enumerate(ids):
                if isinstance(image_id, Tensor):
                    image_id = image_id.item()
                row = {"image_id": image_id, "label": labels[i].item(), "layer_id": layer,
                       "intervention_type": intervention.value, "mask_policy": mask_policy,
                       "mask_seed": seed, "fraction": q,
                       "baseline_prediction": base_pred[i].item(),
                       "clean_prediction": clean_pred[i].item(),
                       "post_intervention_prediction": pred[i].item(),
                       "baseline_margin": base_margin[i].item(),
                       "post_intervention_margin": margin[i].item(),
                       "baseline_loss": base_loss[i].item(), "post_loss": loss[i].item()}
                row.update({key: value[i].item() if isinstance(value, Tensor) else value
                            for key, value in stats.items()})
                if "donor_index" in row:
                    row["donor_image_id"] = ids[row["donor_index"]]
                rows.append(row)
    return pd.DataFrame(rows)
