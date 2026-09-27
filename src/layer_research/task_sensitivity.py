"""True-label margin derivatives at full residual-block boundaries."""
from collections.abc import Mapping
import time

import pandas as pd
import torch

from .feature_extractor import BlockOutputRecorder


def margin_gradient_scores(model, loader, layers, device, blocks_attr="blocks"):
    """Return per-image first-order effects, using one VJP per corrupted batch.

    Accept paired tuples or mappings with x_clean/x_corr/y/image_id. Mapping
    batches may provide aligned clean_outputs[layer] to avoid clean forwards.
    Parameter gradients and requires_grad flags are left untouched. Cost counts
    and synchronized elapsed time are returned in DataFrame.attrs.
    """
    layers = tuple(layers)
    blocks = model.get_submodule(blocks_attr)
    if not layers or len(set(layers)) != len(layers) or any(l < 0 or l >= len(blocks) for l in layers):
        raise ValueError("Provide nonempty unique in-range layers")
    device = torch.device(device)
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    rows, backwards, forwards = [], 0, 0
    for batch in loader:
        if isinstance(batch, Mapping):
            clean, corr, labels, ids = (batch.get(k) for k in ("x_clean", "x_corr", "y", "image_id"))
            donors = batch.get("clean_outputs")
        else:
            clean, corr, labels, ids = batch
            donors = None
        labels = torch.as_tensor(labels, device=device, dtype=torch.long)
        ids = list(ids)
        if labels.shape != (len(corr),) or len(ids) != len(corr):
            raise ValueError("Batch labels and IDs must align with inputs")
        if donors is None:
            with torch.no_grad(), BlockOutputRecorder(model, layers, blocks_attr=blocks_attr) as recorder:
                model(clean.to(device))
                donors = dict(recorder.outputs)
            forwards += 1
        handles, outputs = [], {}
        try:
            with torch.enable_grad():
                for layer in layers:
                    def hook(module, inputs, output, layer=layer):
                        if not isinstance(output, torch.Tensor):
                            raise TypeError("Block output must be a tensor")
                        if not output.requires_grad:
                            output = output.detach().requires_grad_(True)
                        outputs[layer] = output
                        return output
                    handles.append(blocks[layer].register_forward_hook(hook))
                logits = model(corr.to(device))
                forwards += 1
                if logits.ndim != 2 or logits.shape[1] < 2:
                    raise ValueError("Margin requires at least two class logits")
                true = logits.gather(1, labels[:, None]).squeeze(1)
                competitors = logits.scatter(1, labels[:, None], -torch.inf).amax(1)
                gradients = torch.autograd.grad((true - competitors).sum(), [outputs[l] for l in layers])
                backwards += 1
            for layer, gradient in zip(layers, gradients):
                h = outputs[layer].detach()
                donor = donors[layer].to(device=h.device, dtype=h.dtype)
                if donor.shape != h.shape:
                    raise ValueError("Clean donor and corrupted output shapes differ")
                delta = (donor - h).double().flatten(1)
                gradient = gradient.detach().double().flatten(1)
                dot = (gradient * delta).sum(1)
                gn, dn = gradient.norm(dim=1), delta.norm(dim=1)
                values = torch.stack((dot, dot / (gn * dn + 1e-12), gn, dn), 1).cpu().tolist()
                rows.extend((image_id, label, layer, *value)
                            for image_id, label, value in zip(ids, labels.cpu().tolist(), values))
        finally:
            for handle in handles:
                handle.remove()
    if not rows:
        raise ValueError("Cannot score an empty loader")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = pd.DataFrame(rows, columns=["image_id", "label", "layer_id", "dot_full", "dot_norm", "grad_norm", "delta_norm"])
    result.attrs.update(backward_count=backwards, forward_count=forwards,
                        wall_time_seconds=time.perf_counter() - start)
    return result
