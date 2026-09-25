"""Eval-mode extraction at full residual-block boundaries."""
from collections.abc import Sequence, Iterator
from typing import Literal, Self
import torch


def num_blocks(model, blocks_attr="blocks") -> int:
    return len(model.get_submodule(blocks_attr))


class BlockOutputRecorder:
    """Detached snapshots of requested blocks; hooks exist only inside the context."""
    def __init__(self, model, layers: Sequence[int], to_cpu=True, dtype=None, blocks_attr="blocks"):
        self.model = model
        self.layers = tuple(layers)
        self.blocks = model.get_submodule(blocks_attr)
        if len(set(self.layers)) != len(self.layers) or any(i < 0 or i >= len(self.blocks) for i in self.layers):
            raise ValueError("Layer indices must be unique and in range")
        self.to_cpu, self.dtype = to_cpu, dtype
        self.outputs: dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self) -> Self:
        if self._handles:
            raise RuntimeError("Recorder context is already active")
        self.model.eval()
        self.outputs.clear()
        try:
            for layer in self.layers:
                def hook(module, inputs, output, layer=layer):
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("Block output must be a tensor")
                    self.outputs[layer] = output.detach().to(
                        device="cpu" if self.to_cpu else output.device,
                        dtype=self.dtype or output.dtype).clone()
                self._handles.append(self.blocks[layer].register_forward_hook(hook))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def summarize(tokens: torch.Tensor, mode: Literal["cls", "patch_mean", "cls+patch_mean"]) -> torch.Tensor:
    """Summarize single-CLS ViT tokens, excluding the first token from patch means."""
    if tokens.ndim != 3 or tokens.shape[1] < 1:
        raise ValueError("Expected [batch, tokens, channels]")
    if mode == "cls":
        return tokens[:, 0]
    if mode not in ("patch_mean", "cls+patch_mean"):
        raise ValueError(f"Unknown summary mode: {mode}")
    if tokens.shape[1] < 2:
        raise ValueError("Patch summaries require at least one patch token")
    mean = tokens[:, 1:].mean(dim=1)
    return mean if mode == "patch_mean" else torch.cat((tokens[:, 0], mean), dim=-1)


def iter_paired_block_outputs(model, loader, layers, device="cpu", blocks_attr="blocks") -> Iterator[dict]:
    """Yield CPU full-token pairs and logits; no hooks or no-grad state span a yield."""
    layers = tuple(layers)
    model.to(device).eval()
    for clean, corrupted, labels, image_ids in loader:
        with torch.no_grad(), BlockOutputRecorder(model, layers, blocks_attr=blocks_attr) as recorder:
            clean_logits = model(clean.to(device)).detach().cpu()
            clean_outputs = dict(recorder.outputs)
            recorder.outputs.clear()
            corrupted_logits = model(corrupted.to(device)).detach().cpu()
            corrupted_outputs = dict(recorder.outputs)
        yield {"clean": clean_outputs, "corrupted": corrupted_outputs,
               "clean_logits": clean_logits, "corrupted_logits": corrupted_logits,
               "labels": torch.as_tensor(labels).detach().cpu(), "image_ids": list(image_ids)}


@torch.no_grad()
def extract_paired_features(model, loader, layers, summary_modes, device="cpu", blocks_attr="blocks") -> dict:
    """Return clean/corrupted[layer][summary_mode], logits, labels and aligned IDs."""
    layers, modes = tuple(layers), tuple(summary_modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in ("cls", "patch_mean", "cls+patch_mean") for m in modes):
        raise ValueError("Provide unique supported summary modes")
    result = {side: {l: {m: [] for m in modes} for l in layers} for side in ("clean", "corrupted")}
    for key in ("clean_logits", "corrupted_logits", "labels", "image_ids"):
        result[key] = []
    for batch in iter_paired_block_outputs(model, loader, layers, device, blocks_attr):
        for side in ("clean", "corrupted"):
            for layer in layers:
                for mode in modes:
                    # Clone prevents CLS views from retaining full-token storage.
                    result[side][layer][mode].append(summarize(batch[side][layer], mode).clone())
        for key in ("clean_logits", "corrupted_logits", "labels"):
            result[key].append(batch[key])
        result["image_ids"].extend(batch["image_ids"])
    if not result["labels"]:
        raise ValueError("Cannot extract features from an empty loader")
    for side in ("clean", "corrupted"):
        for layer in layers:
            for mode in modes:
                result[side][layer][mode] = torch.cat(result[side][layer][mode])
    for key in ("clean_logits", "corrupted_logits", "labels"):
        result[key] = torch.cat(result[key])
    return result


@torch.no_grad()
def shape_report(model, input_size=(3, 224, 224), blocks_attr="blocks") -> dict[int, tuple]:
    """Run a zero input at the model's current device and floating-point dtype."""
    parameter = next(model.parameters())
    with BlockOutputRecorder(model, range(num_blocks(model, blocks_attr)), blocks_attr=blocks_attr) as recorder:
        model(torch.zeros((1, *input_size), device=parameter.device, dtype=parameter.dtype))
    return {layer: tuple(value.shape) for layer, value in recorder.outputs.items()}
