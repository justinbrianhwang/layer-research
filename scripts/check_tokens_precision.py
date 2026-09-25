"""Compare cached fp16 donors with fp32 recomputed donors (default 64 images)."""
import pandas as pd
import torch
from tqdm import tqdm
from layer_research.feature_extractor import BlockOutputRecorder
from layer_research.patching_engine import make_channel_mask, MaskSpec, run_patched_forward
from _common import Run, parser, loader, conditions, condition_dir, read_cache, table


def main():
    p = parser(__doc__)
    p.set_defaults(limit=64)
    p.add_argument("--split", default="val", choices=["score", "val", "test"])
    run = Run(p.parse_args(), "check_tokens_precision")
    layers = run.cfg["representation"]["layers"]
    base = condition_dir(run.paths, run.args.split, "clean", 0)
    caches = {l: read_cache(base / f"tokens_layer{l}.pt", run) for l in layers}
    name, severity = conditions(run.cfg)[0]
    rows, offset = [], 0
    for clean, corr, y, ids in tqdm(loader(run, run.args.split, name, severity)):
        with torch.no_grad(), BlockOutputRecorder(run.model, layers) as recorder:
            run.model(clean.to(run.device))
        for l in layers:
            cached = caches[l]
            if cached["image_ids"][offset:offset+len(ids)] != list(ids):
                raise ValueError("Cache must contain at least the requested aligned images")
            # Any cached dtype is compared against the fp32 recomputation; fp32 caches should give ~0.
            for q in run.cfg["patching"]["channel_fractions"]:
                mask, _ = make_channel_mask(cached["tokens"].shape[-1], MaskSpec("random_fixed", q, 0))
                args = (run.model, l, corr.to(run.device))
                a, _ = run_patched_forward(*args, cached["tokens"][offset:offset+len(ids)].float(), 1., mask)
                b, _ = run_patched_forward(*args, recorder.outputs[l], 1., mask)
                rows.append(dict(layer_id=l, fraction=q, n_images=len(ids), max_abs_diff=(a-b).abs().max().item()))
        offset += len(ids)
    df = pd.DataFrame(rows).groupby(["layer_id", "fraction"], as_index=False).agg(n_images=("n_images", "sum"), max_abs_diff=("max_abs_diff", "max"))
    df["cache_dtype"] = str(next(iter(caches.values()))["tokens"].dtype)
    table(df, run.paths.tables / "tokens_precision", run)
    print(df.to_string(index=False))
    run.finish()


if __name__ == "__main__":
    main()
