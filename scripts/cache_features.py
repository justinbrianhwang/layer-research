"""Cache summaries at every block and selected full-token donor states."""
import torch
from tqdm import tqdm
from layer_research.feature_extractor import BlockOutputRecorder, summarize
from _common import Run, parser, loader, conditions, condition_dir, read_cache


def main():
    p = parser(__doc__)
    p.add_argument("--include-fit", action="store_true")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--tokens-dtype", choices=["float16", "float32"], default="float16")
    run = Run(p.parse_args(), "cache_features")
    layers = run.cfg["representation"]["layers"]
    for split in (["fit"] if run.args.include_fit else []) + ["score", "val", "test"]:
        for name, severity in [("clean", 0)] + conditions(run.cfg):
            summaries = {l: {m: [] for m in ("cls", "patch_mean")} for l in range(len(run.model.blocks))}
            tokens = {l: [] for l in layers} if name == "clean" else {}
            ids, labels, logits = [], [], []
            sums = {l: 0. for l in layers}
            counts = {l: 0 for l in layers}
            channel_sums = {}
            offset = 0
            collect_channels = split == "score" and name in run.cfg["corruptions"]["observed"]
            donors = {l: read_cache(condition_dir(run.paths, split, "clean", 0) / f"tokens_layer{l}.pt", run)
                      for l in layers} if collect_channels else {}
            for clean, corr, y, batch_ids in tqdm(loader(run, split, name, severity, paired=name != "clean"), desc=f"{split}/{name}/{severity}"):
                with torch.no_grad(), BlockOutputRecorder(run.model, list(summaries)) as recorder:
                    logits.append(run.model((clean if name == "clean" else corr).to(run.device)).cpu())
                ids.extend(batch_ids)
                labels.append(y)
                for l, h in recorder.outputs.items():
                    for mode in summaries[l]:
                        summaries[l][mode].append(summarize(h.float(), mode).clone())
                    if l in tokens:
                        tokens[l].append(h.to(getattr(torch, run.args.tokens_dtype)))
                    if l in sums:
                        sums[l] += h.double().square().sum().item()
                        counts[l] += h.numel()
                    if l in donors:
                        stop = offset + len(batch_ids)
                        if donors[l]["image_ids"][offset:stop] != list(batch_ids):
                            raise ValueError("Channel-stat donor IDs do not match receiver IDs")
                        a = donors[l]["tokens"][offset:stop].float()
                        b = h.to(getattr(torch, run.args.tokens_dtype)).float()
                        delta = (a - b).abs().double().sum((0, 1))
                        channel_sums[l] = channel_sums.get(l, 0) + delta
                offset += len(batch_ids)
            path = condition_dir(run.paths, split, name, severity)
            path.mkdir(parents=True, exist_ok=True)
            meta = dict(image_ids=ids, labels=torch.cat(labels), model_fingerprint=run.fingerprint)
            if collect_channels:
                meta["channel_abs_delta"] = {l: (v / (counts[l] / v.numel())).float()
                                             for l, v in channel_sums.items()}
            torch.save(dict(meta, summaries={l: {m: torch.cat(v) for m, v in modes.items()} for l, modes in summaries.items()}, logits=torch.cat(logits), r_l={l: (sums[l]/counts[l])**.5 for l in layers}), path / "summaries.pt")
            for l, values in tokens.items():
                torch.save(dict(meta, tokens=torch.cat(values)), path / f"tokens_layer{l}.pt")
    run.finish()


if __name__ == "__main__":
    main()
