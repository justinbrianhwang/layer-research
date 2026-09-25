"""Compute label-free D_score metrics and token-level channel statistics."""
import pandas as pd
import torch
from tqdm import tqdm
from layer_research.representation_metrics import layerwise_scores
from _common import Run, parser, conditions, condition_dir, read_cache, table


def main():
    run = Run(parser(__doc__).parse_args(), "compute_metrics", model=False)
    base = condition_dir(run.paths, "score", "clean", 0)
    clean = read_cache(base / "summaries.pt", run)
    layers = run.cfg["representation"]["layers"]
    channels = {l: [] for l in layers}
    frames = []
    for name, severity in tqdm(conditions(run.cfg)):
        path = condition_dir(run.paths, "score", name, severity)
        corr = read_cache(path / "summaries.pt", run, clean["image_ids"])
        features = {side: {l: {m: data["summaries"][l][m] for m in run.cfg["representation"]["summary_modes"]} for l in layers} for side, data in (("clean", clean), ("corrupted", corr))}
        df = layerwise_scores(features, run.cfg["representation"]["metrics"])
        df = df.assign(corruption=name, severity=severity, split="score", is_observed=name in run.cfg["corruptions"]["observed"])
        frames.append(df)
        if name in run.cfg["corruptions"]["observed"]:
            for l in layers:
                channels[l].append(corr["channel_abs_delta"][l])
    table(pd.concat(frames), run.paths.tables / "metrics_score", run)
    torch.save(dict(channel_scores={l: torch.stack(v).mean(0) for l, v in channels.items()}, r_l=clean["r_l"], split="score", model_fingerprint=run.fingerprint), run.paths.cache / "score/channel_stats.pt")
    run.finish()


if __name__ == "__main__":
    main()
