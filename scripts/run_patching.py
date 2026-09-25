"""Stream a cached-donor intervention sweep to Parquet."""
import time
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from layer_research.patching_engine import sweep_layers_budgets
from _common import Run, parser, loader, conditions, condition_dir, read_cache


def main():
    p = parser(__doc__)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--experiment", choices=["E1", "E2"], default="E1")
    p.add_argument("--mask-seeds", type=int)
    p.add_argument("--layers", type=lambda s: [int(v) for v in s.split(",")])
    p.add_argument("--fractions", type=lambda s: [float(v) for v in s.split(",")])
    p.add_argument("--batch-size", type=int)
    run = Run(p.parse_args(), "run_patching")
    cfg, split = run.cfg["patching"], run.args.split
    if run.args.mask_seeds is not None:
        if run.args.mask_seeds <= 0:
            raise ValueError("mask seeds must be positive")
        cfg["mask_seeds"] = run.args.mask_seeds
    if run.args.layers is not None:
        if any(l < 0 or l >= len(run.model.blocks) for l in run.args.layers):
            raise ValueError("layers must be valid block indices")
        run.cfg["representation"]["layers"] = run.args.layers
    if run.args.fractions is not None:
        if any(not 0 < q <= 1 for q in run.args.fractions):
            raise ValueError("fractions must be in (0, 1]")
        cfg["channel_fractions"] = run.args.fractions
    layers = run.cfg["representation"]["layers"]
    base = condition_dir(run.paths, split, "clean", 0)
    clean = read_cache(base / "summaries.pt", run)
    donors = {l: read_cache(base / f"tokens_layer{l}.pt", run, clean["image_ids"])["tokens"] for l in layers}
    stats = read_cache(run.paths.cache / "score/channel_stats.pt", run)
    target = run.paths.raw / f"patching_{split}.parquet"
    writer = None
    schema = None
    try:
        sweep_conditions = conditions(run.cfg)
        sweep_start = time.perf_counter()
        for index, (name, severity) in enumerate(tqdm(sweep_conditions, desc=f"patch {split}")):
            condition_start = time.perf_counter()
            rows_written = 0
            offset = 0
            batches = loader(run, split, name, severity)
            for x, xc, y, ids in batches:
                stop = offset + len(ids)
                if list(ids) != clean["image_ids"][offset:stop]:
                    raise ValueError("Donor IDs do not match receiver IDs")
                batch = dict(x_clean=x, x_corr=xc, y=y, image_id=ids,
                             clean_outputs={l: h[offset:stop].to(run.device, torch.float32) for l, h in donors.items()}, clean_logits=clean["logits"][offset:stop])
                seeds = cfg["mask_seeds"]
                seeds = list(range(seeds)) if isinstance(seeds, int) else seeds
                alphas = cfg["alphas"] if run.args.experiment == "E2" else [cfg["default_alpha"]]
                caps = [None]
                if run.args.experiment == "E2" and cfg.get("norm_cap_rho") is not None:
                    caps.append(cfg["norm_cap_rho"])
                frames = []
                controls = [{"no_intervention": "none", "random_direction_same_norm": "random_direction"}.get(c, c) for c in cfg["controls"]]
                for intervention in ["partial_channel"] + controls:
                    control = intervention != "partial_channel"
                    for cap in ([None] if control else caps):
                        if intervention == "shuffled_donor" and len(ids) < 2:
                            raise ValueError("Shuffled donor needs batches >=2; adjust batch_size")
                        df = sweep_layers_budgets(run.model, [batch], layers,
                            [cfg["channel_fractions"][0]] if control else cfg["channel_fractions"],
                            [cfg["default_alpha"]] if control else alphas, seeds[:1] if control else seeds,
                            run.device, mask_policy=cfg["mask_policy"].replace("random_fixed_nested", "random_fixed"),
                            channel_scores=stats["channel_scores"], r_l=stats["r_l"], norm_cap=cap, intervention=intervention)
                        frames.append(df)
                df = pd.concat(frames, ignore_index=True).assign(corruption=name, severity=severity, split=split,
                    experiment=run.args.experiment, is_observed=name in run.cfg["corruptions"]["observed"], model_fingerprint=run.fingerprint)
                # Explicit optional columns keep a stable Arrow schema across batches.
                for col in ("donor_index", "same_class"):
                    if col not in df:
                        df[col] = float("nan")
                    df[col] = pd.to_numeric(df[col]).astype(float)
                if "donor_image_id" not in df:
                    df["donor_image_id"] = None
                df["donor_image_id"] = df["donor_image_id"].astype("string")
                df["norm_cap"] = pd.to_numeric(df.norm_cap).astype(float)
                arrow = pa.Table.from_pandas(df, preserve_index=False, schema=schema)
                if writer is None:
                    schema = arrow.schema
                    writer = pq.ParquetWriter(target, schema)
                writer.write_table(arrow)
                rows_written += len(df)
                offset = stop
            elapsed = time.perf_counter() - condition_start
            eta = (time.perf_counter() - sweep_start) / (index + 1) * (len(sweep_conditions) - index - 1)
            print(f"{split}/{name}/{severity}: elapsed={elapsed:.2f}s rows_written={rows_written} ETA={eta:.2f}s", flush=True)
    finally:
        if writer is not None:
            writer.close()
    run.finish()


if __name__ == "__main__":
    main()
