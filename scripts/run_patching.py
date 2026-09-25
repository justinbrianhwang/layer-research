"""Stream a cached-donor intervention sweep to Parquet and CSV."""
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
    run = Run(p.parse_args(), "run_patching")
    cfg, split = run.cfg["patching"], run.args.split
    layers = run.cfg["representation"]["layers"]
    base = condition_dir(run.paths, split, "clean", 0)
    clean = read_cache(base / "summaries.pt", run)
    donors = {l: read_cache(base / f"tokens_layer{l}.pt", run, clean["image_ids"])["tokens"].float() for l in layers}
    stats = read_cache(run.paths.cache / "score/channel_stats.pt", run)
    target = run.paths.raw / f"patching_{split}.parquet"
    writer = None
    schema = None
    try:
        for name, severity in tqdm(conditions(run.cfg), desc=f"patch {split}"):
            offset = 0
            batches = loader(run, split, name, severity)
            for x, xc, y, ids in batches:
                stop = offset + len(ids)
                if list(ids) != clean["image_ids"][offset:stop]:
                    raise ValueError("Donor IDs do not match receiver IDs")
                batch = dict(x_clean=x, x_corr=xc, y=y, image_id=ids,
                             clean_outputs={l: h[offset:stop] for l, h in donors.items()}, clean_logits=clean["logits"][offset:stop])
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
                df.to_csv(target.with_suffix(".csv"), mode="w" if offset == 0 and name == conditions(run.cfg)[0][0] and severity == conditions(run.cfg)[0][1] else "a", header=offset == 0 and (name, severity) == conditions(run.cfg)[0], index=False)
                offset = stop
    finally:
        if writer is not None:
            writer.close()
    run.finish()


if __name__ == "__main__":
    main()
