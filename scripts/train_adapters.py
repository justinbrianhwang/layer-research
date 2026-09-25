"""Train exhaustive or single-site adapters, with observed-only fit exposure."""
from dataclasses import asdict
from itertools import product
import json
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from layer_research.adapter_training import BottleneckAdapter, AdaptTrainConfig, train_adapter, evaluate_adapter
from _common import Run, parser, loader, conditions, set_all_seeds, table


def main():
    p = parser(__doc__)
    p.add_argument("--site", type=int)
    run = Run(p.parse_args(), "train_adapters")
    cfg = run.cfg["adapter"]
    layers = [run.args.site] if run.args.site is not None else run.cfg["representation"]["layers"]
    widths = [cfg["default_width"]] if run.args.site is not None else cfg["widths"]
    seeds = range(cfg["training_seeds"]) if isinstance(cfg["training_seeds"], int) else cfg["training_seeds"]
    datasets = [loader(run, "fit", n, s).dataset for n, s in conditions(run.cfg) if n in run.cfg["corruptions"]["observed"]]
    batch_size = run.cfg.get("runtime", {}).get("batch_size", 32)
    fit = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    costs, frames = [], {"val": [], "test": []}
    grid = list(product(layers, widths, seeds))
    for layer, width, seed in tqdm(grid, desc="adapters"):
        set_all_seeds(seed)
        adapter = BottleneckAdapter(run.cfg["model"]["embed_dim"], width).to(run.device)
        train_cfg = AdaptTrainConfig(layer, width, cfg.get("lr", .001), cfg.get("weight_decay", .01), cfg.get("steps", 1000), batch_size, cfg["lambda_clean"], seed, amp=cfg.get("amp", False), lr_schedule=cfg.get("lr_schedule", "constant"))
        result = train_adapter(run.model, adapter, fit, train_cfg, run.device)
        path = run.paths.raw / "adapters" / f"layer_{layer}_width_{width}_seed_{seed}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict={k: v.cpu() for k, v in adapter.state_dict().items()}, model_fingerprint=run.fingerprint), path / "adapter.pt")
        (path / "training.json").write_text(json.dumps(dict(config=asdict(train_cfg), cost=result.cost, loss_curve=result.loss_curve, model_fingerprint=run.fingerprint), indent=2), encoding="utf-8")
        costs.append(dict(result.cost, layer_id=layer, width=width, training_seed=seed, selection_cost=cfg.get("selection_cost", 0)))
        for split in frames:
            for i, (name, severity) in enumerate(conditions(run.cfg)):
                df = evaluate_adapter(run.model, adapter, layer, loader(run, split, name, severity), run.device)
                if i:
                    df = df[df.input_type.ne("clean")]
                frames[split].append(df.assign(width=width, training_seed=seed, split=split, is_observed=df.corruption.isin(run.cfg["corruptions"]["observed"])))
    cost = pd.DataFrame(costs)
    cost["reference_total_compute"] = cost.training_compute.sum() if run.args.site is None else float("nan")
    table(cost, run.paths.tables / "adapter_costs", run)
    for split, values in frames.items():
        table(pd.concat(values, ignore_index=True), run.paths.raw / f"adapter_{split}", run)
    run.finish()


if __name__ == "__main__":
    main()
