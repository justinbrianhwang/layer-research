"""Compute score-split PH distances using a shared clean-fitted PCA per layer."""
import pandas as pd

from layer_research.topology import fit_clean_pca, ph_diagrams, ph_distance
from _common import Run, parser, conditions, condition_dir, read_cache, table


def main():
    p = parser(__doc__)
    p.add_argument("--split", choices=["score"], default="score")
    p.add_argument("--n-points", type=int, default=2000)
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    run = Run(p.parse_args(), "compute_topology", model=False)
    args = run.args
    clean = read_cache(condition_dir(run.paths, args.split, "clean", 0) / "summaries.pt", run)
    rows, costs = [], []
    projections, diagrams = {}, {}
    for layer in run.cfg["representation"]["layers"]:
        for mode in run.cfg["representation"]["summary_modes"]:
            H = clean["summaries"][layer][mode]
            projections[layer, mode] = fit_clean_pca(H, args.pca_dim, args.seed)
            diagrams[layer, mode] = ph_diagrams(H, pca=projections[layer, mode],
                                               n_points=args.n_points, seed=args.seed)
            costs.append(dict(layer=layer, summary_mode=mode, corruption="clean",
                              **diagrams[layer, mode].metadata))
    for name, severity in conditions(run.cfg):
        corr = read_cache(condition_dir(run.paths, args.split, name, severity) / "summaries.pt", run, clean["image_ids"])
        for (layer, mode), projection in projections.items():
            diagram = ph_diagrams(corr["summaries"][layer][mode], pca=projection,
                                  n_points=args.n_points, seed=args.seed)
            distances = ph_distance(diagrams[layer, mode], diagram, metric="both")
            costs.append(dict(layer=layer, summary_mode=mode, corruption=name, severity=severity,
                              diagram=diagram.metadata, distance=distances.metadata))
            for metric, score in distances.items():
                rows.append(dict(layer=layer, metric_name=metric, score=score,
                    n_samples=diagram.metadata["n_points"], summary_mode=mode, label_access="none",
                    corruption=name, severity=severity, split=args.split,
                    is_observed=name in run.cfg["corruptions"]["observed"]))
    if not rows:
        raise ValueError("No topology conditions configured")
    table(pd.DataFrame(rows), run.paths.tables / "topology_score", run)
    run.metric_cost = dict(calls=costs)
    run.finish()


if __name__ == "__main__":
    main()
