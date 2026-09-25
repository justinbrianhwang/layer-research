"""Portable real-data CPU smoke chain; smoke_test.sh delegates here."""
import subprocess
import sys
import time
from pathlib import Path
import yaml
import pandas as pd
from _common import parser, load_config


def main():
    p = parser(__doc__)
    p.set_defaults(device="cpu", limit=20)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg["representation"]["layers"] = [0, 11]
    cfg["patching"].update(channel_fractions=[.1], mask_seeds=1)
    cfg["corruptions"].update(observed=["gaussian_noise"], unseen=["contrast"], severities=[1])
    cfg["evaluation"]["bootstrap_resamples"] = 100
    cfg["runtime"] = dict(batch_size=20, threads=4)
    cfg["output_root"] = "outputs/smoke"
    path = Path(cfg["output_root"])
    path.mkdir(parents=True, exist_ok=True)
    config = path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    start = time.perf_counter()
    succeeded = False
    try:
        for script, extra in [("download_data", []), ("cache_features", []), ("compute_metrics", []),
                              ("run_patching", ["--split", "val"]), ("select_sites", []),
                              ("run_patching", ["--split", "test"]), ("evaluate", [])]:
            subprocess.run([sys.executable, str(Path(__file__).with_name(script + ".py")), "--config", str(config), "--device", args.device, "--limit", str(args.limit), *extra], check=True)
        expected = {"tables/metrics_score": 40, "tables/selections": 16,
                    "tables/E1_effects": 24, "tables/E3_observed": 16, "tables/E3_unseen": 16,
                    "raw/patching_val": args.limit * 24, "raw/patching_test": args.limit * 24}
        for stem, count in expected.items():
            frame = pd.read_parquet(path / "results" / (stem + ".parquet"))
            if len(frame) != count:
                raise AssertionError(f"{stem}: expected {count} rows, got {len(frame)}")
            if not (path / "results" / (stem + ".csv")).exists():
                raise AssertionError(f"Missing CSV for {stem}")
            print(f"Verified {stem}: {count} rows")
        succeeded = True
    finally:
        elapsed = time.perf_counter()-start
        status = "completed" if succeeded else "FAILED (incomplete chain)"
        (path / "wall_time_seconds.txt").write_text(f"{elapsed:.2f}s; {status}\n")
        print(f"Smoke-run wall time: {elapsed:.2f}s; {status}")


if __name__ == "__main__":
    main()
