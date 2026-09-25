"""Acquire ImageNetV2 once and create immutable stratified split IDs."""
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from layer_research.data_protocol import SplitConfig, make_splits, save_splits, load_splits
from _common import Run, parser, records


def main():
    run = Run(parser(__doc__).parse_args(), "download_data", model=False)
    root = Path(run.cfg["data"]["root"])
    if not root.is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        archive = root.parent / "imagenetv2-matched-frequency.tar.gz"
        if not archive.exists():
            url = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
            with urllib.request.urlopen(url) as source, archive.open("wb") as dest, tqdm(total=1264079360, unit="B", unit_scale=True) as bar:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
                    bar.update(len(chunk))
        if archive.stat().st_size != 1264079360:
            raise ValueError("Archive size mismatch; remove incomplete archive before retrying")
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(root.parent, filter="data")
    else:
        print(f"Using extracted data: {root}; download skipped")
    rec = records(run.cfg)
    cfg = SplitConfig(**run.cfg["data"]["split"])
    expected = make_splits([r[2] for r in rec], [r[1] for r in rec], cfg)
    path = Path(run.cfg["data"]["split_file"])
    if path.exists():
        actual = load_splits(path)
        if actual.cfg != cfg or any(set(actual[k]) != set(expected[k]) for k in expected):
            raise ValueError("Existing split differs; refusing to overwrite")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_splits(expected, path)
    labels = {r[2]: r[1] for r in rec}
    for split, ids in expected.items():
        print(split, dict(sorted(Counter(labels[i] for i in ids).items())))
    # Dataset preparation has no model dependency; fingerprint is explicitly null.
    run.finish()


if __name__ == "__main__":
    main()
