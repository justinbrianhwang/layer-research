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
        url = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
        expected = 1264079360
        for attempt in range(1, 6):  # resume-capable retries for flaky hosts
            have = archive.stat().st_size if archive.exists() else 0
            if have == expected:
                break
            if have > expected:
                archive.unlink(); have = 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(request, timeout=60) as source, archive.open("ab" if have else "wb") as dest, tqdm(total=expected, initial=have, unit="B", unit_scale=True) as bar:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
                        bar.update(len(chunk))
            except Exception as error:  # noqa: BLE001 - retry any transport failure
                print(f"download attempt {attempt} failed: {error}", flush=True)
        if archive.stat().st_size != expected:
            raise ValueError("Archive size mismatch after retries; remove incomplete archive before retrying")
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
