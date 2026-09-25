# Vast.ai GPU workflow

This machine has no GPU. All feature caching, patching sweeps and adapter training run on a
rented single-GPU Vast.ai instance. Local machine is for development, CPU tests and analysis.

## Target instance

- 1x RTX 3090 (24 GB) or RTX 4080S. DeiT-S/16 at batch 128 fp16 uses well under 8 GB, so
  the cheapest reliable 24 GB card is sufficient.
- Filters used: `reliability > 0.98`, `inet_down > 200 Mbps`, `disk_space >= 100 GB`,
  `cuda_vers >= 12.4`. Typical price in Sept 2026: $0.15–0.20 / h.
- Image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` (or newer 2.x CUDA 12 runtime).

## Commands (local)

```bash
VAST="$HOME/AppData/Roaming/Python/Python314/Scripts/vastai.exe"   # key already set
$VAST search offers 'gpu_name in [RTX_3090,RTX_4080S,RTX_4090] num_gpus=1 gpu_ram>=20 reliability>0.98 inet_down>200 disk_space>=100 cuda_vers>=12.4 rentable=true' -o dph
$VAST create instance <OFFER_ID> --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime --disk 100 --ssh --direct
$VAST show instances
$VAST ssh-url <INSTANCE_ID>
$VAST destroy instance <INSTANCE_ID>     # ALWAYS destroy when done; billing is hourly
```

## On the instance

```bash
git clone https://github.com/justinbrianhwang/layer-research.git && cd layer-research
pip install -e . timm imagecorruptions scikit-learn pandas pyyaml tqdm
python scripts/download_data.py --config configs/deit_small.yaml   # ImageNetV2, ~1.2 GB, see docs/data_plan.md
python scripts/cache_features.py --config configs/deit_small.yaml
python scripts/run_patching.py   --config configs/deit_small.yaml --experiment E1
```

Results are written to `results/` and synced back with `scp`/`rsync`; raw tensors stay on the
instance and only tables (`.csv`/`.parquet`) come home.

## Data plan

Superseded by `docs/data_plan.md` (ImageNetV2 clean images + generated corruptions). Original note kept below for history.

### Earlier note (ImageNet val idea)

The proposal recommends carving fit/score/val splits from ImageNet **train**. Train is ~140 GB
and impractical on an hourly instance. We instead split the 50k ImageNet **validation** images
by original image id into fit / score / val / test and generate corruptions with the
ImageNet-C code (`imagecorruptions`). Results are therefore reported as *generated-corruption*
evaluation, not official ImageNet-C scores (proposal §6.3 permits this if labelled). The pretrained
backbone never saw val images, so there is no backbone-level leakage.
