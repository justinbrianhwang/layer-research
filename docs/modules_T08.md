# T08: ResNet cross-architecture support

`configs/resnet50.yaml` selects timm `resnet50.a1_in1k` (ImageNet-1k only),
16 sites, GAP summaries, ten patch-mask seeds, adapter width 32 and training
seed 0. Outputs are isolated under `outputs_resnet50/`. The inherited
`embed_dim` field is unused: adapter dimensions come from a dummy forward.

## Residual boundaries and layouts

`resolve_blocks(model, blocks_attr="blocks")` returns a list. With
`resnet_stages`, it concatenates `layer1` through `layer4`: ResNet-50 has
3/4/6/3 bottlenecks and ResNet-18 has 2/2/2/2 basic blocks. Hooks observe the
complete block output **after residual addition and ReLU**, including any
downsample projection. Extraction, patching, adapter attachment, training and
evaluation accept `blocks_attr`; defaults retain the ViT layout and API.

CNN states are `[B,C,H,W]`; `summarize(..., "gap")` averages H and W. CLS and
patch summaries reject CNN states. Channel masks have shape `[C]`, broadcast
over both spatial dimensions, and `exclude_cls` has no effect. Norms flatten
all nonbatch dimensions; `token_count` records H*W. All intervention controls
work on CNN states, with exact full-state replacement. Patcher hooks assert
that the model is in eval mode, freezing BatchNorm running statistics.

`BottleneckAdapter(..., layout="channels_last_4d")` permutes to BHWC, applies
LN/down/activation/up per position, permutes back and adds the residual.
Zero-initialized output projection gives exact initial identity. Parameters
remain `2*d*r + 3*d + r`. The frozen backbone remains in eval mode during
training; gradients pass through later blocks to the adapter.

Scripts propagate the configured block layout. `compute_metrics.py` consumes
GAP caches without a model forward. Channel-difference statistics reduce B/H/W,
retaining C. Model fingerprints flatten tensors before byte reinterpretation
to handle scalar BatchNorm counters; existing ViT fingerprints are unchanged.
Manifest arguments also serialize filesystem Paths as strings.

## Compute accounting

One actual profiling forward counts `nn.Linear` and `nn.Conv2d` multiply-adds
as two FLOPs, including grouped convolutions, residual projections and the
classifier. Per-block costs sum the same counters within each block. Standard
ViT attention additionally counts functional QK-transpose and attention-times-V
matrix multiplies, which Linear hooks cannot see. Biases, activations, pooling,
backbone normalization, softmax, loss and optimizer operations are omitted.
Adapter forward remains `positions * (4*d*r + 5*d)` including approximate LN.

Backward is estimated as **2 times (blocks strictly after the insertion site
+ head + adapter forward)**. This deliberately replaces T04's earlier
one-forward-equivalent frozen-backbone estimate. The insertion block and stem
have no backward because their frozen outputs precede the adapter. These are
shape-based estimates, not hardware instruction measurements.

Measured hook counts below use random, untrained models at 224x224, one image,
width 32, CPU, without downloads. Weight values do not affect these counts.
GFLOPs are decimal billions; forward includes the adapter.

| Model | Site | Forward GFLOPs | Estimated backward GFLOPs | Adapter forward GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| DeiT-S/16 | 0 | 9.207826 | 16.670875 | 0.010061 |
| DeiT-S/16 | 11 | 9.207826 | 0.021658 | 0.010061 |
| ResNet-50 | 0 | 8.285143 | 15.173386 | 0.106775 |
| ResNet-50 | 15 | 8.191715 | 0.034886 | 0.013347 |

Backbone-only forward is 9.197765 GFLOPs for DeiT and 8.178369 GFLOPs for
ResNet. The tiny-ViT regression compares forward and per-block estimates with
the previous analytic estimator, requiring agreement within 10%.

## Caching and disk requirements

ResNet donor caches always use fp32, even if `--tokens-dtype float16` is supplied.
Only requested representation layers are recorded. Clean states alone are
saved as donor tensors; corrupted states are reduced to GAP/channel statistics.
An early site occupies 256*56*56*4 = 3,211,264 bytes per image. All 16 sites
total **22,077,440 bytes/image**, or **44.15 GB (41.12 GiB) per 2,000-image
split**, before serialization overhead. This exceeds the task's rough 20 GB
estimate. Three score/val/test donor splits require about 132.5 GB. Summary
caches are additional. Caching retains one condition's selected donors in
host memory; concatenation can temporarily require another copy. Patching
also retains all requested clean donors in host memory.

Use `--layers` to split cache generation into sequential groups with the same
config, inputs, model and batch size. Summary/RMS/channel-stat dictionaries
merge by layer after fingerprint, image-ID, label and logit checks. Do not run
these cache writes concurrently. For example, replace the driver's cache line:

```bash
for layers in 0,1,2 3,4,5,6 7,8,9,10,11,12 13,14,15; do
  python scripts/cache_features.py --config configs/resnet50.yaml --device cuda --layers "$layers" --tokens-dtype float32
done
```

## ResNet E1 and E4 commands

Run from the repository root on a provisioned GPU machine. The driver contains
this exact sequence; it runs stages afresh rather than treating a partial cache
manifest as proof that every layer is cached.

```bash
python scripts/download_data.py --config configs/resnet50.yaml --device cuda
python scripts/cache_features.py --config configs/resnet50.yaml --device cuda --tokens-dtype float32
python scripts/compute_metrics.py --config configs/resnet50.yaml --device cuda
python scripts/run_patching.py --config configs/resnet50.yaml --device cuda --experiment E1 --split val --mask-seeds 10
python scripts/select_sites.py --config configs/resnet50.yaml --device cuda
python scripts/run_patching.py --config configs/resnet50.yaml --device cuda --experiment E1 --split test --mask-seeds 10
python scripts/evaluate.py --config configs/resnet50.yaml --device cuda
python scripts/train_adapters.py --config configs/resnet50.yaml --device cuda --widths 32 --seeds 1
# Equivalent:
# bash scripts/gpu_run_resnet.sh
```

`--seeds 1` denotes the single seed 0. Adapter training supplies the exhaustive
16-site E4 reference, checkpoints, compute costs and per-image validation/test
rows. E1 evaluation precedes adapter training as in the requested protocol.
The driver does not add an E4 selector/report beyond the existing adapter
outputs. Optional donor verification uses `scripts/check_tokens_precision.py`
with the same config. Pretrained real-data GPU experiments were not run during
implementation: validation was entirely offline, without downloads.

## Tests

`tests/test_T08.py` uses random ResNet-18 CPU models and a tiny ViT. Coverage
includes shapes, GAP, hook cleanup, exact controls, nested channel masks,
random/shuffled controls and caps, eval assertion, first/last-site adapter
gradients and frozen BatchNorm, sweep row counts, FLOP agreement, and an offline
script chain with split-layer cache merging, metrics, patching, precision and
one-step adapter training. Test artifacts use pytest's system-temp basetemp.
