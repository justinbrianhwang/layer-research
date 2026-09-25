# Data plan (decided 2026-09-25, before any experiment was run)

## Constraint

No ImageNet credentials are available on the development machine (no Hugging Face token, no
Kaggle credentials), and ImageNet-1k train/val are gated downloads. The official ImageNet-C tarballs
contain only corrupted validation images, so they cannot supply the clean twin the proposal needs
(§5.1, §8.1). GPU time is rented by the hour, so multi-hundred-GB downloads are also unattractive.

## Decision

Use **ImageNetV2 matched-frequency** (Recht et al., 2019) as the clean image source.

- 10 000 images, exactly 10 per ImageNet-1k class, labelled with the same 1 000 classes.
- Public download from `huggingface.co/datasets/vaishaal/ImageNetV2` (verified reachable, HTTP 200).
- Never used to train DeiT-S, so there is no backbone-level leakage for any split.
- Corruptions are generated with the ImageNet-C reference code (`imagecorruptions` package) at
  severities 1 / 3 / 5 with a fixed per-image seed, for the observed set (gaussian_noise,
  defocus_blur) and the unseen set (contrast, jpeg_compression).

Splits are by original image id, class-stratified: fit 40 % / score 20 % / val 20 % / test 20 %
(4 / 2 / 2 / 2 images per class). Split file: `configs/splits_seed0.json`.

## What this changes in the proposal's wording

| Proposal | This repo |
|---|---|
| ImageNet train for fit/score/val, official ImageNet-C val for test (§6.3) | ImageNetV2 for all four splits, generated corruptions everywhere |
| "ImageNet-C accuracy" | "generated-corruption accuracy on ImageNetV2" (§6.3 requires this labelling) |
| 50 k test images | 2 000 test images (10 per class × 20 %); wider bootstrap CIs, reported as such |

The proposal already anticipates class-balanced subsets (§6.4) and generated corruptions (§6.3).
Nothing in the experimental logic (paired ids, fixed masks, selection before test, regret) changes.

## If ImageNet access becomes available later

`data_protocol` is path-based; pointing `data.root` at an ImageNet val folder and regenerating
`splits_seed*.json` is enough. Results from the two sources will not be mixed in one table.
