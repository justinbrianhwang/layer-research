# T01b — apply corruptions at the model's input resolution (fix to T01)

## Problem

`data_protocol.PairedImageDataset` currently corrupts the **original-resolution** PIL image and then
resizes/crops. ImageNet-C (Hendrycks & Dietterich, 2019) applies each corruption to the image
**after** resize-256 / center-crop-224, i.e. at the network's input resolution. Corrupting before
downscaling averages out noise and changes blur radii, so severities would not mean the same thing
as in the benchmark. Fix this so the pipeline is: geometric preprocessing (resize, crop) → corruption
on the uint8 RGB image at input resolution → ToTensor / Normalize.

## Required change

In `PairedImageDataset.__init__`, build the timm eval transform with
`timm.data.create_transform(**data_config, is_training=False)` as now, then split the resulting
`torchvision.transforms.Compose` into two parts:

- `self.pre_transform`: every transform **before** `ToTensor` (Resize, CenterCrop, ...). Operates on PIL.
- `self.post_transform`: `ToTensor` and everything after it (Normalize). Operates on PIL → tensor.

Fail loudly (`ValueError`) if no `ToTensor` (or timm's `MaybeToTensor`) is found in the pipeline.

`__getitem__` must do:

```python
clean_pil = self.pre_transform(image)                      # PIL at input resolution, e.g. 224x224
corrupted_pil = corrupt_image(clean_pil, self.spec, image_id)
return self.post_transform(clean_pil), self.post_transform(corrupted_pil), label, image_id
```

Expose `self.input_resolution` (H, W) for logging. Keep `corrupt_image` unchanged.

## Tests to add / adjust in `tests/test_T01.py`

- The corrupted tensor equals `post_transform(corrupt_image(pre_transform(img), spec, id))` exactly.
- For a 40x40 source image and input_size 32, the image passed to `corrupt_image` is 32x32 (spy/monkeypatch `corrupt_image` and assert the received size).
- `pre_transform` output for the DeiT config (`resolve_data_config({}, model=timm.create_model("deit_small_patch16_224", pretrained=False))`) is 224x224 and `post_transform` yields shape `(3, 224, 224)`. No download: `pretrained=False`.

## Docs

Update `docs/modules_T01.md` (Data protocol paragraph) to state that corruption happens at input
resolution after resize/crop and before tensor conversion, matching the ImageNet-C protocol.

Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q` and print the summary line.
Do not commit.
