# T04: adapter training

Implements proposal 10.1–10.6, 5.3 and 18.3 for standard timm ViTs/DeiT.
All model execution APIs take an explicit `device`. No downloads are needed for tests.

## API and training

- `BottleneckAdapter(d, r, act="gelu")` computes
  `h + up(act(down(norm(h))))` independently for every token. GELU and ReLU
  are supported. LayerNorm affine parameters and both linear biases are learnable.
  Both up-projection weights and bias start at zero, yielding exact initial identity.
  `extra_params()` is `2*d*r + 3*d + r`; approximate forward FLOPs per token
  are `4*d*r + 5*d` (two FLOPs per multiply-add, plus LayerNorm).
- `attach_adapter(model, layer, adapter)` installs one removable block-output hook.
  It preserves autograd and does not register the adapter as a model submodule.
  The low-level caller owns device placement and handle removal.
- `count_trainable(module)` counts that module's trainable parameters.
  `check_only_adapter_trainable(model, adapter)` validates frozen backbone/classifier
  parameters and learnable adapter parameters; it raises on violations.
- `AdaptTrainConfig` contains the specified layer, width, optimizer, exposure,
  clean-loss, seed, AMP and logging fields, plus `lr_schedule="constant"` or `"cosine"`.
  Cosine uses `lr * (1+cos(pi*step/steps))/2` at zero-based update indices.
- `train_adapter(model, adapter, fit_loader, cfg, device)` returns `TrainResult`
  with `adapter`, `cost`, and `loss_curve`. It trains existing adapter weights;
  seed before constructing an adapter when using this low-level API. It never
  resets learned weights. The orchestration APIs seed initialization themselves.

Training freezes every backbone/classifier parameter, clears stale backbone gradients,
and leaves the model and adapter in eval mode, including after return. Forward execution
retains gradients through downstream blocks and the head. Dropout/drop-path are disabled.
Eval mode also freezes BatchNorm running statistics, but CNN layouts require a separate
attachment and FLOP estimator; the current estimator deliberately assumes standard ViT
attention, MLP, convolutional patch embedding and classification head shapes.

AdamW performs exactly `steps` updates, cycling the loader as needed. Empty or exhausted
one-shot loaders raise. T01 tuple batches `(clean, corrupted, labels, image_ids)` and T02
mapping batches (`x_clean`, `x_corr`, `y`, `image_id`) are supported. The supplied DataLoader
batch size must match the configuration; final partial batches are allowed and counted
by their actual size. Use the same loader settings and seed across sites for equal exposure.
The objective is corrupted CE plus `lambda_clean` times clean CE on the same paired batch.
At zero clean weight the extra clean forward is omitted. Loss records include the first,
last and each `log_every` update, with pre-update losses and the learning rate used.

Python, NumPy, PyTorch and explicit loader/sampler generators are seeded and restored.
Deterministic PyTorch algorithms are enabled temporarily. Reproducibility assumes the same
initial weights, hardware, precision and deterministic dataset; custom dataset RNGs remain
the caller's responsibility. Persistent workers are rejected because their RNG state
cannot be reset here. On CUDA, configure `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process
startup where required by deterministic matrix multiplication. AMP uses CUDA float16 plus
GradScaler, or CPU bfloat16. Nonfinite losses/gradients raise rather than silently skipping
an optimizer update. CUDA/AMP paths are implemented but not GPU-validated by the CPU tests.

## Evaluation and orchestration

`evaluate_adapter(model, adapter, layer, loader, device)` returns per-image corrupted
and clean rows with `input_type`, `image_id`, `label`, `layer_id`, `corruption`, `severity`,
baseline/post predictions, true-label margins and CE losses, plus unadapted
`clean_prediction`. Clean rows use the unadapted clean baseline, `corruption="clean"`,
and severity zero. Corrupted metadata comes from mapping batches or T01's `dataset.spec`;
otherwise it is explicitly `unknown`/null. Empty evaluation returns an empty DataFrame.
Margins subtract the largest competing logit from the true-label logit. Labels and clean
inputs are used to score results, never as adapter inputs. Each inference is independent.
All convenience APIs remove their own hooks even on exceptions and preserve unrelated hooks.
Call them with no pre-attached intervention when an unmodified baseline is required.

`train_selected_site(model, layer, width, seed, fit_loader, val_loader, cfg_base, device,
out_dir, *, selection_cost=0)` trains exactly one externally selected site. It returns
a one-row run table. `train_all_sites(model, layers, widths, seeds, fit_loader, val_loader,
cfg_base, device, out_dir, *, selection_cost=0)` trains the exhaustive Cartesian product
and returns a run table. Candidate lists must be nonempty and unique. Each run writes
`layer_L_width_W_seed_S/adapter.pt` (CPU state_dict), `validation.csv`, and `training.json`
(config, cost, loss history). The exhaustive call also writes `reference.csv`.
Repeated identical run identifiers overwrite those artifacts; use separate output roots
for distinct experiments. Reiterable fit/validation loaders are required for sweeps.

The evaluation module's accuracy helpers compute corrupted `U`, clean accuracy change
and `clean_drop_pp`. These are validation summaries, not automatic acceptance decisions.
Tune hyperparameters/clean-drop tolerance on validation only. Enforce admissibility in
the selector and report any held-out clean constraint violations without filtering them out.

## Compute accounting

A single unadapted profiling forward records actual token/channel shapes with T01's
`BlockOutputRecorder`. For a block with N tokens, D channels and M MLP hidden units,
forward FLOPs are approximated by `8*N*D^2 + 4*N^2*D + 4*N*D*M + 10*N*D`.
Patch convolution uses `2*patches*out_channels*in_channels*kernel_area`;
head/final normalization use `2*D*num_classes + 5*N*D`.
Full forward includes stem, every block, head and adapter. Frozen downstream input-gradient
backward is approximated as one forward-equivalent for each block strictly after the
insertion block and the head, plus twice adapter forward FLOPs for adapter backward.
The insertion block itself has no backward: its output precedes the adapter.
This refines the task's inclusive `blocks >= layer` shorthand for an after-block hook.

`estimated_training_flops`/`training_compute` multiply forward-plus-backward by
`input_views_seen`; `images_seen` counts paired examples, including repeats, and each
pair costs one or two views depending on clean weight. This is a shape-based estimate,
not an instruction profiler: bias, activation, softmax, CE, optimizer and data operations
are omitted. Profiling overhead is reported separately as `profiling_forward_flops`.
`wall_time_seconds` includes loader consumption and profiling; CUDA timing synchronizes
and `peak_memory_bytes` measures peak allocated memory including the resident model.

`selection_cost` is passed through in caller-defined units (use FLOPs for comparison).
`training_compute` always describes only the row's site. `reference_total_compute` is
null for a standalone selected run, and the sum across all runs in the reference table.
Per-run training JSON remains a per-site record. Do not sum repeated reference totals,
or charge exhaustive reference training as if it were required by a realistic selector.
Equal updates/exposure do not imply equal FLOPs.

## Validation

Run `C:/anaconda/envs/layer-research/python.exe -m pytest -q tests/test_adapter_training.py`.
As in T01/T02, prepend the conda environment and its `Library/bin` to process PATH and
set `MKL_THREADING_LAYER=SEQUENTIAL` in this Windows environment. Tests use a tiny random
three-block ViT on CPU. They cover exact identity, parameter counts, frozen weights,
first/last-site gradients, loss reduction, determinism, objective components, clean rows,
margin calculations, hook cleanup including exceptions, and selected/reference costs.
Checkpoint and CSV serialization round-trip in memory to avoid deleting the shared
`.pytest_tmp` directory, which pytest cannot remove in this sandbox. No alternate
repository temporary directories are created.
