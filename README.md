# From Representation Change to Repair

**Budget-Constrained Intervention Site Selection in Vision Models**

> Does the layer where a corrupted image's internal representation changes the most also turn out
> to be the layer where a *small, budget-limited* intervention restores the prediction the most?
> And does that choice survive unseen corruptions and a real, learned repair module?

This repository holds the code, experiment protocol, and running results for the research proposal
in [`vision_intervention_site_selection_proposal.md`](vision_intervention_site_selection_proposal.md)
(Korean, with English title and formulas). The proposal is the source of truth for every design
decision; this README summarizes it and tracks progress.

![Pipeline overview: paired clean/corrupted inputs, frozen ViT with block-boundary probes, two per-block measurements (representation change and budget-limited intervention), site selection fixed before deployment, learned adapter at the selected block, and evaluation by regret, unseen-corruption generalization, and diagnostic-vs-adapter ranking](assets/figs/pipeline.png)

*Figure 1. The full pipeline. Representation change is measured at every block boundary; a fixed-budget
partial patch is applied at every block with the same rule; one block is chosen per model before any
test data is touched; a small learned adapter is then trained at that block with the backbone frozen.*

---

## Status

| Stage | Description | State | Updated |
|---|---|---|---|
| S0 | Project skeleton, conda env, Vast.ai workflow | done | 2026-09-25 |
| T01 | `data_protocol`, `feature_extractor`, `representation_metrics` + tests | in progress (Codex) | 2026-09-25 |
| T02 | `patching_engine` + controls (proposal §8.5) | pending | |
| T03 | `evaluation`, `statistics`, `site_selection` | pending | |
| T04 | `adapter_training` | pending | |
| E1 | Representation change vs. partial-patch recovery | pending | |
| E2 | Budget dependence of site ranking | pending | |
| E3 | Generalization to unseen corruptions | pending | |
| E4 | Diagnostic patching vs. learned adapter | pending | |
| E5 | ResNet re-validation | pending | |
| E6 | Extended metrics (task sensitivity, kNN, TDA) | optional | |

Results are appended to the [Results](#results) section as they land.

---

## The question in one paragraph

Take a fixed, pretrained ViT (non-distilled DeiT-S/16, 12 blocks, d=384). Feed it a clean image and
its corrupted version (ImageNet-C style noise, blur, contrast, JPEG). At every block boundary we can
measure *how much the representation changed* (normalized distance, cosine distance, linear CKA,
neighbourhood preservation). Separately, at every block we can *intervene* with a fixed budget and
measure how much accuracy comes back. The research question is whether the first kind of number
predicts the second, how that depends on the budget, whether a site chosen on observed corruptions
still works on unseen ones, and whether a site that looks good under a privileged diagnostic
intervention is also the right place to put a small learned adapter at deployment time.

The contribution is **not** a new critical layer or a new metric. It is an honest evaluation of
whether representation metrics are a trustworthy basis for the decision "where do I repair".

## Three things kept strictly separate

| Concept | What is measured | What it is not |
|---|---|---|
| Representation change | distance / CKA / kNN between clean and corrupted features per block | proof that the change hurts the prediction |
| Effect on prediction | accuracy, margin, loss before vs. after a specified intervention | proof of the only failure cause |
| Repairability | recovery under a fixed intervention type and budget | an absolute optimum over all possible repairs |

## Two experiment families

**Diagnostic partial patching (privileged).** At block *l*, move the corrupted representation part
of the way toward the clean one on a fixed subset of channels:

`ĥ_l = h_l(x̃) + α · M_{l,q} ⊙ [h_l(x) − h_l(x̃)]`

with channel fraction `q ∈ {0.01, 0.05, 0.10, 0.20}` and strength `α ∈ {0.25, 0.5, 1.0}`. Masks are
random but fixed per seed and nested across budgets. This *uses the clean representation* and is a
diagnostic, not a defense. Full-state replacement is used only as a positive control.

**Learned residual adapter (deployable).** Insert one zero-initialized bottleneck MLP after block *l*,
freeze the backbone and classifier, train only the adapter on observed corruptions plus a clean-
preservation term. At test time it sees only the corrupted image. Bottleneck widths `r ∈ {8, 32, 64}`.

## Primary evaluation: selection regret

For a selection rule *m* that picks a site using only observed data, regret on the test set is the
gap between the best admissible candidate (including "no intervention") and the candidate *m* chose.
Baselines: no intervention, fixed front / middle / back block, random block, metric-based selection,
small direct search, and full validation-set sweep. Uncertainty is a paired bootstrap over original
image ids, with mask seeds and training seeds reported separately.

## Data protocol

| Split | Role |
|---|---|
| `fit` | train adapters |
| `score` | compute representation metrics and fixed masks |
| `val` | choose metric direction, hyper-parameters, clean-accuracy tolerance, final site |
| `test` | evaluate the frozen rule once; never used to revise the rule |

All splits are by **original image id**, so a clean image and all of its corrupted variants stay in
the same split. Observed corruptions: Gaussian noise, defocus blur. Unseen: contrast, JPEG.

**Practical deviation from the proposal (§6.3):** the proposal recommends drawing the calibration
splits from ImageNet *train* (~140 GB). On an hourly GPU rental this is impractical, so the 50 000
ImageNet *validation* images are split by id instead and corruptions are generated with the
ImageNet-C reference code. Numbers are therefore reported as *generated-corruption* evaluation, not
official ImageNet-C scores. See [`docs/vast_workflow.md`](docs/vast_workflow.md).

## Repository layout

```
src/layer_research/
  data_protocol.py           id-based splits, corruption specs, paired dataset
  feature_extractor.py       block-boundary hooks, CLS / patch-mean summaries, shape report
  representation_metrics.py  relative distance, cosine, linear CKA, amplification, kNN preservation
  patching_engine.py         mask policies, partial patching, norm cap, control conditions
  adapter_training.py        frozen backbone, residual bottleneck adapter, compute logging
  site_selection.py          metric / fixed / random / val-sweep selectors
  evaluation.py              accuracy, recovery & new-error sets, regret, unseen evaluation
  statistics.py              paired bootstrap, seed variability, selection stability
configs/                     experiment YAMLs (model, corruptions, budgets, seeds)
scripts/                     cache_features.py, run_patching.py, train_adapters.py, analyze.py
tests/                       CPU-only unit tests (no downloads)
docs/                        Vast.ai workflow, per-task specs for the coding agent, module notes
results/                     tables and figures (raw tensors are not committed)
```

## Setup

```bash
conda env create -f environment.yml
conda activate layer-research
pip install -e .
pytest -q
```

GPU runs use a rented single RTX 3090 on Vast.ai; the exact commands are in
[`docs/vast_workflow.md`](docs/vast_workflow.md). Everything in `tests/` runs on CPU.

## How this repo is developed

Planning, task specification, review, and integration are done by a project-manager agent
(Claude). Module implementation is delegated task-by-task to a coding agent (OpenAI Codex CLI);
each task spec lives in `docs/codex_tasks/`. Every delivered module is checked against the
pre-run verification list in proposal §18.3 (no-intervention reproduction, full-state control,
hook removal, gradient path, split disjointness) before it is used on GPU. The human author
reviews and owns all commits.

## Results

*No experimental results yet. This section is updated after each experiment, with tables rather
than prose, and with the date and commit of the run.*

## Reproducibility

Every result row records: experiment id, checkpoint hash, original image id and split, corruption
type / severity / seed, block, metric name and data split used for scoring, mask policy and seed,
channel count and effective ratio, α, norm cap, actual δ-norm, adapter width and parameter count,
training seed, baseline / clean / post-intervention prediction and margin, and selection, training,
and inference cost (proposal §18.2).

## References

Key references are listed at the end of the proposal: CKA (Kornblith et al., 2019), ViT vs CNN
representations (Raghu et al., 2021), topology of DNNs (Naitzat et al., 2020), representation
similarity grounding (Ding et al., 2021; Klabunde et al., 2025), CLAT (Gopal et al., 2025),
TopoLip (Chen, 2024), NeuroShield-ViT (Islam et al., 2025), Suppress & Diversify (Yang et al.,
2026), activation-patching practice (Zhang & Nanda, 2023; Heimersheim & Nanda, 2024), ImageNet-C
(Hendrycks & Dietterich, 2019), DeiT (Touvron et al., 2021), Shallow-Deep Networks (Kaya et al., 2019).
