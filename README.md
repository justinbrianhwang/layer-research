# From Representation Change to Repair

**Budget-Constrained Intervention Site Selection in Vision Models**

> Does the layer where a corrupted image's internal representation changes the most also turn out
> to be the layer where a *small, budget-limited* intervention restores the prediction the most?
> And does that choice survive unseen corruptions and a real, learned repair module?

This repository holds the code, experiment protocol, and running results for the research proposal
in [`vision_intervention_site_selection_proposal.md`](vision_intervention_site_selection_proposal.md)
(English). The proposal is the source of truth for every design decision; this README summarizes
it and tracks progress.

![Pipeline overview: paired clean/corrupted inputs, frozen ViT with block-boundary probes, two per-block measurements (representation change and budget-limited intervention), site selection fixed before deployment, learned adapter at the selected block, and evaluation by regret, unseen-corruption generalization, and diagnostic-vs-adapter ranking](assets/figs/pipeline.png)

*Figure 1. The full pipeline. Representation change is measured at every block boundary; a fixed-budget
partial patch is applied at every block with the same rule; one block is chosen per model before any
test data is touched; a small learned adapter is then trained at that block with the backbone frozen.*

---

## Status

| Stage | Description | State | Updated |
|---|---|---|---|
| S0 | Project skeleton, conda env, Vast.ai workflow, data plan (ImageNetV2) | done | 2026-09-25 |
| T01 | `data_protocol`, `feature_extractor`, `representation_metrics` + tests | done (15 tests) | 2026-09-25 |
| T02 | `patching_engine` + controls (proposal §8.5) | done (30 tests) | 2026-09-25 |
| T03 | `evaluation`, `statistics`, `site_selection` | done (9 tests; 54 total) | 2026-09-25 |
| T04 | `adapter_training` | done (9 tests; 63 total) | 2026-09-25 |
| T05 | experiment scripts + CPU smoke run + GPU-readiness fixes | done (65 tests; real DeiT-S smoke chain verified on CPU) | 2026-09-25 |
| E1 | Representation change vs. partial-patch recovery | done (val + test sweeps, 20 seeds, 12 conditions) | 2026-09-26 |
| E2 | Budget dependence of site ranking | done for the channel budget q ∈ {0.01, 0.05, 0.10, 0.20} at α = 1 (α grid and norm cap not run) | 2026-09-26 |
| E3 | Generalization to unseen corruptions | done (contrast, JPEG held out) | 2026-09-26 |
| E4 | Diagnostic patching vs. learned adapter | done (width 32 × 3 seeds; widths 8/64 not run) | 2026-09-26 |
| E5 | ResNet re-validation | pending | |
| E6 | Extended metrics (task sensitivity, kNN, TDA) | optional | |

Results are appended to the [Results](#results) section as they land.

---

## The question in one paragraph

Take a fixed, pretrained ViT (non-distilled DeiT-S/16, 12 blocks, $d=384$). Feed it a clean image and
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

Write the clean image as $x$, its corrupted version as $\tilde{x}=T_{c,s}(x)$, the network up to block $l$
as $h_l$ and the rest of the network as $g_l$, so that $F(x)=g_l(h_l(x))$.

**Diagnostic partial patching (privileged).** At block $l$, move the corrupted representation part of
the way toward the clean one on a fixed subset of channels and let the rest of the model run unchanged:

$$
\widehat{h}_l = h_l(\tilde{x}) + \alpha\, M_{l,q} \odot \big[h_l(x) - h_l(\tilde{x})\big]
$$

where $M_{l,q}$ is a channel mask selecting $k=\lfloor qd 
floor$ of the $d$ channels for every token,
$q \in \{0.01, 0.05, 0.10, 0.20\}$ is the channel budget and $\alpha \in \{0.25, 0.5, 1.0\}$ the strength.
Masks are random but fixed per seed and nested across budgets. This *uses the clean representation*
and is a diagnostic, not a defense. Full-state replacement ($M=\mathbf{1}$, $\alpha=1$) trivially gives
$g_l(h_l(x)) = F(x)$ and is used only as a positive control.

**Learned residual adapter (deployable).** Insert one zero-initialized bottleneck MLP after block $l$,
freeze the backbone and classifier, and train only the adapter:

$$
\widehat{h}_l = h_l(\tilde{x}) + A_{\phi,l}\big(h_l(\tilde{x})\big), \qquad
A_{\phi,l}(h) = W_{\mathrm{up}}\, \sigma\big(W_{\mathrm{down}}\, \mathrm{LN}(h)\big)
$$

with the training objective

$$
\mathcal{L}_{\mathrm{adapt}} = \mathbb{E}_{(x,y),c,s}\Big[\ell\big(F_{l,\phi}(T_{c,s}(x)),y\big)
+ \lambda_{\mathrm{clean}}\, \ell\big(F_{l,\phi}(x),y\big)\Big].
$$

At test time the adapter sees only the corrupted image. Bottleneck widths $r \in \{8, 32, 64\}$.

## Primary evaluation: selection regret

The effect of an intervention at block $l$ under budget $B$ is the accuracy change in percentage points,

$$
U_l(B) = 100\,\big[\mathrm{Acc}(F_l^{\mathrm{patch}}) - \mathrm{Acc}(F)\big], \qquad U(\varnothing)=0,
$$

where $\varnothing$ denotes no intervention. For a selection rule $m$ that picks a site
$\widehat{l}_m(B)$ using only observed data, regret on the test set is the gap to the best admissible
candidate $\mathcal{A}_{\mathrm{val}}(B) \subseteq \mathcal{L}\cup\{\varnothing\}$:

$$
\mathrm{Regret}_m(B) = \max_{a \in \mathcal{A}_{\mathrm{val}}(B)} U_{\mathrm{test}}(a,B) - U_{\mathrm{test}}\big(\widehat{l}_m(B),B\big).
$$

The maximum is a post-hoc reference computed after all candidates are evaluated; no selector ever
sees it. Baselines: no intervention, fixed front / middle / back block, random block, metric-based
selection, small direct search, and full validation-set sweep. Sites whose effects differ by less than a
pre-set tolerance $\tau$ are reported as an equivalence set
$\mathcal{L}_\tau(B)=\{\,l : \max_j U_j(B) - U_l(B) \le \tau\,\}$ rather than as a single "critical layer".
Uncertainty is a paired bootstrap over original image ids, with mask seeds and training seeds reported separately.

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

Results are appended as experiments finish, newest last. Generated-corruption evaluation on ImageNetV2
(see [Data protocol](#data-protocol)); these are not official ImageNet-C numbers.

### Experiment A: layer-wise representation change (done, 2026-09-25)

Score split (2 000 ImageNetV2 images, 2 per class), CLS-token summary, severity 3, DeiT-S/16 with
fp32 activations. Full table for all severities, both summaries and every metric:
[`results/tables/metrics_score.csv`](results/tables/metrics_score.csv).

**Relative distance** $\|h_l(x)-h_l(	ilde x)\| / \|h_l(x)\|$ grows monotonically with depth for every corruption,
so "largest change" always points at the last block.

| block | gaussian_noise | defocus_blur | contrast | jpeg_compression |
|---|---|---|---|---|
| 0 | 0.033 | 0.027 | 0.043 | 0.008 |
| 1 | 0.064 | 0.069 | 0.064 | 0.026 |
| 2 | 0.088 | 0.104 | 0.085 | 0.065 |
| 3 | 0.104 | 0.126 | 0.091 | 0.119 |
| 4 | 0.123 | 0.155 | 0.099 | 0.133 |
| 5 | 0.159 | 0.202 | 0.110 | 0.177 |
| 6 | 0.213 | 0.270 | 0.130 | 0.230 |
| 7 | 0.277 | 0.350 | 0.167 | 0.290 |
| 8 | 0.370 | 0.474 | 0.215 | 0.380 |
| 9 | 0.477 | 0.611 | 0.269 | 0.483 |
| 10 | 0.504 | 0.636 | 0.329 | 0.489 |
| 11 | 0.649 | 0.777 | 0.452 | 0.593 |

**1 − linear CKA** does not: for contrast it is largest in the first two blocks and smallest around block 6,
while for blur and noise it is largest at the end. The two metrics therefore disagree on *where* the
representation changes most, which is exactly the ambiguity the selection experiments are meant to resolve.

| block | gaussian_noise | defocus_blur | contrast | jpeg_compression |
|---|---|---|---|---|
| 0 | 0.097 | 0.230 | 0.328 | 0.006 |
| 1 | 0.193 | 0.232 | 0.293 | 0.046 |
| 2 | 0.240 | 0.189 | 0.197 | 0.165 |
| 3 | 0.226 | 0.201 | 0.153 | 0.208 |
| 4 | 0.216 | 0.202 | 0.130 | 0.204 |
| 5 | 0.206 | 0.208 | 0.105 | 0.169 |
| 6 | 0.213 | 0.232 | 0.090 | 0.161 |
| 7 | 0.240 | 0.287 | 0.108 | 0.197 |
| 8 | 0.293 | 0.357 | 0.121 | 0.254 |
| 9 | 0.308 | 0.428 | 0.115 | 0.275 |
| 10 | 0.295 | 0.465 | 0.149 | 0.253 |
| 11 | 0.370 | 0.556 | 0.258 | 0.316 |

**kNN preservation** (10-NN Jaccard between clean and corrupted neighbourhoods; higher is more stable):

| block | gaussian_noise | defocus_blur | contrast | jpeg_compression |
|---|---|---|---|---|
| 0 | 0.191 | 0.189 | 0.108 | 0.571 |
| 1 | 0.199 | 0.204 | 0.219 | 0.450 |
| 2 | 0.172 | 0.216 | 0.277 | 0.279 |
| 3 | 0.189 | 0.223 | 0.295 | 0.247 |
| 4 | 0.182 | 0.209 | 0.313 | 0.230 |
| 5 | 0.198 | 0.201 | 0.360 | 0.246 |
| 6 | 0.198 | 0.185 | 0.368 | 0.262 |
| 7 | 0.204 | 0.166 | 0.370 | 0.256 |
| 8 | 0.195 | 0.146 | 0.349 | 0.237 |
| 9 | 0.208 | 0.133 | 0.362 | 0.238 |
| 10 | 0.211 | 0.137 | 0.333 | 0.255 |
| 11 | 0.161 | 0.107 | 0.235 | 0.198 |



### Experiments B, C, E (E1–E3): diagnostic partial patching, site selection, unseen corruptions (done, 2026-09-26)

Test split, 2 000 images, α = 1, channel masks random-fixed and nested across budgets, 20 mask seeds,
all 12 corruption conditions (4 types × severities 1/3/5). Accuracy change $U$ in percentage points versus
the un-patched corrupted input, averaged over mask seeds and conditions (image-level paired bootstrap for
CIs). Tables: `results/tables/E1_effects.csv` (every condition × site × seed), `E1_seed_variability.csv`,
`E3_observed.csv`, `E3_unseen.csv`, `E1_test_U_by_layer_domain_fraction.csv`.

**Controls (all conditions pooled, validation split).** Full-state replacement and clean-to-clean
replacement reproduce the clean predictions exactly (+17.2 pp, the full clean–corrupted gap);
no-intervention gives 0.00; a random direction with the same per-image norm gives −0.3 to +0.1 pp;
a shuffled donor gives −0.4 to +0.2 pp. The recovery below therefore comes from the clean *direction*,
not from the perturbation size.

**Per-site recovery on the test split** (mean over seeds and conditions):

| domain | q | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| observed | 0.01 | +0.29 | +0.21 | +0.11 | +0.16 | +0.20 | +0.22 | +0.26 | +0.25 | +0.25 | +0.26 | +0.23 | +0.21 |
| observed | 0.05 | +1.42 | +1.21 | +1.11 | +1.29 | +1.56 | +1.56 | +1.60 | +1.59 | +1.67 | +1.86 | +1.86 | +1.61 |
| observed | 0.1 | +1.64 | +1.93 | +2.14 | +2.72 | +3.23 | +3.32 | +3.28 | +3.36 | +3.50 | +3.96 | +4.01 | +3.47 |
| observed | 0.2 | +1.40 | +3.98 | +4.38 | +5.51 | +6.56 | +6.73 | +6.76 | +6.89 | +7.22 | +8.23 | +8.65 | +7.49 |
| unseen | 0.01 | -0.04 | +0.14 | +0.12 | +0.14 | +0.17 | +0.16 | +0.20 | +0.20 | +0.18 | +0.20 | +0.21 | +0.19 |
| unseen | 0.05 | -0.68 | +0.74 | +0.89 | +1.05 | +1.17 | +1.01 | +1.07 | +1.11 | +1.06 | +1.17 | +1.18 | +1.12 |
| unseen | 0.1 | -2.23 | +1.25 | +1.50 | +1.92 | +2.17 | +2.08 | +2.11 | +2.18 | +2.18 | +2.37 | +2.36 | +2.22 |
| unseen | 0.2 | -2.80 | +2.28 | +2.81 | +3.47 | +3.96 | +3.93 | +4.08 | +4.23 | +4.33 | +4.68 | +4.85 | +4.33 |

**Selector regret on the test split**, for the frozen selections made on observed corruptions
(score/val splits only). Regret is measured against the best admissible site under the same budget;
"unseen" evaluates the *same* frozen site on contrast + JPEG, which were never used for selection.

*Budget q = 0.20 (77 of 384 channels):*

| selector | site | observed U (95% CI) | observed regret | unseen U (95% CI) | unseen regret |
|---|---|---|---|---|---|
| fixed_front | 0 | +1.40 (+0.92, +1.89) | 7.25 | -2.80 (-3.25, -2.39) | 7.64 |
| fixed_middle | 6 | +6.76 (+6.33, +7.18) | 1.89 | +4.08 (+3.67, +4.46) | 0.77 |
| fixed_back | 11 | +7.49 (+7.03, +7.98) | 1.16 | +4.33 (+3.91, +4.74) | 0.52 |
| random | 10 | +8.65 (+8.14, +9.18) | 0.00 | +4.85 (+4.41, +5.29) | 0.00 |
| small_search | 11 | +7.49 (+7.03, +7.98) | 1.16 | +4.33 (+3.91, +4.74) | 0.52 |
| val_sweep | 10 | +8.65 (+8.14, +9.18) | 0.00 | +4.85 (+4.41, +5.29) | 0.00 |
| relative_distance/cls | 11 | +7.49 (+7.03, +7.98) | 1.16 | +4.33 (+3.91, +4.74) | 0.52 |
| cosine_distance/cls | 11 | +7.49 (+7.03, +7.98) | 1.16 | +4.33 (+3.91, +4.74) | 0.52 |
| one_minus_cka/cls | 11 | +7.49 (+7.03, +7.98) | 1.16 | +4.33 (+3.91, +4.74) | 0.52 |
| knn_preservation/cls | 11 | +7.49 (+7.03, +7.98) | 1.16 | +4.33 (+3.91, +4.74) | 0.52 |
| amplification_ratio/cls | 0 | +1.40 (+0.92, +1.89) | 7.25 | -2.80 (-3.25, -2.39) | 7.64 |
| relative_distance/patch_mean | 11 | +7.49 (+7.03, +7.98) | 1.16 | +4.33 (+3.91, +4.74) | 0.52 |
| one_minus_cka/patch_mean | 10 | +8.65 (+8.14, +9.18) | 0.00 | +4.85 (+4.41, +5.29) | 0.00 |
| amplification_ratio/patch_mean | 9 | +8.23 (+7.73, +8.73) | 0.42 | +4.68 (+4.26, +5.10) | 0.17 |

*Budget q = 0.05 (19 channels):*

| selector | site | observed U (95% CI) | observed regret | unseen U (95% CI) | unseen regret |
|---|---|---|---|---|---|
| fixed_front | 0 | +1.42 (+1.13, +1.70) | 0.44 | -0.67 (-0.94, -0.40) | 1.85 |
| fixed_middle | 6 | +1.60 (+1.37, +1.81) | 0.27 | +1.07 (+0.89, +1.26) | 0.10 |
| fixed_back | 11 | +1.61 (+1.42, +1.81) | 0.25 | +1.12 (+0.94, +1.31) | 0.05 |
| random | 10 | +1.86 (+1.65, +2.07) | 0.00 | +1.18 (+0.98, +1.37) | 0.00 |
| small_search | 11 | +1.61 (+1.42, +1.81) | 0.25 | +1.12 (+0.94, +1.31) | 0.05 |
| val_sweep | 9 | +1.86 (+1.64, +2.07) | 0.00 | +1.17 (+0.99, +1.36) | 0.00 |
| relative_distance/cls | 11 | +1.61 (+1.42, +1.81) | 0.25 | +1.12 (+0.94, +1.31) | 0.05 |
| cosine_distance/cls | 11 | +1.61 (+1.42, +1.81) | 0.25 | +1.12 (+0.94, +1.31) | 0.05 |
| one_minus_cka/cls | 11 | +1.61 (+1.42, +1.81) | 0.25 | +1.12 (+0.94, +1.31) | 0.05 |
| knn_preservation/cls | 11 | +1.61 (+1.42, +1.81) | 0.25 | +1.12 (+0.94, +1.31) | 0.05 |
| amplification_ratio/cls | 0 | +1.42 (+1.13, +1.70) | 0.44 | -0.67 (-0.94, -0.40) | 1.85 |
| relative_distance/patch_mean | 11 | +1.61 (+1.42, +1.81) | 0.25 | +1.12 (+0.94, +1.31) | 0.05 |
| one_minus_cka/patch_mean | 10 | +1.86 (+1.65, +2.07) | 0.00 | +1.18 (+0.98, +1.37) | 0.00 |
| amplification_ratio/patch_mean | 9 | +1.86 (+1.64, +2.07) | 0.00 | +1.17 (+0.99, +1.36) | 0.00 |

What the diagnostic sweep says:

- **Recovery grows with depth and with budget.** With q = 0.20 the best site (block 10) recovers
  8.7 pp of the 17.2 pp clean–corrupted gap on observed corruptions; block 0 recovers 1.4 pp and
  *hurts* unseen corruptions (−2.8 pp). The site ranking is nearly identical on unseen corruptions,
  so a site frozen on noise + blur transfers to contrast + JPEG (E3): regret on unseen ≤ 0.5 pp for
  every selector except those that choose block 0.
- **Budget dependence is about how much the choice matters, not about which site wins.** At q = 0.01
  every site is inside the 0.5 pp equivalence set (13 candidates including no intervention),
  at q = 0.05 it holds 9 candidates, at q ≥ 0.10 only blocks 9 and 10 remain
  (2 candidates). Metric-based choices only start to cost something at q ≥ 0.10.
- **Which metric you read matters more than whether you read one.** CLS-token distance, cosine,
  1 − CKA and kNN preservation all point at the last block (11), which costs 0.5 to 1.2 pp of
  regret at q ≥ 0.10 relative to block 10; 1 − CKA on patch-mean tokens picks block 10 (regret 0)
  and the patch-mean amplification ratio picks block 9 (regret ≤ 0.4 pp). The CLS amplification
  ratio picks block 0 and is the worst rule in the table (regret 7.3 pp at q = 0.20). A fixed
  "back" rule (block 11) is within 1.2 pp of optimal everywhere and beats every CLS-based metric
  at no cost. The validation sweep (all 12 sites tried on val) reaches zero regret at every budget.

### Diagnostic patching versus learned adapters (E4 link)

Spearman rank correlation between the per-site diagnostic recovery (partial patching, test split)
and the per-site adapter gain (width 32, 3-seed mean, test split):

| domain | q | Spearman ρ | p | best diagnostic site | best adapter site |
|---|---|---|---|---|---|
| observed | 0.05 | -0.94 | 6.99e-06 | 10 | 2 |
| observed | 0.1 | -0.90 | 8.37e-05 | 10 | 2 |
| observed | 0.2 | -0.92 | 1.86e-05 | 10 | 2 |
| unseen | 0.05 | +0.80 | 0.0019 | 10 | 10 |
| unseen | 0.1 | +0.92 | 2.84e-05 | 9 | 10 |
| unseen | 0.2 | +0.98 | 3.09e-08 | 10 | 10 |

- On observed corruptions the two rankings are **strongly anti-correlated**: diagnostic patching says
  "repair late" (block 10), the learned adapter says "repair early" (blocks 1 to 2). The privileged
  clean-direction diagnostic is therefore *not* a usable proxy for where a deployable adapter should go
  (hypothesis H4 in the proposal, in its strong form).
- On unseen corruptions the rankings agree in sign: late sites are best for both, because early
  adapters over-fit the observed blur and late sites generalise. The site that survives a
  clean-accuracy constraint (block 10, see Experiment D) is also the best diagnostic site.
- A representation metric read on the CLS token would have pointed at block 11 for both procedures;
  that is near-optimal for the diagnostic (regret ≤ 1.2 pp) and near-zero gain for the adapter.

### Experiment D: exhaustive adapter reference, 3 training seeds (done, 2026-09-26)

One zero-initialised bottleneck adapter (width 32, 25 760 parameters) trained after each of the 12 blocks
on the fit split with observed corruptions only (Gaussian noise, defocus blur; severities 1/3/5),
1 000 AdamW updates, batch 32, $\lambda_{\mathrm{clean}}=1$, backbone frozen, three training seeds per
site. Accuracy change in percentage points on the **test split** relative to the un-adapted model
(baseline accuracy: observed corruptions 45.6 %, unseen 58.4 %), mean ± std over seeds; per-corruption
columns are seed means. Tables: `results/tables/E4_adapter_test_by_site_3seeds.csv`,
`results/tables/E4_adapter_test_by_site_corruption_3seeds.csv`, costs in `results/tables/adapter_costs.csv`.

| site | observed (noise+blur) | unseen (contrast+JPEG) | clean | gaussian_noise | defocus_blur | contrast | jpeg_compression |
|---|---|---|---|---|---|---|---|
| 0 | +4.93 ± 0.41 | -5.95 ± 0.80 | -2.95 ± 0.18 | +2.72 | +7.14 | -8.08 | -3.82 |
| 1 | +6.10 ± 0.15 | -6.72 ± 0.54 | -3.33 ± 0.73 | +3.38 | +8.83 | -8.88 | -4.56 |
| 2 | +6.20 ± 0.11 | -5.44 ± 0.57 | -3.58 ± 0.35 | +3.29 | +9.12 | -6.54 | -4.33 |
| 3 | +5.18 ± 0.16 | -4.19 ± 0.33 | -3.57 ± 0.31 | +2.25 | +8.12 | -4.84 | -3.53 |
| 4 | +4.05 ± 0.31 | -2.09 ± 0.64 | -3.00 ± 0.41 | +1.21 | +6.90 | -3.47 | -0.72 |
| 5 | +2.38 ± 0.32 | -1.58 ± 0.61 | -3.18 ± 0.60 | -0.42 | +5.17 | -1.96 | -1.21 |
| 6 | +1.09 ± 0.12 | -1.00 ± 0.19 | -2.88 ± 0.23 | -1.36 | +3.53 | -1.42 | -0.57 |
| 7 | +0.89 ± 0.45 | -0.94 ± 0.11 | -3.00 ± 0.31 | -1.14 | +2.92 | -0.98 | -0.89 |
| 8 | +0.80 ± 0.18 | -0.12 ± 0.12 | -2.50 ± 0.17 | -0.87 | +2.47 | +0.18 | -0.43 |
| 9 | +0.40 ± 0.33 | -0.14 ± 0.09 | -1.67 ± 0.23 | -1.02 | +1.83 | +0.01 | -0.29 |
| 10 | +0.77 ± 0.08 | +0.51 ± 0.34 | -1.33 ± 0.20 | -0.43 | +1.97 | +0.56 | +0.47 |
| 11 | +0.22 ± 0.17 | -0.83 ± 0.07 | -1.73 ± 0.21 | -0.83 | +1.27 | -0.67 | -0.98 |

Seed variability is small (std ≤ 0.8 pp), so the pattern is stable:

- **The best observed-corruption site is the worst unseen-corruption site.** Blocks 1 to 2 gain
  6 pp on noise + blur (driven by defocus blur, +9 pp) and lose 5 to 7 pp on contrast + JPEG and
  3 to 4 pp of clean accuracy. Late sites (block 10) gain less than 1 pp but are the only sites that
  are non-negative on every corruption family.
- **The realistic selection rule is decided by the clean-accuracy tolerance, not by the metric.**
  Selecting on the validation split with the proposal's rule (best observed gain among sites whose
  clean drop is within $arepsilon_{\mathrm{clean}}$):

| tolerance $arepsilon_{\mathrm{clean}}$ (pp) | admissible sites (val) | chosen | test observed U | test unseen U |
|---|---|---|---|---|
| 0.5 | none | no intervention | 0.00 | 0.00 |
| 1.5 | 10, 11 | 10 | +0.77 | +0.51 |
| 3.0 | 9, 10, 11 | 10 | +0.77 | +0.51 |
| 5.0 | all but 5 | 1 | +6.10 | −6.72 |

- Training cost falls monotonically with depth (a block-0 adapter needs about 1.9× the training
  FLOPs of a block-11 adapter) because gradients must flow through every downstream block.
- Comparison with the diagnostic patching sweep follows once the test sweep finishes; on the
  validation split, diagnostic partial patching favours the *last* blocks (9 to 11), which is the
  opposite end of the network from the adapter optimum on observed corruptions.

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
