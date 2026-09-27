# Does Representation Change Predict Where to Repair?
## Reliability of Layer Selection in Vision Models under a Limited Intervention Budget

**English title:** From Representation Change to Repair: Budget-Constrained Intervention Site Selection in Vision Models  
**Document type:** Research proposal — research questions, testable hypotheses, and experimental design  
**Subject:** Block-wise representations in image classification models and robustness to natural image corruptions  
**Literature check date:** September 24, 2026

> **Core question**  
> Which layer-wise representation metrics go beyond identifying large representation changes to predict effective repair sites under a limited modification budget, and do these choices remain effective for unseen corruptions and actual learned adaptation?

---

## 1. Research Overview

This study analyzes the relationship between **layer-wise representation changes** observed in vision models and **the recovery achieved through actual modification**. We measure how internal representations differ as clean and corrupted images pass through a model, apply limited interventions using the same rule at each layer, and compare the resulting recovery in predictive performance.

The central aim is to **evaluate how reliably existing representation metrics inform the decision of “where to repair,”** rather than to declare a new vulnerable layer. We examine correlations between layer-wise distances or similarities and recovery effects, but ultimately evaluate how much recovery performance is lost because of the site selected by a metric.

To this end, we clearly distinguish two types of experiments. The first is **diagnostic partial patching**, which supplies part of the clean representation of the same image. The second is **actual learned repair**, which applies a small adapter in a test environment where clean inputs are unavailable. By testing whether effective diagnostic sites are also favorable for actual adaptation, we examine the connection between internal representation analysis and usable repair methods.

The initial scope is limited to one fixed ViT-family classifier, single-block interventions, and natural image corruptions. A ResNet-family model is used for validation on another architecture. TDA is not a mandatory component; it is introduced as an extension only if it provides additional value for repair site selection beyond distance, similarity, and neighborhood-structure metrics.

**The hypotheses and numerical settings in this document are pre-experimental proposals. Performance improvements, the superiority of particular layers, and claims of research priority are not assumed to be established results.**

## 2. Background and Motivation

### 2.1 Observing Layer-Wise Representations Is Insufficient

Prior work has compared neural representation similarity, analyzed layer-wise structures in CNNs and ViTs, and studied topological changes in data as it passes through layers. CKA, CNN–ViT representation comparisons, and analyses of topological changes in neural networks are starting points for this study, not novel contributions in themselves. [R1], [R2], [R3]

Robustness research has also proposed CLAT, which selects important layers for training; TopoLip, which connects topology and Lipschitz perspectives; NeuroShield-ViT, which analyzes perturbation propagation in ViTs and intervenes on early neurons; and S&D, which studies robust computational paths under natural corruptions. Thus, describing the work as “observing layer-wise perturbations and intervening at important layers” is insufficient to establish its distinctiveness. [R6], [R7], [R8], [R9]

Building on this prior work, we specifically evaluate **the conditions and limitations under which observational metrics can serve as criteria for intervention site selection**.

### 2.2 Three Concepts to Distinguish

| Concept | Core question | Measurement in this study | Interpretive limitation |
|---|---|---|---|
| **Representation change** | Where do representations of clean and corrupted inputs differ substantially? | Normalized distance, cosine distance, CKA, neighborhood-structure changes, etc. | A large change does not necessarily harm prediction. |
| **Effect on prediction** | How much does a specified internal intervention change the final decision? | Pre/post-intervention differences in accuracy, correct-class margin, and loss | This is the effect of the specified intervention, not proof of a unique cause of failure. |
| **Repairability** | Where is repair effective when the means and budget for modification are limited? | Budget-specific partial-patching and learned-adaptation effects | This depends on the intervention type, budget, training data, and optimization conditions. |

The primary question is **how well the first measurement predicts the third outcome**. The second provides an experimental means of linking observation to recovery.

### 2.3 A Large Representation Change May Not Identify a Good Repair Site

The following are possibilities to test, not phenomena established in advance.

A large representation change may occur in directions that are unimportant for final classification. Conversely, even a small change may substantially affect prediction if its direction is sensitive with respect to the decision boundary. Furthermore, a site may be effective when the clean representation is supplied externally, yet estimating the correction from the corrupted representation alone may be difficult.

We therefore do not equate the magnitude of representation differences, sensitivity of the output, and learnability of the actual correction. Treating the connection between representation similarity metrics and functional differences as a question requiring validation also links this study to prior metric-evaluation research. [R4], [R5]

## 3. Related Work and Positioning

| Study | Established main content | Use and distinction in this study |
|---|---|---|
| **CKA — Kornblith et al., ICML 2019** [R1] | Proposes a method for comparing neural representation similarity. | Used as a basic representation metric; its predictive value for repair sites is evaluated separately. |
| **Do Vision Transformers See Like Convolutional Neural Networks? — Raghu et al., NeurIPS 2021** [R2] | Compares internal representation structures and information transmission in CNNs and ViTs. | Provides context for interpreting architectural differences. Differences between two models are not generalized into laws for entire architecture families. |
| **Topology of Deep Neural Networks — Naitzat et al., JMLR 2020** [R3] | Analyzes layer-wise topological changes and simplification in the classification settings studied. | Supports avoiding the assumption that topological change itself implies failure or information loss. |
| **Grounding Representation Similarity — Ding et al., NeurIPS 2021** [R4] | Evaluates whether representation metrics are sensitive to functionally important changes and insensitive to irrelevant ones. | We do not claim to be the first study to connect representation metrics to function. |
| **ReSi — Klabunde et al., ICLR 2025** [R5] | Benchmarks multiple representation similarity metrics against explicit evaluation criteria. | Defines budget-constrained repair site selection as the evaluation task, distinct from general metric evaluation. |
| **CLAT — Gopal et al., ICML 2025** [R6] | Uses criticality to selectively fine-tune a subset of important layers. | A major competing study. We distinguish the original metric from a version adapted to natural corruptions. |
| **TopoLip — Chen, released in 2024** [R7] | Analyzes robustness by linking topological perspectives and Lipschitz continuity in layer-wise analysis. | Merely adding TDA is not treated as a distinguishing contribution. |
| **NeuroShield-ViT — Islam et al., released in 2025** [R8] | Analyzes adversarial perturbation propagation in ViTs and proposes interventions on vulnerable early neurons. | We do not claim that a discrepancy between the site of observed change and the site of effective intervention is an entirely new observation. |
| **S&D — Yang et al., listed as accepted at ICML 2026** [R9] | Analyzes and exploits robust paths for natural corruptions and compares application sites. | We exclude claims that prior work lacks interventions or site comparisons. |
| **Activation patching methodology** [R10], [R11] | Examines interpretive differences arising from intervention construction and evaluation criteria. | We specify intervention operators, masks, corruption procedures, and evaluation metrics. Observations in language models are not directly treated as facts about vision models. |

### 3.1 Proposed Axes of Distinction

The candidate contribution is to connect the following three axes within a single evaluation problem.

**Budget dependence:** Evaluate how effective sites and metric-based selection performance change with the fraction of modifiable channels, intervention strength, or adapter capacity.

**Cross-corruption generalization:** Fix a site selected on particular corruptions and evaluate its effectiveness on corruptions not used for selection. This is strictly distinguished from reselecting sites using unseen corruptions.

**Connection between diagnostics and actual adaptation:** Evaluate how well site rankings from diagnostic experiments using clean representations predict site rankings for adapters that receive only corrupted inputs at test time.

This combination defines the research scope to be tested. The literature review in this proposal does not establish that no identical setting exists; final originality claims will be adjusted based on the detailed settings and experimental results of directly competing methods.

### 3.2 Claims We Will Not Make

We will not claim “the first layer-wise representation analysis,” “the first discovery of a critical layer,” “the first functional validation of representation metrics,” “the first layer-wise TDA analysis,” or that “all previous studies considered only correlations.”

Our goal is to **identify site-selection criteria and failure conditions appropriate to the defined repair problem**, within the scope supported by the results.

## 4. Objectives, Questions, and Hypotheses

### 4.1 Research Objective

We connect layer-wise representation metrics, limited diagnostic interventions, and actual learned adaptation within a common data-splitting and evaluation framework. The results will determine whether representation metrics can serve as a basis for repair site selection beyond their role as observational tools.

### 4.2 Research Questions

| Number | Research question | Primary evaluation |
|---|---|---|
| **RQ1** | Do layers with larger representation changes exhibit greater performance recovery under limited partial patching? | Layer-wise rank relationships, selection regret, performance relative to fixed sites |
| **RQ2** | Do effective sites and the predictive value of metrics vary with the intervention budget? | Budget-specific recovery curves, site-ranking changes, practically meaningful performance differences |
| **RQ3** | Do sites selected on observed corruptions remain effective on unseen corruptions? | Unseen-corruption performance at fixed sites, generalization by corruption and family |
| **RQ4** | Do site rankings from diagnostic partial patching predict site rankings for actual adapters? | Diagnostic–adapter rank relationships, actual adaptation regret, selection cost |
| **RQ5 — Optional extension** | Do topological metrics provide additional value beyond distance, similarity, and neighborhood-structure metrics? | Differences in selection performance under identical conditions, computational cost, stability |

### 4.3 Testable Hypotheses and Falsifiability

**H1. Representation-change magnitude alone may be insufficient to explain repair sites.**  
We hypothesize that selecting sites with large normalized distances or CKA changes may incur meaningful performance losses relative to an exhaustive reference evaluation of limited interventions. However, this hypothesis is weakened if simple metrics achieve low regret under most conditions.

**H2. Good sites may depend on the intervention budget.**  
We hypothesize that site rankings may differ between modifying only a few channels and making broader modifications. However, a change in the top-ranked site is not interpreted as meaningful budget dependence if the performance difference falls within the uncertainty range.

**H3. Site-selection performance on observed corruptions may differ from that on unseen corruptions.**  
Metrics that depend excessively on corruption characteristics may perform poorly on unseen corruptions. Conversely, if fixed early, middle, or late sites are consistently sufficient, the need for more complex selection methods is reduced.

**H4. Diagnostic repairability and actual learned repairability may not fully coincide.**  
Partial patching that supplies clean representations does not need to estimate correction values, whereas an actual adapter must learn them from corrupted representations. A small discrepancy would support using diagnostic experiments as a useful site-search tool. A large discrepancy would identify limits on the applicability of diagnostic results.

The superiority of TDA is not a baseline hypothesis. RQ5 is tested only when there is evidence that additional metrics are needed.

## 5. Problem Definition and Scope

### 5.1 Inputs, Model, and Representations

Let $(x,y)$ denote a clean image and its ground-truth label, and $(c,s)$ the corruption type and severity. The corrupted image is

$$
\widetilde{x}=T_{c,s}(x).
$$

Let $h_l$ denote the model up to block $l$, and $g_l$ the remaining blocks and classifier.

$$
F(x)=g_l(h_l(x)).
$$

The block output is defined to include the entire state needed for subsequent computation. For the first model, we use block boundaries after the attention and MLP residual additions. In architectures where multiple branches or separate states cross a boundary, replacing a single tensor is not considered replacement of the entire state.

### 5.2 Selection Unit: One Fixed Site per Model, Not per Image

The basic task is to **select one intervention site per model using calibration data from observed corruptions and fix it before deployment**. Dynamic routing that reselects the site for each input image is outside the basic scope.

This distinction matters. In this design, CKA and dataset-level topological metrics are aggregate metrics over multiple images. They are not directly interpreted as per-image risk or online site-selection scores.

In unseen-corruption evaluation, the site is not changed based on the corruption type. We retain the site selected on average across multiple observed corruptions. Results from separate selection for each corruption are reported only as supplementary diagnostic analyses.

### 5.3 Distinguishing Three Types of Budget

| Budget | Meaning | Main quantities recorded |
|---|---|---|
| **Diagnostic intervention budget $B_{\mathrm{patch}}$** | How much clean representation is supplied? | Modified channel fraction $q$, interpolation strength $\alpha$, modification norm cap $\rho$, mask policy |
| **Actual adaptation budget $B_{\mathrm{adapt}}$** | What resources are allowed for training and deploying the adapter? | Additional parameters, additional computation, training computation, memory |
| **Site-selection budget $B_{\mathrm{select}}$** | What resources are spent finding a good site? | Metric computation, number of backward passes, number of sites tested, training used for search |

Replacing 5% of channels and training a given number of parameters are not equivalent budgets. The diagnostic–adaptation connection compares site rankings and selection effects under their respective budgets, without assuming numerical equivalence between them.

### 5.4 Including No Intervention as a Candidate

We denote no intervention by $\varnothing$ and define its effect relative to the baseline model as $U(\varnothing)=0$. For the set of possible sites $\mathcal{L}$, the final candidate set is $\mathcal{L}\cup\{\varnothing\}$.

If every intervention reduces performance or exceeds the permitted decrease on clean inputs, no intervention may be the correct choice. The study does not assume that a repair module must be inserted.

### 5.5 The Goal Is Not to Prove an Absolutely Optimal Repair Site

This study evaluates **relative recovery effects within a specified intervention type, set of candidate sites, budget, and training procedure**. It does not seek the optimal site across all possible repair methods, including other adapter types or larger training budgets.

In particular, diagnostic partial patching is a reference experiment with access to clean representations. It is not called a mathematical upper bound on actual adaptation performance. The clean model can also misclassify inputs, and we do not exclude the possibility that learned adaptation may outperform it under certain conditions.

## 6. Data, Models, and Observation Sites

### 6.1 Base Model

The first candidate model is **non-distilled DeiT-S/16** pretrained on ImageNet. The official implementation, `deit_small_patch16_224`, uses 12 blocks and 384-dimensional embeddings. Fixing the input size and token count keeps representation sizes constant across blocks, facilitating intervention-budget comparisons. [R13], [R14]

We start with this model to isolate site effects by reducing differences in layer dimensions and output shapes, not because we assume that this architecture is superior. Actual runs will fix the checkpoint identifier, file hash, implementation version, and preprocessing settings.

One ResNet-family checkpoint is used for validation across architectures. Claims about CNN–ViT differences must also consider differences in pretraining data, training methods, clean performance, and model capacity. Comparing one model from each family does not justify universal conclusions about all CNNs and ViTs.

### 6.2 Task and Corruption Types

The analysis is limited to image classification. We use ImageNet-C as a starting point for evaluating natural image corruptions. ImageNet-C benchmarks robustness to common image corruptions, rather than adversarial worst-case perturbations. [R12]

Candidate corruptions for initial exploration are as follows.

| Role | Candidate corruptions | Purpose |
|---|---|---|
| Observed corruptions | Gaussian noise, defocus blur | Construct metrics and site selection using different forms of corruption. |
| Unseen corruptions | contrast, JPEG compression | Evaluate generalization to corruptions not used for selection. |
| Extended evaluation | A broader range of noise, blur, weather, and digital corruptions | Check whether results depend on just two particular corruptions. |

The observed/unseen split above is a prespecified baseline example. It will not be changed after analysis to obtain a favorable combination. Formal generalization analysis will also hold out entire corruption families, in addition to individual corruption types.

Unseen severity and unseen corruption type are separate problems and are evaluated separately. Results obtained by changing only the severity of the same noise corruption are not called generalization to a new corruption type.

### 6.3 Splitting Data by Original Image

| Split | Role | Permitted use |
|---|---|---|
| $D_{\mathrm{fit}}$ | Adapter training | Use observed corruptions, clean images, and training labels. |
| $D_{\mathrm{score}}$ | Estimation of representation metrics and fixed masks | Compute site-selection scores and fixed masks from clean–corrupted pairs. |
| $D_{\mathrm{val}}$ | Validation and final site selection | Finalize metric direction, hyperparameters, clean-accuracy tolerance, and selection rules. |
| $D_{\mathrm{test}}$ | Final evaluation | Evaluate the fixed rules. Do not revise selection rules after seeing the results. |

All splits are made by **original image ID**. The clean version and all corruption/severity variants of the same image must remain in the same split.

The recommended default is to partition ImageNet training images into adaptation training, scoring, and validation splits, while reserving official validation images and their corresponding ImageNet-C images for final evaluation. Calibration corruptions are generated from training images using fixed generation code and preprocessing.

Official ImageNet-C is reserved for evaluation, with priority given to evaluation using the released image files as instructed in the authors' repository. When corruptions are regenerated instead of using the official files, results are explicitly labeled as generated-corruption evaluation to avoid confusing them with the same ImageNet-C scores. [R12]

The fact that the pretrained backbone has seen ImageNet training images is described separately from data leakage in the newly trained adapter and site-selection procedure. These splits do not remove the pretrained model's data history.

### 6.4 Using Only a Subset of the Evaluation Set

A fixed, class-balanced subset may be used for preliminary validation. We save the original image list and sampling seed and use the same image set for all sites and metrics.

Subset results are distinguished from full-benchmark results. Even when only some classes are sampled, the classifier's output classes are not arbitrarily reduced. Evaluating a subset of images from the full ImageNet class set must not be confused with a separate classification task.

### 6.5 Observation Sites and Representation Summaries

For ViTs, we observe the residual-stream output of each block. Actual interventions use the full token tensor, while aggregate representation analysis separately compares the following summaries.

- Use the CLS-token representation.
- Use the mean representation of patch tokens.
- Combine the two representations or use a fixed token sample when needed.

For CNNs, residual-block outputs and spatially average-pooled representations are the default candidates. If metrics are computed on pooled representations while interventions are applied to the original feature maps, we explicitly state **the difference between the measured and intervened representations**.

Tokens are not treated as independent image samples to inflate sample size. We also distinguish the statistical units of image-level and token-level analyses.

## 7. Experiment A — Measuring Layer-Wise Representation Change

### 7.1 Basic Distances

We denote the representation difference for the same original image as

$$
\Delta h_l(x,c,s)=h_l(x)-h_l(T_{c,s}(x)).
$$

The baseline normalized-distance candidate is

$$
D_l^{\mathrm{rel}}=
\mathbb{E}_{x}\left[
\frac{\|\Delta h_l(x,c,s)\|_F}
{\|h_l(x)\|_F+\epsilon}
\right].
$$

We record raw distances, normalized distances, and layer-wise activation norms together. We do not assume that one normalization makes all layers functionally comparable, and examine sensitivity to alternative normalizations such as clean-reference RMS.

We also compare cosine distance, using recovery outcomes to determine whether its insensitivity to norm changes is beneficial or loses useful information. All averages use the same corruptions, severities, and original images.

### 7.2 CKA

Let $H_l$ and $\widetilde{H}_l$ be matrices whose rows contain representations of the same image set. Linear CKA is computed after subtracting column means. [R1]

$$
\operatorname{CKA}(H_l,\widetilde{H}_l)=
\frac{\|H_l^\top\widetilde{H}_l\|_F^2}
{\|H_l^\top H_l\|_F\,
 \|\widetilde{H}_l^\top\widetilde{H}_l\|_F}.
$$

We use $1-\operatorname{CKA}$ as a candidate representation-change score. It measures relational structure across an image set, not the error probability of an individual image.

We fix sample size, centering, representation summaries, and numerical precision. For representations with near-zero variance or small denominators, we record instability in the definition rather than substituting arbitrary valid-looking scores. We also examine score and ranking stability as a function of sample size.

### 7.3 Metric Families for Comparison

| Category | Candidates | Purpose of comparison |
|---|---|---|
| Direct change magnitude | Normalized distance, cosine distance | The simplest site-selection criteria. |
| Aggregate similarity | CKA, optionally Procrustes-family distances | Evaluate the functional validity of representation-structure comparisons. |
| Layer-wise amplification | Ratios and differences of change magnitudes between consecutive blocks | Distinguish layers with large cumulative changes from layers where change increases. |
| Existing criticality | Original CLAT metric and an explicit corruption-based variant | Compare against directly relevant layer-selection criteria. |
| Task sensitivity | Margin or loss gradient in the clean direction | Test whether accounting for directions important to the output improves selection. |
| Geometric structure | Nearest-neighbor preservation, within-/between-class distances | Relatively simple structural metrics preceding TDA. |
| Optional extensions | PH-based distances, TopoLip-related metrics | Test for additional value beyond existing metrics. |

Metrics that use gradients or ground-truth labels are not claimed to operate under the same information conditions as label-free metrics. Results tables report both the information accessed and computational cost.

### 7.4 Precise Scope of the CLAT Comparison

CLAT originally uses feature weakness defined under adversarial perturbations and criticality ratios between consecutive layers. If we compute similar ratios from natural-corruption pairs, this is a **CLAT-inspired corruption-based variant**, not a complete reproduction of the original method. [R6]

We therefore distinguish the original metric, the corruption-based variant, and the actual CLAT fine-tuning procedure. Comparing only the variant metric is not evidence of a “better defense than CLAT.” Handling of ratio explosions caused by small denominators and the reference state for the first block are also defined in advance.

### 7.5 Do Not Reverse Metric Direction Post Hoc

Selecting the layer with the largest representation change and selecting the layer with the smallest change are different selection methods. Both may be compared, but the preferred direction is not determined from final test results.

Direction selection and combinations of metrics are determined only on the validation split of observed corruptions. This prevents choosing the most favorable interpretation based on test-set scores.

## 8. Experiment B — Limited Diagnostic Partial Patching

### 8.1 Intervention Operator

We move the output of block $l$ for a corrupted input partway toward the clean representation of the same image.

$$
\widehat h_l
=
 h_l(\widetilde{x})
 +\alpha M_{l,q}\odot
 \big[h_l(x)-h_l(\widetilde{x})\big].
$$

$M_{l,q}$ is the mask specifying where to modify the representation, $q$ is the modification fraction, and $\alpha$ is the patching strength. The remaining model $g_l$ is then executed unchanged.

This experiment uses the **privileged condition of knowing the corresponding clean representation**. We do not describe the model as repairing itself or present this as a defense algorithm usable at test time. Interpreting activation patching requires specifying both the intervention procedure and evaluation criteria. [R10], [R11]

### 8.2 The Default Mask Is Defined over Channels

For a ViT representation of shape $N\times d$, the default experiment selects $k=\lfloor qd\rfloor$ of the $d$ channels and applies the same mask to those channels across all tokens. Input settings are fixed so that $N$ and $d$ are identical across sites.

Converting a fraction to an integer channel count can change the effective fraction, so both the nominal fraction and actual number of modified channels are saved. $q=0$ is treated as a separate no-intervention condition.

The two default mask policies are as follows.

| Policy | Definition | Role |
|---|---|---|
| **Random fixed mask** | Generate a channel ordering with a prespecified seed and select the first $k$ channels. | Baseline site comparison that minimizes mask-optimization effects. |
| **Fixed mask based on observed data** | Compute per-channel change statistics on $D_{\mathrm{score}}$ and fix the highest-scoring channels. | Examine how data-driven selection affects site rankings under comparable intervention budgets. |

Random masks are repeated across multiple seeds. Within each seed, the mask for a smaller budget is contained in that for a larger budget, avoiding confounding budget changes with mask replacement. We do not assume that the same channel index has the same meaning across layers.

Optimizing masks using each test image's clean–corrupted difference uses additional privileged information and is reserved for a separate diagnostic condition. The baseline site comparison does not permit different mask optimization for different metrics.

### 8.3 Budget Settings and Fairness

Initial candidate settings are $q\in\{0.01,0.05,0.10,0.20\}$ and $\alpha\in\{0.25,0.50,1.00\}$. These form a starting grid for observing different modification scales, not a claim about expected performance.

Even with identical modification fractions and strengths, actual displacement can differ. We therefore record the following together.

**Modification scope:** Record the selected channel count, actual number of modified elements, and token count.

**Modification magnitude:** Record the norm of the actual $\delta_l=\widehat h_l-h_l(\widetilde{x})$ and its magnitude normalized by clean-reference RMS.

**Computational cost:** Separate the costs of extracting clean representations, extracting corrupted representations, and executing the suffix after intervention.

In a supplementary experiment, we apply a norm cap while preserving the modification direction.

$$
\delta_l^{\mathrm{cap}}
=
\delta_l\cdot
\min\left(1,
\frac{\rho\sqrt{n_l}\,r_l}{\|\delta_l\|_F+\epsilon}
\right),
$$

where $n_l$ is the number of representation elements and $r_l$ is the per-element RMS reference computed on observed clean data. Modifications already below the norm cap are not artificially enlarged.

We report **matched modification-fraction/strength conditions** separately from **conditions with an additional norm cap**. We check whether site rankings persist under both, without claiming to have perfectly matched every cost and effect magnitude across all layers simultaneously.

### 8.4 Avoiding Trivial Recovery through Full-State Replacement

If a deterministic model can be decomposed as $F=g_l\circ h_l$ and the entire state required for subsequent computation is replaced with the exact clean state, then

$$
g_l(\widehat h_l)=g_l(h_l(x))=F(x).
$$

Thus, recovering the clean output through full-state replacement alone does not establish that the layer is important or is an optimal repair site.

Full-state replacement is used only as a **positive control for implementation verification** and excluded from the main site-selection results. It restores the “clean model's output,” which is not always the “correct answer.”

Token interventions in ViTs require additional care. Replacing the entire CLS token read by the classifier at the final block with its clean value can effectively supply the clean classification result directly, even at a small token fraction. The default experiment therefore uses partial channel patching. If token experiments are added, CLS inclusion and replacement of the entire readout state are marked as separate conditions.

### 8.5 Required Controls and Implementation Verification

| Check | Expected behavior or purpose |
|---|---|
| $q=0$ or $\alpha=0$ | The output must match the pre-intervention output for the corrupted input within numerical tolerance. |
| Supply the same clean representation to a clean input | The output must remain unchanged because the difference is zero. |
| Replace the entire effective state with the clean state | Verify reproduction of the clean output at deterministic boundaries. |
| Random-direction intervention with the same norm | Check whether arbitrary movement, rather than movement in the clean direction, produces a similar effect. |
| Patching with shuffled original-image correspondence | Check whether correspondence to the same image matters for the effect. |
| Reevaluate after removing hooks | Verify that intervention state does not persist into the next experiment. |

Shuffled donors can change semantic and class information, so they are not interpreted as a pure “information-quantity control.” Same-class and different-class donors are distinguished when needed.

### 8.6 Measuring Effects on Prediction

The primary outcome is the accuracy change over the full evaluation set.

$$
U_l^{\mathrm{patch}}(B,c,s)
=
100\left[
\operatorname{Acc}(F_l^{\mathrm{patch}})
-
\operatorname{Acc}(F)
\right].
$$

Accuracy in this equation is on the $[0,1]$ scale, and $U$ is measured in percentage points. Both accuracies are computed on the same corrupted image set. Results across mask seeds are reported as the mean and variability of independent runs, not as an ensemble that combines predictions.

As a continuous secondary metric, we also compute the change in the correct-class margin.

$$
m(z,y)=z_y-\max_{k\ne y}z_k.
$$

Using only correct-class probability can make it difficult to separate recovery from changes in confidence or calibration, so we examine accuracy, margin, and loss together. Normalized recovery rates whose denominators may be small or negative are not used as primary evaluation metrics.

### 8.7 Evaluating Recovery and New Errors Together

In addition to overall accuracy, the following sets are fixed before intervention.

**Corruption-induced failure set:** Images classified correctly when clean but incorrectly after corruption. Recovery on this set shows the direct repair effect.

**Set remaining correct after corruption:** We measure the fraction of images that were correct before intervention but become incorrect afterward. We do not conceal cases where fixing some errors damages more previously correct predictions.

We report results for the full set, the recovery set, and the set at risk of new errors together. Favorable analysis subsets are not reconstructed based on post-intervention outcomes.

## 9. Experiment C — Evaluating the Reliability of Representation Metrics for Site Selection

### 9.1 Comparing Observational Scores with Actual Effects

For each corruption, severity, and budget condition, we compare the following two vectors.

$$
\mathbf{s}=(s_1,\ldots,s_L),\qquad
\mathbf{u}(B)=(U_1(B),\ldots,U_L(B)).
$$

$\mathbf{s}$ contains site scores produced by a representation metric, and $\mathbf{u}$ contains effects measured by directly applying the specified intervention.

We compute layer-wise Spearman or Kendall rank relationships, but do not judge metric utility from correlation alone. We also account for rank uncertainty with few layers and dependence between adjacent layers.

### 9.2 Core Metric: Site-Selection Regret

Let $\widehat l_m(B)$ denote the site selected using only observed data. Let $\mathcal A_{\mathrm{val}}(B)$ be the candidate set satisfying conditions established on the validation split, such as clean-performance constraints, including no intervention.

Regret in the final evaluation is

$$
\operatorname{Regret}_m(B)
=
\max_{a\in\mathcal A_{\mathrm{val}}(B)}U_{\mathrm{test}}(a,B)
-
U_{\mathrm{test}}(\widehat l_m(B),B).
$$

This is **performance forgone within the same candidate set because of metric-based selection**. Values closer to zero indicate choices closer to the exhaustive-evaluation reference.

The maximum on the right-hand side is a **post-hoc comparison reference** constructed after all final test evaluations. The actual selection algorithm does not use it to choose a site. Taking the largest observed value in a finite sample can also introduce optimistic bias. Selection methods are always fixed on observed data, and uncertainty in the reference is reported.

Checking candidate admissibility also incurs evaluation cost. A low-cost selection method may first test or train only its chosen site, check the validation conditions, and fall back to no intervention if they are not met. Admissibility information about other sites obtained through exhaustive evaluation is not supplied to the selector for free. If validation labels are used to decide whether to intervene, the full selection procedure is labeled as using labels even when the representation score itself is label-free.

As a strong practically usable baseline, we separately include **selection of one site after evaluating all sites on the validation split**. This method is expensive to select with but uses no test labels, distinguishing it from the post-hoc reference.

### 9.3 Baseline Site-Selection Methods

| Baseline | Selection method | Rationale |
|---|---|---|
| No intervention | Retain the original model. | Determine whether repair is needed at all. |
| Fixed early site | Select the first candidate block. | Compare a simple early-intervention hypothesis. |
| Fixed middle site | Select a prespecified middle block. | Provide a fixed-site baseline. |
| Fixed late site | Select the last candidate block. | Examine the advantage of proximity to the classifier. |
| Random site | Select a site from a prespecified distribution. | Determine whether metrics provide useful information. |
| Representation-based selection | Select using scores computed on observed corruptions. | The main subject of evaluation. |
| Small direct search | Measure actual intervention effects on some data or at some sites. | Determine whether direct testing is preferable to metric computation. |
| Exhaustive validation selection | Select based on all candidates' validation performance. | A practically usable, high-cost baseline. |
| Exhaustive test-evaluation reference | Take the maximum final performance across all fixed candidates. | A post-hoc regret reference, not a deployment method. |

If complex metrics do not outperform fixed sites, we state this explicitly. If representation analysis costs more than a small direct search, we do not claim it is an efficient selection method merely because it predicts rankings.

### 9.4 Assessing Budget Dependence

It is insufficient to observe only whether the identity of the top-ranked site changes with the budget. We jointly examine crossings of site-specific recovery curves, the magnitudes of performance differences, and bootstrap confidence intervals.

When several sites perform essentially equally, we report an **equivalence set of sites** rather than a single “critical layer.” Small numerical differences are not interpreted as major structural discoveries.

### 9.5 Evaluation on Unseen Corruptions

Scores, masks, sites, and hyperparameters are fixed on observed corruptions before evaluating unseen corruptions. The default selection is one fixed site chosen for a mixture of observed corruptions.

Clean–corrupted pairs from unseen corruptions may be used to construct final diagnostic plots, but reselecting the deployed site based on those plots is not unseen-corruption generalization evaluation. Such target-informed analyses are separately labeled as post-hoc diagnostics.

We report both per-corruption performance and overall averages. We check whether average improvements conceal large deteriorations on particular corruptions.

## 10. Experiment D — Connecting to Actual Learned Adaptation

### 10.1 Frozen Backbone and a Single Adapter

A single small residual adapter is inserted after each candidate block.

$$
\widehat h_l=h_l(\widetilde{x})+A_{\phi,l}(h_l(\widetilde{x})).
$$

The default candidate module is a bottleneck MLP applied identically token-wise or at each spatial position.

$$
A_{\phi,l}(h)=W_{\mathrm{up}}\,
\sigma\!\left(W_{\mathrm{down}}\operatorname{LN}(h)\right).
$$

The backbone and original classifier are frozen, and only $A_{\phi,l}$ is trained. Normalization trainability, bias use, and initialization are identical across candidate sites. The module is configured to reproduce the original model's output initially, for example by zero-initializing the output projection.

For the base ViT, representation dimensions are identical across blocks, so using the same bottleneck width facilitates direct comparison of added parameters and forward computation. Candidate bottleneck widths may be $r\in\{8,32,64\}$, for example, but budgets are recorded using measured parameter counts and computation.

This bottleneck width controls the capacity of a nonlinear module and does not have the same meaning as the channel-patching fraction in diagnostic experiments.

### 10.2 Training Objective

The default objective combines classification loss on observed corruptions with a term for maintaining clean performance.

$$
\mathcal L_{\mathrm{adapt}}
=
\mathbb E_{(x,y),c,s}
\left[
\ell(F_{l,\phi}(T_{c,s}(x)),y)
+\lambda_{\mathrm{clean}}\ell(F_{l,\phi}(x),y)
\right].
$$

$\lambda_{\mathrm{clean}}$ and optimization settings are determined only on the validation split. Every site receives the same tuning range.

Clean feature matching may be added as an optional extension, but is evaluated separately from the basic classification objective. Training primarily for feature matching and then observing agreement with diagnostic clean patching is not by itself evidence of general repair-site predictability.

Clean inputs or a clean teacher are used only within the permitted training split. At test time, the adapted model receives only one corrupted image, without its clean version, ground-truth label, corruption type, or statistics from unseen corruptions.

### 10.3 Controlling the Site Dependence of Training Cost

Even with frozen backbone parameters, gradients must pass through downstream computation to reach the adapter. Thus, early and late sites may require different training computation despite using identical modules.

The main experiment uses the same data exposure and number of optimizer updates and reports actual training computation. Supplementary experiments match cumulative training computation. **We do not claim to guarantee both equal update counts and equal training FLOPs simultaneously.**

Implementation must distinguish freezing backbone parameters from running the entire forward pass with gradients disabled. The gradient path with respect to inputs must remain intact downstream of the adapter. For CNNs, we specify whether BatchNorm statistics are frozen so that they do not change differently across sites.

### 10.4 Distinguishing Three Diagnostic–Adaptation Connections

| Connection | Specific question | Evaluation |
|---|---|---|
| Representation metric → partial patching | Are sites with large representation changes effective when some clean information is injected? | $s_l$ and $U_l^{\mathrm{patch}}$ |
| Partial patching → actual adaptation | Are sites where clean-direction interventions are effective also favorable for learned adaptation? | $U_l^{\mathrm{patch}}$ and $U_l^{\mathrm{adapt}}$ |
| Representation metric → actual adaptation | Can simple representation measurements select an actual deployment site? | Actual adaptation regret at the metric-selected site |

All three relationships must be reported to distinguish sources of failure. Failure of the representation metric to predict diagnostic effects and successful prediction of diagnostic effects that do not transfer to learned adaptation lead to different conclusions.

### 10.5 Actual Selection Procedure and Exhaustive-Evaluation Reference

The default practical selection procedure chooses a site from observed data and trains an adapter only at that site. Results from training modules at every site are retained as an exhaustive-evaluation reference for research comparisons.

The efficiency of diagnostic selection is evaluated as **diagnostic computation cost + training cost of the selected adapter**. A selector constructed after training modules at all sites is not reported as a method that reduces training cost.

We also distinguish the total computation actually spent producing reference results in the research experiments from the computation required to use each selection method independently.

### 10.6 Clean-Performance Constraint for Actual Adaptation

Actual adaptation is evaluated jointly for improvement on corrupted inputs and changes in clean performance. We define the clean-accuracy tolerance as $\varepsilon_{\mathrm{clean}}$ and select candidates within the tolerance established on validation data.

Exceeding the tolerance at test time is reported as an actual violation; candidates are not silently removed after inspecting test results. Harmless diagnostic clean-to-clean patching is not evidence that an actual adapter preserves clean performance.

## 11. Conditions for Developing New Site-Selection Methods

### 11.1 Creating a New Metric Is Not a Required Goal

If existing distances or fixed-site selection perform sufficiently well, we do not introduce a new metric unnecessarily. In that case, the contribution is to establish conditions under which simple selection suffices for the defined repair problem.

A new selection method is developed when existing metrics consistently exhibit large regret and their failures can be linked to measurable features.

### 11.2 Task Sensitivity Is a Strong Baseline to Compare First

A simple local approximation explains why representation-difference magnitude alone cannot determine the direction of the effect. For margin $m(g_l(h),y)$ and a small intervention $\delta_l$, consider

$$
\Delta m
\approx
\left\langle
\nabla_{h_l}m(g_l(h_l(\widetilde{x})),y),\delta_l
\right\rangle.
$$

Thus, even a large change may yield little margin improvement if it is not aligned with directions important to the output. This is a local approximation for small interventions, not a guarantee for large partial patches or nonlinear regions.

A directional-sensitivity score based on this approximation is included as an interpretable baseline. We explicitly account for its label and backward-computation requirements and do not claim it as a novel original theory.

### 11.3 Candidate Budget-Aware Selectors

As an extension, we may examine a simple recovery predictor that takes representation change, task sensitivity, normalized depth, and intervention budget as inputs.

$$
\widehat U_l(B)
=q_\theta\big(D_l,S_l,l/L,B\big).
$$

Here, $S_l$ denotes additional features such as task sensitivity. We first use simple regression or ranking models and compare them with depth-only and depth-and-budget-only models.

Training recovery targets are obtained only from observed corruptions and permitted development data. Derived samples of the same original image are not split between training and validation, and validation groups samples by corruption type. A few dozen layer-wise observations from one model are not treated as a large independent training dataset.

If training a selector requires actual adaptation outcomes from every site, that cost is included. We do not claim search savings without demonstrating that this cost is recouped on additional models or new conditions.

## 12. Purpose and Validation Conditions of the TDA Extension

### 12.1 Use TDA When It Answers a New Question

The purpose of TDA is not merely to add the fact that “topological structure was computed at each layer.” Prior work already connects layer-wise topological changes to robustness, so additional explanatory value is required for the following question. [R3], [R7]

> Under the same modification budget, do changes in topological structure enable selection of more effective repair sites than normalized distance, CKA, and nearest-neighbor structure?

If basic metrics are sufficient, TDA is not added. If it is added, we compare site-selection performance, sample stability, and computational cost.

### 12.2 Objects of Analysis

Layer-wise representations of the same image set are constructed as follows.

$$
\mathcal H_l=\{h_l(x_i)\}_{i=1}^{n},\qquad
\widetilde{\mathcal H}_l=\{h_l(\widetilde{x}_i)\}_{i=1}^{n}.
$$

After the required representation summarization, we compute persistent homology and use differences between clean and corrupted persistence diagrams as candidate metrics. We initially consider homology dimensions $H_0$ and $H_1$, for which computation and interpretation are feasible.

Changes between diagrams computed separately at each layer are not the same concept as a single persistence module equipped with maps between layers. The default TDA extension **compares diagrams obtained from each layer's point cloud**; it does not claim to define rigorous topological lifetimes along network depth.

### 12.3 Items to Control

| Item | Control principle |
|---|---|
| Samples | Compare the same original images and class composition, and check stability across multiple subsamples. |
| Representation scale | Apply the same normalization, determined from observed clean data, to clean and corrupted representations. |
| Dimensionality reduction | Fit PCA or similar methods on permitted observed data and fix them. Distinguish this from refitting on every test set. |
| Distance and filtration | Fix the distance function, filtration range, and homology dimensions. |
| Diagram processing | Predefine handling of infinite intervals, empty diagrams, and small persistence values. |
| Labels | Distinguish overall from class-conditional structure and indicate whether labels are used. |
| Computation | Record throughput by sample size and feature dimension, peak memory, and total site-selection cost. |

We do not assume that topological changes after dimensionality reduction faithfully preserve changes in the original high-dimensional representations. If small sample sizes make class-specific structure unstable, the scope of interpretation is limited accordingly.

### 12.4 Topological Simplification Does Not Mean Failure

Prior work has observed topological simplification during successful classification. Thus, reductions in connected components or holes are not directly used as evidence of information collapse or misclassification. [R3]

The topology of unlabeled point clouds may not adequately reflect changes in class semantics or sample correspondence. Similar diagrams do not establish equivalent function; within-/between-class structure and actual intervention outcomes are examined together.

Generic PH distances and TopoLip are not the same metric. An implementation is labeled a TopoLip reproduction only if it follows the paper's definition exactly; modifications are explicitly identified as variants. [R7]

### 12.5 Criteria for Adopting TDA

TDA is considered a useful extension if it reduces unseen-corruption selection regret relative to basic metrics or reproducibly distinguishes conditions in which existing metrics fail. A single weak correlation or visually different diagrams do not establish its necessity.

If a cheaper structural metric such as nearest-neighbor preservation achieves the same performance, there is little reason to adopt computationally expensive TDA as the main method.

## 13. Evaluation Metrics and Statistical Design

### 13.1 Primary and Secondary Evaluation

**Primary evaluation:** Use mean accuracy improvement on unseen corruptions and actual adaptation regret at sites selected using only observed data. The adaptation budget, corruption set, and averaging weights are fixed in advance. Direct performance differences against simple site-selection methods and exhaustive validation selection are also reported.

**Secondary evaluation:** Use diagnostic partial-patching regret, margin change, site-rank relationships, recovery rate on corruption-induced failures, new-error rate, clean-accuracy change, budget-specific recovery curves, and site-selection cost.

Experiments using only some corruptions are explicitly reported as mean accuracy/error on that corruption set. Names such as standard mCE are used only when the original benchmark definition and corruption composition are followed. [R12]

### 13.2 Aggregation Units and Uncertainty

The default is a paired bootstrap over original images. A clean image and its corruption/severity variants are resampled together to preserve correspondence. When class balance is required, resampling is performed within classes before aggregation.

Variability across mask seeds and adapter-training seeds is recorded separately. If image and seed uncertainty are combined, the hierarchical resampling procedure is specified. Results across layers, tokens, and corruption severities are not all treated as independent samples.

Initial settings may use 20 diagnostic mask seeds and 3 learned-adaptation seeds. Adequacy is assessed based on observed variability and computational resources, without choosing repeat counts to favor particular outcomes.

### 13.3 Multiple Comparisons and Selection Bias

Primary hypotheses, primary evaluation metrics, and the primary budget are specified first; the rest are labeled exploratory analyses. Joint hypothesis testing across multiple metrics, budgets, and corruptions uses prespecified multiple-comparison control, such as the Holm method.

Effect sizes and confidence intervals take priority over p-values alone. Because exhaustive-evaluation maxima may be optimistic due to sample noise, claims are not based solely on gaps to post-hoc references; we also report differences against feasible baselines selected on validation data.

### 13.4 Do Not Force a Unique Optimal Site

When differences between sites are small, we report the set of sites within tolerance $\tau$.

$$
\mathcal L_{\tau}(B)
=
\left\{l:
\max_j U_j(B)-U_l(B)\leq\tau
\right\}.
$$

$\tau$ is set in advance based on validation data and a criterion for practically meaningful performance differences. It is not changed after inspecting test results to make more sites appear favorable.

We evaluate not only whether a metric identifies one exact layer, but also whether it selects a member of the equivalence set and what actual regret it incurs.

## 14. Basic and Extended Experimental Scope

### 14.1 Minimum Configuration for the Core Validation

The following is a starting plan that can be directly instantiated as an experimental configuration. All numbers are initial candidates to be fixed independently of results; adjustments for computational resources are documented with their reasons and settings.

| Item | Default configuration |
|---|---|
| Model | One non-distilled DeiT-S/16 checkpoint |
| Task | ImageNet image classification with the original classifier retained |
| Intervention sites | Outputs of all 12 blocks as candidates |
| Observed corruptions | Gaussian noise, defocus blur |
| Unseen corruptions | contrast, JPEG compression |
| Corruption severity | 1, 3, and 5 for preliminary comparisons; all severities of the chosen benchmark for extensions |
| Representation metrics | Normalized distance, cosine distance, CKA, explicit corruption-based criticality variant |
| Default mask | Random fixed channel mask |
| Diagnostic budgets | Channel fractions of 1%, 5%, 10%, and 20%, with default $\alpha=1$ |
| Supplementary diagnostic conditions | $\alpha=0.25,0.5$, norm cap, fixed masks based on observed data |
| Actual adaptation | Single residual bottleneck MLP, default bottleneck width 32 |
| Extended adaptation budgets | Bottleneck widths 8, 32, and 64, with actual parameters and computation reported |
| Main baselines | No intervention, fixed early/middle/late sites, random site, metric selection, small direct search, exhaustive validation selection |
| Reproducibility | Image-ID splits, fixed seeds, saved original and post-intervention predictions |

We do not initially run every combination of metrics, masks, budgets, and models. The default configuration tests the observation–diagnostic–adaptation connections, and only factors needed to distinguish the results are added.

### 14.2 Core Experiment Groups

| Experiment | Factors varied | Question examined |
|---|---|---|
| **E1. Representation–partial-patching relationship** | Block site and representation metric | Do observed change magnitudes predict diagnostic effects? |
| **E2. Budget-specific site selection** | Modification fraction, strength, and norm cap | Do changes in site rankings reflect meaningful budget effects? |
| **E3. Unseen-corruption generalization** | Evaluation corruption types and families | Does performance generalize with selection fixed? |
| **E4. Diagnostic–learned-adaptation connection** | Site and adapter capacity | Do diagnostic results remain useful for deployable repair? |
| **E5. Validation across architectures** | ViT- and ResNet-family models | Do results depend on a single checkpoint? |
| **E6. Metric extensions** | Task sensitivity, neighborhood structure, TDA | Is there additional selection value beyond basic metrics? |

E1–E4 form the core scope. E5 provides validation to broaden the scope of claims, and E6 is an extension conducted when its need has been established.

### 14.3 Topics Excluded from the Basic Scope

Object detection and segmentation, full autonomous-driving systems, multimodal fusion, comparisons spanning all model architectures, simultaneous multisite interventions, and per-input dynamic routing are outside the basic scope.

Adversarial attacks are a separate extension from natural corruptions. Improvements on natural corruptions are not described as improvements in adversarial robustness. If attack evaluation is added, attacks must target the actual adapted model; attacks generated only against the original backbone do not support general defense claims.

## 15. Main Risks and Interpretation Principles

| Risk | Incorrect interpretation | Response |
|---|---|---|
| Full-state or full-readout replacement | Interpreting the result as discovering an important layer at low cost. | Separate implementation checks from the main experiment and use only partial interventions for primary evaluation. |
| Different representation dimensions | Assuming that the same channel fraction means the same resources. | Record modified element counts, norms, added parameters, and computation together. |
| Coordinate dependence of representations | Assuming that particular channels are universal semantic units. | Limit baseline results to the checkpoint's coordinate system and mask policy. |
| Post-hoc site selection | Presenting a site chosen after observing unseen corruptions as a generalization result. | Fix selection rules on observed data and distinguish them from post-hoc references. |
| Confounding with mask optimization | Confusing site effects with better channel-selection effects. | Compare sites within the same mask policy. |
| Privileged diagnostic information | Interpreting clean-representation injection as practically feasible repair. | Separate diagnostic partial patching from adaptation using only corrupted inputs. |
| Adapter limitations | Interpreting a small module's failure as irreversible information loss. | Analyze dependence on capacity, training amount, and objective, without claiming information loss. |
| Seed and sample noise | Declaring a structural transition merely because the top layer changes. | Examine confidence intervals, effect sizes, and equivalence sets together. |
| Semantic changes under severe corruption | Failing to distinguish damaged visual input information from internal model failure. | Combine severity-specific results with sample inspection, without assuming semantic preservation unconditionally. |
| Topological simplification | Treating simplification itself as failure. | Validate separately through function and actual intervention outcomes. |
| Omitted selection costs | Describing expensive metrics or exhaustive training as low-cost selection. | Account separately for metric, search, and training costs. |
| Generalization across model architectures | Generalizing one checkpoint's results to all CNNs or ViTs. | Record training conditions and capacity and limit the scope of claims. |

Representations constructed through partial patching may fall outside the distribution naturally produced by the original model. We examine sensitivity through intervention-strength comparisons and norm controls, without claiming to eliminate all distributional differences introduced by artificial interventions.

CKA measures representation similarity, while probing performance measures how well a particular readout extracts information. Neither is used as a direct measure of mutual information or proof of irreversible information loss. [R1], [R4]

## 16. Possible Outcomes and Research Decision Criteria

This section specifies how possible outcomes will be interpreted, rather than predicting performance values.

### 16.1 Existing Representation Metrics Predict Actual Adaptation Sites Well

This would support using representation analysis as a proxy for site search. However, practical selection claims also require cost advantages over direct search and stability on unseen corruptions.

### 16.2 Metrics Predict Diagnostic Effects but Transfer Poorly to Actual Adaptation

Rather than concluding that representation metrics are wrong, the central result would be **the difference between supplying clean information through intervention and learning to produce corrections**. We test whether this discrepancy persists when adapter capacity and objectives change.

### 16.3 Effective Sites Vary with Budget or Corruption

This may support conditional site selection rather than a single universal critical layer. Any new selection method is tested for whether accounting for budget reduces actual regret relative to fixed sites and existing metrics.

### 16.4 Fixed Sites Are Sufficient under Most Conditions

This would indicate limited need for complex metrics. Even if representation differences appear interesting, we explicitly state the limitation if they provide no practical site-selection advantage. We do not retrospectively select favorable corruptions to manufacture a need for a new method.

### 16.5 Site Differences Are Small or Unstable across Repeated Experiments

We do not make strong critical-layer claims. We report equivalence sets, sensitivity to sample size, and the effects of mask and training seeds, and consider sites with simpler computation or implementation for practical selection.

### 16.6 TDA Provides No Additional Benefit

TDA is excluded from the main method or retained as a limitations analysis. It is not retained merely because it is topological. Even if it improves performance, cost and stability must also be satisfactory to justify its use.

### 16.7 Criteria for Establishing a Research Contribution

The study must go beyond a single visualization or correlation to demonstrate **reproducible site-selection performance or reproducible conditions of selection failure**. Results that do not support the hypotheses can be reported, but the absence of differences in a small preliminary experiment does not automatically establish a general negative conclusion or a paper contribution.

## 17. Expected Contributions and Deliverables

### 17.1 Expected Contributions

**Problem definition:** Specify an explicit task for evaluating layer-wise representation changes through budget-constrained repair site selection.

**Validation framework:** Separate the three relationships among representation metrics, diagnostic interventions, and actual adaptation, and jointly evaluate site-selection regret, unseen corruptions, and cost.

**Empirical findings:** Determine when simple metrics or fixed sites suffice and when they fail. Propose improved site-selection methods only when supported by results.

### 17.2 Planned Main Figures and Tables

| Deliverable | Content |
|---|---|
| Layer-wise representation changes versus diagnostic effects | Present change-magnitude and recovery curves side by side for the same model and corruption. |
| Site–budget recovery table or heatmap | Show whether effective sites vary with modification scale. |
| Diagnostic–adaptation comparison | Present the relationship between partial-patching and learned-adaptation effects at each site. |
| Site-selection regret table | Compare fixed sites, existing metrics, and direct search by budget. |
| Unseen-corruption performance table | Show whether selections fixed on observed data remain effective for other corruptions. |
| Cost–performance comparison | Present selection effects relative to costs including metric computation and module training. |
| Controls and sensitivity table | Present stability with respect to norms, masks, seeds, and representation summaries. |

The pre-experimental proposal does not present hypothetical accuracies or expected gains as actual results.

### 17.3 Reproducibility Deliverables

Research deliverables include experimental configurations, original-image split lists, checkpoint identifiers, feature-extraction, intervention, and adapter-training code, metric-computation code, raw results tables, and analysis scripts.

Original images from public datasets are handled according to their terms of use. For data unsuitable for redistribution, we consider providing only image IDs and reproduction procedures.

## 18. Implementation and Logging Requirements

### 18.1 Module Separation

| Module | Responsibility |
|---|---|
| `data_protocol` | Split original image IDs; fix corruption types, severities, and seeds |
| `feature_extractor` | Define block boundaries, extract CLS/patch representations, verify shapes |
| `representation_metrics` | Compute normalized distances, CKA, and other metrics |
| `patching_engine` | Implement mask policies, partial patching, norm caps, and controls |
| `adapter_training` | Freeze the backbone, train adapters, record costs |
| `site_selection` | Select sites from observed data; implement no-intervention and fixed-site baselines |
| `evaluation` | Evaluate overall accuracy, recovery/new errors, regret, and unseen corruptions |
| `statistics` | Analyze paired bootstrap uncertainty, seed variability, and selection stability |

Only necessary representations are cached. Rather than unconditionally saving every token, layer, and corruption result, we distinguish original predictions, summary representations for metrics, and representations required for interventions. We also verify whether storage precision affects intervention results.

### 18.2 Required Fields in Results Tables

Each result must allow tracking of at least the following information.

```text
experiment_id
model_id / checkpoint_hash / implementation_version
original_image_id / split / label
corruption_type / severity / corruption_seed
layer_id / intervention_boundary / representation_summary
metric_name / score / score_data_split / label_access
intervention_type / mask_policy / mask_seed
channel_count / effective_channel_ratio / alpha / norm_cap / actual_delta_norm
adapter_width / trainable_parameters / training_seed
baseline_prediction / clean_prediction / post_intervention_prediction
baseline_margin / post_intervention_margin
selection_cost / training_compute / inference_overhead
```

For aggregate scores such as CKA and PH, the image-set ID and sample size are saved separately. Per-image and aggregate results are not treated as independent samples with the same row-level unit.

### 18.3 Pre-Run Checks

For diagnostics and evaluation, the model is fixed in evaluation mode so that stochastic components such as dropout and stochastic depth do not confound clean–corrupted differences. Numerical precision and output-comparison tolerances are recorded as well.

First verify reproduction of the unmodified model's outputs, clean–corrupted ID correspondence, actual tensor shapes at block boundaries, no-intervention and full-state-replacement controls, gradient paths, and the absence of overlapping original IDs across splits.

Also verify that site-selection rules do not access test labels or unseen-corruption scores. If these checks fail, differences between sites are not interpreted as research findings.

## 19. Separate Follow-Up Question: Distinguishing Information Loss from Information Use

The following question is related to this study but remains outside its scope.

> When a late layer produces an incorrect answer, has the information needed for the correct answer disappeared, or does it remain in the representation but fail to inform the final decision?

Overthinking, where an intermediate classifier is correct but the final classifier is wrong, has already been studied in Shallow-Deep Networks. Thus, merely attaching linear probes to intermediate layers is not claimed as an independent new contribution. [R15]

Extending this direction would require controlling factors such as shape, color, and background and combining readouts of varying complexity with internal model interventions. This would distinguish **information that can be read out**, **information actually used for classification**, and **information usable by an admissible adapter**.

Failure of learned adaptation in this study is failure under a particular module and training procedure. It does not imply that information is unrecoverable by every possible readout and intervention.

## 20. Final Research Proposal

This study starts by **not treating “where representations change substantially” and “where a small modification restores the decision” as the same question**.

We compare layer-wise change magnitudes with limited partial-patching effects in one ViT model and test whether the resulting site selections transfer to unseen corruptions and actual adapters. We first evaluate whether simple metrics and fixed-site selection suffice, adding budget-aware selectors or TDA only when their need is established.

A successful outcome need not introduce a more complex method or identify a single critical layer. The central goal is to **explain, through controlled experiments and cost evaluation, when evidence from internal representation analysis is valid for actual repair decisions and when it is not**.

---

## References and Official Resources

Findings from the literature are interpreted within each study's model, data, and intervention conditions. The links below point to the checked original papers or official author resources. Release years and conference publication years are distinguished where they differ.

### R1. CKA

Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. **Similarity of Neural Network Representations Revisited.** ICML 2019, PMLR 97:3519–3529.  
[Official paper page](https://proceedings.mlr.press/v97/kornblith19a.html)

### R2. Comparing CNN and ViT Representations

Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, Alexey Dosovitskiy. **Do Vision Transformers See Like Convolutional Neural Networks?** NeurIPS 2021.  
[Author-released paper](https://arxiv.org/abs/2108.08810)

### R3. Layer-Wise Topological Changes

Gregory Naitzat, Andrey Zhitnikov, Lek-Heng Lim. **Topology of Deep Neural Networks.** Journal of Machine Learning Research, 21(184):1–40, 2020.  
[Official paper page](https://jmlr.org/papers/v21/20-345.html)

### R4. Representation Metrics and Functional Differences

Frances Ding, Jean-Stanislas Denain, Jacob Steinhardt. **Grounding Representation Similarity Through Statistical Testing.** NeurIPS 2021. The arXiv version is titled *Grounding Representation Similarity with Statistical Testing*.  
[Official paper page](https://proceedings.neurips.cc/paper/2021/hash/0c0bf917c7942b5a08df71f9da626f97-Abstract.html) · [Author-released paper](https://arxiv.org/abs/2108.01661)

### R5. ReSi

Max Klabunde, Tassilo Wald, Tobias Schumacher, Klaus Maier-Hein, Markus Strohmaier, Florian Lemmerich. **ReSi: A Comprehensive Benchmark for Representational Similarity Measures.** ICLR 2025. The initial arXiv release was in 2024; the checked v2 is a 2025 revision.  
[Author-released paper and conference listing](https://arxiv.org/abs/2408.00531)

### R6. CLAT

Bhavna Gopal, Huanrui Yang, Jingyang Zhang, Mark Horton, Yiran Chen. **Boosting Adversarial Robustness with CLAT: Criticality Leveraged Adversarial Training.** ICML 2025, PMLR 267:20142–20161. The initial version was released in 2024 with a different title from the final conference paper.  
[Official paper page](https://proceedings.mlr.press/v267/gopal25a.html) · [Released paper](https://arxiv.org/abs/2408.10204)

### R7. TopoLip

Baiyuan Chen. **Is Smoothness the Key to Robustness? A Comparison of Attention and Convolution Models Using a Novel Metric.** arXiv:2410.17628, 2024.  
[Released paper](https://arxiv.org/abs/2410.17628)

### R8. NeuroShield-ViT

Chashi Mahiul Islam, Samuel Jacob Chacko, Mao Nishino, Xiuwen Liu. **Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers.** arXiv:2502.04679, 2025. It appears in the conference proceedings under the title *NeuroShield-ViT: Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers*, and the ISVC 2025 proceedings were published online in 2026.  
[Released paper](https://arxiv.org/abs/2502.04679) · [Official publisher page](https://link.springer.com/chapter/10.1007/978-3-032-14492-8_3)

### R9. Suppress and Diversify

Jiangang Yang, Wenhui Shi, Xiaoran Xu, Wenyue Chong, Luqing Luo, Jing Xing, Jian Liu. **Suppress and Diversify: Refining Robust Pathways for Corruption Robustness.** arXiv:2608.06712, 2026. The authors' public page lists it as accepted at ICML 2026.  
[Released paper](https://arxiv.org/abs/2608.06712) · [HTML text](https://arxiv.org/html/2608.06712v1)

### R10. Evaluation Methods for Activation Patching

Fred Zhang, Neel Nanda. **Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.** arXiv:2309.16042, released in 2023.  
[Released paper](https://arxiv.org/abs/2309.16042)

### R11. Interpreting Activation Patching

Stefan Heimersheim, Neel Nanda. **How to use and interpret activation patching.** arXiv:2404.15255, 2024.  
[Released paper](https://arxiv.org/abs/2404.15255)

### R12. ImageNet-C and Evaluation Resources

Dan Hendrycks, Thomas Dietterich. **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.** ICLR 2019.  
[Official paper page](https://openreview.net/forum?id=HJz6tiCqYm) · [Released paper](https://arxiv.org/abs/1903.12261) · [Authors' official data and code repository](https://github.com/hendrycks/robustness)

### R13. DeiT

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou. **Training data-efficient image transformers & distillation through attention.** ICML 2021, PMLR 139:10347–10357.  
[Official paper page](https://proceedings.mlr.press/v139/touvron21a.html)

### R14. Official DeiT Model Definition

Facebook Research. **DeiT official implementation — models.py.** This resource documents the configuration and checkpoint links for non-distilled `deit_small_patch16_224`. Runs pin a specific code version rather than the mutable default branch.  
[Official model definition](https://github.com/facebookresearch/deit/blob/main/models.py)

### R15. Overthinking

Yigitcan Kaya, Sanghyun Hong, Tudor Dumitras. **Shallow-Deep Networks: Understanding and Mitigating Network Overthinking.** ICML 2019, PMLR 97:3301–3310.  
[Official paper page](https://proceedings.mlr.press/v97/kaya19a.html)

[R1]: https://proceedings.mlr.press/v97/kornblith19a.html
[R2]: https://arxiv.org/abs/2108.08810
[R3]: https://jmlr.org/papers/v21/20-345.html
[R4]: https://proceedings.neurips.cc/paper/2021/hash/0c0bf917c7942b5a08df71f9da626f97-Abstract.html
[R5]: https://arxiv.org/abs/2408.00531
[R6]: https://proceedings.mlr.press/v267/gopal25a.html
[R7]: https://arxiv.org/abs/2410.17628
[R8]: https://arxiv.org/abs/2502.04679
[R9]: https://arxiv.org/abs/2608.06712
[R10]: https://arxiv.org/abs/2309.16042
[R11]: https://arxiv.org/abs/2404.15255
[R12]: https://arxiv.org/abs/1903.12261
[R13]: https://proceedings.mlr.press/v139/touvron21a.html
[R14]: https://github.com/facebookresearch/deit/blob/main/models.py
[R15]: https://proceedings.mlr.press/v97/kaya19a.html
