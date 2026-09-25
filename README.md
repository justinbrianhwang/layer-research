# Layer Research: Budget-Constrained Intervention Site Selection in Vision Models

Research code for the proposal in `vision_intervention_site_selection_proposal.md`
("From Representation Change to Repair").

## Setup

```bash
conda env create -f environment.yml
conda activate layer-research
pip install -e .
pytest
```

## Layout (mirrors proposal section 18.1)

| module | responsibility |
|---|---|
| `data_protocol` | image-ID splits, corruption type/severity/seed fixing |
| `feature_extractor` | block boundaries, CLS/patch summaries, shape checks |
| `representation_metrics` | normalized distance, cosine, CKA, criticality variants |
| `patching_engine` | mask policies, partial patching, norm cap, controls |
| `adapter_training` | frozen backbone, residual bottleneck adapter, cost logging |
| `site_selection` | metric-based / fixed / random / val-exhaustive selectors |
| `evaluation` | accuracy, recovery / new-error sets, regret, unseen corruptions |
| `statistics` | paired bootstrap, seed variability, selection stability |

GPU runs are executed on a rented Vast.ai instance; see `docs/vast_workflow.md`.
