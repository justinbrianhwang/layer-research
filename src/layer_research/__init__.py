"""Tools for reproducible vision representation research."""
from .data_protocol import (
    SplitConfig, SplitResult, make_splits, save_splits, load_splits, verify_disjoint,
    OBSERVED_CORRUPTIONS, UNSEEN_CORRUPTIONS, DEFAULT_SEVERITIES,
    CorruptionSpec, corrupt_image, PairedImageDataset,
)
from .feature_extractor import (
    BlockOutputRecorder, num_blocks, summarize, extract_paired_features,
    iter_paired_block_outputs, shape_report,
)
from .representation_metrics import (
    relative_distance, raw_distance, activation_norm, cosine_distance, linear_cka,
    one_minus_cka, amplification_ratio, knn_preservation, layerwise_scores,
)

__all__ = [
    "SplitConfig", "SplitResult", "make_splits", "save_splits", "load_splits",
    "verify_disjoint", "OBSERVED_CORRUPTIONS", "UNSEEN_CORRUPTIONS", "DEFAULT_SEVERITIES",
    "CorruptionSpec", "corrupt_image", "PairedImageDataset", "BlockOutputRecorder",
    "num_blocks", "summarize", "extract_paired_features", "iter_paired_block_outputs",
    "shape_report", "relative_distance", "raw_distance", "activation_norm",
    "cosine_distance", "linear_cka", "one_minus_cka", "amplification_ratio",
    "knn_preservation", "layerwise_scores",
]

from .patching_engine import (
    MaskSpec, InterventionType, make_channel_mask, nested_masks, Patcher,
    run_patched_forward, sweep_layers_budgets,
)

__all__ += [
    "MaskSpec", "InterventionType", "make_channel_mask", "nested_masks", "Patcher",
    "run_patched_forward", "sweep_layers_budgets",
]

from .evaluation import (
    accuracy_gain_pp, recovery_and_new_error, margin_change, clean_accuracy_change_pp,
    layer_effect_table, aggregate_over_seeds, regret, equivalence_set,
)
from .site_selection import (
    Selection, select_fixed, select_random, select_by_metric, select_val_sweep,
    select_small_search, apply_admissibility, rank_correlation,
)
from .statistics import (
    paired_bootstrap, bootstrap_layer_gain, bootstrap_regret, seed_variability,
    selection_stability, holm_correction,
)

__all__ += [
    "accuracy_gain_pp", "recovery_and_new_error", "margin_change", "clean_accuracy_change_pp",
    "layer_effect_table", "aggregate_over_seeds", "regret", "equivalence_set",
    "Selection", "select_fixed", "select_random", "select_by_metric", "select_val_sweep",
    "select_small_search", "apply_admissibility", "rank_correlation",
    "paired_bootstrap", "bootstrap_layer_gain", "bootstrap_regret", "seed_variability",
    "selection_stability", "holm_correction",
]
