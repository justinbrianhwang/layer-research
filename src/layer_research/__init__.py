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
