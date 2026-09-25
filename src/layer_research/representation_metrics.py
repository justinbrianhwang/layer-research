"""Label-free metrics on aligned, image-level representation matrices."""
import logging
import numpy as np
import pandas as pd
import torch

_LOG = logging.getLogger(__name__)


def _matrix(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or 0 in value.shape or not np.isfinite(value).all():
        raise ValueError("Expected a nonempty finite [n, d] matrix")
    return value


def _pair(H, Ht):
    H, Ht = _matrix(H), _matrix(Ht)
    if H.shape != Ht.shape:
        raise ValueError("Paired representations must have identical shapes")
    return H, Ht


def _epsilon(eps):
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def relative_distance(H, Ht, eps=1e-8):
    """Mean per-image L2 change divided by clean L2 norm plus eps."""
    _epsilon(eps)
    H, Ht = _pair(H, Ht)
    return float(np.mean(np.linalg.norm(H - Ht, axis=1) / (np.linalg.norm(H, axis=1) + eps)))


def raw_distance(H, Ht):
    """Mean per-image L2 change."""
    H, Ht = _pair(H, Ht)
    return float(np.linalg.norm(H - Ht, axis=1).mean())


def activation_norm(H):
    """Mean clean per-image L2 norm."""
    return float(np.linalg.norm(_matrix(H), axis=1).mean())


def cosine_distance(H, Ht, eps=1e-8):
    """Mean one minus cosine; zero-vector cosine is defined as zero."""
    _epsilon(eps)
    H, Ht = _pair(H, Ht)
    H = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), eps)
    Ht = Ht / np.maximum(np.linalg.norm(Ht, axis=1, keepdims=True), eps)
    return float(np.mean(1 - np.clip(np.sum(H * Ht, axis=1), -1, 1)))


def linear_cka(H, Ht, eps=1e-8):
    """Column-centered linear CKA in float64; unstable denominator yields NaN."""
    _epsilon(eps)
    H, Ht = _pair(H, Ht)
    H, Ht = H - H.mean(axis=0), Ht - Ht.mean(axis=0)
    denominator = np.linalg.norm(H.T @ H) * np.linalg.norm(Ht.T @ Ht)
    if denominator < eps:
        _LOG.warning("Linear CKA denominator %g is below eps %g", denominator, eps)
        return float("nan")
    return float(np.linalg.norm(H.T @ Ht) ** 2 / denominator)


def one_minus_cka(H, Ht, eps=1e-8):
    """Representation change defined as one minus linear CKA."""
    return 1 - linear_cka(H, Ht, eps)


def amplification_ratio(scores, eps=1e-8):
    """First layer is 1.0; later ratios use score[l] / max(score[l-1], eps)."""
    _epsilon(eps)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Scores must be a finite nonnegative sequence")
    return np.concatenate(([1.0], scores[1:] / np.maximum(scores[:-1], eps))) if len(scores) else np.array([])


def knn_preservation(H, Ht, k=10):
    """Mean Jaccard of Euclidean k-NN sets, excluding self; ties use row order."""
    H, Ht = _pair(H, Ht)
    n = len(H)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < n_samples")
    # Row-wise distances bound working memory to O(n*d), rather than O(n*n).
    overlaps = []
    for i in range(n):
        sets = []
        for values in (H, Ht):
            distances = np.sum((values - values[i]) ** 2, axis=1)
            distances[i] = np.inf
            sets.append(set(np.argsort(distances, kind="stable")[:k]))
        overlaps.append(len(sets[0] & sets[1]) / len(sets[0] | sets[1]))
    return float(np.mean(overlaps))


def layerwise_scores(features: dict, metric_names: list[str]) -> pd.DataFrame:
    """Score extraction dictionaries; amplification uses consecutive relative distances."""
    metrics = {f.__name__: f for f in (relative_distance, raw_distance, activation_norm,
               cosine_distance, linear_cka, one_minus_cka, knn_preservation)}
    unknown = set(metric_names) - set(metrics) - {"amplification_ratio"}
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    clean, corrupted = features["clean"], features["corrupted"]
    if clean.keys() != corrupted.keys():
        raise ValueError("Clean and corrupted layers differ")
    rows, previous = [], {}
    for layer in sorted(clean):
        if clean[layer].keys() != corrupted[layer].keys():
            raise ValueError("Clean and corrupted summary modes differ")
        for mode in clean[layer]:
            H, Ht = _pair(clean[layer][mode], corrupted[layer][mode])
            if "image_ids" in features and len(H) != len(features["image_ids"]):
                raise ValueError("Representations and image IDs differ in length")
            for name in metric_names:
                if name == "amplification_ratio":
                    score = relative_distance(H, Ht)
                    value = score / max(previous[mode], 1e-8) if mode in previous else 1.0
                else:
                    value = metrics[name](H) if name == "activation_norm" else metrics[name](H, Ht)
                rows.append((layer, name, value, len(H), mode, "none"))
            previous[mode] = relative_distance(H, Ht)
    return pd.DataFrame(rows, columns=["layer", "metric_name", "score", "n_samples", "summary_mode", "label_access"])
