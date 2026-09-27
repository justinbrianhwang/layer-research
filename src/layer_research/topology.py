"""Persistent homology on aligned clouds, with clean-fitted shared PCA.

Optional ripser/persim imports are lazy. Returned dict subclasses carry cost and
infinite-bar counts in .metadata, keeping metric keys directly iterable.
"""
import time
import tracemalloc
import os
import threading
from contextlib import contextmanager

import numpy as np
from sklearn.decomposition import PCA

from .representation_metrics import _matrix, _pair


class PHResult(dict):
    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata or {}


def _rss_reader():
    """Read resident bytes without an optional process-monitor dependency."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        handle = kernel.GetCurrentProcess()

        def read():
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not get_info(handle, ctypes.byref(counters), counters.cb):
                raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
            return counters.WorkingSetSize
        return read
    if os.path.exists("/proc/self/statm"):
        def read():
            with open("/proc/self/statm", encoding="ascii") as source:
                return int(source.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return read
    return None


@contextmanager
def _cost(metadata):
    read_rss = _rss_reader()
    peak = [read_rss() if read_rss else 0]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(.01):
            peak[0] = max(peak[0], read_rss())

    monitor = threading.Thread(target=sample, daemon=True) if read_rss else None
    if monitor:
        monitor.start()
    owned = not tracemalloc.is_tracing()
    if owned:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        yield
    finally:
        stopped.set()
        if monitor:
            monitor.join()
            peak[0] = max(peak[0], read_rss())
        traced_peak = tracemalloc.get_traced_memory()[1]
        metadata.update(wall_time_seconds=time.perf_counter() - start,
                        peak_memory_bytes=peak[0] if read_rss else traced_peak,
                        peak_traced_memory_bytes=traced_peak,
                        memory_measurement=("process RSS sampled every 10 ms (includes resident baseline)"
                                            if read_rss else "tracemalloc (excludes untraced native allocations)"))
        if owned:
            tracemalloc.stop()


def fit_clean_pca(X, pca_dim=32, seed=0):
    """Fit once on the full clean score cloud; None disables projection."""
    X = _matrix(X)
    if pca_dim is None:
        return None
    if isinstance(pca_dim, bool) or not isinstance(pca_dim, (int, np.integer)) or pca_dim < 1:
        raise ValueError("pca_dim must be a positive integer or None")
    return PCA(n_components=min(pca_dim, *X.shape), svd_solver="full", random_state=seed).fit(X)


def ph_diagrams(X, maxdim=1, pca_dim=32, seed=0, n_points=None, *, pca=None):
    """Compute finite diagrams; pca must be fitted on the clean cloud for pairs.

    Row sampling uses a private RNG, so aligned clouds use identical row indices.
    Infinite bars are removed and counted separately for every homology degree.
    """
    from ripser import ripser

    X = _matrix(X)
    if isinstance(maxdim, bool) or not isinstance(maxdim, (int, np.integer)) or maxdim < 0:
        raise ValueError("maxdim must be a nonnegative integer")
    if n_points is not None and (isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)) or n_points < 1):
        raise ValueError("n_points must be a positive integer or None")
    result = PHResult()
    with _cost(result.metadata):
        if pca is None:
            pca = fit_clean_pca(X, pca_dim, seed)
        if pca is not None:
            X = pca.transform(X)
        indices = (np.sort(np.random.default_rng(seed).choice(len(X), n_points, replace=False))
                   if n_points is not None and n_points < len(X) else np.arange(len(X)))
        diagrams = ripser(X[indices], maxdim=maxdim)["dgms"]
        dropped = {}
        for dim, diagram in enumerate(diagrams):
            finite = np.isfinite(diagram).all(axis=1)
            result[dim] = diagram[finite].copy()
            dropped[dim] = int((~finite).sum())
        result.metadata.update(infinite_bars_dropped=dropped, n_points=len(indices), row_indices=indices.tolist())
    return result


def ph_distance(H, Ht, *, dims=(0, 1), metric="bottleneck", pca_dim=32,
                seed=0, n_points=2000, pca=None):
    """Compare point clouds with one clean-fitted projection (or diagram dicts).

    metric='both' computes both supported distances without repeating ripser.
    """
    import persim

    dims = tuple(dims)
    if not dims or len(set(dims)) != len(dims) or any(not isinstance(d, int) or d < 0 for d in dims):
        raise ValueError("dims must be unique nonnegative integers")
    if metric not in ("bottleneck", "wasserstein", "both"):
        raise ValueError("Unknown PH distance metric")
    result = PHResult()
    with _cost(result.metadata):
        if not isinstance(H, dict):
            H, Ht = _pair(H, Ht)
            projection = pca if pca is not None else fit_clean_pca(H, pca_dim, seed)
            H, Ht = [ph_diagrams(X, max(dims), pca_dim=None, seed=seed,
                                 n_points=n_points, pca=projection) for X in (H, Ht)]
        dropped = {}
        for dim in dims:
            diagrams, counts = [], []
            for cloud in (H, Ht):
                diagram = np.asarray(cloud[dim], dtype=float).reshape(-1, 2)
                finite = np.isfinite(diagram).all(axis=1)
                counts.append(int((~finite).sum()) + getattr(cloud, "metadata", {}).get("infinite_bars_dropped", {}).get(dim, 0))
                diagrams.append(diagram[finite])
            dropped[dim] = dict(clean=counts[0], corrupted=counts[1])
            for name in (("bottleneck", "wasserstein") if metric == "both" else (metric,)):
                result[f"ph_{name}_H{dim}"] = float(getattr(persim, name)(*diagrams))
        result.metadata.update(infinite_bars_dropped=dropped,
                               diagrams=[getattr(cloud, "metadata", {}) for cloud in (H, Ht)])
    return result
