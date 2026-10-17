"""Timing, peak-memory and latency measurement (REVIEW-FEEDBACK.md §3.8). """

from __future__ import annotations

import os
import statistics
import threading
import time
from typing import Callable

import numpy as np
import psutil


class PeakMemorySampler:
    """Samples RSS of this process every `interval` seconds while active."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss / (1024 * 1024)
            self.peak_rss_mb = max(self.peak_rss_mb, rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak_rss_mb = self._proc.memory_info().rss / (1024 * 1024)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def measure_batch_latency(
    score_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    repeats: int = 5,
    warmup: int = 3,
) -> dict[str, float]:
    """Median microseconds per row over full-array scoring passes."""
    for _ in range(warmup):
        score_fn(X)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        score_fn(X)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return {"batch_rows": int(len(X)), "batch_median_s": med, "batch_us_per_row": med / len(X) * 1e6}


def measure_single_flow_latency(
    score_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    n: int = 1000,
    warmup: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """p50/p95 milliseconds for scoring one row at a time."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=n + warmup)
    times = []
    for i, j in enumerate(idx):
        row = X[j : j + 1]
        t0 = time.perf_counter()
        score_fn(row)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt * 1e3)
    arr = np.asarray(times)
    return {"single_n": n, "single_p50_ms": float(np.percentile(arr, 50)), "single_p95_ms": float(np.percentile(arr, 95))}


def torch_peak_memory_mb() -> float | None:
    try:
        import torch

        if torch.backends.mps.is_available():
            return float(torch.mps.driver_allocated_memory() / (1024 * 1024))
    except Exception:
        return None
    return None