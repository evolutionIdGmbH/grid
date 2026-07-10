"""SS6 batch scheduling contract, overlap half (DESIGN.md §6, M6).

The contract: masks are computed on CPU overlapped with the GPU forward pass;
a request whose mask is not ready at sampling time is skipped for the round,
never served an approximate mask.

Realization against vLLM 0.24 V1 (no per-step defer hook exists for RUNNING
requests — admission-time compile gating is vLLM-native via its async grammar
executor, which holds WAITING requests without stalling the batch):

- ``schedule(guide, state)`` is called from ``accept_tokens`` — the moment the
  successor state is known — and ONLY when that state's mask is not already
  T1-warm (``GridGuide.is_mask_warm``): the warm steady state must never
  queue behind the pool (unconditional scheduling serialized every request's
  step behind the worker queue + GIL ping-pong — the G8 batched-TPOT
  pathology). A worker thread builds the cold state's mask; the ms-scale cold
  trie walk runs with the GIL RELEASED (kernel v4 walk detach), so the
  scheduler thread keeps moving. The build lands in the shared write-back
  MaskCache; by the time the scheduler asks fill_bitmask, the entry is warm.
- ``wait(state)`` at fill time blocks only for the un-hidden remainder of an
  in-flight build (the bounded residual G8's adversarial arm measures);
  approximate masks are never substituted, by design.

Duplicate scheduling of a configuration is deduplicated per live state object;
racing builds of the same cache key are safe by construction (publish is
idempotent by content hash, OBL-KEY1).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


def _nice_this_thread() -> None:
    """Lower THIS thread's scheduling priority (Linux; no-op elsewhere).
    GRID_POOL_NICE (default 10, 0 disables): pool workers doing cold walks
    share the host with the live engine loop — under CPU/memory-bandwidth
    contention during a fresh schema's window, the engine must win. Uses
    per-thread setpriority(PRIO_PROCESS, tid) via ctypes (os.nice would
    renice the whole process)."""
    import ctypes
    import os
    import sys

    if sys.platform != "linux":
        return
    try:
        nice = min(int(os.environ.get("GRID_POOL_NICE", "10")), 19)
    except ValueError:
        nice = 10
    if nice <= 0:
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        tid = libc.syscall(186)  # SYS_gettid on x86_64; aarch64 uses 178
        if tid < 0:
            tid = libc.syscall(178)
        if tid > 0:
            libc.setpriority(0, tid, nice)  # PRIO_PROCESS
    except Exception:
        pass  # scheduling hint only; never fail a build over it


class MaskPrefetcher:
    """Successor-state mask builder: hide cold walks behind the forward pass."""

    def __init__(self, max_workers: int = 1) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="grid-mask-prefetch",
                                        initializer=_nice_this_thread)
        self._inflight: dict[int, tuple[Any, Future]] = {}  # id(state) -> (state ref, future)
        self._lock = threading.Lock()
        self.stats = {"scheduled": 0, "deduped": 0, "waits": 0,
                      "wait_ms_total": 0.0, "wait_ms_max": 0.0, "errors": 0}

    def schedule(self, guide, state) -> None:
        """Kick a background build of `state`'s mask (idempotent per state)."""
        key = id(state)
        with self._lock:
            if key in self._inflight:
                self.stats["deduped"] += 1
                return
            fut = self._pool.submit(self._build, guide, state)
            # hold the state ref: id() keys stay valid while the entry lives
            self._inflight[key] = (state, fut)
            self.stats["scheduled"] += 1

    def _build(self, guide, state) -> None:
        # _mask_ids walks + publishes + registers on a miss; near-free on a hit.
        # No audit records are written on the mask path (only _advance appends).
        guide._mask_ids(state)

    def done(self, state) -> bool:
        """Non-blocking readiness probe (W6 defer chassis): True when no
        build for `state` is in flight, or the in-flight build has finished
        (including errored — fill's `wait` swallows the error and the fill
        path recomputes synchronously, so a dead future must read ready).
        No side effects: the entry stays in `_inflight` for wait() to
        consume."""
        with self._lock:
            got = self._inflight.get(id(state))
        return got is None or got[1].done()

    def wait(self, state, timeout: float | None = None) -> float:
        """Block until `state`'s scheduled build completes (no-op when none is
        in flight). Returns the milliseconds actually waited — the residual
        the forward window did not hide."""
        with self._lock:
            got = self._inflight.pop(id(state), None)
        if got is None:
            return 0.0
        _state, fut = got
        t0 = time.perf_counter()
        try:
            fut.result(timeout=timeout)
        except Exception:
            # the fill path recomputes synchronously and surfaces real errors
            self.stats["errors"] += 1
        waited = (time.perf_counter() - t0) * 1e3
        self.stats["waits"] += 1
        self.stats["wait_ms_total"] += waited
        self.stats["wait_ms_max"] = max(self.stats["wait_ms_max"], waited)
        return waited

    def drop(self, state) -> None:
        """Forget a scheduled build (rollback/reset paths); the build itself
        still completes and warms the cache — never wasted, never wrong."""
        with self._lock:
            self._inflight.pop(id(state), None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
