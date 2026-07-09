"""Serving-side machinery for the SS6 batch scheduling contract (M6):

- singleflight: E17 RegistrySlot semantics — one build per fingerprint,
  N waiters share the result or the exception, FAILED negatively cached.
- prefetch: overlap — successor-state masks build on a worker thread behind
  the GPU forward window; fill-time waits are the bounded residual.
"""

from grid.serving.prefetch import MaskPrefetcher
from grid.serving.singleflight import SingleFlight

__all__ = ["MaskPrefetcher", "SingleFlight"]
