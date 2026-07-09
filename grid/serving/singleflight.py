"""E17 RegistrySlot single-flight (DESIGN.md §4/E17).

One slot per requested fingerprint: ⊳PENDING → READY | FAILED(err, ttl).
All concurrent waiters on PENDING receive the same result or the same
exception object. FAILED is negatively cached with a TTL to prevent recompile
storms on known-bad fingerprints; TTL expiry removes the slot and the next
request re-enters PENDING. READY values are treated as immutable and shared.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PENDING, READY, FAILED = "PENDING", "READY", "FAILED"


@dataclass
class _Slot:
    state: str = PENDING
    event: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    error: BaseException | None = None
    expiry: float = 0.0  # FAILED only


class SingleFlight:
    """get_or_build(key, builder): the E17 state machine over a dict of slots."""

    def __init__(self, failed_ttl_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self._slots: dict[Any, _Slot] = {}
        self._lock = threading.Lock()
        self._ttl = failed_ttl_s
        self._clock = clock
        self.stats = {"builds": 0, "joined": 0, "failures": 0, "negative_hits": 0,
                      "ready_hits": 0}

    def get_or_build(self, key: Any, builder: Callable[[], Any]) -> Any:
        while True:
            with self._lock:
                slot = self._slots.get(key)
                if slot is not None and slot.state == FAILED and self._clock() >= slot.expiry:
                    del self._slots[key]  # TTL expiry: next request re-enters PENDING
                    slot = None
                if slot is None:
                    slot = _Slot()
                    self._slots[key] = slot
                    building = True
                else:
                    building = False
                    if slot.state == READY:
                        self.stats["ready_hits"] += 1
                        return slot.value
                    if slot.state == FAILED:
                        self.stats["negative_hits"] += 1
                        raise slot.error
                    self.stats["joined"] += 1

            if building:
                self.stats["builds"] += 1
                try:
                    value = builder()
                except BaseException as exc:
                    with self._lock:
                        slot.state = FAILED
                        slot.error = exc
                        slot.expiry = self._clock() + self._ttl
                    self.stats["failures"] += 1
                    slot.event.set()
                    raise
                with self._lock:
                    slot.state = READY
                    slot.value = value
                slot.event.set()
                return value

            slot.event.wait()
            with self._lock:
                if slot.state == READY:
                    return slot.value
                if slot.state == FAILED:
                    self.stats["negative_hits"] += 1
                    raise slot.error
            # PENDING again should be impossible (event only sets on resolve);
            # loop defensively rather than deadlock.

    def evict(self, key: Any) -> None:
        """READY supersede / manual invalidation (E17 'artifact superseded')."""
        with self._lock:
            slot = self._slots.get(key)
            if slot is not None and slot.state != PENDING:
                del self._slots[key]
