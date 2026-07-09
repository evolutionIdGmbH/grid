"""E14 AuditLog: hash-chained per-token records (DESIGN.md SS5 E14, G10a).

Record: (step, config_hash, mask_entry_id, chosen_token, blocked_count,
instruction_kind, prev_record_hash). Every step appends — including each token of
a Write span and the EOS step (the chain tail for non-error stops).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

GENERATE, WRITE, EOS = "GENERATE", "WRITE", "EOS"


@dataclass(frozen=True)
class AuditRecord:
    step: int
    config_hash: int
    mask_entry_id: str | None    # None <=> instruction_kind in {WRITE, EOS}
    chosen_token: int
    blocked_count: int
    instruction_kind: str
    record_hash: str = ""

    def compute_hash(self, prev_hash: str) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(prev_hash.encode())
        h.update(
            f"{self.step}|{self.config_hash}|{self.mask_entry_id}|"
            f"{self.chosen_token}|{self.blocked_count}|{self.instruction_kind}".encode()
        )
        return h.hexdigest()


@dataclass
class AuditLog:
    records: list[AuditRecord] = field(default_factory=list)
    sealed: bool = False
    seal_info: dict = field(default_factory=dict)

    GENESIS = "grid-audit-genesis"

    def append(self, step: int, config_hash: int, mask_entry_id: str | None,
               chosen_token: int, blocked_count: int, kind: str) -> AuditRecord:
        assert not self.sealed, "audit log is sealed"
        assert (mask_entry_id is None) == (kind in (WRITE, EOS)), "E14: entry id iff GENERATE"
        prev = self.records[-1].record_hash if self.records else self.GENESIS
        rec = AuditRecord(step, config_hash, mask_entry_id, chosen_token, blocked_count, kind)
        rec = AuditRecord(**{**rec.__dict__, "record_hash": rec.compute_hash(prev)})
        self.records.append(rec)
        return rec

    def seal(self, stop_reason: str, artifact_fingerprints: dict[str, str], flags: dict | None = None) -> None:
        self.seal_info = {
            "stop_reason": stop_reason,
            "chain_head": self.records[-1].record_hash if self.records else self.GENESIS,
            "artifacts": dict(artifact_fingerprints),
            "flags": dict(flags or {}),
        }
        self.sealed = True

    def verify_chain(self) -> bool:
        prev = self.GENESIS
        for rec in self.records:
            if rec.record_hash != rec.compute_hash(prev):
                return False
            prev = rec.record_hash
        if self.sealed and self.records:
            return self.seal_info.get("chain_head") == self.records[-1].record_hash
        return True
