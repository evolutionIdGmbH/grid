"""GRID error taxonomy (DESIGN.md SS7).

Design rule: generation-time exceptions are bugs (DeadEndError, IllegalTransition,
LexerHypothesisOverflow, IdentifierMaskBypassError). Everything user-fixable fails
at compile/verify/call time.
"""

from __future__ import annotations


class GridError(Exception):
    """Base class for all GRID errors."""


class GrammarInvalid(GridError):
    """E1/E2 validation failure. Fix the grammar/policy; never raised at generation time."""


class LALRConflictError(GridError):
    """E4 compile failure; ``report`` lists conflict states/lookaheads."""

    def __init__(self, report: list) -> None:
        super().__init__(f"LALR(1) conflicts: {report[:5]}{'...' if len(report) > 5 else ''}")
        self.report = report


class EmptyLanguageError(GrammarInvalid):
    """E2 verify: L(G_role) is empty. Policy misconfiguration; refuse the role."""


class LexiconBuildError(GridError):
    """E3 MATERIALIZING -> INVALID. Policy/schema author fixes."""


class TrieBuildError(GridError):
    """E5 BUILDING -> FAILED. Tokenizer-adapter defect; file, don't catch."""


class LexerHypothesisOverflow(GridError):
    """INV-LEX1 breach at runtime. Compile-side bug (H_max wrong); file, don't catch."""


class DeadEndError(GridError):
    """Empty mask at SS6 step 8. Bug by theorem: abort generation, dump state. G5 asserts zero."""


class IdentifierMaskBypassError(GridError):
    """Generic-IDENT cache entry consulted at an identifier position (E11 key-type guard).

    Active in ALL builds. Always a bug; G6 injects the condition and asserts it fires.
    """


class ProcessorReuseError(GridError):
    """E13 FINISHED reuse (including after stop_at/ERROR stops via finish())."""


class AuditFlushError(GridError):
    """E14 strict-mode flush failure."""


class IllegalTransition(GridError):
    """A state machine attempted a transition not present in its statechart YAML."""

    def __init__(self, entity: str, from_state: str, trigger: str) -> None:
        super().__init__(f"{entity}: illegal transition from {from_state!r} on {trigger!r}")
        self.entity = entity
        self.from_state = from_state
        self.trigger = trigger


class StaleArtifactError(GridError):
    """Audit replay against a missing archived namespace. Archival misconfiguration."""
