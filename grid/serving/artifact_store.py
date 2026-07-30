"""Versioned on-disk compile-artifact store (flag: GRID_PERF_ARTIFACT_STORE).

Warm deployments reload compile artifacts instead of rebuilding them:

- namespace ``schema_src``: canonical-schema key -> (.grid source, recorded
  set), wired inside :func:`grid.jsonschema.compile_json_schema`;
- namespace ``scanner``: grammar key -> pickled ScannerDFA (lazy facades are
  skipped — their warmth comes from ``component``);
- namespace ``lalr``: projection key -> pickled LALRTables;
- namespace ``component`` (S3): (kind, budget, pattern) -> pickled
  TerminalDFA or a breach marker — CROSS-SCHEMA (patterns recur corpus-wide),
  sub-flag GRID_PERF_STORE_COMPONENTS;
- namespace ``trie`` (S3): (tokenizer fingerprint, slicer variant) ->
  pickled TokenTrie, sub-flag GRID_PERF_STORE_TRIE; the S2 slicer flag is a
  build-time input (TrieSlices are baked into the payload), so each flag
  state has its own entry — a slice-carrying trie must never leak into a
  slicer-off process (whose kill-switch contract is the full walk), nor the
  reverse;
- namespace ``journal`` (S3): blake2b(grammar_src) -> ContextJournal
  snapshot (walk-miss keys/contexts only, never masks), sub-flag
  GRID_PERF_STORE_JOURNAL; timing-only by construction.

Deferred namespaces (S3 step 8, reserved key shapes under
:func:`kernel_fingerprint`, no payloads this epoch): persisted T2 mask blobs
(#20b) and RustWalker ingestion arenas.

Entries live under ``<root>/<code_epoch>/<namespace>/<key>.bin``. code_epoch
content-hashes the package version, the Python major.minor, and every pipeline
source module, so any engine change invalidates the store wholesale. Loads
verify an envelope (format, epoch, namespace, key) and self-heal (unlink +
rebuild) on any mismatch or corruption; writes are atomic (per-writer tmp —
pid, thread id, counter — + os.replace), so concurrent writers, threads
included, at worst duplicate work with identical content. Failed builds raise
before any put, so error outcomes reproduce
exactly on warm runs. With the flag off every helper is a pure passthrough to
the underlying builder.

Security: entries are pickles, and unpickling executes code. The store is safe
only because artifacts are self-produced under a user-owned directory (default
``~/.cache/grid``). Never point GRID_CACHE_DIR at a shared or world-writable
path.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import pickle
import sys
import threading
import warnings
from functools import lru_cache
from pathlib import Path

from grid import perf_flags
from grid.grammar.projection import RoleProjection
from grid.grammar.spec import DialectGrammar
from grid.lalr.compile import LALRTables, compile_tables
from grid.lexer.dfa import ScannerDFA, build_scanner

# 2: tier-1 schema_src canon switched from sorted-key to insertion-order JSON
#    (compile_schema output depends on dict insertion order, so sorted-key
#    entries alias order-variant schemas and must never be served; the keying
#    module is not in _EPOCH_MODULES, so the epoch alone would not evict them)
FORMAT = 2

# Every module whose source participates in producing a stored artifact —
# broader than the writer call sites: Terminal.priority and role_shape_hash
# semantics live in spec.py/projection.py. factored.py produces stored
# payloads two ways (ScannerDFA arrays under GRID_PERF_FACTORED_SCANNER via
# materialize, and the per-terminal component namespace), trie/build.py the
# trie namespace; rx/nfa/subset are the E2 split of dfa.py — their sources
# feed every scanner and component payload just as when they lived inside
# dfa.py; grammar/parts.py renders the schema_src TEXT payload on P2
# direct-emission store misses (render_text), so an emitter change must
# evict stored text. The epoch is a directory name, so adding a module here
# wholesale-invalidates without a FORMAT bump.
_EPOCH_MODULES = (
    "grid.lexer.dfa",
    "grid.lexer.factored",
    "grid.lexer.nfa",
    "grid.lexer.rx",
    "grid.lexer.subset",
    "grid.lalr.compile",
    "grid.jsonschema.compiler",
    "grid.jsonschema.normalize",
    "grid.jsonschema.rx",
    "grid.grammar.parts",
    "grid.grammar.spec",
    "grid.grammar.projection",
    "grid.grammar.reduction",
    "grid.trie.build",
)

_put_warned = False
_tmp_counter = itertools.count()


def enabled() -> bool:
    return perf_flags.artifact_store_enabled()


def root() -> Path:
    override = os.environ.get("GRID_CACHE_DIR")
    return Path(override) if override else Path.home() / ".cache" / "grid"


@lru_cache(maxsize=1)
def code_epoch() -> str:
    import importlib
    import importlib.metadata

    h = hashlib.blake2b(digest_size=16)
    try:
        version = importlib.metadata.version("grid-guardrail")
    except importlib.metadata.PackageNotFoundError:
        version = "0+src"
    h.update(version.encode())
    # a Python upgrade invalidates wholesale rather than risking cross-version
    # unpickle errors mid-serving
    h.update(f"|py{sys.version_info[0]}.{sys.version_info[1]}|".encode())
    for name in _EPOCH_MODULES:
        mod = importlib.import_module(name)
        h.update(name.encode())
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()


def get(namespace: str, key: str) -> object | None:
    if not enabled():
        return None
    try:  # a broken store must never break a compile: degrade to a miss
        # (code_epoch reads module sources off disk — pyc-only deployments)
        path = root() / code_epoch() / namespace / f"{key}.bin"
        blob = path.read_bytes()
    except Exception:
        return None
    try:
        env = pickle.loads(blob)
        if (
            isinstance(env, dict)
            and env.get("format") == FORMAT
            and env.get("epoch") == code_epoch()
            and env.get("namespace") == namespace
            and env.get("key") == key
        ):
            return env["payload"]
    except Exception:
        pass
    try:  # corrupt or foreign entry: self-heal so the rebuild's put wins
        path.unlink()
    except OSError:
        pass
    return None


def put(namespace: str, key: str, payload: object) -> None:
    global _put_warned
    if not enabled():
        return
    try:
        d = root() / code_epoch() / namespace
        d.mkdir(parents=True, exist_ok=True)
        blob = pickle.dumps(
            {"format": FORMAT, "epoch": code_epoch(), "namespace": namespace,
             "key": key, "payload": payload},
            protocol=5,
        )
        tmp = d / (f"{key}.bin.tmp.{os.getpid()}"
                   f".{threading.get_ident()}.{next(_tmp_counter)}")
        tmp.write_bytes(blob)
        os.replace(tmp, d / f"{key}.bin")
    except Exception as exc:  # a broken store must never break a compile
        if not _put_warned:
            _put_warned = True
            warnings.warn(
                f"GRID artifact store: put failed, continuing without persistence ({exc})",
                stacklevel=2,
            )


def _order_key(terminal_order: tuple[str, ...]) -> str:
    # DialectGrammar.fingerprint hashes terminals sorted by NAME and omits
    # declaration order, but terminal ids and priority tie-breaks are
    # positional in terminal_order: two sources with the same terminal set in
    # different declaration order share a fingerprint yet need different
    # artifacts, so the order is hashed into the store key explicitly.
    return hashlib.blake2b(",".join(terminal_order).encode(), digest_size=16).hexdigest()


def load_or_build_scanner(grammar: DialectGrammar) -> ScannerDFA:
    """Drop-in for ``build_scanner(grammar.terminals, grammar.terminal_order)``."""
    if not enabled():
        return build_scanner(grammar.terminals, grammar.terminal_order)
    key = f"{grammar.fingerprint}:{_order_key(grammar.terminal_order)}"
    hit = get("scanner", key)
    if isinstance(hit, ScannerDFA):
        return hit
    dfa = build_scanner(grammar.terminals, grammar.terminal_order)
    if not getattr(dfa, "lazy", False):
        # lazy facades (over-budget factored products) hold locks + demand
        # state: never picklable, and their redeploy warmth comes from the
        # component namespace instead — skip the put rather than tripping
        # the one-shot store-degraded warning on a TypeError
        put("scanner", key, dfa)
    return dfa


# breach-marker payload for the component namespace: (pattern, is_literal)
# breached this exact budget — deterministic (subset_construct is FIFO), so
# the warm path skips the doomed eager attempt and builds the demand-interned
# component directly. Versioned string, never a pickle of live state.
_COMPONENT_BREACH = "component-breach:v1"


def component_key(pattern: str, is_literal: bool, budget: int | None) -> str:
    """Component-namespace key. Unlike the exact-schema-keyed namespaces the
    identity is the (pattern, is_literal) pair the schema compiler already
    dedupes terminals on — cross-schema by construction (STRING_RX / format
    patterns / keyword literals recur corpus-wide). The RESOLVED budget is
    part of the key because the artifact depends on it (breach vs eager is a
    budget property; the in-memory memo keys on it for the same reason)."""
    h = hashlib.blake2b(digest_size=16)
    h.update(b"L" if is_literal else b"R")
    h.update(str(budget).encode())
    h.update(b"|")
    h.update(pattern.encode("utf-8", "surrogatepass"))
    return h.hexdigest()


def load_or_build_component(pattern: str, is_literal: bool, budget: int | None):
    """Drop-in for ``factored._build_component(pattern, is_literal, budget)``
    (call it with the RESOLVED budget, as factored._component does).

    Payloads: TerminalDFA (frozen tuple dataclass, pickle-roundtrip-equal)
    for components that terminate within budget; the _COMPONENT_BREACH
    sentinel for budget breaches — the substring-union family's capped
    attempt is seconds of subset construction per member (S3 step-1
    measurement), and the LazyTerminalDFA it produces holds locks/closures
    that can never be pickled, so the marker persists the DECISION and the
    warm path rebuilds the cheap NFA artifacts only."""
    from grid.lexer.factored import TerminalDFA, _build_component, _lazy_component

    if not (enabled() and perf_flags.store_components_enabled()):
        return _build_component(pattern, is_literal, budget)
    key = component_key(pattern, is_literal, budget)
    hit = get("component", key)
    if isinstance(hit, TerminalDFA):
        return hit
    if hit == _COMPONENT_BREACH:
        return _lazy_component(pattern, is_literal)
    # GrammarInvalid (bad regex) raises HERE, before any put — error
    # outcomes reproduce exactly on warm runs (store law)
    comp = _build_component(pattern, is_literal, budget)
    put("component", key,
        comp if isinstance(comp, TerminalDFA) else _COMPONENT_BREACH)
    return comp


@lru_cache(maxsize=1)
def kernel_fingerprint() -> str | None:
    """blake2b of the grid_core extension binary (None without the kernel).

    RESERVED-KEY component only this epoch: grid_core exports no
    __version__/blob-format constant, and both deferred namespaces below key
    on the exact kernel build because their payloads are kernel-native:

    - T2 mask blobs (CANDIDATES #20b): MaskEntryV7.blob is register_blob's
      own export format — a wrong-key hit is the forbidden served-wrong-mask
      class, so persistence waits for a served-mask-parity gate. Reserved
      key: (dialect, schema_fp, tokenizer_fingerprint, vocab_size,
      kernel_fingerprint(), blob-format const).
    - RustWalker ingestion arenas: reserved key (scanner key, trie
      fingerprint, kernel_fingerprint(), kernel word width); blocked on the
      RUST_SCANNER un-hold / size-gated-dispatch decision.

    No payload is written under either shape this epoch (S3 step 8)."""
    try:
        import grid_core

        so = next(
            p for p in Path(grid_core.__file__).parent.iterdir()
            if p.suffix == ".so" or ".so." in p.name or p.suffix == ".pyd"
        )
        return hashlib.blake2b(so.read_bytes(), digest_size=16).hexdigest()
    except Exception:
        return None


def journal_key(grammar_src: str) -> str:
    """The dialect identity: blake2b of the grammar source — the same scope
    as the registry's T2 pools and journals (one journal per dialect)."""
    return hashlib.blake2b(grammar_src.encode(), digest_size=16).hexdigest()


def load_or_restore_journal(grammar_src: str):
    """ContextJournal for this dialect, restored from the ``journal``
    namespace and bound for self-flush when the store + sub-flag are on;
    otherwise exactly today's fresh in-memory journal. Only ever called from
    the GRID_ADMIT_WARM=1 wiring (_GuideRegistry._build), so restore is
    additionally inert without that switch. Payloads are keys/contexts only
    (tier-i generic/genN tuples, tier-ii ident frozensets) — a restored entry
    can warm an entry nobody consults, never serve a wrong mask: tier-i only
    feeds T2-donor adoption and tier-ii feeds exact walks."""
    from grid.serving.journal import ContextJournal

    journal = ContextJournal()
    if not (enabled() and perf_flags.store_journal_enabled()):
        return journal
    key = journal_key(grammar_src)
    try:  # warm-hint only: a corrupt payload restores nothing
        hit = get("journal", key)
        if hit is not None:
            journal.restore(hit)
    except Exception:  # pragma: no cover - defensive
        pass
    journal.bind_store(key)
    return journal


def load_or_build_trie(adapter):
    """Drop-in for ``build_trie(adapter)`` (grid/trie/build.py).

    Namespace ``trie`` keyed by (tokenizer fingerprint, slicer variant). The
    fingerprint is computed from the (token bytes, id) table — the build's
    SOLE input besides the S2 flag — via the same single pass a cold build
    consumes, so a hit costs one token_bytes iteration + unpickle and a miss
    never iterates twice. The pure-Python DFS build this skips measured
    178 ms on gpt2/50k vocab and runs once per process on the first-request
    path.

    GRID_PERF_SLICER is read at BUILD time and bakes TrieSlices into the
    returned object, so the key carries the flag state: without it a
    slicer-on deployment's entry would hand a slice-carrying trie to a
    slicer-off process (voiding the kill switch's full-walk contract
    cross-process), and a slicer-off entry would silently disable a
    slicer-on deployment. Both variants coexist per fingerprint; within one
    variant the payload is deterministic (coverage-based slices=None under
    s1 is a function of the entries, hence of the fingerprint)."""
    from grid.trie.build import (
        TokenTrie,
        _build_from_entries,
        _entries_fingerprint,
        _token_entries,
    )

    if not (enabled() and perf_flags.store_trie_enabled()):
        from grid.trie.build import build_trie

        return build_trie(adapter)
    entries = _token_entries(adapter)
    fp = _entries_fingerprint(entries)
    key = f"{fp}:{'s1' if perf_flags.slicer_enabled() else 's0'}"
    hit = get("trie", key)
    if isinstance(hit, TokenTrie) and hit.tokenizer_fingerprint == fp:
        return hit
    trie = _build_from_entries(entries)
    put("trie", key, trie)
    return trie


def load_or_compile_tables(
    proj: RoleProjection, identifier_terminals: frozenset[str] = frozenset()
) -> LALRTables:
    """Drop-in for ``compile_tables(proj, identifier_terminals)``."""
    if not enabled() or proj.state != "CACHED":
        # non-CACHED projections must raise compile_tables' own ValueError
        return compile_tables(proj, identifier_terminals)
    expected_fp = f"{proj.base.fingerprint}:{proj.role_shape_hash}"
    # identifier_terminals is a compile_tables input NOT covered by
    # LALRTables.fingerprint (it only sets identifier_terminal_ids), so it is
    # part of the key
    ident_key = hashlib.blake2b(
        ",".join(sorted(identifier_terminals)).encode(), digest_size=16
    ).hexdigest()
    key = f"{expected_fp}:{_order_key(proj.base.terminal_order)}:{ident_key}"
    hit = get("lalr", key)
    if isinstance(hit, LALRTables) and hit.fingerprint == expected_fp:
        return hit
    tables = compile_tables(proj, identifier_terminals)
    put("lalr", key, tables)
    return tables
