"""Versioned on-disk compile-artifact store (flag: GRID_PERF_ARTIFACT_STORE).

Warm deployments reload compile artifacts instead of rebuilding them:

- namespace ``schema_src``: canonical-schema key -> (.grid source, recorded
  set), wired inside :func:`grid.jsonschema.compile_json_schema`;
- namespace ``scanner``: grammar key -> pickled ScannerDFA;
- namespace ``lalr``: projection key -> pickled LALRTables.

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
# broader than the three writer call sites: Terminal.priority and
# role_shape_hash semantics live in spec.py/projection.py.
_EPOCH_MODULES = (
    "grid.lexer.dfa",
    "grid.lalr.compile",
    "grid.jsonschema.compiler",
    "grid.jsonschema.normalize",
    "grid.jsonschema.rx",
    "grid.grammar.spec",
    "grid.grammar.projection",
    "grid.grammar.reduction",
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
    put("scanner", key, dfa)
    return dfa


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
