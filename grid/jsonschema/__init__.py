"""JSON Schema -> GRID grammar compilation (the 0.2.x coverage epoch).

Public API:
    compile_json_schema(schema, strict=False) -> (grammar_source, recorded)

`recorded` is the set of constraint names present in the schema but not
enforced by the grammar (default mode records them; strict=True raises
Unsupported instead — the llguidance-style declared-non-support convention).
"""

from grid import perf_flags
from grid.jsonschema.compiler import Unsupported, compile_schema

__all__ = ["compile_json_schema", "Unsupported"]


def compile_json_schema(schema, strict: bool = False):
    """Compile a JSON Schema into .grid grammar source."""
    # Flag pre-check BEFORE the serving import: grid.serving pulls the
    # journal/prefetch/projection/statecharts/yaml chain (~3-7ms per
    # process), which flag-off fast builds must not pay — this uniform
    # per-process cost was the entire fast-schema p50 1.12 regression in
    # the combined bake-off. Same reader artifact_store.enabled() wraps
    # (grid/perf_flags.py), so the short-circuit can never diverge from
    # the store's own decision.
    if not perf_flags.artifact_store_enabled():
        return compile_schema(schema, strict=strict)

    from grid.serving import artifact_store

    if not artifact_store.enabled():
        return compile_schema(schema, strict=strict)

    import hashlib
    import json

    # Tier-1 store key: canonical schema JSON + mode. The canon PRESERVES dict
    # insertion order (sort_keys=False): compile_schema output depends on it
    # (keyword-terminal numbering and rule naming follow property order), so
    # dict-equal schemas differing only in insertion order must never share an
    # entry — and the round-trip check below cannot split them, because dict
    # equality ignores order. Schemas that don't canonicalize faithfully
    # (non-str keys, tuples, NaN, ...) bypass the store: the round-trip check
    # rejects any input json.dumps would alias.
    try:
        canon = json.dumps(schema, sort_keys=False, separators=(",", ":"), ensure_ascii=False)
        faithful = json.loads(canon) == schema
    except (TypeError, ValueError):
        faithful = False
    if not faithful:
        return compile_schema(schema, strict=strict)
    key = hashlib.blake2b(
        canon.encode() + b"\x00" + (b"strict" if strict else b"lax"), digest_size=16
    ).hexdigest()
    hit = artifact_store.get("schema_src", key)
    if isinstance(hit, tuple) and len(hit) == 2:
        src, recorded = hit
        return src, set(recorded)
    # Unsupported propagates uncached: error outcomes reproduce exactly warm
    src, recorded = compile_schema(schema, strict=strict)
    artifact_store.put("schema_src", key, (src, tuple(sorted(recorded))))
    return src, recorded
