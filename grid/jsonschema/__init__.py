"""JSON Schema -> GRID grammar compilation (the 0.2.x coverage epoch).

Public API:
    compile_json_schema(schema, strict=False) -> (grammar_source, recorded)
    compile_json_schema_grammar(schema, strict=False)
        -> (FROZEN DialectGrammar, recorded)

`recorded` is the set of constraint names present in the schema but not
enforced by the grammar (default mode records them; strict=True raises
Unsupported instead — the llguidance-style declared-non-support convention).
"""

from grid import perf_flags
from grid.jsonschema.compiler import Unsupported, compile_schema

__all__ = ["compile_json_schema", "compile_json_schema_grammar", "Unsupported"]


def _store_key(schema, strict: bool) -> str | None:
    """Tier-1 artifact-store key, or None for schemas the store must bypass.

    Canonical schema JSON + mode. The canon PRESERVES dict insertion order
    (sort_keys=False): compile_schema output depends on it (keyword-terminal
    numbering and rule naming follow property order), so dict-equal schemas
    differing only in insertion order must never share an entry — and the
    round-trip check below cannot split them, because dict equality ignores
    order. Schemas that don't canonicalize faithfully (non-str keys, tuples,
    NaN, ...) bypass the store: the round-trip check rejects any input
    json.dumps would alias."""
    import hashlib
    import json

    try:
        canon = json.dumps(schema, sort_keys=False, separators=(",", ":"),
                           ensure_ascii=False)
        faithful = json.loads(canon) == schema
    except (TypeError, ValueError):
        faithful = False
    if not faithful:
        return None
    return hashlib.blake2b(
        canon.encode() + b"\x00" + (b"strict" if strict else b"lax"),
        digest_size=16,
    ).hexdigest()


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

    key = _store_key(schema, strict)
    if key is None:
        return compile_schema(schema, strict=strict)
    hit = artifact_store.get("schema_src", key)
    if isinstance(hit, tuple) and len(hit) == 2:
        src, recorded = hit
        return src, set(recorded)
    # Unsupported propagates uncached: error outcomes reproduce exactly warm
    src, recorded = compile_schema(schema, strict=strict)
    artifact_store.put("schema_src", key, (src, tuple(sorted(recorded))))
    return src, recorded


def compile_json_schema_grammar(schema, strict: bool = False,
                                *, unify_string_values: bool = False):
    """Compile a JSON Schema to a (FROZEN DialectGrammar, recorded) pair.

    The grammar-object twin of compile_json_schema for schema->mask compile
    pipelines. Flag-off (GRID_PERF_DIRECT_EMIT unset/"0") this is exactly
    compile_json_schema + spec.load. Flag-on it builds the grammar straight
    from the compiler's GrammarParts manifest (DialectGrammar.from_parts:
    no text render, no regex re-parse); .grid text remains available via
    grid.grammar.parts.render_text as the debug/audit form.

    Artifact-store interplay (GRID_PERF_ARTIFACT_STORE): the schema_src
    namespace stores TEXT — store HITS return it through spec.load in both
    flag states, so direct emission only speeds store MISSES (which still
    render once for the put) and store-off builds. unify_string_values (the
    LALR-conflict retry knob) always bypasses the store, exactly as its
    direct compile_schema call sites did."""
    from grid.grammar import spec

    if not perf_flags.direct_emit_enabled():
        if unify_string_values:
            src, recorded = compile_schema(schema, strict=strict,
                                           unify_string_values=True)
        else:
            src, recorded = compile_json_schema(schema, strict=strict)
        return spec.load(src), recorded

    from grid.jsonschema.compiler import compile_schema_parts

    if not unify_string_values and perf_flags.artifact_store_enabled():
        from grid.serving import artifact_store

        if artifact_store.enabled():
            key = _store_key(schema, strict)
            if key is not None:
                hit = artifact_store.get("schema_src", key)
                if isinstance(hit, tuple) and len(hit) == 2:
                    src, recorded = hit
                    return spec.load(src), set(recorded)
                from grid.grammar.parts import render_text

                parts, recorded = compile_schema_parts(schema, strict=strict)
                # store format stays text; the miss still skips the re-parse
                artifact_store.put(
                    "schema_src", key,
                    (render_text(parts), tuple(sorted(recorded))))
                return spec.DialectGrammar.from_parts(parts), recorded
    parts, recorded = compile_schema_parts(
        schema, strict=strict, unify_string_values=unify_string_values)
    return spec.DialectGrammar.from_parts(parts), recorded
