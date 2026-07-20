"""JSON Schema -> GRID grammar compilation (the 0.2.x coverage epoch).

Public API:
    compile_json_schema(schema, strict=False) -> (grammar_source, recorded)

`recorded` is the set of constraint names present in the schema but not
enforced by the grammar (default mode records them; strict=True raises
Unsupported instead — the llguidance-style declared-non-support convention).
"""

from grid.jsonschema.compiler import Unsupported, compile_schema

__all__ = ["compile_json_schema", "Unsupported"]


def compile_json_schema(schema, strict: bool = False):
    """Compile a JSON Schema into .grid grammar source."""
    return compile_schema(schema, strict=strict)
