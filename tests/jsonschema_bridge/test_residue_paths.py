"""Path-qualified residue: compile_schema_with_paths locates each recorded
constraint on an instance-shaped path, without disturbing the legacy set or
the compiled grammar (byte-identical to compile_schema)."""

import pytest

from grid.jsonschema import Unsupported, compile_schema_with_paths
from grid.jsonschema.compiler import compile_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "multipleOf": 0.25},
        "tags": {"type": "array", "items": {"type": "string"},
                 "uniqueItems": True},
        "user": {
            "type": "object",
            "properties": {"ratio": {"type": "number", "multipleOf": 0.1}},
            "additionalProperties": False,
        },
    },
    "required": ["score"],
    "additionalProperties": False,
}


def test_paths_locate_property_level_records():
    _src, _rec, paths = compile_schema_with_paths(SCHEMA)
    assert paths["$.score"] == {"multipleOf"}
    assert paths["$.tags"] == {"uniqueItems"}


def test_paths_descend_into_nested_objects():
    _src, _rec, paths = compile_schema_with_paths(SCHEMA)
    assert paths["$.user.ratio"] == {"multipleOf"}


def test_paths_descend_into_array_items():
    schema = {"type": "object", "properties": {
        "rows": {"type": "array", "items": {
            "type": "object",
            "properties": {"x": {"type": "number", "multipleOf": 2}},
            "additionalProperties": False}}},
        "additionalProperties": False}
    _src, _rec, paths = compile_schema_with_paths(schema)
    assert paths["$.rows[*].x"] == {"multipleOf"}


def test_non_identifier_keys_use_bracket_segments():
    schema = {"type": "object", "properties": {
        "weird key!": {"type": "number", "multipleOf": 3}},
        "additionalProperties": False}
    _src, _rec, paths = compile_schema_with_paths(schema)
    assert paths["$['weird key!']"] == {"multipleOf"}


def test_legacy_set_and_grammar_are_unchanged():
    src_legacy, rec_legacy = compile_schema(SCHEMA)
    src_paths, rec_paths, paths = compile_schema_with_paths(SCHEMA)
    assert src_legacy == src_paths
    assert rec_legacy == rec_paths
    # every located name appears in the legacy set and vice versa
    names = {feat for feats in paths.values() for feat in feats}
    assert names == rec_paths


def test_strict_error_carries_path_attribute():
    with pytest.raises(Unsupported) as ei:
        compile_schema_with_paths(SCHEMA, strict=True)
    assert str(ei.value).startswith("strict: ")
    assert getattr(ei.value, "path", "").startswith("$.")


def test_clean_schema_has_empty_paths():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}},
              "required": ["a"], "additionalProperties": False}
    _src, rec, paths = compile_schema_with_paths(schema)
    assert rec == set() and paths == {}
