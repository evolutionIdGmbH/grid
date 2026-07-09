"""SS4.4 entry points: generate.cfg / generate.sql (singledispatch on model type)."""

from __future__ import annotations

from functools import singledispatch

from grid.audit.log import AuditLog
from grid.generate.api import GridSequenceGeneratorAdapter
from grid.grammar import spec
from grid.grammar.projection import RoleProjection
from grid.guide import GridGuide
from grid.lalr.compile import compile_tables
from grid.lalr.reserve import ReserveTable
from grid.lexer.dfa import build_scanner
from grid.models.mock import MockModel
from grid.processors import GridLogitsProcessor
from grid.samplers import multinomial
from grid.trie.build import build_trie


def build_guide(
    grammar_source: str,
    adapter,
    projection: RoleProjection | None = None,
    lexicons=None,
    schema_fingerprint: str | None = None,
    identifier_terminals: frozenset[str] = frozenset({"TABLE_NAME", "COLUMN_NAME"}),
    audit: bool = False,
) -> GridGuide:
    """Assemble the full artifact chain: grammar -> projection -> LALR -> scanner ->
    trie -> reserve -> guide."""
    grammar = spec.load(grammar_source)
    proj = projection if projection is not None else RoleProjection.full(grammar).build()
    tables = compile_tables(proj, identifier_terminals if lexicons is not None else frozenset())
    dfa = build_scanner(grammar.terminals, grammar.terminal_order)
    trie = build_trie(adapter)
    reserve = ReserveTable(tables=tables, dfa=dfa, adapter=adapter, lexicons=lexicons)
    return GridGuide(
        tables=tables, dfa=dfa, trie=trie, adapter=adapter,
        lexicons=lexicons, schema_fingerprint=schema_fingerprint,
        reserve=reserve, audit=AuditLog() if audit else None,
    )


@singledispatch
def cfg(model, grammar_source: str, sampler=None, audit: bool = False) -> GridSequenceGeneratorAdapter:
    raise NotImplementedError(f"generate.cfg not implemented for model {type(model)}")


@cfg.register(MockModel)
def _cfg_mock(model: MockModel, grammar_source: str, sampler=None, audit: bool = False):
    guide = build_guide(grammar_source, model.tokenizer, audit=audit)
    processor = GridLogitsProcessor(model.tokenizer, guide)
    return GridSequenceGeneratorAdapter(model, processor, sampler or multinomial(), mode="cfg")


@singledispatch
def sql(model, grammar_source: str, policy=None, schema=None, sampler=None, audit: bool = True):
    raise NotImplementedError(f"generate.sql not implemented for model {type(model)}")


def _sql_impl(model, grammar_source: str, policy=None, schema=None, sampler=None,
              audit: bool = True):
    grammar = spec.load(grammar_source)
    proj = policy.projection(grammar) if policy is not None else RoleProjection.full(grammar).build()
    tables = compile_tables(proj, frozenset({"TABLE_NAME", "COLUMN_NAME"}))
    lexicons = schema.lexicons(tables, policy) if schema is not None else None
    dfa = build_scanner(grammar.terminals, grammar.terminal_order)
    trie = build_trie(model.tokenizer)
    reserve = ReserveTable(tables=tables, dfa=dfa, adapter=model.tokenizer, lexicons=lexicons)
    guide = GridGuide(
        tables=tables, dfa=dfa, trie=trie, adapter=model.tokenizer,
        lexicons=lexicons, schema_fingerprint=schema.fingerprint if schema is not None else None,
        reserve=reserve, audit=AuditLog() if audit else None,
    )
    processor = GridLogitsProcessor(model.tokenizer, guide)
    return GridSequenceGeneratorAdapter(model, processor, sampler or multinomial(), mode="sql")


sql.register(MockModel)(_sql_impl)


def _cfg_impl(model, grammar_source: str, sampler=None, audit: bool = False):
    guide = build_guide(grammar_source, model.tokenizer, audit=audit)
    processor = GridLogitsProcessor(model.tokenizer, guide)
    return GridSequenceGeneratorAdapter(model, processor, sampler or multinomial(), mode="cfg")


try:  # transformers is an optional extra (DESIGN.md SS12)
    from grid.models.transformers_model import TransformersModel

    cfg.register(TransformersModel)(_cfg_impl)
    sql.register(TransformersModel)(_sql_impl)
except ImportError:  # pragma: no cover
    pass
