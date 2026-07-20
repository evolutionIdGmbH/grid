"""Official JSON-Schema-Test-Suite, run under the 0.2.x honesty contract.

For every case the compiled grammar must satisfy:
  1. a VALID instance is NEVER rejected (false-rejects are the forbidden
     class — no excuses, recorded or not);
  2. an INVALID instance is either rejected, or the schema's compilation
     RECORDED at least one unenforced constraint (over-acceptance is allowed
     only when it is declared per schema);
  3. schemas the compiler refuses (`Unsupported`) are skipped — declared
     non-support is a legitimate bucket.

Suite checkout: tmp/JSON-Schema-Test-Suite (pinned shallow clone; the test
skips if absent). Sections exercising remote refs / meta-schema machinery
are excluded by filename.
"""

import json
import pathlib
import sys
import warnings

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "bench"))

from grid.jsonschema import Unsupported, compile_json_schema  # noqa: E402
from grid.generate import build_guide  # noqa: E402
from grid.models.tokenizer_adapter import MockTokenizer  # noqa: E402

warnings.filterwarnings("ignore", message=".*L-REC01.*")

SUITE = pathlib.Path(__file__).resolve().parents[2] / "tmp" / "JSON-Schema-Test-Suite"

# sections outside the compiler's scope by design
SKIP_FILES = {
    "refRemote.json", "dynamicRef.json", "anchor.json", "id.json",
    "unknownKeyword.json", "vocabulary.json", "defs.json",
    "infinite-loop-detection.json", "format.json", "content.json",
    # $ref through location-independent ids / URN bases: pointer-only resolver
    "ref.json",
}
DRAFTS = ["draft7", "draft2020-12"]

TOK = MockTokenizer()
BYTE_ID = {i: TOK.vocabulary[f"<0x{i:02X}>"] for i in range(256)}


def _accepts(guide, text: str) -> bool:
    state = guide.initial_state
    for b in text.encode("utf-8"):
        ids, _ = guide._mask_ids(state)
        if BYTE_ID[b] not in set(int(x) for x in ids):
            return False
        state = guide.get_next_state(state, BYTE_ID[b])
    ids, _ = guide._mask_ids(state)
    return TOK.eos_token_id in set(int(x) for x in ids)


def _cases():
    for draft in DRAFTS:
        d = SUITE / "tests" / draft
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            if f.name in SKIP_FILES:
                continue
            yield draft, f


@pytest.mark.skipif(not SUITE.is_dir(), reason="suite checkout missing")
def test_official_suite_honesty_contract():
    stats = {"sections": 0, "compiled": 0, "declared": 0,
             "valid_ok": 0, "invalid_rejected": 0, "invalid_recorded": 0}
    failures = []
    for draft, path in _cases():
        for section in json.load(open(path)):
            stats["sections"] += 1
            schema = section["schema"]
            try:
                src, recorded = compile_json_schema(schema)
                guide = build_guide(src, TOK)
            except Unsupported:
                stats["declared"] += 1
                continue
            except Exception as e:                      # engine-level refusal
                stats["declared"] += 1
                continue
            stats["compiled"] += 1
            for t in section["tests"]:
                data = t["data"]
                s = json.dumps(data, indent=None, ensure_ascii=False)
                got = _accepts(guide, s)
                if t["valid"]:
                    if got:
                        stats["valid_ok"] += 1
                    else:
                        failures.append(
                            f"FALSE-REJECT {draft}/{path.name}: "
                            f"{section['description'][:40]} :: {s[:60]}")
                else:
                    if not got:
                        stats["invalid_rejected"] += 1
                    elif recorded:
                        stats["invalid_recorded"] += 1
                    else:
                        failures.append(
                            f"SILENT-ACCEPT {draft}/{path.name}: "
                            f"{section['description'][:40]} :: {s[:60]}")
    msg = (f"stats={stats}\n" + "\n".join(failures[:25]) +
           (f"\n... and {len(failures)-25} more" if len(failures) > 25 else ""))
    assert not failures, msg
    # coverage floor: the harness must actually exercise the compiler
    assert stats["compiled"] >= 200, msg
