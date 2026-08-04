"""Repo-wide DSPy signature audit: which signatures GRID can mask-enforce,
which carry a recorded residue (and on which output fields), and which it
declares unsupported.

    python -m grid.integrations.dspy_check path/to/pipeline.py my.module ...
    python -m grid.integrations.dspy_check --strict src/signatures.py

Targets are .py files or dotted module names; each is imported and scanned
for ``dspy.Signature`` subclasses. Default mode reports; ``--strict`` exits
non-zero when any signature is less than fully mask-enforceable, which is
the CI-gate shape (fail the PR when a teammate's ``set[str]`` slips in, not
the pipeline three weeks later).

Exit codes: 0 ok; 1 --strict violation; 2 a target failed to import.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

from grid.integrations.dspy_adapter import SignatureNotEnforceable
from grid.jsonschema import Unsupported

__all__ = ["audit_module", "main"]


def _load(target: str):
    if target.endswith(".py") or "/" in target or target == ".":
        path = Path(target).resolve()
        name = "_grid_check_" + path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {target}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(target)


def _signatures(mod):
    import dspy

    out = []
    for name, val in sorted(vars(mod).items()):
        if (isinstance(val, type) and issubclass(val, dspy.Signature)
                and val is not dspy.Signature):
            out.append((name, val))
    return out


def audit_module(mod, adapter=None) -> list[tuple[str, str, str]]:
    """[(signature name, status, detail)] for every Signature in ``mod``.

    status: "enforceable" (empty residue), "recorded" (residue named per
    output field), or "declared" (GRID refuses the schema up front).
    """
    from grid.integrations.dspy_adapter import GridJSONAdapter

    ad = adapter or GridJSONAdapter()   # non-strict: collect the FULL residue
    rows = []
    for name, sig in _signatures(mod):
        try:
            paths = ad.recorded_paths_for(sig)
        except (SignatureNotEnforceable, Unsupported) as e:
            rows.append((name, "declared", str(e)))
            continue
        if not paths:
            rows.append((name, "enforceable", ""))
        else:
            detail = "; ".join(
                f"{p}: {', '.join(sorted(feats))}"
                for p, feats in sorted(paths.items()))
            rows.append((name, "recorded", detail))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m grid.integrations.dspy_check",
        description="Audit DSPy signatures for GRID mask-enforceability.")
    ap.add_argument("targets", nargs="+",
                    help=".py files or dotted module names")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless every signature is fully "
                         "mask-enforceable (CI gate)")
    args = ap.parse_args(argv)

    try:
        import dspy  # noqa: F401
    except ImportError:
        print("dspy_check requires dspy (pip install dspy)", file=sys.stderr)
        return 2

    all_rows: list[tuple[str, str, str]] = []
    failed_imports = 0
    for target in args.targets:
        try:
            mod = _load(target)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"IMPORT FAILED  {target}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            failed_imports += 1
            continue
        rows = audit_module(mod)
        if not rows:
            print(f"(no dspy.Signature classes in {target})")
        all_rows.extend(rows)
        for name, status, detail in rows:
            line = f"{status.upper():12s} {name}"
            if detail:
                line += f"  [{detail}]"
            print(line)

    n = len(all_rows)
    bad = [r for r in all_rows if r[1] != "enforceable"]
    print(f"\n{n} signature(s): {n - len(bad)} enforceable, "
          f"{sum(1 for r in bad if r[1] == 'recorded')} with recorded "
          f"residue, {sum(1 for r in bad if r[1] == 'declared')} declared "
          "unsupported")
    if failed_imports:
        return 2
    if args.strict and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
