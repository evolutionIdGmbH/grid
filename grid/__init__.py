"""GRID: Grammar-Railed Decoding."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("grid-guardrail")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"
