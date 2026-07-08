"""Adapter registry and auto-detection.

Adapters are tried most-specific first (a recognised tool) and fall back to the
generic shapes. Detection is structural, so renaming a file never fools it. An
unrecognised shape fails loudly rather than being silently mis-parsed.
"""

from __future__ import annotations

from typing import Protocol

from ..core.schema import EvalData
from .deepeval import DeepEvalAdapter
from .generic import GenericRecordsAdapter, NativeNestedAdapter
from .openevals import OpenEvalsAdapter
from .promptfoo import PromptfooAdapter


class Adapter(Protocol):
    source_format: str

    def detect(self, raw) -> bool: ...
    def parse(self, raw) -> EvalData: ...


class UnknownFormatError(ValueError):
    """Raised when no adapter recognises the input."""


# Order matters: specific formats before generic fallbacks.
REGISTRY: list[Adapter] = [
    PromptfooAdapter(),
    DeepEvalAdapter(),
    OpenEvalsAdapter(),
    NativeNestedAdapter(),
    GenericRecordsAdapter(),
]


def detect_adapter(raw) -> Adapter:
    for adapter in REGISTRY:
        if adapter.detect(raw):
            return adapter
    raise UnknownFormatError(
        "Could not recognise this evaluation format. EvalTrust looked for "
        "promptfoo results, a DeepEval test-results export, a nested "
        "{\"examples\": [...]} structure, or a list of records with model/score "
        "fields. Provide per-example scores in one of those shapes, or a CSV."
    )
