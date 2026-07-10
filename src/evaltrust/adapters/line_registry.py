"""Line-format adapter registry.

Detection is structural, so renaming a file never fools it. Specific formats go
before generic ones. When no adapter claims the rows, ingest uses its existing
record path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .common import Record
from .lm_eval import LMEvalAdapter


class LineAdapter(Protocol):
    source_format: str

    def detect_lines(self, rows: list[dict]) -> bool: ...
    def parse_lines(
        self, rows: list[dict], *, path: Path | None = None
    ) -> tuple[list[Record], dict]: ...


# Order matters: specific formats before generic fallbacks. There is no generic
# entry. Unclaimed rows keep using ingest's existing dicts_to_records path.
LINE_REGISTRY: list[LineAdapter] = [
    LMEvalAdapter(),
]


def detect_line_adapter(rows: list[dict]) -> LineAdapter | None:
    for adapter in LINE_REGISTRY:
        if adapter.detect_lines(rows):
            return adapter
    return None
