"""Version markers for the machine-readable output.

- ``SCHEMA_VERSION``: the shape of the JSON payloads. Bump on any change to the
  structure of ``to_dict()`` output so consumers can detect it.
- ``METHODOLOGY_VERSION``: the set of audit methods and decision thresholds that
  produced a verdict. Bump when a check's statistic or decision rule changes, so
  a stored report is traceable to how it was computed.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.0"
METHODOLOGY_VERSION = "1.0"
