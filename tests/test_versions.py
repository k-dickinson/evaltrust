"""Every machine-readable payload must carry version markers so downstream
tooling can depend on the shape (schema) and on the method that produced the
numbers (methodology)."""

import json

import evaltrust
from evaltrust.versions import METHODOLOGY_VERSION, SCHEMA_VERSION
from evaltrust.core.schema import EvalData, Example


def _two_model_data():
    exs = [Example(id=str(i), scores={"A": i % 2, "B": (i + 1) % 2}) for i in range(40)]
    return EvalData(models=["A", "B"], examples=exs, source_format="native")


def test_versions_are_nonempty_strings():
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION
    assert isinstance(METHODOLOGY_VERSION, str) and METHODOLOGY_VERSION


def test_public_api_exposes_versions():
    assert evaltrust.SCHEMA_VERSION == SCHEMA_VERSION
    assert evaltrust.METHODOLOGY_VERSION == METHODOLOGY_VERSION


def test_report_to_dict_carries_versions():
    d = evaltrust.audit(_two_model_data()).to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["methodology_version"] == METHODOLOGY_VERSION
    assert json.loads(json.dumps(d, allow_nan=False)) == d  # round-trips exactly


def test_suite_to_dict_carries_versions():
    suite = {"acc": _two_model_data(), "safety": _two_model_data()}
    d = evaltrust.audit_suite(suite).to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["methodology_version"] == METHODOLOGY_VERSION
    assert json.loads(json.dumps(d, allow_nan=False)) == d  # round-trips exactly
