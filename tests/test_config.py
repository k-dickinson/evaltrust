"""Tests for AuditConfig: one place for a team's statistical policy, loadable
from a config file so it can be checked into a repo."""

import pytest

from evaltrust.config import AuditConfig


def test_defaults_match_the_documented_values():
    c = AuditConfig()
    assert c.alpha == 0.05
    assert c.equivalence_margin == 0.05
    assert c.saturation_fraction == 0.95
    assert c.judge_agreement_threshold == 0.8
    assert c.judge_correlation_threshold == 0.8


def test_from_dict_ignores_unknown_keys():
    c = AuditConfig.from_dict({"alpha": 0.01, "nonsense": 123})
    assert c.alpha == 0.01


def test_correction_defaults_to_bonferroni():
    assert AuditConfig().correction == "bonferroni"


def test_correction_is_loadable_from_a_toml(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text('correction = "holm"\n')
    assert AuditConfig.load(start_dir=str(tmp_path)).correction == "holm"


def test_load_reads_a_dedicated_toml(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text(
        "alpha = 0.01\nequivalence_margin = 0.1\njudge_agreement_threshold = 0.9\n")
    c = AuditConfig.load(start_dir=str(tmp_path))
    assert c.alpha == 0.01
    assert c.equivalence_margin == 0.1
    assert c.judge_agreement_threshold == 0.9


def test_both_judge_thresholds_round_trip_through_dedicated_toml(tmp_path):
    # The agreement floor and the correlation floor are separate keys that both
    # load from a repo's config.
    (tmp_path / ".evaltrust.toml").write_text(
        "judge_agreement_threshold = 0.7\njudge_correlation_threshold = 0.9\n")
    c = AuditConfig.load(start_dir=str(tmp_path))
    assert c.judge_agreement_threshold == 0.7
    assert c.judge_correlation_threshold == 0.9


def test_judge_correlation_threshold_round_trips_through_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.evaltrust]\njudge_correlation_threshold = 0.6\n")
    assert AuditConfig.load(start_dir=str(tmp_path)).judge_correlation_threshold == 0.6


def test_load_reads_pyproject_tool_table(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.evaltrust]\nalpha = 0.02\nsaturation_fraction = 0.9\n")
    c = AuditConfig.load(start_dir=str(tmp_path))
    assert c.alpha == 0.02
    assert c.saturation_fraction == 0.9


def test_dedicated_file_wins_over_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.evaltrust]\nalpha = 0.02\n")
    (tmp_path / ".evaltrust.toml").write_text("alpha = 0.01\n")
    assert AuditConfig.load(start_dir=str(tmp_path)).alpha == 0.01


def test_load_with_no_config_returns_defaults(tmp_path):
    assert AuditConfig.load(start_dir=str(tmp_path)) == AuditConfig()


def test_explicit_path_is_read(tmp_path):
    p = tmp_path / "policy.toml"
    p.write_text("alpha = 0.005\n")
    assert AuditConfig.load(path=str(p)).alpha == 0.005


# ---------------------------------------------------------------------------
# New fields: gated_metrics and metric_weights
# ---------------------------------------------------------------------------

def test_gated_metrics_defaults_to_empty_frozenset():
    assert AuditConfig().gated_metrics == frozenset()
    assert isinstance(AuditConfig().gated_metrics, frozenset)


def test_metric_weights_defaults_to_empty_mapping():
    from types import MappingProxyType
    cfg = AuditConfig()
    assert dict(cfg.metric_weights) == {}
    assert isinstance(cfg.metric_weights, MappingProxyType)


def test_audit_config_is_hashable_with_weights():
    cfg = AuditConfig(metric_weights={"correctness": 2.0, "style": 1.0})
    # Must not raise TypeError: unhashable type.
    h = hash(cfg)
    assert isinstance(h, int)


def test_audit_config_equal_weight_order_independent():
    """Configs with the same weights in different insertion order must be equal."""
    cfg1 = AuditConfig(metric_weights={"a": 1.0, "b": 3.0})
    cfg2 = AuditConfig(metric_weights={"b": 3.0, "a": 1.0})
    assert cfg1 == cfg2
    assert hash(cfg1) == hash(cfg2)


def test_metric_weights_is_immutable():
    cfg = AuditConfig(metric_weights={"correctness": 2.0})
    with pytest.raises(TypeError):
        cfg.metric_weights["safety"] = 99.0  # type: ignore[index]


def test_zero_weight_raises_value_error():
    with pytest.raises(ValueError, match="positive"):
        AuditConfig(metric_weights={"correctness": 0.0})


def test_negative_weight_raises_value_error():
    with pytest.raises(ValueError, match="positive"):
        AuditConfig(metric_weights={"safety": -5.0})


def test_nan_weight_raises_value_error():
    import math
    with pytest.raises(ValueError, match="finite"):
        AuditConfig(metric_weights={"correctness": math.nan})


def test_inf_weight_raises_value_error():
    import math
    with pytest.raises(ValueError, match="finite"):
        AuditConfig(metric_weights={"correctness": math.inf})


def test_from_dict_coerces_gated_metrics_list_to_frozenset():
    cfg = AuditConfig.from_dict({"gated_metrics": ["safety", "toxicity"]})
    assert isinstance(cfg.gated_metrics, frozenset)
    assert cfg.gated_metrics == frozenset({"safety", "toxicity"})


def test_from_dict_coerces_metric_weights_to_mapping_proxy():
    from types import MappingProxyType
    cfg = AuditConfig.from_dict({"metric_weights": {"correctness": 3.0}})
    assert isinstance(cfg.metric_weights, MappingProxyType)
    assert cfg.metric_weights["correctness"] == 3.0


def test_from_dict_rejects_zero_weight():
    with pytest.raises(ValueError, match="positive"):
        AuditConfig.from_dict({"metric_weights": {"correctness": 0.0}})


def test_from_dict_rejects_negative_weight():
    with pytest.raises(ValueError, match="positive"):
        AuditConfig.from_dict({"metric_weights": {"style": -1.0}})


def test_from_dict_rejects_nan_weight():
    import math
    with pytest.raises(ValueError, match="finite"):
        AuditConfig.from_dict({"metric_weights": {"correctness": math.nan}})


def test_from_dict_rejects_inf_weight():
    import math
    with pytest.raises(ValueError, match="finite"):
        AuditConfig.from_dict({"metric_weights": {"correctness": math.inf}})


def test_load_gated_and_weights_from_toml(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text(
        'gated_metrics = ["safety"]\n'
        '[metric_weights]\ncorrectness = 3.0\nstyle = 1.0\n'
    )
    cfg = AuditConfig.load(start_dir=str(tmp_path))
    assert cfg.gated_metrics == frozenset({"safety"})
    assert cfg.metric_weights["correctness"] == 3.0
    assert cfg.metric_weights["style"] == 1.0


def test_load_zero_weight_from_toml_raises(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text(
        '[metric_weights]\ncorrectness = 0.0\n'
    )
    with pytest.raises(ValueError, match="positive"):
        AuditConfig.load(start_dir=str(tmp_path))


def test_dataclass_replace_preserves_immutability():
    """dataclasses.replace on a config with weights must still produce a valid config."""
    from dataclasses import replace
    cfg = AuditConfig(metric_weights={"correctness": 2.0})
    cfg2 = replace(cfg, alpha=0.01)
    assert cfg2.alpha == 0.01
    assert cfg2.metric_weights["correctness"] == 2.0
    # Still immutable
    with pytest.raises(TypeError):
        cfg2.metric_weights["new"] = 1.0  # type: ignore[index]


def test_bare_string_gated_metrics_raises_value_error():
    """gated_metrics = "safety" (missing brackets) must raise, not silently
    produce frozenset({'s','a','f','e','t','y'})."""
    with pytest.raises(ValueError, match="bare string"):
        AuditConfig(gated_metrics="safety")


def test_from_dict_bare_string_gated_metrics_raises_value_error():
    """TOML typo: gated_metrics = "safety" instead of ["safety"] must raise."""
    with pytest.raises(ValueError, match="bare string"):
        AuditConfig.from_dict({"gated_metrics": "safety"})

def test_from_dict_warns_on_unknown_keys_with_a_suggestion():
    with pytest.warns(UserWarning, match=r"alpah.*did you mean 'alpha'"):
        c = AuditConfig.from_dict({"alpah": 0.01})
    assert c.alpha == 0.05  # typo ignored, default kept


def test_from_dict_warns_on_dash_for_underscore_typo():
    with pytest.warns(UserWarning, match=r"equivalence-margin.*equivalence_margin"):
        AuditConfig.from_dict({"equivalence-margin": 0.1})


def test_from_dict_strict_raises_listing_unknown_keys():
    with pytest.raises(ValueError, match=r"alpah"):
        AuditConfig.from_dict({"alpah": 0.01}, strict=True)


def test_explicit_config_path_with_typo_errors(tmp_path):
    p = tmp_path / "policy.toml"
    p.write_text("alpah = 0.01\n")
    with pytest.raises(ValueError, match=r"alpah.*did you mean 'alpha'"):
        AuditConfig.load(path=str(p))


def test_discovered_config_with_typo_warns_but_loads(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text("alpah = 0.01\nseed = 7\n")
    with pytest.warns(UserWarning, match=r"alpah"):
        c = AuditConfig.load(start_dir=str(tmp_path))
    assert c.seed == 7          # known keys still apply
    assert c.alpha == 0.05      # the typo didn't silently set alpha
