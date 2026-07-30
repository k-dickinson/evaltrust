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
    assert c.bayesian is False
    assert c.win_rate is False
    assert c.run_aware is False
    assert c.run_aware_future_runs is None


def test_win_rate_participates_in_equality_and_hash():
    default_a = AuditConfig()
    default_b = AuditConfig()
    enabled = AuditConfig(win_rate=True)

    assert default_a == default_b
    assert hash(default_a) == hash(default_b)
    assert enabled != default_a
    assert hash(enabled) != hash(default_a)


def test_run_aware_fields_participate_in_equality_and_hash():
    default_a = AuditConfig()
    default_b = AuditConfig()
    enabled = AuditConfig(run_aware=True, run_aware_future_runs=3)
    future_count_only = AuditConfig(run_aware_future_runs=3)

    assert default_a == default_b
    assert hash(default_a) == hash(default_b)
    assert enabled != default_a
    assert hash(enabled) != hash(default_a)
    assert future_count_only != default_a
    assert hash(future_count_only) != hash(default_a)


@pytest.mark.parametrize("future_runs", [None, True, 0, -1, 1.5, "3"])
def test_run_aware_requires_a_positive_real_integer(future_runs):
    with pytest.raises(ValueError, match="run_aware_future_runs"):
        AuditConfig(run_aware=True, run_aware_future_runs=future_runs)


@pytest.mark.parametrize("future_runs", [True, 0, -1, 1.5, "3"])
def test_future_run_count_is_inert_when_run_aware_is_off(future_runs):
    config = AuditConfig(run_aware=False, run_aware_future_runs=future_runs)
    assert config.run_aware_future_runs == future_runs


def test_run_aware_fields_load_from_toml(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text(
        "run_aware = true\nrun_aware_future_runs = 4\n"
    )
    config = AuditConfig.load(start_dir=str(tmp_path))
    assert config.run_aware is True
    assert config.run_aware_future_runs == 4


def test_bayesian_is_loadable_from_dict_and_toml(tmp_path):
    assert AuditConfig.from_dict({"bayesian": True}).bayesian is True
    (tmp_path / ".evaltrust.toml").write_text("bayesian = true\n")
    assert AuditConfig.load(start_dir=str(tmp_path)).bayesian is True


def test_bayesian_participates_in_equality_and_hash():
    disabled = AuditConfig(bayesian=False)
    enabled = AuditConfig(bayesian=True)
    assert disabled != enabled
    assert hash(disabled) != hash(enabled)


def test_from_dict_warns_and_ignores_unknown_keys():
    # Unknown keys warn (so a typo isn't silent) but don't stop the known keys
    # from applying.
    with pytest.warns(UserWarning, match="nonsense"):
        c = AuditConfig.from_dict({"alpha": 0.01, "nonsense": 123})
    assert c.alpha == 0.01


def test_correction_defaults_to_bonferroni():
    assert AuditConfig().correction == "bonferroni"


def test_correction_is_loadable_from_a_toml(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text('correction = "holm"\n')
    assert AuditConfig.load(start_dir=str(tmp_path)).correction == "holm"


def test_all_pairs_defaults_off():
    assert AuditConfig().all_pairs is False


def test_all_pairs_loads_from_dict():
    assert AuditConfig.from_dict({"all_pairs": True}).all_pairs is True


def test_all_pairs_loads_from_toml(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text("all_pairs = true\n")
    assert AuditConfig.load(start_dir=str(tmp_path)).all_pairs is True


def test_all_pairs_participates_in_equality_and_hash():
    disabled = AuditConfig(all_pairs=False)
    enabled = AuditConfig(all_pairs=True)
    assert disabled != enabled
    assert hash(disabled) != hash(enabled)


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
# gated_metrics
# ---------------------------------------------------------------------------

def test_gated_metrics_defaults_to_empty_frozenset():
    assert AuditConfig().gated_metrics == frozenset()
    assert isinstance(AuditConfig().gated_metrics, frozenset)


def test_from_dict_coerces_gated_metrics_list_to_frozenset():
    cfg = AuditConfig.from_dict({"gated_metrics": ["safety", "toxicity"]})
    assert isinstance(cfg.gated_metrics, frozenset)
    assert cfg.gated_metrics == frozenset({"safety", "toxicity"})


def test_load_gated_metrics_from_toml(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text('gated_metrics = ["safety"]\n')
    cfg = AuditConfig.load(start_dir=str(tmp_path))
    assert cfg.gated_metrics == frozenset({"safety"})


def test_dataclass_replace_preserves_gated_metrics():
    """dataclasses.replace must produce a valid config with policy fields intact."""
    from dataclasses import replace
    cfg = AuditConfig(gated_metrics=frozenset({"correctness"}))
    cfg2 = replace(cfg, alpha=0.01)
    assert cfg2.alpha == 0.01
    assert cfg2.gated_metrics == frozenset({"correctness"})


# ---------------------------------------------------------------------------
# metric_weights was removed (#153). A config that still sets it goes through
# the standard unknown-key path: explicit --config errors, discovered configs
# warn and ignore. Nothing silently does nothing.
# ---------------------------------------------------------------------------

def test_metric_weights_is_no_longer_a_config_field():
    with pytest.raises(TypeError):
        AuditConfig(metric_weights={"correctness": 2.0})
    assert not hasattr(AuditConfig(), "metric_weights")


def test_explicit_config_with_metric_weights_errors(tmp_path):
    p = tmp_path / "policy.toml"
    p.write_text('[metric_weights]\ncorrectness = 3.0\n')
    with pytest.raises(ValueError, match=r"Unknown config key.*metric_weights"):
        AuditConfig.load(path=str(p))


def test_discovered_toml_with_metric_weights_warns_and_ignores(tmp_path):
    (tmp_path / ".evaltrust.toml").write_text(
        'seed = 7\n[metric_weights]\ncorrectness = 3.0\n')
    with pytest.warns(UserWarning, match=r"metric_weights"):
        c = AuditConfig.load(start_dir=str(tmp_path))
    assert c.seed == 7           # known keys still apply
    assert c == AuditConfig(seed=7)


def test_pyproject_with_metric_weights_warns_and_ignores(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.evaltrust]\nalpha = 0.01\n'
        '[tool.evaltrust.metric_weights]\ncorrectness = 3.0\n')
    with pytest.warns(UserWarning, match=r"metric_weights"):
        c = AuditConfig.load(start_dir=str(tmp_path))
    assert c.alpha == 0.01
    assert c == AuditConfig(alpha=0.01)


def test_default_config_equality_and_hash_survive_field_removal():
    assert AuditConfig() == AuditConfig()
    assert hash(AuditConfig()) == hash(AuditConfig())


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
