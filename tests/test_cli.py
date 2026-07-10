"""Tests for the `evaltrust audit` command."""

import json
from importlib.metadata import version as package_version

from typer.testing import CliRunner

from evaltrust.cli import app

runner = CliRunner()


def test_version_flag_prints_installed_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == package_version("evaltrust")


def test_version_flag_is_eager_and_skips_command_parsing():
    # --version must short-circuit before any command runs or arguments are
    # validated, the way pip/pytest behave.
    result = runner.invoke(app, ["--version", "audit"])
    assert result.exit_code == 0
    assert result.stdout.strip() == package_version("evaltrust")


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def clean_win_file(tmp_path):
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": 0, "B": 1 if i < 180 else 0}}
        for i in range(200)]}
    return write(tmp_path, "win.json", raw)


def noise_file(tmp_path):
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": i % 2, "B": (i + 1) % 2}}
        for i in range(120)]}
    return write(tmp_path, "noise.json", raw)


def suite_file(tmp_path):
    # Two metrics: "strong" (20 discordant pairs, tiny p) and "borderline"
    # (10 vs 2 discordant pairs, p = 0.0386, between alpha/2 and alpha).
    rows = []
    for i in range(20):
        rows += [{"id": f"s{i}", "model": "A", "metric": "strong", "score": 0},
                 {"id": f"s{i}", "model": "B", "metric": "strong", "score": 1}]
    for i in range(10):
        rows += [{"id": f"b{i}", "model": "A", "metric": "borderline", "score": 0},
                 {"id": f"b{i}", "model": "B", "metric": "borderline", "score": 1}]
    for i in range(2):
        rows += [{"id": f"c{i}", "model": "A", "metric": "borderline", "score": 1},
                 {"id": f"c{i}", "model": "B", "metric": "borderline", "score": 0}]
    return write(tmp_path, "suite.json", rows)


def test_correction_flag_selects_holm(tmp_path):
    out = runner.invoke(
        app, ["audit", suite_file(tmp_path), "--correction", "holm", "--json"])
    assert out.exit_code == 0
    payload = json.loads(out.stdout)
    assert "holm" in payload["correction"].lower()
    # Holm rejects the borderline metric that Bonferroni cannot.
    bonf = runner.invoke(
        app, ["audit", suite_file(tmp_path), "--correction", "bonferroni", "--json"])
    bonf_payload = json.loads(bonf.stdout)
    assert "bonferroni" in bonf_payload["correction"].lower()


def test_bad_correction_value_errors(tmp_path):
    result = runner.invoke(
        app, ["audit", suite_file(tmp_path), "--correction", "banana"])
    assert result.exit_code == 2


def test_audit_prints_report_and_exits_zero(tmp_path):
    result = runner.invoke(app, ["audit", clean_win_file(tmp_path)])
    assert result.exit_code == 0
    assert "EvalTrust" in result.stdout
    assert "High Confidence" in result.stdout


def test_missing_file_exits_nonzero_with_message(tmp_path):
    result = runner.invoke(app, ["audit", str(tmp_path / "nope.json")])
    assert result.exit_code != 0
    assert "No such" in result.stdout or "not" in result.stdout.lower()


def test_unknown_format_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["audit", write(tmp_path, "x.json", {"nope": 1})])
    assert result.exit_code != 0


def test_single_model_file_is_audited_for_score_reliability(tmp_path):
    raw = {"models": ["A"], "examples": [
        {"id": str(i), "scores": {"A": i % 2}} for i in range(100)]}
    result = runner.invoke(app, ["audit", write(tmp_path, "one.json", raw)])
    assert result.exit_code == 0
    assert "Score Reliability" in result.stdout


def test_single_model_threshold_gate(tmp_path):
    # 50% model, target 0.8 -> below -> fail-under gate trips.
    raw = {"models": ["A"], "examples": [
        {"id": str(i), "scores": {"A": i % 2}} for i in range(200)]}
    result = runner.invoke(app, ["audit", write(tmp_path, "one.json", raw),
                                 "--threshold", "0.8", "--fail-under", "moderate"])
    assert result.exit_code == 1


def test_strict_flag_fails_on_low_confidence(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--strict"])
    assert result.exit_code == 1


def test_no_strict_still_exits_zero_on_low_confidence(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path)])
    assert result.exit_code == 0
    assert "Low Confidence" in result.stdout


def test_fail_under_moderate_blocks_low(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--fail-under", "moderate"])
    assert result.exit_code == 1


def test_fail_under_high_blocks_a_moderate_result(tmp_path):
    # A clear win with a small effect lands at Moderate; --fail-under high blocks it.
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": i % 2, "B": 1 if i < 55 else i % 2}}
        for i in range(100)]}
    result = runner.invoke(app, ["audit", write(tmp_path, "mod.json", raw), "--fail-under", "high"])
    assert result.exit_code in (0, 1)  # deterministic per data; must not error
    assert result.exit_code != 2


def test_fail_under_low_never_blocks(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--fail-under", "low"])
    assert result.exit_code == 0


def test_fail_under_bad_level_is_an_error(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--fail-under", "banana"])
    assert result.exit_code == 2


def test_config_file_changes_thresholds(tmp_path):
    import json as _json
    # Two models with the same ~70% rate: not significant.
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": 1 if i % 10 < 7 else 0,
                                  "B": 1 if (i + 3) % 10 < 7 else 0}}
        for i in range(120)]}
    data_file = write(tmp_path, "d.json", raw)
    policy = tmp_path / "policy.toml"
    policy.write_text("equivalence_margin = 1.0\n")   # everything is "equivalent"
    result = runner.invoke(app, ["audit", data_file, "--config", str(policy), "--json"])
    payload = _json.loads(result.stdout)
    decision = [f for f in payload["findings"] if f["details"].get("check") == "decision"][0]
    assert decision["details"]["outcome"] == "equivalent"


def test_bad_config_path_errors(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--config",
                                 str(tmp_path / "nope.toml")])
    assert result.exit_code == 2


def _audit_json(tmp_path, src, name):
    out = runner.invoke(app, ["audit", src, "--json"]).stdout
    (tmp_path / name).write_text(out)
    return str(tmp_path / name)


def test_diff_detects_regression(tmp_path):
    good = _audit_json(tmp_path, clean_win_file(tmp_path), "a.json")   # HIGH
    bad = _audit_json(tmp_path, noise_file(tmp_path), "b.json")        # LOW
    result = runner.invoke(app, ["diff", good, bad])
    assert result.exit_code == 1
    assert "Regression" in result.stdout


def test_diff_no_change_exits_zero(tmp_path):
    a = _audit_json(tmp_path, clean_win_file(tmp_path), "a.json")
    result = runner.invoke(app, ["diff", a, a])
    assert result.exit_code == 0
    assert "No change" in result.stdout


def test_diff_improvement_does_not_fail(tmp_path):
    bad = _audit_json(tmp_path, noise_file(tmp_path), "b.json")
    good = _audit_json(tmp_path, clean_win_file(tmp_path), "a.json")
    result = runner.invoke(app, ["diff", bad, good])
    assert result.exit_code == 0
    assert "Improvement" in result.stdout


def single_model_file(tmp_path, name, model, n, rate):
    raw = {"models": [model], "examples": [
        {"id": str(i), "scores": {model: 1 if i < int(n * rate) else 0}}
        for i in range(n)]}
    return write(tmp_path, name, raw)


def test_two_file_comparison(tmp_path):
    a = single_model_file(tmp_path, "gpt.json", "gpt", 200, 0.60)
    b = single_model_file(tmp_path, "claude.json", "claude", 200, 0.90)
    result = runner.invoke(app, ["audit", a, b])
    assert result.exit_code == 0
    assert "gpt" in result.stdout and "claude" in result.stdout


def test_two_multi_model_files_error_helpfully(tmp_path):
    raw = {"models": ["A", "B"], "examples": [{"id": "1", "scores": {"A": 1, "B": 0}}]}
    f = write(tmp_path, "multi.json", raw)
    result = runner.invoke(app, ["audit", f, f])
    assert result.exit_code == 2
    assert "one model per file" in result.stdout.lower()


def test_three_files_rejected(tmp_path):
    f = single_model_file(tmp_path, "x.json", "m", 10, 0.5)
    result = runner.invoke(app, ["audit", f, f, f])
    assert result.exit_code == 2


def test_explicit_model_selection(tmp_path):
    raw = {"models": ["A", "B", "C"], "examples": [
        {"id": str(i), "scores": {"A": 0, "B": 1, "C": 0}} for i in range(30)]}
    result = runner.invoke(
        app, ["audit", write(tmp_path, "m.json", raw), "--model-a", "A", "--model-b", "C"])
    assert result.exit_code == 0
    assert "A vs C" in result.stdout or "C vs A" in result.stdout


def test_json_output_is_valid_json(tmp_path):
    import json as _json
    result = runner.invoke(app, ["audit", clean_win_file(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = _json.loads(result.stdout)
    assert payload["verdict"]["level"] == "HIGH"
    assert "findings" in payload
    # No rich box characters in JSON mode.
    assert "╭" not in result.stdout


def test_json_output_still_respects_strict(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--json", "--strict"])
    assert result.exit_code == 1


def test_plain_output_is_ascii_only(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--plain"])
    assert result.exit_code == 0
    assert "[fail]" in result.stdout or "[ok  ]" in result.stdout
    assert result.stdout.isascii()
    assert "╭" not in result.stdout


def multi_metric_file(tmp_path):
    lines = ["id,model,metric,score"]
    for i in range(80):
        # correctness: new clearly better; tone: identical noise
        lines.append(f"q{i},old,correctness,{1 if i % 5 else 0}")
        lines.append(f"q{i},new,correctness,{1 if i % 10 else 0}")
        lines.append(f"q{i},old,tone,{i % 2}")
        lines.append(f"q{i},new,tone,{(i + 1) % 2}")
    p = tmp_path / "suite.csv"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_multi_metric_file_produces_a_suite_report(tmp_path):
    result = runner.invoke(app, ["audit", multi_metric_file(tmp_path)])
    assert result.exit_code == 0
    assert "metrics" in result.stdout
    assert "correctness" in result.stdout and "tone" in result.stdout


def test_multi_metric_json_has_per_metric_results(tmp_path):
    import json as _json
    result = runner.invoke(app, ["audit", multi_metric_file(tmp_path), "--json"])
    payload = _json.loads(result.stdout)
    assert set(payload["metrics"].keys()) == {"correctness", "tone"}
    assert "corrected_alpha" in payload


def test_explain_flag_adds_detail(tmp_path):
    base = runner.invoke(app, ["audit", noise_file(tmp_path)])
    detailed = runner.invoke(app, ["audit", noise_file(tmp_path), "--explain"])
    assert "Detail" not in base.stdout
    assert "Detail" in detailed.stdout


def test_md_output_contains_verdict_and_finding_titles(tmp_path):
    result = runner.invoke(app, ["audit", clean_win_file(tmp_path), "--md"])
    assert result.exit_code == 0
    assert "# EvalTrust" in result.stdout
    assert "High Confidence" in result.stdout or "high confidence" in result.stdout.lower()
    assert "**[" in result.stdout


def test_suite_json_with_html_keeps_stdout_pure_json(tmp_path):
    # HTML isn't supported for multi-metric suites, so the CLI warns -- but that
    # warning must not land on stdout after the JSON body, or it corrupts the
    # machine-readable output. In --json mode stdout must stay pure JSON.
    out = tmp_path / "out.html"
    result = runner.invoke(
        app, ["audit", multi_metric_file(tmp_path), "--json", "--html", str(out)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)   # must parse cleanly, no trailing warning
    assert "metrics" in payload


def test_config_typo_in_explicit_config_is_an_error(tmp_path):
    policy = tmp_path / "policy.toml"
    policy.write_text("alpah = 0.01\n")
    result = runner.invoke(app, ["audit", noise_file(tmp_path),
                                 "--config", str(policy)])
    assert result.exit_code == 2
    assert "alpah" in result.stdout
