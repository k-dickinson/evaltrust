# Security Policy

## Supported versions

EvalTrust is pre-1.0; security fixes are made against the latest released
version.

| Version | Supported |
|---------|-----------|
| 0.7.x   | ✅        |
| < 0.7   | ❌        |

## Reporting a vulnerability

Please report suspected vulnerabilities privately rather than opening a public
issue.

- Preferred: open a private report via GitHub Security Advisories
  (**Security** tab → **Report a vulnerability**) on the
  [repository](https://github.com/k-dickinson/evaltrust/security/advisories/new).

Include enough detail to reproduce the issue: affected version, a minimal input
file if relevant, and the impact you observed. We aim to acknowledge reports
within a few days and will keep you updated on the fix and disclosure timeline.

## Scope

EvalTrust runs locally and offline: it reads an evaluation results file and prints
a report. It makes **no network calls**, sends **no telemetry or analytics**, and
requires **no credentials or account**. The most relevant concerns are therefore
around parsing untrusted input files — for example, a malformed or maliciously
crafted results file causing unexpected behavior. Reports in that area are
especially welcome.

## Dependencies and SBOM

The runtime dependency footprint is deliberately small — `numpy`, `scipy`,
`rich`, and `typer` (plus `tomli` on Python < 3.11). Dependencies are declared
with lower bounds in `pyproject.toml` so downstream environments resolve
compatible versions; nothing is fetched at runtime.

A CycloneDX Software Bill of Materials can be generated for any installed
version with:

```bash
make sbom        # writes sbom.json (CycloneDX) for the current environment
```

For a release-grade SBOM covering only the package and its runtime
dependencies, run that in a clean environment after `pip install evaltrust`.
