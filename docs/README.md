# EvalTrust documentation

EvalTrust checks whether an eval's conclusion is real or just noise. You ran an
eval and got "Model B beats Model A by 1.5 points" - EvalTrust does the statistics
that tell you whether to believe it, before you ship on a gap that might be luck.

- [**Design & philosophy**](design.md) - what EvalTrust is, the problem it solves,
  and the principles behind it.
- [**Architecture**](architecture.md) - the pipeline, the modules, and how they
  fit together.
- [**Checks and methods**](checks.md) - every check, its statistical method, and
  its thresholds.
- [**Integrations**](integrations.md) - using EvalTrust standalone, embedded in
  your eval/pytest, or in CI.
- [**Configuration**](configuration.md) - set your team's thresholds once in a
  config file.
- [**Input formats**](input-formats.md) - what you can feed EvalTrust and how it
  detects each format.
- [**Writing a format adapter**](adapters.md) - a guide for contributors adding
  support for a new tool.

For contribution guidelines see [`CONTRIBUTING.md`](../CONTRIBUTING.md).
