# EvalLab

**An auditor for LLM evaluations.** It doesn't tell you how good your model is —
it tells you whether you can *trust* the evaluation you used to decide.

```bash
pip install evallab
evallab audit results.json
```

Point it at the output of your existing eval (DeepEval, Promptfoo, LangSmith,
OpenEvals, or a CSV) and EvalLab reports whether "Model B beat Model A" is real
evidence or just noise — and if the evidence is weak, why, how it knows, and how
to fix it.

See `docs/superpowers/specs/` for the design.
