# Normal-theory predictive reruns

The fixed-example predictive rerun method estimates the realized aggregate
`B - A` gap when the same examples are rerun at one explicit common future run
count. Every example has equal weight. A and B run streams and different
examples' run streams are independent in v1. List position does not imply
covariance.

For example `i`, model `M`, observed run count `R_Mi`, and future run count
`R*`, the method uses

```text
T_hat = mean_i(mean(B_i) - mean(A_i))

q_Mi = (s2_Mi / m^2) * (1 / R_Mi + 1 / R*)
V_hat = sum_Mi(q_Mi)

df = V_hat^2 / sum_Mi(q_Mi^2 / (R_Mi - 1))
```

The fitted Welch-Satterthwaite normal-theory approximation reports
`normal_theory_probability_a_better` for the strict event
`future B - A < 0`. This is an estimated rerun probability under the working
model. It is not a posterior probability that A is intrinsically better, and it
is not universally calibrated across arbitrary run distributions.

The accompanying range contains `central_mass` of the fitted approximation. A
`central_mass` of 0.95 describes model mass, not a calibrated 95% interval or a
coverage guarantee.

## Measured limitation

Simulation with centered, variance-scaled LogNormal(0, 1) run noise measured
central-range coverage around 0.933 with 8 examples, 3 observed runs, and 1
future run. Coverage was around 0.937 with 25 examples, 6 observed runs, and 1
future run. Behavior moved back toward nominal as more independent draws were
averaged.

Binary, ordinal, rounded, skewed, heavy-tailed, and shared-shock data can
violate the working model. A shared shock is especially important because v1
has no run identity from which to estimate covariance.

## Limited evidence

When every stream has one observed run, the method returns the observed
equal-example point as a point range. It returns no probability, prediction
variance, or degrees of freedom. It does the same when every identified stream
has observed zero variance. Observed constancy does not prove that future runs
are deterministic. Mixed singleton and identified streams are rejected so the
caller can apply and count its explicit skip policy.

Issue #130 contains only the pure numerical primitive. Configuration, CLI
eligibility, help text, findings, and report wiring remain in #131. Existing
mean-gap inference, significance tests, confidence intervals, TOST results,
effect sizes, precision advice, leaders, and verdicts are unchanged.
