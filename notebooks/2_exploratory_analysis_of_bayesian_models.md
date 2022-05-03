---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Poetry
    language: python
    name: poetry-kernel
---

# Exploratory analysis of Bayesian models


# Imports

```python
%config InlineBackend.figure_format="retina"
```

```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from scipy import stats
```

## Life after inference


## Understanding your assumptions


**Prior predictive checks**
are computing a prior predictive distribution by sampling from the model
without taking into account the data.
This can be used to understand affects of choices of priors.

If priors give bad values:

- Rethinkg model
- Use prior that reduces invalid results
- See if data is informative enough to exclude nonsensical values

For example a logistic regression
can favor more and more extreme values
as the number of predictors increases.
A strong regularizing prior—like
the Laplace distribution—
can keep model away from extreme values.



### Understanding your predictions


**Posterior predictive checks**
evaluate how close the synthetic observations are
to the actual observations.

Can plot them againt each other to visially inspect,
or use Baeysian p-value—probability
that simulated test statistic $T_{sim}$
is less or equal than the observed statistic $T_{obs}$.
Usually the ideal value is $p_{B} = 0.5$—half
the time it's below,
and the other half above.

$$
p_{B} = p(T_{sim} \leq T_{obs} \mid \tilde Y)
$$

Can plot with `az.plot_bpv(..., kind="p_value")`.

Arviz funcitons for plots:

- **`az.plot_ppc(..)`**
  plots generated model data vs actual data
- **`az.plot_bpv(..., kind="p_values")`**
  plots the proportion of predicted values
  that are less than or equal to the observed data (p-value)
  compared to the expected distribution for a dataset of the same size
  as the observed data
- **`az.plot_bpv(..., kind="u_values")`**
  plots the proportion of predicted values that are less or equal than the observed per observation.
  Ideal case is white line—a uniform distribution.
  Grey band is where we expect to see 94% of Uniform-like curves,
  even good models have deviations from a perfect Uniform.

![Posterior check plots](images/chapter_2/posterior_predictive_many_examples.png)



## Diagnosing numerical inference


Bad chains:

- Are not independent,
  and show correlation
- Are not identically distributed
- Have some regions that are not sampled well

```python
rng = np.random.default_rng(42)
good_chains = stats.beta.rvs(2, 5, size=(2, 2000))
bad_chains0 = rng.normal(
    np.sort(good_chains, axis=None), 0.05, size=4000
).reshape(2, -1)  # Values not independent and identically distributed

# Portions have high correlation
bad_chains1 = good_chains.copy()
for i in rng.integers(1900, size=4):
    bad_chains1[i % 2 :, i : i + 100] = rng.beta(i, 950, size=100)

chains = {
    "good_chains": good_chains,
    "bad_chains0": bad_chains0,
    "bad_chains1": bad_chains1,
}
```

### Effective sample size


Samples from MCMC methods are **autocorrelated**—there
is a similarrity between values as a function of the time lag between them—and
the amount of information contained in the sample
is less than what one would get from an iid sample
of the same size.

**Effective Sample Size (ESS)**
is an estimator
that takes autocorrelation into account
and provides the numers of draws we would have
if our sample was iid.

```python
az.ess(chains)
```

`az.ess` returns `bulk-ESS` by default—which
assesses how the center of the distribution
is resolved.
`tail-ESS` corresponds to
5 and 95 percentiles.
`az.ess(..., method="quantile")`
for specific quantiles.

```python
_, (top_ax, bottom_ax) = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(20, 7))
az.plot_ess(chains, kind="local", ax=top_ax)  # Small intervals between 2 quantiles
az.plot_ess(chains, kind="quantile", ax=bottom_ax)
plt.tight_layout();
```

In plots above,
dashed line represents
the minimum suggested value for sufficient sample size—400.


### Potential scale reduction factor $\hat R$


MCMC will converge eventually,
but need to test if it converged now.
Can run multiple chains
and see if they look similar.
**$\hat R$**
is a numerical reprentation of this idea—it
is the standard deviation of all samples of $\theta$
divided by the root mean quare of the seperated within-chain standard deviations.
Ideal value is 1—
variance between chains should be the same
as variance within-chain.
Practically $\hat R \lessapprox 1.01$  is okay.

```python
az.rhat(chains)
```

### Monte carlo standard error


MCMC methods introduce an additional layer of uncertainty
as we approximate the posterior with a finite number of samples.
**Monte Carlo standard error (MCSE)**
takes into account that the samples
are not truly independent of each other.
MCSEE required domain expertise.
If we want to report a value of an estimated parameter to the second decimal,
MCSE must be below that second decimal.
Check MCSE after checking
that ESS is high enough
and $\hat R$ is low enough.

```python
az.mcse(chains)
```

```python
az.plot_mcse(chains);
```

Ideally we want MCSE to be small across all regions of the parameter space.

```python
az.summary(chains, kind="diagnostics")
```

### Trace plots


**Trace plots** are often the first plot you make after inference.
Draws the sampled values at each iteration step.
Check if different chains
converge to the same distributuion.
Check autocorrelation.

```python
az.plot_trace(chains)
plt.tight_layout();
```

Left column:
one KDE per chain.
Right column:
sampled values per chain per step.

Good chains have only small differences between distributions,
and ordered values should have no pattern,
and be difficult to distinguish chains from each other.
Random peaks that are inconsistent from trace to trace—like
`bad_chains1`—are
suspicious.


### Autocorrelation plots


Autocorrelation decreases the actual amount of information in a sample.

```python
az.plot_autocorr(chains, combined=True);
```

### Rank plots





**Rank plots**
are histograms of the ranked samples.
Ranks are computed for all chains,
but plots are per chain.
If all chains are targeting the same distribution,
we expect a Uniform distribution.

```python
az.plot_rank(chains, kind="bars");
```

Can also plot vertical lines.
Vertical lines above the dashed line
indicate an exces sample value.
Below the line is a lack of sampled values.
The shorter the line,
the better.

```python
az.plot_rank(chains, kind="vlines");
```

Rank plots are more sensitive thatn trace plots.
`az.plot_trace(..., kind="rank_bars")` or `az.plot_trace(..., kind="rank_vlines")`.


### Divergences


Besides studying the generated samples,
can also monitor the innner workings
of the sampling method.



```python
with pm.Model() as model_0:
    θ1 = pm.Normal("θ1", 0, 1, testval=0.1)
    θ2 = pm.Uniform("θ2", -θ1, θ1)
    idata_0 = pm.sample(return_inferencedata=True)

with pm.Model() as model_1:
    θ1 = pm.HalfNormal("θ1", 1 / (1 - 2 / np.pi) ** 0.5)
    θ2 = pm.Uniform("θ2", -θ1, θ1)
    idata_1 = pm.sample(return_inferencedata=True)

with pm.Model() as model_1bis:
    θ1 = pm.HalfNormal("θ1", 1 / (1 - 2 / np.pi) ** 0.5)
    θ2 = pm.Uniform("θ2", -θ1, θ1)
    idata_1bis = pm.sample(target_accept=0.95, return_inferencedata=True)
```

```python
for i, idata in enumerate([idata_0, idata_1, idata_1bis]):
    az.plot_trace(idata, kind="rank_vlines")
    az.plot_pair(idata, divergences=True)
plt.tight_layout();
```

Each of the vertical bars represents a divergence in the trace plots,
and in the pair plots they are the teal points.

`model_0` has $\theta1$ as a Normal centered at 0,
which is negative half the time.
`model_1` tries to reparamaterize.
`model_1bis` increases `target_accept`
and looks good,
but need to check ESS and $\hat R$
to be sure.


### Sampler parameters and other diagnostics

Solutions:

- Increase `target_accept`
  if divergences originate from numerical imprecision.
- Increase `tune`—which
  further adjusts the sampler parameters—can
  help to increase ESS or lower $\hat R$
- Increase the number of draws can help with convergence—but
  is the least productive solution.
- Reparameterizxation
- Improve model structue
- More informative priors
- Change model

Samples:

- Use 200–300 samples for testing
- Use 2,000–4,000 when more comfertable

Additional diagnostics:

- Parellel plots
- Seperation plots


## Model comparison


Can compare models using generalization error,
or out-of-sample predictive accuracy—and
estimate of how well a model behaves at predicting data
not used to fit it.

One generic metric is the logrithmic scoring rule—the
**expected log pointwise predictive density (ELPD)**.

$$
\text{ELPD} = \sum_{i=1}^{n} \int p_t(\tilde y_i) \; \log p(\tilde y_i \mid y_i) \; d\tilde y_i
$$

where $p_t(\tilde y_i)$ is the distribution of the true data-generating process for $\tilde y_i$
and $p(\tilde y_i \mid y_i)$ is the posterior predictive distribution.

For real problems we do not know $p_t(\tilde y_i)$,
so we can use the deviance:

$$
\sum_{i=1}^{n} \log \int \ p(y_i \mid \boldsymbol{\theta}) \; p(\boldsymbol{\theta} \mid y) d\boldsymbol{\theta}
$$

To compute this
we use the same data used to fit the model,
and on average overestimate the ELPD—which
leads to overfitting.
Cross-validation helps alleviate this problem.


### Cross-validation and LOO


**Cross-validation (CV)**
is estimateing out-of-sample predictive accuracy.
You re-fit the model many times,
each time excluding a different portion of the data.
The excluded portion is used to measure the accuracy of the model.
This is repeated many times
and the results are averaged over the runs.

$$
\text{ELPD}_\text{LOO-CV} = \sum_{i=1}^{n} \log
    \int \ p(y_i \mid \boldsymbol{\theta}) \; p(\boldsymbol{\theta} \mid y_{-i}) d\boldsymbol{\theta}
$$

In practive we don't know $\boldsymbol{\theta}$ 
and need to compute $n$ posteriors.
This is expensice,
and we can approximate $\text{ELPD}_\text{LOO-CV}$
from a single fit to the data
by using Pareto smoothed importance sampling leave-one-out cross validation PSIS-LOO-CV—or
just LOO.

```python
rng = np.random.default_rng(42)
y_obs = rng.normal(0, 1, size=100)
idatas_cmp = {}

# Generate data from Skewnormal likelihood model
# with fixed mean and skewness and random standard deviation
with pm.Model() as mA:
    σ = pm.HalfNormal("σ", 1)
    y = pm.SkewNormal("y", 0, σ, alpha=1, observed=y_obs)
    # This pattern is likely to change in PyMC 4.0
    trace_A = pm.sample(return_inferencedata=False)
    posterior_predictive_A = pm.sample_posterior_predictive(trace_A)
    idataA = az.from_pymc3(
        trace=trace_A,
        posterior_predictive=posterior_predictive_A,
    )
idatas_cmp["mA"] = idataA

# Generate data from Normal likelihood model
# with fixed mean with random standard deviation
with pm.Model() as mB:
    σ = pm.HalfNormal("σ", 1)
    y = pm.Normal("y", 0, σ, observed=y_obs)
    trace_B = pm.sample(return_inferencedata=False)
    posterior_predictive_B = pm.sample_posterior_predictive(trace_B)
    idataB = az.from_pymc3(
        trace=trace_B,
        posterior_predictive=posterior_predictive_B
    )
idatas_cmp["mB"] = idataB

# Generate data from Normal likelihood model
# with random mean and random standard deviation
with pm.Model() as mC:
    μ = pm.Normal("μ", 0, 1)
    σ = pm.HalfNormal("σ", 1)
    y = pm.Normal("y", μ, σ, observed=y_obs)
    trace_C = pm.sample(return_inferencedata=False)
    posterior_predictive_C = pm.sample_posterior_predictive(trace_B)
    idataC = az.from_pymc3(
        trace=trace_C,
        posterior_predictive=posterior_predictive_C,
    )
idatas_cmp["mC"] = idataC
```

```python
az.loo(idataA)
```

```python
compare_df = az.compare(idatas_cmp)
compare_df
```

- `rank`
  Ranks the model from highest predictive accuracy
- `loo`
  lists ELPD values
- `p_loo`
  list values for the penalization term.
  This is kinda the estaimted effective number of parameteres,
  it can be lower than the actual number of parameters
  when model has more structure—
  like a hierarchical model—or
  can be much higher when the model has weak predictive capability
  and may indicate misspecification for model.
- `d_loo`
  list of relative differences
  between the value of LOO
  for the top and current model.
- `weight`
  is the weight for each model—the
  probability of each model given the data
  amoung other compared models
- `se`
  The standard error for the ELPD
- `dse`
  The standard errors of the differences between two values
  of the ELPD.
  `dse` is not necessarity the same as `se`,
  because the uncertaintly about the ELPD can be correlated between models.
  dse is always 0 for top rank models.
- `warning`
  if LOO may not be reliable.
- `loo_scale`
  is the scale of reported values.
  Default is log scale.

```python
az.plot_compare(compare_df);
```

Can plot as well.
Where open circle is `loo`,
black dots are predictive accuracy without `p_loo`,
Black segments are error for LOO (`se`),
and grey segments are erorrs of the difference `dse`
between LOO value and best ranked value.

`mA` is worst,
but `mB` and `mC` are close.
Rule of thumb—LOO difference below 4 is small.
Difference is that `mB` the mean is fixed at 0,
and `mC` it has a prior—which
results in a penalty.
This is why `p_loo` is larger for `mC`,
and black dot on plot
(unpenalized ELPD)
and open dot
($\text{ELPD}_\text{LOO-CV}$)
is larger as well.
`dse` is also lower
than respective `se`—indicating
correlation.
