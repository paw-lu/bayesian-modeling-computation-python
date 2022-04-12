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

# Probability background

<!-- #region tags=[] -->
## Probability
<!-- #endregion -->

- **Sample space** $\mathcal{X}$ is the set of all possible events from an **experiment**
- **Event** $A$ is a subset of $\mathcal{X}$


## Conditional probability


$$
P(A \mid B) = \frac{P(A, B)}{P(B)}
$$

$A$ & $B$

$$
P(A, B) = P(A \cap B)
$$ 

$A$ given $B$

$$
P(A \mid B)
$$



## Discrete random variables and distributions


- **Probability Mass Function (PMF)**
  is a function that return probabilities of discrete values.
  All values sum up to 1.
- **Cumulative Distribution Function (CDF)**
  is a function that returns probabilty that value is less than or equal to parameter.
  It is monotonically increasing,
  right-continuous,
  converge to 0 at $- \infty$,
  1 at $-\infty$.


### Discrete uniform distribution


$$
P(X = x) = {\frac {1}{b - a + 1}} = \frac{1}{n}
$$

For interval $[a, b]$,
otherwise $P(X = x) = 0$,
where $n=b-a+1$ is the total number of values $x$ can take.


### Binomial distribution


A **Bernoulli trial** is an experiment with only two possible outcomes.
$n$ indepentent Bernoulli trials
with a success probability $p$.
This distribution of $X$ is a Binomial distribution.

$$
P(X = x) = \frac{n!}{x!(n-x)!}p^x(1-p)^{n-x}
$$

Only considers total number of successes,
not order.
First term is Binomial Coefficient—it
computes all possible combinations of $x$ elements
taken from a set of $n$ elements.
Second term counts the number of $x$ successes
in $n$ trials.

At $n = 1$,
Binomial distribution is known as
Bernoulli distribution.


### Poisson distribution


The probability that $x$ events happen
during a fixed time interval.
These events occur with an average rate $\mu$
and independent from each other.
Used when there are a large number of trials,
each with a small probability of success:

- Radioactive decay
- Number of car accidents

$$
P(X = x)  = \frac{\mu^{x} e^{-\mu}}{x!}, x = 0, 1, 2, \dots
$$

It is an infite set.

As $\mu$ increases,
Poisson approximates a (discrete) Normal distribution.
A Binomial distribution can be approximated with a Poisson
when $n >> p$.

$$
\text{Pois}(\mu=np) \approx \text{Bin}(n, p)
$

```python


```
