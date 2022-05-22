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

<!-- #region tags=[] -->
# Splines
<!-- #endregion -->

<!-- #region tags=[] -->
## Imports
<!-- #endregion -->

```python
import pymc3 as pm
```

## Polynomial regression


Linear model can be written as

$$
\mathbb{E}[Y]= \beta_0 + \beta_1 X
$$

which looks like

$$
\mathbb{E}[Y]= \beta_0 + \beta_1 X + \beta_2 X^2 + \cdots + \beta_m X^m
$$

for a **polynomial regression**.
As we increase the order of the polynomial—$m$—we
increase the flexibility of the curve.
Problem is that polynomials act globally—when
we apply a polynomial of degree $m$
we are saying the relationship between the $Y$ and $X$ is of degree $m$
for the entire dataset.

This can lead to curves that are too flexible,
and prone to overfitting.

<!-- #region tags=[] -->
## Expanding the feature space
<!-- #endregion -->

Polynomial regression
is a method to **expand the feature space**.
Beyond polynomials,
we can expand features as:

$$
\mathbb{E}[Y]= \beta_0 + \beta_1 B_{1}(X_{1}) + \beta_2 B_{2}(X_{2}) + \cdots + \beta_m B_{m}(X_{m})
$$

where $B_i$ are arbitrary functions—**basis
functions**.

Besides polynomials,
basis functions
can be a power of two,
lograrithms,
square root,
etc.

Can use indicator functions
like $I(c_i \leq x_k < c_j)$
to break up the original $\boldsymbol{X}$ predictor into (non-overlapping) subsets
and then fit the polynomial locally—only
inside these subsets.
This is fitting **piecewise polynomials**.

![piecewise](images/chapter_5/piecewise.png)

In chart above,
blue line is true function we try to approximate.
Black line are piecewise polynomials of order 1–4—respectively.
Dashed verticals are limits of each subdomain.

Same idea can be extended to more than one predictor,
and be combined with an inverse link function $\phi$.
These are known as **Generalized Additive Models (GAM)**.

$$
\mathbb{E}[Y]= \phi \left(\sum_i^p f(X_i)\right)
$$

