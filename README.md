[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/bMND8GGu)
# Advanced AI Statistics Lab
## Sparse Joint PMF, Conditional PMF, Covariance, and Correlation

This lab focuses on two-variable probability concepts:

- Joint PMF
- Marginal PMFs
- Conditional PMFs
- Probability over a set
- Expectation
- Covariance
- Correlation
- Independence
- Variance identity

---

# Question 1 — Sparse 4 by 4 Joint PMF

You are given the following joint PMF:

|      | y=0  | y=1  | y=2  | y=3  |
|------|------|------|------|------|
| x=0  | 0.10 | 0.05 | 0.00 | 0.00 |
| x=1  | 0.15 | 0.20 | 0.05 | 0.00 |
| x=2  | 0.00 | 0.10 | 0.15 | 0.05 |
| x=3  | 0.00 | 0.00 | 0.05 | 0.10 |

Tasks:

1. Implement the joint PMF.
2. Compute marginal PMFs PX(x) and PY(y).
3. Compute conditional PMF P(X=x given Y=y).
4. Return the full conditional distribution of X given Y=y.
5. Compute P(X+Y > 3).
6. Check whether X and Y are independent.

---

# Question 2 — Expectation, Covariance, and Correlation

Using the same joint PMF table:

Tasks:

1. Compute E[X].
2. Compute E[Y].
3. Compute E[XY].
4. Compute Var(X).
5. Compute Var(Y).
6. Compute Cov(X,Y).
7. Compute the correlation coefficient:

rho_XY = Cov(X,Y) / sqrt( Var(X) * Var(Y) )

8. Compute Var(X+Y).

9. Verify:

Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)

---

# Run Locally

Install requirements:

```bash
pip install numpy pytest
