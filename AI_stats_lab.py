## `AI_stats_lab.py`


import numpy as np


# -------------------------------------------------
# Sparse 4 by 4 Joint PMF
# -------------------------------------------------

def joint_pmf(x, y):
    """
    Joint PMF table:

             y=0   y=1   y=2   y=3
    x=0      0.10  0.05  0.00  0.00
    x=1      0.15  0.20  0.05  0.00
    x=2      0.00  0.10  0.15  0.05
    x=3      0.00  0.00  0.05  0.10
    """
    pmf = {
        (0, 0): 0.10, (0, 1): 0.05, (0, 2): 0.00, (0, 3): 0.00,
        (1, 0): 0.15, (1, 1): 0.20, (1, 2): 0.05, (1, 3): 0.00,
        (2, 0): 0.00, (2, 1): 0.10, (2, 2): 0.15, (2, 3): 0.05,
        (3, 0): 0.00, (3, 1): 0.00, (3, 2): 0.05, (3, 3): 0.10,
    }
    return pmf.get((x, y), 0.00)


def marginal_px(x):
    """
    Compute PX(x) by summing joint_pmf(x, y) over y = 0,1,2,3.
    """
    return sum(joint_pmf(x, y) for y in range(4))


def marginal_py(y):
    """
    Compute PY(y) by summing joint_pmf(x, y) over x = 0,1,2,3.
    """
    return sum(joint_pmf(x, y) for x in range(4))


def conditional_pmf_x_given_y(x, y):
    """
    Compute P(X=x given Y=y).

    P(X=x given Y=y) = joint_pmf(x,y) / PY(y)

    If PY(y) is zero, return 0.
    """
    py = marginal_py(y)
    if py == 0:
        return 0
    return joint_pmf(x, y) / py


def conditional_distribution_x_given_y(y):
    """
    Return conditional distribution of X given Y=y
    as dictionary:

    {
        0: P(X=0 given Y=y),
        1: P(X=1 given Y=y),
        2: P(X=2 given Y=y),
        3: P(X=3 given Y=y)
    }
    """
    return {x: conditional_pmf_x_given_y(x, y) for x in range(4)}


def probability_sum_greater_than_3():
    """
    Compute P(X + Y > 3).
    """
    return sum(joint_pmf(x, y) for x in range(4) for y in range(4) if x + y > 3)


def independence_check():
    """
    Return True if X and Y are independent.

    X and Y are independent if:

    joint_pmf(x,y) = PX(x) * PY(y)

    for every x and y.
    """
    for x in range(4):
        for y in range(4):
            if not np.isclose(joint_pmf(x, y), marginal_px(x) * marginal_py(y)):
                return False
    return True


# -------------------------------------------------
# Expectation, Covariance, and Correlation
# -------------------------------------------------

def expected_x():
    """
    Compute E[X].
    """
    return sum(x * marginal_px(x) for x in range(4))


def expected_y():
    """
    Compute E[Y].
    """
    return sum(y * marginal_py(y) for y in range(4))


def expected_xy():
    """
    Compute E[XY].
    """
    return sum(x * y * joint_pmf(x, y) for x in range(4) for y in range(4))


def variance_x():
    """
    Compute Var(X).
    """
    ex = expected_x()
    return sum((x - ex) ** 2 * marginal_px(x) for x in range(4))


def variance_y():
    """
    Compute Var(Y).
    """
    ey = expected_y()
    return sum((y - ey) ** 2 * marginal_py(y) for y in range(4))


def covariance_xy():
    """
    Compute Cov(X,Y).

    Cov(X,Y) = E[XY] - E[X]*E[Y]
    """
    return expected_xy() - expected_x() * expected_y()


def correlation_xy():
    """
    Compute correlation coefficient:

    rho_XY = Cov(X,Y) / sqrt( Var(X) * Var(Y) )
    """
    denominator = np.sqrt(variance_x() * variance_y())
    if denominator == 0:
        return 0
    return covariance_xy() / denominator


def variance_sum():
    """
    Compute Var(X+Y).
    """
    expected_sum = expected_x() + expected_y()
    return sum(((x + y) - expected_sum) ** 2 * joint_pmf(x, y) for x in range(4) for y in range(4))


def variance_identity_check():
    """
    Verify:

    Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)

    Return True if the identity holds, else False.
    """
    return bool(np.isclose(variance_sum(), variance_x() + variance_y() + 2 * covariance_xy()))
