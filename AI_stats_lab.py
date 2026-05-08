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
        (0, 0): 0.10,
        (0, 1): 0.05,
        (0, 2): 0.00,
        (0, 3): 0.00,

        (1, 0): 0.15,
        (1, 1): 0.20,
        (1, 2): 0.05,
        (1, 3): 0.00,

        (2, 0): 0.00,
        (2, 1): 0.10,
        (2, 2): 0.15,
        (2, 3): 0.05,

        (3, 0): 0.00,
        (3, 1): 0.00,
        (3, 2): 0.05,
        (3, 3): 0.10
    }

    return pmf.get((x, y), 0.0)


def marginal_px(x):
    """
    Compute PX(x) by summing joint_pmf(x, y) over y = 0,1,2,3.
    """

    total = 0

    for y in range(4):
        total += joint_pmf(x, y)

    return total


def marginal_py(y):
    """
    Compute PY(y) by summing joint_pmf(x, y) over x = 0,1,2,3.
    """

    total = 0

    for x in range(4):
        total += joint_pmf(x, y)

    return total


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

    distribution = {}

    for x in range(4):
        distribution[x] = conditional_pmf_x_given_y(x, y)

    return distribution


def probability_sum_greater_than_3():
    """
    Compute P(X + Y > 3).
    """

    probability = 0

    for x in range(4):
        for y in range(4):

            if x + y > 3:
                probability += joint_pmf(x, y)

    return probability


def independence_check():
    """
    Return True if X and Y are independent.

    X and Y are independent if:

    joint_pmf(x,y) = PX(x) * PY(y)

    for every x and y.
    """

    for x in range(4):
        for y in range(4):

            left_side = joint_pmf(x, y)
            right_side = marginal_px(x) * marginal_py(y)

            if not np.isclose(left_side, right_side):
                return False

    return True


# -------------------------------------------------
# Expectation, Covariance, and Correlation
# -------------------------------------------------

def expected_x():
    """
    Compute E[X].
    """

    expectation = 0

    for x in range(4):
        expectation += x * marginal_px(x)

    return expectation


def expected_y():
    """
    Compute E[Y].
    """

    expectation = 0

    for y in range(4):
        expectation += y * marginal_py(y)

    return expectation


def expected_xy():
    """
    Compute E[XY].
    """

    expectation = 0

    for x in range(4):
        for y in range(4):
            expectation += x * y * joint_pmf(x, y)

    return expectation


def variance_x():
    """
    Compute Var(X).
    """

    ex = expected_x()

    ex2 = 0

    for x in range(4):
        ex2 += (x ** 2) * marginal_px(x)

    return ex2 - (ex ** 2)


def variance_y():
    """
    Compute Var(Y).
    """

    ey = expected_y()

    ey2 = 0

    for y in range(4):
        ey2 += (y ** 2) * marginal_py(y)

    return ey2 - (ey ** 2)


def covariance_xy():
    """
    Compute Cov(X,Y).

    Cov(X,Y) = E[XY] - E[X]*E[Y]
    """

    return expected_xy() - (expected_x() * expected_y())


def correlation_xy():
    """
    Compute correlation coefficient:

    rho_XY = Cov(X,Y) / sqrt( Var(X) * Var(Y) )
    """

    cov = covariance_xy()
    var_x = variance_x()
    var_y = variance_y()

    return cov / np.sqrt(var_x * var_y)


def variance_sum():
    """
    Compute Var(X+Y).
    """

    values = []
    probabilities = []

    for x in range(4):
        for y in range(4):

            values.append(x + y)
            probabilities.append(joint_pmf(x, y))

    expected_sum = 0

    for i in range(len(values)):
        expected_sum += values[i] * probabilities[i]

    expected_sum_square = 0

    for i in range(len(values)):
        expected_sum_square += (values[i] ** 2) * probabilities[i]

    return expected_sum_square - (expected_sum ** 2)


def variance_identity_check():
    """
    Verify:

    Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y)

    Return True if the identity holds, else False.
    """

    left_side = variance_sum()

    right_side = (
        variance_x()
        + variance_y()
        + 2 * covariance_xy()
    )

    return np.isclose(left_side, right_side)
