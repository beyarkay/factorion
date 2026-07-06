"""Dependency-free statistical helpers for multi-seed run comparison.

Paired t-test when both sides share the same seeds (removes between-seed
variance, dramatically more power), Welch's t-test otherwise.
"""

from __future__ import annotations

import math


# ── Statistical helpers (no scipy dependency) ──────────────────────


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch_t_test(
    a: list[float], b: list[float]
) -> tuple[float, float, float]:
    """Welch's t-test for two independent samples with unequal variances.

    Returns (t_statistic, degrees_of_freedom, p_value).
    p_value is two-tailed, computed via the regularized incomplete beta function.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 0.0, 1.0

    mean_a, mean_b = mean(a), mean(b)
    var_a = stdev(a) ** 2
    var_b = stdev(b) ** 2

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, n_a + n_b - 2, 1.0

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else n_a + n_b - 2

    p_value = _t_distribution_p_value(abs(t_stat), df) * 2  # two-tailed
    p_value = min(p_value, 1.0)

    return t_stat, df, p_value


def _t_distribution_p_value(t: float, df: float) -> float:
    """One-tailed p-value from the t-distribution using the regularized
    incomplete beta function.  P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2).
    """
    x = df / (df + t * t)
    return 0.5 * _regularized_incomplete_beta(x, df / 2.0, 0.5)


def _regularized_incomplete_beta(
    x: float, a: float, b: float, max_iter: int = 200, tol: float = 1e-12
) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction
    (Lentz's algorithm)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the symmetry relation when x > (a+1)/(a+b+2) for convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a, max_iter, tol)

    # Front factor: x^a * (1-x)^b / (a * B(a,b))
    ln_front = (
        a * math.log(x)
        + b * math.log(1.0 - x)
        - math.log(a)
        - _ln_beta(a, b)
    )
    front = math.exp(ln_front)

    # Continued fraction (modified Lentz)
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    return front * f


def _ln_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def paired_t_test(
    a: list[float], b: list[float]
) -> tuple[float, float, float]:
    """Paired t-test for matched samples (same seeds run on both branches).

    Computes differences d_i = a_i - b_i, then tests whether mean(d) != 0.
    Returns (t_statistic, degrees_of_freedom, p_value).
    p_value is two-tailed.
    """
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0, 0.0, 1.0

    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_d = mean(diffs)
    std_d = stdev(diffs)

    if std_d == 0:
        return 0.0, n - 1, 1.0

    t_stat = mean_d / (std_d / math.sqrt(n))
    df = n - 1

    p_value = _t_distribution_p_value(abs(t_stat), df) * 2  # two-tailed
    p_value = min(p_value, 1.0)

    return t_stat, df, p_value


def cohens_d(a: list[float], b: list[float], paired: bool = False) -> float:
    """Cohen's d effect size.

    When paired=True, uses the standard deviation of the differences
    (appropriate for paired designs).  Otherwise uses the pooled SD.
    """
    if paired:
        if len(a) != len(b) or len(a) < 2:
            return 0.0
        diffs = [ai - bi for ai, bi in zip(a, b)]
        sd = stdev(diffs)
        if sd == 0:
            return 0.0
        return mean(diffs) / sd

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = stdev(a) ** 2
    var_b = stdev(b) ** 2
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (mean(a) - mean(b)) / pooled_std
