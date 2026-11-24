from pathlib import Path

import numpy as np
import pytest

from zcatcmp.problems import BudgetExhausted, EvaluationLog, LoggedZCAT, PymooZCAT, bounds
from zcatcmp.seeding import shared_doe

GOLDEN = np.load(Path(__file__).parent / "data" / "zcat_golden.npz")


@pytest.mark.parametrize("M", [2, 3])
@pytest.mark.parametrize("p", range(1, 21))
def test_matches_golden(p, M):
    """Mismos valores que el ZCAT de jMetal usado en code/surrogate_ELM (.env original)."""
    X, F = GOLDEN[f"X_{p}_{M}"], GOLDEN[f"F_{p}_{M}"]
    zcat = LoggedZCAT(p, M, EvaluationLog(len(X), 30, M))
    got = np.array([zcat.evaluate_x(x) for x in X])
    np.testing.assert_allclose(got, F, rtol=1e-12, atol=1e-12)


def test_pymoo_view_shares_log_and_values():
    X, F = GOLDEN["X_7_3"], GOLDEN["F_7_3"]
    log = EvaluationLog(len(X), 30, 3)
    out = {}
    PymooZCAT(LoggedZCAT(7, 3, log))._evaluate(X, out)
    np.testing.assert_allclose(out["F"], F, rtol=1e-12, atol=1e-12)
    assert log.n == len(X)
    np.testing.assert_array_equal(log.X, X)


def test_budget_is_exact():
    log = EvaluationLog(5, 30, 2)
    zcat = LoggedZCAT(1, 2, log)
    X = GOLDEN["X_1_2"]
    for x in X[:5]:
        zcat.evaluate_x(x)
    with pytest.raises(BudgetExhausted):
        zcat.evaluate_x(X[5])
    assert log.n == 5


def test_shared_doe_is_deterministic_lhs():
    A = shared_doe(3, 2, 0, 600, 30)
    np.testing.assert_array_equal(A, shared_doe(3, 2, 0, 600, 30))
    assert not np.array_equal(A, shared_doe(3, 2, 1, 600, 30))
    assert not np.array_equal(A, shared_doe(3, 3, 0, 600, 30))
    lb, ub = bounds(30)
    U = (A - lb) / (ub - lb)
    assert U.min() >= 0 and U.max() <= 1
    # propiedad LHS: un punto por estrato en cada dimensión
    for j in range(30):
        assert len(np.unique(np.floor(U[:, j] * 600))) == 600
