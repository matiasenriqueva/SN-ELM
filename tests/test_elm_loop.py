"""El bucle reescrito reproduce exactamente a SurrogateNSGAII de code/surrogate_ELM.

El original predice cada solución por separado (X de 1 fila) y el reescrito en
lote. Según la plataforma, BLAS suma en distinto orden para cada forma de matriz
(diferencias ~1e-16) y un empate de dominancia o de crowding basta para que dos
trayectorias se separen. Por eso la comparación de trayectorias usa en ambos lados
una ELM que predice fila por fila: así prueba la lógica del bucle de forma exacta
en cualquier máquina, y ``test_batch_prediction_matches_rowwise`` acota aparte la
diferencia numérica entre ambas formas de predecir.
"""
import numpy as np
import pytest
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import DominanceComparator

from tests._original import original_archive, original_snsgaii, original_surrogate, requires_original
from zcatcmp.elm_loop import ELMLoopNSGAII
from zcatcmp.problems import EvaluationLog, LoggedZCAT
from zcatcmp.seeding import seed_everything, shared_doe
from zcatcmp.surrogates import ELM, SurrogateModel

N_DOE, BUDGET, NS, G = 60, 120, 5, 3


class _RowwiseELM(ELM):
    """Misma ELM, pero cada fila se predice como una matriz de 1 fila (igual que el original)."""

    def predict(self, X):
        return np.vstack([super().predict(x[None, :]) for x in X])


@pytest.mark.parametrize("act", ["sigmoid", "gaussian", "multiquadric"])
def test_batch_prediction_matches_rowwise(act):
    rng = np.random.default_rng(0)
    X, Y = rng.uniform(-1, 1, (600, 30)), rng.normal(size=(600, 3))
    np.random.seed(1)
    elm = ELM(121, C=32.0, activation=act)
    elm.fit(X, Y)
    rowwise = _RowwiseELM(121, C=32.0, activation=act)
    rowwise._W, rowwise._b, rowwise._beta = elm._W, elm._b, elm._beta
    np.testing.assert_allclose(elm.predict(X[:50]), rowwise.predict(X[:50]), rtol=1e-9, atol=1e-9)


def _random_fill(problem):
    lb, ub = np.asarray(problem.lower_bound), np.asarray(problem.upper_bound)
    return problem.solutions_from([lb + np.random.random(len(lb)) * (ub - lb)])[0]


@requires_original
@pytest.mark.parametrize("p,M,act", [(1, 2, "multiquadric"), (12, 3, "sigmoid"), (19, 2, "multiquadric")])
def test_same_trajectory_as_original(monkeypatch, p, M, act):
    X_doe = shared_doe(p, M, 0, N_DOE, 30)
    nh = 4 * 30 + 1

    # --- original: init_progress + step hasta agotar el presupuesto ---
    old_problem = LoggedZCAT(p, M, EvaluationLog(10 * BUDGET, 30, M))

    def fake_lhs(n, problem, use_scipy=True):
        return problem.solutions_from(X_doe) if n == N_DOE else [_random_fill(problem)]

    monkeypatch.setattr(original_archive.SurrogateArchive, "lhs", staticmethod(fake_lhs))
    seed_everything(42)
    old = original_snsgaii.SurrogateNSGAII(
        problem=old_problem, population_size=50, offspring_size=50,
        crossover=SBXCrossover(probability=0.9, distribution_index=20),
        mutation=PolynomialMutation(probability=2.0 / 30, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=DominanceComparator()),
        surrogate=original_surrogate.ELMSurrogate(nh, C=32.0, activation=act),
        evaluation_budget=BUDGET, n_initial=N_DOE, n_select=NS, inner_generations=G,
    )
    old.init_progress()
    while old._real_evals < BUDGET:
        old.step()
    X_old = np.array([s.variables for s in old._archive.PE])

    # --- reescrito ---
    log = EvaluationLog(BUDGET, 30, M)
    new_problem = LoggedZCAT(p, M, log)
    seed_everything(42)
    model = SurrogateModel(_RowwiseELM(nh, C=32.0, activation=act), new_problem.lower_bound,
                           new_problem.upper_bound, N_DOE)
    new = ELMLoopNSGAII(new_problem, model, X_doe, BUDGET, n_select=NS, inner_generations=G)
    new.run()

    assert log.n == BUDGET
    np.testing.assert_allclose(log.X, X_old, rtol=0, atol=1e-12)
