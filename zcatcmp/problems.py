"""ZCAT (jMetalPy) con registro de evaluaciones reales y corte exacto del presupuesto.

Todos los algoritmos evalúan a través de ``LoggedZCAT``: cada evaluación real queda
registrada en orden en un ``EvaluationLog``. El resultado de una corrida se obtiene
de ese registro (no del algoritmo), así que todos se miden igual: el conjunto no
dominado de las primeras ``N_EVAL`` evaluaciones reales.
"""
from __future__ import annotations

import numpy as np
from jmetal.core.solution import FloatSolution
from pymoo.core.problem import Problem

from . import config
from .zcat_jmetal import ZCAT


class BudgetExhausted(Exception):
    """Se intentó evaluar más allá del presupuesto."""


class EvaluationLog:
    def __init__(self, budget: int, n_var: int, n_obj: int):
        self.budget = budget
        self.X = np.empty((budget, n_var))
        self.F = np.empty((budget, n_obj))
        self.n = 0

    def record(self, x, f) -> None:
        self.X[self.n] = x
        self.F[self.n] = f
        self.n += 1

    @property
    def exhausted(self) -> bool:
        return self.n >= self.budget


class LoggedZCAT(ZCAT):
    """ZCAT que registra cada evaluación y lanza ``BudgetExhausted`` al pasarse."""

    def __init__(self, problem_id: int, n_obj: int, log: EvaluationLog, n_var: int = config.N_VAR):
        super().__init__(problem_id, number_of_variables=n_var, number_of_objectives=n_obj, **config.ZCAT_KWARGS)
        self.log = log

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if self.log.exhausted:
            raise BudgetExhausted
        super().evaluate(solution)
        self.log.record(solution.variables, solution.objectives)
        return solution

    def evaluate_x(self, x) -> np.ndarray:
        sol = FloatSolution(self.lower_bound, self.upper_bound, self.number_of_objectives())
        sol.variables = list(map(float, x))
        return np.asarray(self.evaluate(sol).objectives)

    def solutions_from(self, X) -> list:
        """Soluciones jMetal con las variables dadas (sin evaluar).

        Usa ``create_solution`` y sobreescribe las variables, igual que
        ``SurrogateArchive.lhs`` del código original, para consumir el mismo
        flujo de ``random``.
        """
        sols = []
        for row in X:
            sol = self.create_solution()
            sol.variables = row.tolist()
            sols.append(sol)
        return sols


class PymooZCAT(Problem):
    """Vista pymoo de un ``LoggedZCAT`` (comparte registro y presupuesto)."""

    def __init__(self, zcat: LoggedZCAT):
        super().__init__(
            n_var=zcat.number_of_variables(),
            n_obj=zcat.number_of_objectives(),
            xl=np.asarray(zcat.lower_bound, dtype=float),
            xu=np.asarray(zcat.upper_bound, dtype=float),
        )
        self.zcat = zcat

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.zcat.evaluate_x(x) for x in X])


def bounds(n_var: int = config.N_VAR):
    idx = np.arange(1.0, n_var + 1.0)
    return -0.5 * idx, 0.5 * idx
