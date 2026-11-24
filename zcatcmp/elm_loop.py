"""Bucle ELM-NSGA-II (code/surrogate_ELM/snsgaii.py) con cualquier surrogado.

Misma lógica que ``SurrogateNSGAII``, con cuatro cambios que no alteran el
algoritmo (ver tests/test_elm_loop.py):

* Los hijos de cada generación interna se aproximan en lote (antes, uno a uno).
  La selección sólo mira a la población actual, así que el orden de consumo de
  números aleatorios es el mismo; las predicciones difieren sólo por el redondeo
  de BLAS (~1e-16), que según la plataforma puede romper un empate distinto.
* El filtro de novedad usa una matriz con las variables de P_E.
* El diseño inicial se recibe desde fuera (LHS compartido entre algoritmos).
* ``run`` controla el presupuesto con las evaluaciones reales. El ``run``
  heredado de jMetal evaluaba antes 50 soluciones aleatorias que no entraban al
  archivo, y su contador podía cortar la corrida hasta 50 evaluaciones antes
  tras una iteración sin candidatos.
"""
from __future__ import annotations

import copy
from typing import Callable, List

import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.util.comparator import DominanceComparator
from jmetal.util.density_estimator import CrowdingDistanceDensityEstimator as CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from . import config
from .problems import LoggedZCAT
from .surrogates import SurrogateModel


class ELMLoopNSGAII(NSGAII):
    def __init__(
        self,
        problem: LoggedZCAT,
        model: SurrogateModel,
        X_doe: np.ndarray,
        budget: int,
        n_select: int,
        inner_generations: int,
        population_size: int = config.POP_SIZE,
        epsilon: float = config.EPSILON,
        fill_sampler: Callable[[], FloatSolution] | None = None,
    ):
        n = problem.number_of_variables()
        super().__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size,
            mutation=PolynomialMutation(probability=config.PM_FACTOR / n, distribution_index=config.PM_ETA),
            crossover=SBXCrossover(probability=config.SBX_PROB, distribution_index=config.SBX_ETA),
            selection=BinaryTournamentSelection(comparator=DominanceComparator()),
            termination_criterion=StoppingByEvaluations(budget),
        )
        self.model = model
        self.X_doe = X_doe
        self.budget = budget
        self.n_select = n_select
        self.inner_G = inner_generations
        self.epsilon = epsilon
        self.fill_sampler = fill_sampler or self._random_solution
        self.PE: List[FloatSolution] = []
        self._PE_X = None
        self._real_evals = 0
        self.empty_iterations = 0

    # ------------------------------------------------------------------
    def run(self):
        self.init_progress()
        consecutive_empty = 0
        while self._real_evals < self.budget:
            if self.step():
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                self.empty_iterations += 1
                if consecutive_empty >= config.MAX_EMPTY_ITERATIONS:
                    break

    def init_progress(self) -> None:
        initial = self.problem.solutions_from(self.X_doe)
        for sol in initial:
            self.problem.evaluate(sol)
        self._real_evals = len(initial)
        self.PE = list(initial)
        self._PE_X = np.array([s.variables for s in initial])
        self.model.training.add_initial(self._PE_X, np.array([s.objectives for s in initial]))
        self.solutions = copy.deepcopy(initial[: self.population_size])
        self.evaluations = self._real_evals

    def step(self) -> int:
        """Un ciclo surrogado; devuelve cuántas evaluaciones reales hizo."""
        self.model.fit()
        candidates = self._select_candidates(self._inner_nsgaii())
        if not candidates:
            return 0
        evaluated = []
        for sol in candidates:
            self.problem.evaluate(sol)
            self._real_evals += 1
            evaluated.append(sol)
            if self._real_evals >= self.budget:
                break
        X_new = np.array([s.variables for s in evaluated])
        self.PE.extend(evaluated)
        self._PE_X = np.vstack([self._PE_X, X_new])
        self.model.training.add(X_new, np.array([s.objectives for s in evaluated]))
        self.evaluations = self._real_evals
        return len(evaluated)

    # ------------------------------------------------------------------
    def _approximate(self, sols: List[FloatSolution]) -> None:
        F = self.model.predict(np.array([s.variables for s in sols]))
        for sol, f in zip(sols, F):
            sol.objectives = f.tolist()

    def _random_solution(self) -> FloatSolution:
        lb = np.asarray(self.problem.lower_bound)
        ub = np.asarray(self.problem.upper_bound)
        return self.problem.solutions_from([lb + np.random.random(len(lb)) * (ub - lb)])[0]

    def _inner_nsgaii(self) -> List[FloatSolution]:
        pop = copy.deepcopy(get_non_dominated_solutions(self.PE)[: self.population_size])
        while len(pop) < self.population_size:
            pop.append(self.fill_sampler())
        self._approximate(pop)

        for _ in range(self.inner_G):
            offspring = []
            for _ in range(self.offspring_population_size):
                parents = [self.selection_operator.execute(pop), self.selection_operator.execute(pop)]
                child = self.crossover_operator.execute(parents)[0]
                self.mutation_operator.execute(child)
                offspring.append(child)
            self._approximate(offspring)
            pop = self._nsga2_select(pop + offspring, self.population_size)
        return pop

    @staticmethod
    def _nsga2_select(combined: List[FloatSolution], size: int) -> List[FloatSolution]:
        ranking = FastNonDominatedRanking()
        ranking.compute_ranking(combined)
        result, fi = [], 0
        while fi < ranking.get_number_of_subfronts():
            front = ranking.get_subfront(fi)
            if len(result) + len(front) > size:
                CrowdingDistance().compute_density_estimator(front)
                front.sort(key=lambda s: s.attributes.get("crowding_distance", 0), reverse=True)
                result.extend(front[: size - len(result)])
                break
            result.extend(front)
            fi += 1
        return result

    def _select_candidates(self, pop: List[FloatSolution]) -> List[FloatSolution]:
        ranking = FastNonDominatedRanking()
        ranking.compute_ranking(pop)
        front0 = ranking.get_subfront(0)
        CrowdingDistance().compute_density_estimator(front0)
        front0.sort(key=lambda s: s.attributes.get("crowding_distance", 0), reverse=True)
        eps, selected = self.epsilon, []
        while not selected and eps < 10.0:
            for sol in front0:
                if self._is_novel(sol, eps):
                    selected.append(sol)
                if len(selected) >= self.n_select:
                    break
            if not selected:
                eps *= 2.0
        return selected[: self.n_select]

    def _is_novel(self, sol: FloatSolution, eps: float) -> bool:
        return bool(np.min(np.linalg.norm(self._PE_X - np.asarray(sol.variables), axis=1)) >= eps)

    def result(self) -> List[FloatSolution]:
        return get_non_dominated_solutions(self.PE)

    def get_name(self) -> str:
        return "ELMLoopNSGAII"
