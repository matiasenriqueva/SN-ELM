"""Construcción y ejecución de cada variante a partir de un ``Job``.

Variantes
---------
ELM-<S>      bucle ELM-NSGA-II con surrogado S (ELM, RBF o KRG)
SSA-<S>      SSANSGA2 de pysamoo con surrogado S y operadores igualados
KRVEA          K-RVEA (pysamoo) con la población de RVEA continua entre actualizaciones,
               como en el original; grilla del bucle
KRVEA-pysamoo  K-RVEA de pysamoo tal cual (RVEA desde población aleatoria) con u=5, w_max=20
NSGAII       NSGA-II sin surrogado, mismos operadores que los NSGA-II internos
LHS          N_E puntos LHS (control de muestreo puro)

Todas las variantes con surrogado parten del mismo LHS inicial y entrenan sobre
el mismo tipo de archivo P_T con tope (``surrogates.TrainingSet``).
"""
from __future__ import annotations

import numpy as np
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.normalization import normalize
from pymoo.util.roulette import RouletteWheelSelection
import pysamoo.algorithms.krvea as _krvea_module
from pysamoo.algorithms.krvea import KRVEA
from pysamoo.algorithms.ssansga2 import SSANSGA2
from sklearn.cluster import KMeans

from . import config
from .elm_loop import ELMLoopNSGAII
from .problems import BudgetExhausted, LoggedZCAT, PymooZCAT
from .seeding import derive_seed, lhs
from .surrogates import ScheduledKriging, SurrogateModel, TrainingSet, make_surrogate


# ======================================================================
# Adaptadores para pysamoo
# ======================================================================

class _PredictedProblem(Problem):
    def __init__(self, model: SurrogateModel, n_obj: int):
        super().__init__(n_var=len(model.lb), n_obj=n_obj, xl=model.lb, xu=model.ub)
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.model.predict(X)


class FixedSurrogate:
    """Sustituye al ``Surrogate`` de pysamoo por un modelo fijo (sin selección)."""

    def __init__(self, model: SurrogateModel, n_obj: int):
        self.model = model
        self.n_obj = n_obj

    def validate(self, *args, **kwargs):
        pass

    def fit(self, archive: Population) -> None:
        X, F = archive.get("X", "F")
        self.model.training.sync(X, F)
        self.model.fit()

    def problem(self) -> Problem:
        return _PredictedProblem(self.model, self.n_obj)


class MatchedSSANSGA2(SSANSGA2):
    """SSANSGA2 con los mismos operadores que el bucle ELM-NSGA-II.

    ``_infill`` es el de pysamoo (commit f65993f) con dos cambios:

    * el NSGA2 interno recibe SBX(0.9, eta=20) y PM(2/n, eta=20) en vez de los
      valores por defecto de pymoo, para que la ablación no confunda bucle con
      operadores;
    * si tras eliminar duplicados no queda ningún candidato nuevo, se repite la
      búsqueda interna desde una población aleatoria. En pysamoo original una
      iteración sin candidatos no evalúa nada y el bucle no termina nunca. Tras
      ``MAX_EMPTY_ITERATIONS`` intentos vacíos se aborta la corrida con error.
    """

    inner_retries = 0

    def _infill(self):
        self.surrogate.fit(self._archive)
        problem = self.surrogate.problem()
        n = problem.n_var
        for attempt in range(config.MAX_EMPTY_ITERATIONS):
            use_archive = self.surr_sampling == "current" and attempt == 0
            sampling = self._archive if use_archive else FloatRandomSampling()
            algorithm = NSGA2(
                pop_size=self.surr_pop_size,
                sampling=sampling,
                crossover=SBX(prob=config.SBX_PROB, eta=config.SBX_ETA),
                mutation=PM(prob=1.0, prob_var=config.PM_FACTOR / n, eta=config.PM_ETA),
            )
            inner_seed = int(self.random_state.integers(1, 2**31 - 1))
            res = minimize(problem, algorithm, ("n_gen", self.surr_n_gen), seed=inner_seed, verbose=False)
            cand = DefaultDuplicateElimination(epsilon=self.surr_eps_elim).do(res.pop, self._archive)
            if len(cand) > 0:
                break
            self.inner_retries += 1
        else:
            raise RuntimeError(f"SSANSGA2: {config.MAX_EMPTY_ITERATIONS} búsquedas internas sin candidatos nuevos")

        if len(cand) <= self.n_infills:
            return Population.new(X=cand.get("X"))

        ideal = res.opt.get("F").min(axis=0)
        nadir = res.opt.get("F").max(axis=0) + 1e-16
        vals = normalize(cand.get("F"), ideal, nadir)
        kmeans_seed = int(self.random_state.integers(1, 2**31 - 1))
        kmeans = KMeans(n_clusters=self.n_infills, random_state=kmeans_seed).fit(vals)
        groups = [[] for _ in range(self.n_infills)]
        for k, i in enumerate(kmeans.labels_):
            groups[i].append(k)
        S = []
        for group in groups:
            if len(group) > 0:
                rank = cand[group].get("crowding").argsort().argsort().astype(float)
                selection = RouletteWheelSelection(rank, larger_is_better=True)
                S.append(group[selection.next(random_state=self.random_state)])
        return Population.new(X=cand[S].get("X"))


class CappedKRVEA(KRVEA):
    """K-RVEA de pysamoo con cambios en la gestión del modelo.

    Comunes a todas las variantes Kriging: entrena sobre P_T (tope N_TRAIN) en vez de
    todo el archivo, y reestima θ cada ``theta_every`` ajustes (``ScheduledKriging``).

    ``warm_start``: pysamoo crea en cada actualización un RVEA nuevo con población
    aleatoria y lo evoluciona sólo ``w_max`` generaciones sobre el surrogado; en K-RVEA
    original (Chugh et al., 2018) la población de RVEA continúa entre actualizaciones.
    Con ``warm_start=True`` el primer RVEA parte de P_T y los siguientes de la población
    final del anterior. La selección de candidatos es la de pysamoo, sin cambios.
    """

    def __init__(self, n_train: int, theta_every: int = 1, warm_start: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.training = TrainingSet(n_train)
        self.kriging = ScheduledKriging(self.surrogate_proto, theta_every)
        self.warm_start = warm_start
        self._population = None

    def _fit_models(self, proto, X, F):
        self.kriging.fit(X, F)
        return self.kriging.models

    def _make_rvea(self, ref_dirs):
        start = self._population if self._population is not None else self.training.X
        return RVEA(ref_dirs=ref_dirs, sampling=start.copy())

    def _minimize_and_keep(self, *args, **kwargs):
        res = minimize(*args, **kwargs)
        self._population = res.pop.get("X").copy()
        return res

    def _infill(self):
        full = self._archive
        X, F = full.get("X", "F")
        self.training.sync(X, F)
        self._archive = Population.new(X=self.training.X.copy(), F=self.training.F.copy())
        # KRVEA._infill usa fit_per_objective, RVEA y pymoo_minimize del módulo krvea: se
        # sustituyen sólo durante esta llamada (cada proceso ejecuta una corrida a la vez)
        patches = {"fit_per_objective": self._fit_models}
        if self.warm_start:
            patches.update(RVEA=self._make_rvea, pymoo_minimize=self._minimize_and_keep)
        originals = {name: getattr(_krvea_module, name) for name in patches}
        for name, value in patches.items():
            setattr(_krvea_module, name, value)
        try:
            return super()._infill()
        finally:
            for name, value in originals.items():
                setattr(_krvea_module, name, value)
            self._archive = full


# ======================================================================
# Ejecución
# ======================================================================

def run_variant(job: dict, zcat: LoggedZCAT, X_doe: np.ndarray, seed: int) -> dict:
    """Ejecuta la variante del job; termina por presupuesto (``BudgetExhausted``)."""
    variant = job["variant"]
    M, n_var, budget = job["M"], job["n_var"], job["budget"]
    loop, params = job.get("loop") or {}, job.get("model")
    theta_every = job.get("krg_theta_every", 1)

    if variant == "NSGAII":
        algo = NSGAII(
            problem=zcat,
            population_size=config.POP_SIZE,
            offspring_population_size=config.POP_SIZE,
            mutation=PolynomialMutation(probability=config.PM_FACTOR / n_var, distribution_index=config.PM_ETA),
            crossover=SBXCrossover(probability=config.SBX_PROB, distribution_index=config.SBX_ETA),
            termination_criterion=StoppingByEvaluations(budget),
        )
        algo.run()
        return {}

    if variant == "LHS":
        for x in lhs(budget, n_var, derive_seed("lhs", job["problem"], M, job["seed"])):
            zcat.evaluate_x(x)
        return {}

    def surrogate_model(kind):
        surrogate = make_surrogate(kind, n_var, params, krg_theta_every=theta_every)
        return SurrogateModel(surrogate, zcat.lower_bound, zcat.upper_bound, job["n_train"])

    loop_kind, _, model_kind = variant.partition("-")

    if loop_kind == "ELM":
        model = surrogate_model(model_kind)
        algo = ELMLoopNSGAII(zcat, model, X_doe, budget, n_select=loop["NS"], inner_generations=loop["G"])
        algo.run()
        return {"empty_iterations": algo.empty_iterations, **_kriging_info(model.surrogate)}

    pymoo_problem = PymooZCAT(zcat)
    common = dict(n_initial_doe=len(X_doe), sampling=X_doe)

    if loop_kind == "SSA":
        model = surrogate_model(model_kind)
        algo = MatchedSSANSGA2(
            n_infills=loop["NS"], surr_n_gen=loop["G"], surr_pop_size=config.POP_SIZE,
            surrogate=FixedSurrogate(model, M), **common,
        )
    elif variant in ("KRVEA", "KRVEA-pysamoo"):
        algo = CappedKRVEA(job["n_train"], theta_every=theta_every, warm_start=variant == "KRVEA",
                           n_infills=loop["NS"], w_max=loop["G"], **common)
    else:
        raise ValueError(f"variante desconocida: {variant}")

    # copy_algorithm=False: los contadores del algoritmo quedan en ``algo`` aunque la
    # corrida termine con BudgetExhausted (un lote de puntos que cruza el presupuesto)
    try:
        minimize(pymoo_problem, algo, ("n_evals", budget), seed=seed, verbose=False, copy_algorithm=False)
    except BudgetExhausted:
        pass
    if loop_kind == "SSA":
        return {"inner_retries": algo.inner_retries, **_kriging_info(model.surrogate)}
    return _kriging_info(algo.kriging)


def _kriging_info(surrogate) -> dict:
    if not isinstance(surrogate, ScheduledKriging):
        return {}
    return {"krg_theta_every": surrogate.theta_every, "krg_fits": surrogate.n_fits,
            "krg_theta_fits": surrogate.n_theta_fits}
