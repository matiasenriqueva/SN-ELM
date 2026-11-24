"""Surrogados intercambiables y gestión común del archivo de entrenamiento.

Para que la ablación compare sólo lo que debe, todos los bucles (ELM-NSGA-II,
SSANSGA2, KRVEA) usan la misma política de datos:

* ``TrainingSet``: archivo P_T con tope ``N_TRAIN``; una vez lleno, cada nueva
  solución reemplaza a la más cercana en el espacio de objetivos (igual que
  ``SurrogateArchive`` en code/surrogate_ELM).
* ``SurrogateModel``: P_T + normalización (X a [-1, 1], Y estandarizado con las
  estadísticas de P_T) + un ``Surrogate`` cualquiera.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np


# ======================================================================
# Modelos (trabajan con X e Y ya normalizados)
# ======================================================================

class Surrogate(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """X:(N, n), Y:(N, M)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """X:(N, n) -> (N, M)."""


class ELM(Surrogate):
    """ELM regularizada de salida múltiple (Huang et al., 2012).

    ``sigmoid`` y ``multiquadric`` son idénticas a ``ELMSurrogate`` de
    code/surrogate_ELM/surrogate.py (mismo consumo de ``np.random``).

    ``gaussian`` corrige el original, que sorteaba el ancho con ``randn``: con
    ``b < 0`` el término ``exp(-d^2 b)`` crece sin control (H ~ 1e80 con n=30) y
    con ``b > 0`` casi todas las neuronas quedan en cero. Aquí el ancho es
    positivo y se escala con la dimensión, ``b_k ~ U(0.2, 2) / n``, de modo que
    ``d^2 b_k`` es de orden 1 para distancias típicas entre las entradas en
    [-1, 1]^n y los centros ~ N(0, I).
    """

    ACTIVATIONS = ("sigmoid", "gaussian", "multiquadric")

    def __init__(self, n_hidden: int, C: float = 1.0, activation: str = "sigmoid"):
        if activation not in self.ACTIVATIONS:
            raise ValueError(f"activación desconocida: {activation}")
        self.NH = n_hidden
        self.C = C
        self.act = activation
        self._W = self._b = self._beta = None

    def _H(self, X: np.ndarray) -> np.ndarray:
        if self.act == "sigmoid":
            return 1.0 / (1.0 + np.exp(-(X @ self._W + self._b)))
        d2 = np.sum((X[:, None, :] - self._W.T[None, :, :]) ** 2, axis=2)
        if self.act == "gaussian":
            return np.exp(-d2 * self._b)
        return np.sqrt(d2 + self._b ** 2)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        NT, ND = X.shape
        self._W = np.random.randn(ND, self.NH)
        if self.act == "gaussian":
            self._b = np.random.uniform(0.2, 2.0, self.NH) / ND
        else:
            self._b = np.random.randn(self.NH)
        H = self._H(X)
        if self.NH <= NT:
            A = np.eye(self.NH) / self.C + H.T @ H
            self._beta = np.linalg.solve(A, H.T @ Y)
        else:
            A = np.eye(NT) / self.C + H @ H.T
            self._beta = H.T @ np.linalg.solve(A, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._H(X) @ self._beta


class PerObjective(Surrogate):
    """Un modelo de pysurrogate por objetivo (copias del prototipo)."""

    def __init__(self, prototype):
        self.prototype = prototype
        self.models = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.models = [copy.deepcopy(self.prototype) for _ in range(Y.shape[1])]
        for m, model in enumerate(self.models):
            model.fit(X, Y[:, m])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([model.predict(X).y[:, 0] for model in self.models])


class ScheduledKriging(Surrogate):
    """Kriging por objetivo con θ reestimado cada ``theta_every`` ajustes.

    Estimar θ por máxima verosimilitud es ~98% del costo de un ajuste (≈8 s frente
    a ≈0.16 s con θ fijo, 600 puntos, n=30). Entre reestimaciones el modelo se
    reajusta a los datos actuales con el último θ estimado; con θ fijo en el óptimo
    las predicciones son idénticas a las del modelo optimizado. ``theta_every=1``
    reestima en cada ajuste.

    Ojo: ``Kriging.fit(optimize=False)`` usa ``self.theta`` (el valor inicial, 1.0),
    no el último estimado; por eso se copia el θ estimado a ``model.theta``.
    """

    def __init__(self, prototype, theta_every: int = 1):
        if getattr(prototype, "ARD", False):
            raise ValueError("ScheduledKriging asume θ isotrópico (ARD=False)")
        self.prototype = prototype
        self.theta_every = theta_every
        self.models = []
        self.thetas: list[float] = []
        self.n_fits = 0
        self.n_theta_fits = 0

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        optimize = self.n_fits % self.theta_every == 0 or len(self.thetas) != Y.shape[1]
        models = []
        for m in range(Y.shape[1]):
            model = copy.deepcopy(self.prototype)
            if not optimize:
                model.theta = self.thetas[m]
            model.fit(X, Y[:, m], optimize=optimize)
            models.append(model)
        if optimize:
            self.thetas = [float(np.ravel(model.model.model["theta"])[0]) for model in models]
            self.n_theta_fits += 1
        self.models = models
        self.n_fits += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([model.predict(X).y[:, 0] for model in self.models])


def make_kriging(theta_every: int = 1) -> ScheduledKriging:
    # El mismo Kriging que usa KRVEA en pysamoo (correlación exponencial):
    # el kernel por defecto de pysurrogate (RationalQuadratic) es ~10x más caro.
    from pysamoo.algorithms._ego import default_kriging
    return ScheduledKriging(default_kriging(), theta_every)


def make_surrogate(kind: str, n_var: int, params: dict | None = None, krg_theta_every: int = 1) -> Surrogate:
    params = params or {}
    if kind == "ELM":
        return ELM(params["nh_factor"] * n_var + 1, C=params["C"], activation=params["act"])
    if kind == "RBF":
        from pysurrogate.models import RBF
        return PerObjective(RBF())
    if kind == "KRG":
        return make_kriging(krg_theta_every)
    raise ValueError(f"surrogado desconocido: {kind}")


# ======================================================================
# Archivo de entrenamiento y normalización
# ======================================================================

class TrainingSet:
    """P_T con tope; reemplazo de la solución más cercana en objetivos."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.X = None
        self.F = None
        self.n_seen = 0        # cuántas soluciones del archivo completo ya se procesaron

    def add_initial(self, X: np.ndarray, F: np.ndarray) -> None:
        self.X = np.array(X[: self.capacity], dtype=float)
        self.F = np.array(F[: self.capacity], dtype=float)
        self.n_seen = len(X)
        for x, f in zip(X[self.capacity:], F[self.capacity:]):
            self._replace_closest(x, f)

    def add(self, X: np.ndarray, F: np.ndarray) -> None:
        for x, f in zip(X, F):
            if len(self.X) < self.capacity:
                self.X = np.vstack([self.X, x])
                self.F = np.vstack([self.F, f])
            else:
                self._replace_closest(x, f)
        self.n_seen += len(X)

    def sync(self, X_all: np.ndarray, F_all: np.ndarray) -> None:
        """Incorpora lo que haya de nuevo en un archivo que sólo crece."""
        if self.X is None:
            self.add_initial(X_all, F_all)
        elif len(X_all) > self.n_seen:
            self.add(X_all[self.n_seen:], F_all[self.n_seen:])

    def _replace_closest(self, x, f) -> None:
        idx = int(np.argmin(np.linalg.norm(self.F - f, axis=1)))
        self.X[idx] = x
        self.F[idx] = f


class SurrogateModel:
    """P_T + normalización + surrogado. Predice en la escala original."""

    def __init__(self, surrogate: Surrogate, lower, upper, capacity: int):
        self.surrogate = surrogate
        self.lb = np.asarray(lower, dtype=float)
        self.ub = np.asarray(upper, dtype=float)
        self.training = TrainingSet(capacity)
        self._mu = self._std = None

    def normalize_X(self, X: np.ndarray) -> np.ndarray:
        return 2.0 * (X - self.lb) / (self.ub - self.lb + 1e-8) - 1.0

    def fit(self) -> None:
        F = self.training.F
        self._mu = F.mean(axis=0)
        self._std = F.std(axis=0) + 1e-8
        self.surrogate.fit(self.normalize_X(self.training.X), (F - self._mu) / self._std)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.surrogate.predict(self.normalize_X(np.atleast_2d(X))) * self._std + self._mu
