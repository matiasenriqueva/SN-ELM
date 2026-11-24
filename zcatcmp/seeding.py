"""Semillas derivadas de forma determinista y diseño inicial (LHS) compartido."""
from __future__ import annotations

import hashlib
import random

import numpy as np
from scipy.stats import qmc

from .problems import bounds


def derive_seed(*parts) -> int:
    """Entero de 31 bits estable entre plataformas a partir de las partes dadas."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    """``random`` lo usan los operadores de jMetal; ``np.random`` el ELM."""
    random.seed(seed)
    np.random.seed(seed)


def lhs(n: int, n_var: int, seed: int) -> np.ndarray:
    lb, ub = bounds(n_var)
    unit = qmc.LatinHypercube(d=n_var, rng=np.random.default_rng(seed)).random(n)
    return qmc.scale(unit, lb, ub)


def shared_doe(problem_id: int, n_obj: int, seed: int, n: int, n_var: int) -> np.ndarray:
    """El mismo LHS para todos los algoritmos en (problema, M, semilla)."""
    return lhs(n, n_var, derive_seed("doe", problem_id, n_obj, seed))
