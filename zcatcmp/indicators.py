"""Indicadores en el espacio normalizado f_i / i^2 (el frente de ZCAT cumple f_i in [0, i^2]).

HV con punto de referencia 1.1 en cada objetivo; IGD+ y épsilon aditivo contra
los frentes de referencia de data/reference_fronts (copiados de code/resources/reference_fronts).
"""
from __future__ import annotations

from functools import lru_cache

import moocore
import numpy as np

from . import config


def scale(n_obj: int) -> np.ndarray:
    return np.arange(1, n_obj + 1, dtype=float) ** 2


@lru_cache(maxsize=None)
def reference_front(problem_id: int, n_obj: int) -> np.ndarray:
    path = config.REFERENCE_FRONTS / f"ZCAT{problem_id}.{n_obj}D.pf"
    return np.loadtxt(path) / scale(n_obj)


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    return moocore.is_nondominated(F, keep_weakly=False)


def front_indicators(F: np.ndarray, problem_id: int, n_obj: int) -> tuple[float, float, float]:
    """(HV, IGD+, EP) de un conjunto de puntos (se filtran los dominados)."""
    Fn = F[nondominated_mask(F)] / scale(n_obj)
    R = reference_front(problem_id, n_obj)
    hv = float(moocore.hypervolume(Fn, ref=np.full(n_obj, config.HV_REF)))
    return hv, float(moocore.igd_plus(Fn, ref=R)), float(moocore.epsilon_additive(Fn, ref=R))


def anytime_curve(F_all: np.ndarray, n: int, problem_id: int, n_obj: int, checkpoints) -> dict:
    """Indicadores del archivo no dominado tras las primeras k evaluaciones."""
    evals, hv, igdp, ep = [], [], [], []
    for k in checkpoints:
        if k > n:
            break
        h, i, e = front_indicators(F_all[:k], problem_id, n_obj)
        evals.append(int(k)); hv.append(h); igdp.append(i); ep.append(e)
    return {"curve_evals": evals, "curve_hv": hv, "curve_igdp": igdp, "curve_ep": ep}
