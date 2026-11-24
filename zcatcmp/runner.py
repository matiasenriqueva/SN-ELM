"""Ejecución de un trabajo en un proceso: no escribe en disco, devuelve una fila."""
from __future__ import annotations

import json
import signal
import socket
import time
import traceback
from datetime import datetime, timezone

import numpy as np

from . import config
from .algorithms import run_variant
from .indicators import anytime_curve, front_indicators, nondominated_mask
from .problems import BudgetExhausted, EvaluationLog, LoggedZCAT
from .seeding import derive_seed, seed_everything, shared_doe


class JobTimeout(Exception):
    """La corrida superó ``max_seconds``."""


def _on_alarm(signum, frame):
    raise JobTimeout


def run_job(job: dict) -> dict:
    p, M, s = job["problem"], job["M"], job["seed"]
    n_var, budget = job["n_var"], job["budget"]

    # Misma semilla para todas las variantes en (problema, M, semilla): números
    # aleatorios comunes, que reducen la varianza de las comparaciones pareadas.
    seed = derive_seed("run", p, M, s)
    seed_everything(seed)
    log = EvaluationLog(budget, n_var, M)
    zcat = LoggedZCAT(p, M, log, n_var)
    X_doe = shared_doe(p, M, s, job["n_doe"], n_var)

    status, error, info = "ok", None, {}
    max_seconds = job.get("max_seconds")
    use_alarm = bool(max_seconds) and hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(max_seconds))
    cpu0, wall0 = time.process_time(), time.perf_counter()
    try:
        info = run_variant(job, zcat, X_doe, seed)
    except BudgetExhausted:
        pass
    except JobTimeout:
        status, error = "timeout", f"superó el máximo de {max_seconds} s con {log.n} evaluaciones"
    except Exception:
        status, error = "error", traceback.format_exc(limit=30)
    finally:
        if use_alarm:
            signal.alarm(0)
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0

    n = log.n
    if status == "ok" and n < budget:
        status = "incomplete"

    row = {
        **{k: job[k] for k in ("key", "stage", "variant", "config_id", "problem", "M", "seed")},
        "config": json.dumps({"loop": job.get("loop"), "model": job.get("model")}, sort_keys=True),
        "status": status, "error": error, "n_evals": n,
        "cpu_time": cpu, "wall_time": wall,
        "hv": None, "igdp": None, "ep": None,
        "curve_evals": [], "curve_hv": [], "curve_igdp": [], "curve_ep": [],
        "front_F": None, "front_X": None,
        "info": json.dumps(info or {}),
        "host": socket.gethostname(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": job.get("code_version", "unknown"),
    }
    if n > 0:
        F, X = log.F[:n], log.X[:n]
        row["hv"], row["igdp"], row["ep"] = front_indicators(F, p, M)
        row.update(anytime_curve(F, n, p, M, job.get("checkpoints", config.CHECKPOINTS)))
        mask = nondominated_mask(F)
        row["front_F"] = F[mask].tolist()
        if job.get("store_X"):
            row["front_X"] = X[mask].tolist()
    return row


def quiet_worker() -> None:
    """Inicializa cada proceso de cálculo.

    Ctrl+C en la terminal llega a todo el grupo de procesos: los de cálculo lo
    ignoran y es el principal quien los detiene (guardando antes lo terminado).
    SIGTERM vuelve al comportamiento por defecto (terminar), por si se heredó el
    manejador del proceso principal.
    """
    import logging
    import warnings

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    logging.getLogger("jmetal").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(all="ignore")
