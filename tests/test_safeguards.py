"""Salvaguardas contra corridas que no terminan."""
import signal

import pytest
from pymoo.core.population import Population

from tests.test_runner import small_job
from zcatcmp import algorithms, config
from zcatcmp.runner import run_job

LOOP = dict(G=2, NS=5)


@pytest.fixture
def empty_candidates(monkeypatch):
    """La eliminación de duplicados devuelve 'sin candidatos' en las primeras ``n`` llamadas."""
    real = algorithms.DefaultDuplicateElimination

    def install(n):
        calls = {"n": 0}

        class Patched:
            def __init__(self, epsilon):
                self.real = real(epsilon=epsilon)

            def do(self, pop, *others):
                calls["n"] += 1
                return Population() if calls["n"] <= n else self.real.do(pop, *others)

        monkeypatch.setattr(algorithms, "DefaultDuplicateElimination", Patched)
    return install


def test_ssa_retries_when_no_new_candidates(empty_candidates):
    empty_candidates(3)
    row = run_job(small_job("SSA-RBF", LOOP, None))
    assert row["status"] == "ok" and row["n_evals"] == 80
    assert '"inner_retries": 3' in row["info"]


def test_ssa_aborts_after_too_many_empty_searches(empty_candidates, monkeypatch):
    monkeypatch.setattr(config, "MAX_EMPTY_ITERATIONS", 4)
    empty_candidates(10 ** 6)
    row = run_job(small_job("SSA-RBF", LOOP, None))
    assert row["status"] == "error" and "sin candidatos nuevos" in row["error"]
    assert row["n_evals"] == 40          # sólo el diseño inicial


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM sólo existe en Linux/Unix")
def test_job_timeout():
    job = dict(small_job("ELM-ELM", dict(G=20, NS=5), config.ELM_DEFAULT, budget=2000, n_doe=600),
               max_seconds=2)
    row = run_job(job)
    assert row["status"] == "timeout" and row["n_evals"] < 2000
    assert signal.alarm(0) == 0          # la alarma quedó desactivada
