"""Cada variante corre de punta a punta con un presupuesto pequeño, exacto y reproducible."""
import numpy as np
import pytest

from zcatcmp import config
from zcatcmp.runner import run_job
from zcatcmp.stages import config_id

LOOP = dict(G=2, NS=5)
CASES = [
    ("ELM-ELM", LOOP, config.ELM_DEFAULT),
    ("ELM-ELM", LOOP, dict(config.ELM_DEFAULT, act="gaussian")),
    ("ELM-RBF", LOOP, None),
    ("ELM-KRG", LOOP, None),
    ("SSA-ELM", LOOP, config.ELM_DEFAULT),
    ("SSA-RBF", LOOP, None),
    ("SSA-KRG", LOOP, None),
    ("KRVEA", LOOP, None),
    ("KRVEA-pysamoo", LOOP, None),
    ("NSGAII", None, None),
    ("LHS", None, None),
]


def small_job(variant, loop, model, M=2, problem=4, seed=0, budget=80, n_doe=40):
    cid = config_id(loop, model)
    return {
        "key": f"test|{variant}|{cid}|ZCAT{problem}|M{M}|s{seed}", "stage": "test",
        "variant": variant, "config_id": cid, "loop": loop, "model": model,
        "problem": problem, "M": M, "seed": seed, "n_var": 30,
        "budget": budget, "n_doe": n_doe, "n_train": n_doe,
        "checkpoints": (40, 60, 80), "store_X": True,
    }


@pytest.mark.parametrize("M", [2, 3])
@pytest.mark.parametrize("variant,loop,model", CASES)
def test_variant_runs_exact_budget(variant, loop, model, M):
    row = run_job(small_job(variant, loop, model, M=M))
    assert row["status"] == "ok", row["error"]
    assert row["n_evals"] == 80
    assert 0 <= row["hv"] <= config.HV_REF ** M
    assert np.isfinite([row["igdp"], row["ep"]]).all()
    assert row["curve_evals"] == [40, 60, 80]
    assert row["curve_hv"] == sorted(row["curve_hv"])          # el archivo nunca empeora
    assert len(row["front_F"]) == len(row["front_X"]) >= 1


@pytest.mark.parametrize("variant,loop,model", [CASES[0], CASES[4], CASES[7]])
def test_reproducible(variant, loop, model):
    a = run_job(small_job(variant, loop, model, problem=9, M=3))
    b = run_job(small_job(variant, loop, model, problem=9, M=3))
    assert a["front_F"] == b["front_F"] and a["hv"] == b["hv"]


def test_surrogate_variants_share_initial_design():
    """Las primeras n_doe evaluaciones son el mismo LHS en todas las variantes con surrogado."""
    from zcatcmp.indicators import anytime_curve  # noqa: F401  (import de humo)

    rows = [run_job(small_job(v, l, m, problem=2)) for v, l, m in (CASES[0], CASES[5], CASES[7])]
    assert len({r["curve_hv"][0] for r in rows}) == 1


def test_error_is_reported_not_raised():
    row = run_job(small_job("NO-EXISTE", LOOP, None))
    assert row["status"] == "error" and "variante desconocida" in row["error"]


@pytest.mark.parametrize("variant", ["ELM-KRG", "SSA-KRG", "KRVEA", "KRVEA-pysamoo"])
def test_kriging_theta_schedule_is_applied(variant):
    import json

    job = dict(small_job(variant, dict(G=2, NS=5), None, budget=100), krg_theta_every=3)
    info = json.loads(run_job(job)["info"])
    assert info["krg_theta_every"] == 3
    assert info["krg_fits"] >= 4 and info["krg_theta_fits"] == -(-info["krg_fits"] // 3)


def test_krvea_warm_start_reuses_population(monkeypatch):
    """KRVEA arranca cada RVEA desde la población anterior; KRVEA-pysamoo desde una aleatoria."""
    from zcatcmp import algorithms

    starts = []
    real_make = algorithms.CappedKRVEA._make_rvea

    def spy(self, ref_dirs):
        rvea = real_make(self, ref_dirs)
        starts.append(np.array(rvea.initialization.sampling))
        return rvea

    monkeypatch.setattr(algorithms.CappedKRVEA, "_make_rvea", spy)
    warm = run_job(small_job("KRVEA", LOOP, None, budget=100))
    assert warm["status"] == "ok" and len(starts) >= 3
    assert len(starts[0]) == 40                     # primer RVEA: P_T (n_train=40 en el test)
    assert len(starts[1]) != 40                     # siguientes: población final del RVEA anterior
    n_calls = len(starts)
    cold = run_job(small_job("KRVEA-pysamoo", LOOP, None, budget=100))
    assert len(starts) == n_calls                   # pysamoo tal cual no usa _make_rvea
    assert cold["front_F"] != warm["front_F"]
