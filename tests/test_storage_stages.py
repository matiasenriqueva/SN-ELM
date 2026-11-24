import json

import pytest

from zcatcmp import config, stages
from zcatcmp.runner import run_job
from zcatcmp.selection import select
from zcatcmp.storage import ResultWriter, completed_keys, load_results
from tests.test_runner import small_job


def test_writer_roundtrip_and_resume(tmp_path):
    rows = [run_job(small_job("LHS", None, None, seed=s)) for s in range(3)]
    rows[1] = dict(rows[1], status="error")
    writer = ResultWriter(tmp_path, flush_every=2)
    for r in rows:
        writer.add(r)
    writer.flush()
    assert len(list(tmp_path.glob("part-*.parquet"))) == 2
    assert not list(tmp_path.glob(".*.tmp"))
    df = load_results(tmp_path)
    assert len(df) == 3 and df["hv"].notna().all()
    assert completed_keys(tmp_path) == {rows[0]["key"], rows[2]["key"]}


def test_stage_sizes_and_unique_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    pilot = stages.build_jobs("pilot")
    A = stages.build_jobs("A")
    assert len(A) == 27 * 20 * 2 * 10
    assert len({j["key"] for j in A}) == len(A)
    assert {j["problem"] for j in pilot} == set(config.PILOT_PROBLEMS)
    with pytest.raises(FileNotFoundError):
        stages.build_jobs("B")

    models = [config.ELM_DEFAULT, dict(config.ELM_DEFAULT, act="sigmoid")]
    (tmp_path / "selection_A.json").write_text(json.dumps({"2": models, "3": models}))
    B = stages.build_jobs("B")
    cheap_per_M = 2 * 6 + 6 + 2 * 6             # ELM-ELM (2 modelos) + SSA-ELM + ELM-RBF + SSA-RBF
    kriging_per_M = 3 * 6                       # ELM-KRG + SSA-KRG + KRVEA
    assert len(B) == (cheap_per_M * len(config.SEEDS_TUNING) + kriging_per_M * len(config.SEEDS_TUNING_KRIGING)) * 20 * 2
    assert {j["seed"] for j in B if j["variant"] == "KRVEA"} == set(config.SEEDS_TUNING_KRIGING)
    assert {j["seed"] for j in B if j["variant"] == "ELM-RBF"} == set(config.SEEDS_TUNING)
    assert len({j["key"] for j in B}) == len(B)
    assert B[0]["variant"] in ("ELM-KRG", "SSA-KRG", "KRVEA")    # lo caro primero


def test_selection_picks_best_rank():
    import pandas as pd

    rows = []
    for cid, base in (("good", 1.0), ("bad", 0.5)):
        for p in (1, 2, 3):
            for s in (0, 1):
                rows.append(dict(M=2, variant="ELM-ELM", config_id=cid, problem=p, seed=s, status="ok",
                                 hv=base + 0.01 * s,
                                 config=json.dumps({"loop": None, "model": {"tag": cid}})))
    chosen = select(pd.DataFrame(rows), "A", top_k=1)
    assert chosen == {"2": [{"tag": "good"}]}
