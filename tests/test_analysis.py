import json

import numpy as np
import pandas as pd
import pytest

from zcatcmp import analysis as an

# calidad de base (HV) y costo de cada variante en los datos sintéticos
VARIANTS = {"ELM-KRG": (0.80, 2000), "SSA-KRG": (0.79, 1900), "KRVEA": (0.76, 1800), "ELM-ELM": (0.70, 40),
            "SSA-ELM": (0.70, 110), "NSGAII": (0.66, 0.3), "ELM-RBF": (0.64, 250), "SSA-RBF": (0.63, 300),
            "KRVEA-pysamoo": (0.45, 1700), "LHS": (0.05, 0.1)}


def synthetic(seeds=10, problems=20, noise=0.01, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    evals = list(range(100, 2001, 100))
    rows = []
    for M in (2, 3):
        for p in range(1, problems + 1):
            for variant, (base, cpu) in VARIANTS.items():
                for s in range(seeds):
                    hv = base * (1 - 0.1 * (M - 2)) + rng.normal(0, noise)
                    curve = [hv * min(1.0, (e / 2000) ** 0.5) for e in evals]
                    rows.append(dict(variant=variant, config_id="cfg", problem=p, M=M, seed=s, status="ok",
                                     hv=hv, igdp=1 - hv, ep=1 - hv, cpu_time=cpu * (1 + 0.1 * rng.random()),
                                     curve_evals=evals, curve_hv=curve, curve_igdp=[1 - c for c in curve],
                                     curve_ep=[1 - c for c in curve], config=json.dumps({"loop": None, "model": None})))
    return pd.DataFrame(rows)


def test_holm_matches_reference():
    np.testing.assert_allclose(an.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])


def test_a12():
    assert an.a12([3, 4], [1, 2]) == 1.0 and an.a12([1, 2], [3, 4]) == 0.0 and an.a12([1], [1]) == 0.5


def test_pairwise_detects_known_differences():
    df = an.prepare(synthetic())
    pairs = an.pairwise(df, "hv", "ELM-ELM")
    wtl = an.win_tie_loss(pairs).set_index(["M", "algo"])
    assert wtl.loc[(2, "NSGAII"), "+"] == 20          # ELM-ELM (0.70) > NSGA-II (0.66) en todo
    assert wtl.loc[(2, "ELM-KRG"), "-"] == 20         # Kriging mejor
    assert wtl.loc[(2, "SSA-ELM"), "="] >= 17         # misma distribución que el control: casi todo empate
    # IGD+ va en sentido contrario y debe dar lo mismo
    wtl_igdp = an.win_tie_loss(an.pairwise(df, "igdp", "ELM-ELM")).set_index(["M", "algo"])
    assert wtl_igdp.loc[(2, "NSGAII"), "+"] == 20


def test_friedman_ranks_and_ablation():
    df = an.prepare(synthetic())
    res = {r["M"]: r for r in an.friedman(df, "hv", "ELM-ELM")}
    ranks = res[2]["avg_ranks"]
    assert ranks.index[0] == "ELM-KRG" and ranks.index[-1] == "LHS"
    assert res[2]["p"] < 1e-10 and res[2]["cd"] > 0
    table = an.ablation_ranks(ranks)
    assert table.loc["ELM", "KRG"] < table.loc["ELM", "ELM"] < table.loc["ELM", "RBF"]
    posthoc = res[2]["posthoc"].set_index("algo")
    assert posthoc.loc["LHS", "p_holm"] < 0.05


def test_anytime_and_reach():
    df = an.prepare(synthetic())
    curves = an.anytime(df, "hv")
    final = curves[curves["evals"] == 2000].set_index(["M", "algo"])["relative"]
    assert final.loc[(2, "ELM-KRG")] == pytest.approx(1.0, abs=0.02)
    reach = an.evals_to_reach(df, "NSGAII").set_index(["M", "algo"])
    assert reach.loc[(2, "ELM-KRG"), "median_evals"] < 2000
    assert reach.loc[(2, "LHS"), "reached"] == 0


def test_analyze_writes_outputs(tmp_path, monkeypatch):
    import analyze
    from zcatcmp import config

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(analyze, "load_results", lambda *a, **k: synthetic(seeds=5))
    monkeypatch.setattr("sys.argv", ["analyze.py", "--stage", "C"])
    analyze.main()
    out = tmp_path / "analysis" / "C"
    for name in ("wtl_hv.csv", "wtl_hv.tex", "cd_hv.pdf", "ablation_hv.pdf", "cost_quality.pdf",
                 "anytime_hv.pdf", "evals_to_reach_nsgaii.csv", "friedman_stats.csv"):
        assert (out / name).exists(), name


def test_factor_effects_rank_the_dominant_factor_first(tmp_path, monkeypatch):
    import analyze
    from zcatcmp import config
    from zcatcmp.stages import config_id, grid

    rng = np.random.default_rng(1)
    effect = {"act": {"sigmoid": 0.0, "gaussian": -0.05, "multiquadric": 0.05}, "nh_factor": {2: 0.0, 4: 0.002, 8: 0.004}}
    rows = []
    for model in grid(config.ELM_GRID):
        loop = config.LOOP_FIXED_A
        for M in (2, 3):
            for p in range(1, 21):
                for s in range(5):
                    hv = 0.6 + effect["act"][model["act"]] + effect["nh_factor"][model["nh_factor"]] + rng.normal(0, 0.003)
                    rows.append(dict(variant="ELM-ELM", config_id=config_id(loop, model), problem=p, M=M, seed=s,
                                     status="ok", hv=hv, igdp=1 - hv, ep=1 - hv,
                                     config=json.dumps({"loop": loop, "model": model})))
    df = pd.DataFrame(rows)
    levels, effects = an.factor_effects(df, "hv")
    by_factor = effects[effects["M"] == 2].set_index("factor")["effect"]
    assert by_factor.idxmax() == "act" and by_factor["act"] > 3 * by_factor["C"]
    best_act = levels[(levels["M"] == 2) & (levels["factor"] == "act")].sort_values("avg_rank").iloc[0]["level"]
    assert best_act == "multiquadric"

    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(analyze, "load_results", lambda *a, **k: df)
    monkeypatch.setattr("sys.argv", ["analyze.py", "--stage", "A"])
    analyze.main()
    assert (tmp_path / "analysis" / "A" / "factor_effects.tex").exists()


def test_loop_effects_relative_ranks():
    rows = []
    for variant, best_G in (("ELM-ELM", 5), ("ELM-KRG", 20)):
        for G in (5, 10, 20):
            for NS in (5, 10):
                loop = dict(G=G, NS=NS)
                for p in range(1, 11):
                    for s in range(3):
                        hv = 0.6 - 0.01 * abs(G - best_G) / 5 + 0.002 * (NS == 10)
                        rows.append(dict(variant=variant, config_id=f"G{G}-NS{NS}", problem=p, M=2, seed=s,
                                         status="ok", hv=hv, config=json.dumps({"loop": loop, "model": None})))
    eff = an.loop_effects(pd.DataFrame(rows), "hv").set_index(["variant", "factor", "level"])["relative_rank"]
    assert eff.between(0, 1).all()
    assert eff[("ELM-ELM", "G", 5)] < eff[("ELM-ELM", "G", 20)]
    assert eff[("ELM-KRG", "G", 20)] < eff[("ELM-KRG", "G", 5)]
    assert eff[("ELM-ELM", "NS", 10)] < eff[("ELM-ELM", "NS", 5)]


def test_reference_coverage_distinguishes_full_from_patchy_fronts():
    from zcatcmp.indicators import reference_front, scale

    R = reference_front(1, 3) * scale(3)
    full, patch = R[::5], R[R[:, 0] < 0.2][::5]
    rows = [dict(M=3, problem=1, algo=algo, front_F=F.tolist()) for algo, F in (("full", full), ("patch", patch))]
    cov = an.reference_coverage(pd.DataFrame(rows), radii=(0.1,)).set_index("algo")["coverage"]
    assert cov["full"] > 0.95 and cov["patch"] < 0.5
