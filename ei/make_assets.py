"""Figuras, tablas y material suplementario de la versión para Evolutionary Intelligence.

    python ei/make_assets.py            # desde comparison/, tras correr las etapas A, B y C

Rehace el análisis de las etapas B y C sin las variantes descartadas (results/analysis/B_ei/ y
C_ei/), deja las figuras y el material suplementario en ei/paper/ con los nombres que pide la
revista, y los datos de las tablas del manuscrito en ei/tables/.

Reglas de artwork de la revista (tamaño grande): 84 mm a una columna o 174 mm a página
completa, Arial de 8 a 12 pt, sin títulos dentro de la figura.
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analyze  # noqa: E402
from zcatcmp import config, plots  # noqa: E402
from zcatcmp.storage import load_results  # noqa: E402

EXCLUDED = ("LHS", "KRVEA-pysamoo", "SSA-RBF")
METRICS = ["hv", "igdp", "ep"]
METRIC_NAMES = {"hv": "HV", "igdp": "IGD+", "ep": "EP"}
NAMES = {"ELM-ELM": "SN-ELM", "ELM-RBF": "SN-RBF", "ELM-KRG": "SN-KRG", "NSGAII": "NSGA-II"}

# Título de trabajo: se fija al reescribir el manuscrito
TITLE = "SN-ELM: an extreme learning machine-assisted NSGA-II for expensive multi-objective optimization"
JOURNAL = "Evolutionary Intelligence"
AUTHORS = "M. Vergara, Y. Salgueiro"
CONTACT = "Universidad de Talca, Chile. Corresponding author: mavergara19@utalca.cl"

ANALYSIS = config.RESULTS_DIR / "analysis"
C, B, A = ANALYSIS / "C_ei", ANALYSIS / "B_ei", ANALYSIS / "A"
PAPER, TABLES = ROOT / "ei" / "paper", ROOT / "ei" / "tables"
DIAGRAM = ANALYSIS / "sn_elm_diagram.pdf"
FIGURES = {"Fig1": DIAGRAM, "Fig2": B / "loop_effect_G_hv.pdf", "Fig3": C / "cd_hv.pdf", "Fig4": C / "ablation_hv.pdf",
           "Fig5": C / "cost_quality.pdf", "Fig6": C / "anytime_hv.pdf", "Fig7": C / "fronts_M2.pdf",
           "Fig8": C / "fronts_M3.pdf", "ESM_1": C / "fronts_M3_all.pdf"}


def analyse() -> None:
    plots.FULL_WIDTH = 174 * plots.MM
    analyze.ONLINE_RESOURCE_1 = (
        f"Online Resource 1\n\n{TITLE}\n\n{JOURNAL}\n{AUTHORS}\n{CONTACT}\n\n"
        "Non-dominated set of the median-HV run on the 20 ZCAT problems with three objectives.\n"
        "Small dots: reference front. Points beyond 1.25 times the extent of the front are omitted.")
    for stage, out, run in (("C", C, lambda df, out: analyze.comparison(df, out, "ELM-ELM", METRICS)),
                            ("B", B, lambda df, out: analyze.loop(df, out, METRICS))):
        df = load_results(config.RESULTS_DIR / stage, columns=analyze.COLUMNS)
        out.mkdir(parents=True, exist_ok=True)
        run(df[~df["variant"].isin(EXCLUDED)], out)
    plots.sn_elm_diagram(DIAGRAM)


def tables() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for M in (2, 3):
        cpu = pd.read_csv(C / f"cost_quality_M{M}.csv", index_col="algo")["cpu_mean_s"]
        for metric in METRICS:
            ranks = pd.read_csv(C / f"friedman_ranks_{metric}_M{M}.csv", index_col="algo")["avg_rank"]
            rows += [dict(M=M, algo=a, column=METRIC_NAMES[metric], value=r) for a, r in ranks.items()]
        rows += [dict(M=M, algo=a, column="CPU per run (s)", value=v) for a, v in cpu.items()]
    ranks = pd.DataFrame(rows).pivot_table(index="algo", columns=["column", "M"], values="value")
    ranks = ranks.sort_values(("HV", 2)).rename(index=NAMES)
    ranks.to_csv(TABLES / "ranks_cpu.csv")

    wtl = pd.concat([pd.read_csv(C / f"wtl_{m}.csv").assign(indicator=METRIC_NAMES[m]) for m in METRICS])
    wtl["algo"] = wtl["algo"].replace(NAMES)
    wtl.to_csv(TABLES / "wtl.csv", index=False)

    posthoc = pd.concat([pd.read_csv(C / f"friedman_posthoc_{m}_M{M}.csv").assign(indicator=METRIC_NAMES[m], M=M)
                         for m in METRICS for M in (2, 3)])
    posthoc["algo"] = posthoc["algo"].replace(NAMES)
    posthoc.to_csv(TABLES / "posthoc_vs_sn_elm.csv", index=False)

    selection = json.loads((config.RESULTS_DIR / "selection_B.json").read_text())
    loop = [dict(M=int(M), variant=NAMES.get(v, v), G=c["loop"]["G"], NS=c["loop"]["NS"], model=c["model"])
            for M, variants in selection.items() for v, c in variants.items() if v not in EXCLUDED]
    pd.DataFrame(loop).to_csv(TABLES / "selection_B.csv", index=False)
    shutil.copyfile(A / "factor_effects.csv", TABLES / "factor_effects_A.csv")
    for name in ("friedman_stats.csv", "coverage_reference_front.csv", "evals_to_reach_nsgaii.csv"):
        shutil.copyfile(C / name, TABLES / name)


def per_problem_table() -> pd.DataFrame:
    frames = []
    for metric in METRICS:
        summary = pd.read_csv(C / f"summary_{metric}.csv")
        pairs = pd.read_csv(C / f"pairwise_{metric}.csv")
        merged = summary.merge(pairs, on=["M", "problem", "algo"], how="left")
        merged.insert(0, "indicator", METRIC_NAMES[metric])
        frames.append(merged)
    table = pd.concat(frames).rename(columns={"algo": "algorithm", "outcome": "SN-ELM vs algorithm"})
    table["algorithm"] = table["algorithm"].replace(NAMES)
    return table.drop(columns=["p"])


def paper_files() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    for name, source in FIGURES.items():
        shutil.copyfile(source, PAPER / f"{name}.pdf")
    header = (f"# Online Resource 2. {TITLE}. {JOURNAL}. {AUTHORS}. {CONTACT}\n"
              "# Per-problem results of the final comparison: median and quartiles over 30 runs, and comparison of "
              "SN-ELM against each algorithm (two-sided Mann-Whitney U, Holm-corrected p, A12; "
              "+ better, = no difference, - worse).\n")
    with open(PAPER / "ESM_2.csv", "w", encoding="utf-8", newline="") as handle:
        handle.write(header)
        per_problem_table().to_csv(handle, index=False, float_format="%.6g")


def main() -> None:
    analyse()
    tables()
    paper_files()
    print(f"\nfiguras y material suplementario en {PAPER}\ntablas en {TABLES}")


if __name__ == "__main__":
    main()
