"""Genera las tablas y figuras de una etapa en results/analysis/<etapa>/.

    python analyze.py --stage C                 # comparación final (control: ELM-ELM)
    python analyze.py --stage A                 # importancia de los hiperparámetros del ELM
    python analyze.py --stage B                 # efecto de G y N_S dentro de cada variante
    python analyze.py --stage C --control SSA-KRG
    python analyze.py --stage C --exclude LHS KRVEA-pysamoo SSA-RBF --tag ei   # -> analysis/C_ei/

Funciona igual en el servidor o en el PC tras copiar results/ con scp.
"""
import argparse

import numpy as np
import pandas as pd

from zcatcmp import analysis as an
from zcatcmp import config, plots
from zcatcmp.indicators import front_indicators
from zcatcmp.storage import load_results

COLUMNS = ["variant", "config_id", "config", "problem", "M", "seed", "status", "hv", "igdp", "ep",
           "cpu_time", "curve_evals", "curve_hv", "curve_igdp", "curve_ep", "front_F"]
FRONT_PROBLEMS = (1, 11, 14, 17)          # una por clase de frente de ZCAT
# La revista pide título, revista, autores y correo del autor de contacto en cada archivo suplementario
ONLINE_RESOURCE_1 = (
    "Online Resource 1\n\n"
    "A controlled cost-quality comparison of ELM-, RBF- and Kriging-assisted\n"
    "multi-objective evolutionary algorithms on the ZCAT benchmark\n\n"
    "Soft Computing\n"
    "M. Vergara, Y. Salgueiro\n"
    "Universidad de Talca, Chile. Corresponding author: mavergara19@utalca.cl\n\n"
    "Non-dominated set of the median-HV run on the 20 ZCAT problems with three objectives.\n"
    "Small dots: reference front. Points beyond 1.25 times the extent of the front are omitted."
)


def comparison(df: pd.DataFrame, out, control: str, metrics) -> None:
    df = an.prepare(df)
    if control not in set(df["algo"]):
        raise SystemExit(f"el control {control!r} no está en los resultados: {sorted(df['algo'].unique())}")
    stats_rows, cost_tables = [], {}
    for metric in metrics:
        name = an.METRIC_NAMES[metric]
        an.summary(df, metric).to_csv(out / f"summary_{metric}.csv", index=False)

        pairs = an.pairwise(df, metric, control)
        pairs.to_csv(out / f"pairwise_{metric}.csv", index=False)
        wtl = an.win_tie_loss(pairs)
        wtl.to_csv(out / f"wtl_{metric}.csv", index=False)
        an.to_latex(wtl, out / f"wtl_{metric}.tex", fmt="{:.0f}",
                    caption=f"{control} against each variant per problem ({name}; Mann-Whitney U with Holm, "
                            r"$\alpha=0.05$): + better, = no difference, - worse.", label=f"tab:wtl-{metric}")
        print(f"\n== {name}: victorias/empates/derrotas de {control} ==\n{wtl.to_string(index=False)}")

        ablation, cd_panels = {}, {}
        for res in an.friedman(df, metric, control):
            M = res["M"]
            stats_rows.append({k: res[k] for k in ("M", "metric", "chi2", "p", "k", "n", "cd")})
            ranks = res["avg_ranks"].rename("avg_rank").reset_index().rename(columns={"index": "algo"})
            ranks.to_csv(out / f"friedman_ranks_{metric}_M{M}.csv", index=False)
            if res["posthoc"] is not None:
                res["posthoc"].to_csv(out / f"friedman_posthoc_{metric}_M{M}.csv", index=False)
            cd_panels[M] = (res["avg_ranks"], res["cd"])
            ablation[M] = an.ablation_ranks(res["avg_ranks"])
            ablation[M].to_csv(out / f"ablation_{metric}_M{M}.csv")
            if metric == metrics[0]:
                cost_tables[M] = an.cost_quality(df, res["avg_ranks"], M)
            print(f"Friedman {name} M={M}: chi2={res['chi2']:.1f} p={res['p']:.2g} | rangos: "
                  + ", ".join(f"{a} {r:.2f}" for a, r in res["avg_ranks"].items()))
        plots.cd_figure(cd_panels, out / f"cd_{metric}.pdf")
        if all(t.notna().any().any() for t in ablation.values()):
            plots.ablation_plot(ablation, name, out / f"ablation_{metric}.pdf")

    pd.DataFrame(stats_rows).to_csv(out / "friedman_stats.csv", index=False)
    for M, table in cost_tables.items():
        table.to_csv(out / f"cost_quality_M{M}.csv")
    plots.cost_quality_plot(cost_tables, an.METRIC_NAMES[metrics[0]], out / "cost_quality.pdf")

    curves = an.anytime(df, "hv")
    curves.to_csv(out / "anytime_hv.csv", index=False)
    plots.anytime_plot(curves, "HV", out / "anytime_hv.pdf")

    if "front_F" in df.columns:
        coverage = an.reference_coverage(df)
        coverage.to_csv(out / "coverage_reference_front.csv", index=False)
        table = coverage[coverage["radius"] == 0.1].pivot_table(index="algo", columns="M", values="coverage")
        print(f"\n== cobertura del frente de referencia (radio 0.1) ==\n{table.round(3).to_string()}")

    if 2 in cost_tables and "front_F" in df.columns:
        best = cost_tables[2].index[0]              # mejor rango medio en el primer indicador
        algos = [a for a in dict.fromkeys([best, control, "NSGAII"]) if a in set(df["algo"])]
        fronts = an.median_run_fronts(df, 2, FRONT_PROBLEMS, algos)
        reference = {p: np.loadtxt(config.REFERENCE_FRONTS / f"ZCAT{p}.2D.pf") for p in FRONT_PROBLEMS}
        plots.fronts_plot(fronts, reference, FRONT_PROBLEMS, algos, 2, out / "fronts_M2.pdf")

    if 3 in cost_tables and "front_F" in df.columns:
        best = cost_tables[3].index[0]
        algos = [a for a in dict.fromkeys([best, control, "NSGAII"]) if a in set(df["algo"])]
        problems = sorted(df["problem"].unique())
        fronts = an.median_run_fronts(df, 3, problems, algos)
        hv = {k: front_indicators(F, k[0], 3)[0] for k, F in fronts.items()}
        reference = {p: np.loadtxt(config.REFERENCE_FRONTS / f"ZCAT{p}.3D.pf") for p in problems}
        outside = plots.fronts_plot_3d(fronts, reference, [p for p in FRONT_PROBLEMS if p in problems], algos, hv,
                                       out / "fronts_M3.pdf")
        plots.fronts_pages_3d(fronts, reference, problems, algos, hv, out / "fronts_M3_all.pdf",
                              front_matter=ONLINE_RESOURCE_1)
        print(f"\nfrentes M=3: {outside:.1%} de los puntos queda fuera de los ejes (1.25 x frente de referencia)")
    if "NSGAII" in set(df["algo"]):
        reach = an.evals_to_reach(df, "NSGAII", "hv")
        reach.to_csv(out / "evals_to_reach_nsgaii.csv", index=False)
        print(f"\n== evaluaciones para alcanzar el HV final de NSGA-II ==\n{reach.to_string(index=False)}")


def loop(df: pd.DataFrame, out, metrics) -> None:
    for metric in metrics:
        effects = an.loop_effects(df, metric)
        effects.to_csv(out / f"loop_effects_{metric}.csv", index=False)
        if metric == metrics[0]:
            plots.loop_effect_plot(effects, out / f"loop_effect_G_{metric}.pdf")
            table = effects.pivot_table(index=["variant"], columns=["factor", "level", "M"], values="relative_rank")
            print(table.round(2).to_string())


def factors(df: pd.DataFrame, out, metrics) -> None:
    effects, global_tests = [], []
    for metric in metrics:
        levels, eff = an.factor_effects(df, metric)
        levels.to_csv(out / f"factor_levels_{metric}.csv", index=False)
        effects.append(eff)
        global_tests += an.friedman_configs(df, metric)
    effects = pd.concat(effects)
    table = effects.pivot_table(index="factor", columns=["metric", "M"], values="effect")
    table.columns = [f"{an.METRIC_NAMES[m]} M={M}" for m, M in table.columns]
    table = table.sort_values(table.columns[0], ascending=False).reset_index()
    table.to_csv(out / "factor_effects.csv", index=False)
    an.to_latex(table, out / "factor_effects.tex", fmt="{:.1f}",
                caption="Effect size of each ELM hyperparameter: range of the mean rank between its "
                        "best and worst level (stage A).", label="tab:factor-effects")
    pd.DataFrame(global_tests).to_csv(out / "friedman_configs.csv", index=False)
    print(table.to_string(index=False))
    print(pd.DataFrame(global_tests).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--control", default="ELM-ELM")
    ap.add_argument("--metrics", nargs="+", default=["hv", "igdp", "ep"], choices=list(an.ORIENT))
    ap.add_argument("--exclude", nargs="+", default=[], metavar="VARIANT", help="variantes que no entran al análisis")
    ap.add_argument("--tag", help="sufijo de la carpeta de salida: analysis/<etapa>_<tag>/")
    args = ap.parse_args()

    df = load_results(config.RESULTS_DIR / args.stage, columns=COLUMNS)
    if df.empty:
        raise SystemExit(f"no hay resultados en {config.RESULTS_DIR / args.stage}")
    unknown = set(args.exclude) - set(df["variant"])
    if unknown:
        raise SystemExit(f"variantes desconocidas en --exclude: {sorted(unknown)}")
    df = df[~df["variant"].isin(args.exclude)]
    out = config.RESULTS_DIR / "analysis" / (f"{args.stage}_{args.tag}" if args.tag else args.stage)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "A":
        factors(df, out, args.metrics)
    elif args.stage == "B":
        loop(df, out, args.metrics)
    else:
        comparison(df, out, args.control, args.metrics)
    print(f"\nresultados en {out}")


if __name__ == "__main__":
    main()
