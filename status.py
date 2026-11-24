"""Resumen del avance y del costo de una etapa (se puede correr mientras la etapa sigue).

    python status.py --stage pilot
    python status.py --stage pilot --estimate  # proyecta el costo de A, B y C a partir del piloto
    python status.py --stage B --remaining     # tiempo restante real de la etapa en curso
"""
import argparse

import pandas as pd

from zcatcmp import config, stages
from zcatcmp.storage import load_results

# Si aún no existen las selecciones de A/B, se proyecta con unas provisionales: el costo
# estimado sólo depende de (variante, M), no de qué configuración resulte elegida.
_PROVISIONAL = {
    "A": {str(M): [config.ELM_DEFAULT, dict(config.ELM_DEFAULT, act="sigmoid")] for M in config.OBJECTIVES},
    "B": {str(M): {v: {"loop": dict(G=10, NS=10), "model": config.ELM_DEFAULT if v.endswith("-ELM") else None}
                   for v in config.TUNED_VARIANTS} for M in config.OBJECTIVES},
}


def jobs_for(stage: str) -> tuple[list[dict], bool]:
    """Trabajos de la etapa; el bool indica si se usó una selección provisional."""
    try:
        return stages.build_jobs(stage), False
    except FileNotFoundError:
        real = stages.load_selection
        stages.load_selection = lambda s: real(s) if stages.selection_path(s).exists() else _PROVISIONAL[s]
        try:
            return stages.build_jobs(stage), True
        finally:
            stages.load_selection = real


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    return (ok.groupby(["variant", "config_id", "M"])["cpu_time"]
            .agg(runs="count", cpu_mean_s="mean", cpu_max_s="max").round(1).reset_index())


def estimate(pilot: pd.DataFrame, stage: str, workers: int) -> float:
    """Asigna a cada trabajo de la etapa el tiempo medio del piloto de su variante y M.

    En el piloto cada variante con bucle se mide en su config. barata (G10-NS10) y
    en la más cara (G20-NS5); se usa el promedio de ambas como estimación de la grilla.
    Los tiempos del piloto se midieron con el servidor lleno, así que ya incluyen la
    competencia entre procesos. Devuelve las horas de reloj estimadas.
    """
    per = pilot[pilot["status"] == "ok"].groupby(["variant", "M"])["cpu_time"].mean()
    jobs, provisional = jobs_for(stage)
    total, missing = 0.0, set()
    for job in jobs:
        key = (job["variant"], job["M"])
        if key in per.index:
            total += per[key]
        else:
            missing.add(key)
    hours = total / 3600 / workers
    note = " (selección provisional)" if provisional else ""
    print(f"etapa {stage}: {len(jobs)} trabajos, ~{total / 3600:.0f} h-CPU, ~{hours:.1f} h con {workers} procesos{note}")
    if missing:
        print(f"  sin tiempo de piloto para: {sorted(missing)} (no sumados)")
    return hours


def remaining(stage: str, workers: int) -> None:
    """Tiempo restante de una etapa en curso, ponderado por el costo de cada corrida pendiente.

    El "restante" del log supone que todas las corridas cuestan lo mismo que las ya
    terminadas; como las caras (Kriging) van primero, lo sobreestima mucho. Aquí cada
    corrida pendiente se estima con el CPU medio observado en esta etapa para su
    (variante, config., M); si aún no hay, con su (variante, M); y si tampoco, con el
    piloto corregido por cuánto más lentas resultaron las corridas reales.
    """
    jobs, _ = jobs_for(stage)
    df = load_results(config.RESULTS_DIR / stage, columns=["key", "variant", "config_id", "M", "status", "cpu_time"])
    ok = df[df["status"] == "ok"]
    done = set(ok["key"])
    pending = [j for j in jobs if j["key"] not in done]
    by_config = ok.groupby(["variant", "config_id", "M"])["cpu_time"].mean()
    by_variant = ok.groupby(["variant", "M"])["cpu_time"].mean()
    pilot = load_results(config.RESULTS_DIR / "pilot", columns=["variant", "M", "status", "cpu_time"])
    by_pilot = pilot[pilot["status"] == "ok"].groupby(["variant", "M"])["cpu_time"].mean()
    ratios = [by_variant[k] / by_pilot[k] for k in by_variant.index if k in by_pilot.index]
    slowdown = float(pd.Series(ratios).median()) if ratios else 1.0

    total, unknown = 0.0, 0
    for job in pending:
        cfg, var = (job["variant"], job["config_id"], job["M"]), (job["variant"], job["M"])
        if cfg in by_config.index:
            total += by_config[cfg]
        elif var in by_variant.index:
            total += by_variant[var]
        elif var in by_pilot.index:
            total += by_pilot[var] * slowdown
        else:
            unknown += 1
    hours = total / 3600 / workers
    print(f"\netapa {stage}: {len(jobs) - len(pending)} de {len(jobs)} terminadas, {len(pending)} pendientes")
    print(f"restante estimado: ~{hours:.1f} h (~{hours / 24:.1f} días) con {workers} procesos"
          f" | corridas reales vs piloto: x{slowdown:.2f}")
    if unknown:
        print(f"  {unknown} corridas sin referencia de costo (no sumadas)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--estimate", action="store_true", help="proyecta A, B y C a partir del piloto")
    ap.add_argument("--remaining", action="store_true", help="tiempo restante de la etapa en curso")
    ap.add_argument("--workers", type=int, default=47)
    args = ap.parse_args()

    df = load_results(config.RESULTS_DIR / args.stage,
                      columns=["variant", "config_id", "M", "status", "n_evals", "cpu_time", "error"])
    print(f"etapa {args.stage}: {len(df)} filas")
    if df.empty:
        return
    print(df.groupby(["variant", "status"]).size().unstack(fill_value=0).to_string())
    with pd.option_context("display.max_rows", 300, "display.width", 160):
        print(cost_table(df).to_string(index=False))
    errors = df[df["status"] != "ok"]
    if len(errors):
        print("\nprimer error:\n", errors["error"].dropna().head(1).to_string(index=False))

    if args.estimate:
        pilot = load_results(config.RESULTS_DIR / "pilot", columns=["variant", "M", "status", "cpu_time"])
        print()
        hours = sum(estimate(pilot, stage, args.workers) for stage in ("A", "B", "C"))
        print(f"total A+B+C: ~{hours:.0f} h (~{hours / 24:.1f} días)")
    if args.remaining:
        remaining(args.stage, args.workers)


if __name__ == "__main__":
    main()
