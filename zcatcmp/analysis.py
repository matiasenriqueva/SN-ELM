"""Análisis estadístico de la campaña (funciones puras sobre DataFrames de resultados).

Comparación (etapa C)
    * ``pairwise``: por problema, Mann-Whitney U (rank-sum) del control contra cada
      variante, Holm sobre las variantes del problema y tamaño de efecto Â12.
    * ``friedman``: por M, Friedman sobre las medianas por problema, post-hoc de Holm
      contra el control (Demšar, 2006) y diferencia crítica de Nemenyi.
    * ``ablation_ranks``: rango medio de las 6 variantes de la ablación (bucle × modelo).
    * ``anytime`` / ``evals_to_reach``: curvas de HV relativo y evaluaciones necesarias
      para alcanzar el HV final de una referencia.
Factores (etapa A)
    * ``factor_effects``: rango medio por nivel de cada hiperparámetro del ELM y tamaño
      de efecto (amplitud entre el mejor y el peor nivel).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import friedmanchisquare, mannwhitneyu, norm, studentized_range

from . import config
from .indicators import reference_front, scale
from .selection import average_ranks

ORIENT = {"hv": 1.0, "igdp": -1.0, "ep": -1.0}     # +1: mayor es mejor
METRIC_NAMES = {"hv": "HV", "igdp": "IGD+", "ep": "EP"}


# ======================================================================
# Preparación
# ======================================================================

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sólo corridas 'ok' y una columna ``algo``: la variante, o variante|config si hay varias."""
    df = df[df["status"] == "ok"].copy()
    n_cfg = df.groupby(["M", "variant"])["config_id"].transform("nunique")
    df["algo"] = np.where(n_cfg > 1, df["variant"] + "|" + df["config_id"], df["variant"])
    return df


def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def a12(x, y) -> float:
    """Vargha-Delaney: P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x, dtype=float)[:, None], np.asarray(y, dtype=float)[None, :]
    return float(((x > y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


# ======================================================================
# Comparación
# ======================================================================

def summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    g = df.groupby(["M", "problem", "algo"])[metric]
    out = pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75), "runs": g.size()})
    return out.reset_index()


def pairwise(df: pd.DataFrame, metric: str, control: str, alpha: float = 0.05) -> pd.DataFrame:
    """'+' el control es significativamente mejor, '-' peor, '=' sin diferencia."""
    rows = []
    sign = ORIENT[metric]
    for (M, problem), g in df.groupby(["M", "problem"]):
        x = sign * g.loc[g["algo"] == control, metric].to_numpy()
        if len(x) == 0:
            continue
        others = [a for a in sorted(g["algo"].unique()) if a != control]
        stats = []
        for algo in others:
            y = sign * g.loc[g["algo"] == algo, metric].to_numpy()
            p = 1.0 if np.array_equal(np.sort(x), np.sort(y)) else mannwhitneyu(x, y, alternative="two-sided").pvalue
            stats.append((algo, p, a12(x, y)))
        p_holm = holm([s[1] for s in stats]) if stats else []
        for (algo, p, effect), ph in zip(stats, p_holm):
            outcome = "=" if ph >= alpha else ("+" if effect > 0.5 else "-")
            rows.append(dict(M=M, problem=problem, algo=algo, p=p, p_holm=ph, a12=effect, outcome=outcome))
    return pd.DataFrame(rows)


def win_tie_loss(pairs: pd.DataFrame) -> pd.DataFrame:
    counts = pairs.groupby(["M", "algo", "outcome"]).size().unstack(fill_value=0)
    return counts.reindex(columns=["+", "=", "-"], fill_value=0).reset_index()


def friedman(df: pd.DataFrame, metric: str, control: str, alpha: float = 0.05) -> list[dict]:
    """Friedman por M sobre las medianas por problema (1 = mejor rango)."""
    results = []
    for M, g in df.groupby("M"):
        med = g.groupby(["problem", "algo"])[metric].median().unstack().dropna(axis=1)
        ranks = (-ORIENT[metric] * med).rank(axis=1)
        avg = ranks.mean().sort_values()
        k, n = med.shape[1], med.shape[0]
        stat, p = friedmanchisquare(*[med[c] for c in med.columns]) if k >= 3 else (np.nan, np.nan)
        se = np.sqrt(k * (k + 1) / (6.0 * n))
        cd = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2) * se
        posthoc = None
        if control in avg.index:
            others = [a for a in avg.index if a != control]
            z = (avg[others] - avg[control]) / se
            pv = 2 * (1 - norm.cdf(np.abs(z)))
            posthoc = pd.DataFrame({"algo": others, "rank_diff": (avg[others] - avg[control]).to_numpy(),
                                    "z": z.to_numpy(), "p": pv, "p_holm": holm(pv)})
        results.append(dict(M=M, metric=metric, chi2=stat, p=p, k=k, n=n, avg_ranks=avg, cd=cd, posthoc=posthoc))
    return results


def ablation_ranks(avg_ranks: pd.Series) -> pd.DataFrame:
    """Rango medio de las variantes de la ablación como tabla bucle × modelo."""
    table = pd.DataFrame(index=list(config.SURROGATE_LOOPS), columns=list(config.SURROGATE_MODELS), dtype=float)
    for algo, rank in avg_ranks.items():
        variant = algo.split("|")[0]
        if variant in config.ABLATION_VARIANTS:
            loop, model = variant.split("-")
            table.loc[loop, model] = rank
    return table


def cost_quality(df: pd.DataFrame, avg_ranks: pd.Series, M: int) -> pd.DataFrame:
    cpu = df[df["M"] == M].groupby("algo")["cpu_time"].mean()
    return pd.DataFrame({"cpu_mean_s": cpu, "avg_rank": avg_ranks}).dropna().sort_values("avg_rank")


def _curves(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = f"curve_{metric}"
    long = df[["M", "problem", "algo", "seed", "curve_evals", col]].explode(["curve_evals", col])
    long = long.rename(columns={"curve_evals": "evals", col: "value"}).dropna()
    return long.astype({"evals": int, "value": float})


def anytime(df: pd.DataFrame, metric: str = "hv") -> pd.DataFrame:
    """Mediana sobre semillas, relativa al mejor valor final por problema, promediada sobre problemas."""
    med = _curves(df, metric).groupby(["M", "problem", "algo", "evals"])["value"].median().reset_index()
    final = med[med["evals"] == med["evals"].max()]
    if ORIENT[metric] > 0:
        best = final.groupby(["M", "problem"])["value"].max().rename("best")
        med = med.join(best, on=["M", "problem"])
        med["relative"] = med["value"] / med["best"].where(med["best"] > 0)
    else:
        best = final.groupby(["M", "problem"])["value"].min().rename("best")
        med = med.join(best, on=["M", "problem"])
        med["relative"] = med["best"] / med["value"].where(med["value"] > 0)
    return med.groupby(["M", "algo", "evals"])["relative"].mean().reset_index()


def evals_to_reach(df: pd.DataFrame, reference: str = "NSGAII", metric: str = "hv") -> pd.DataFrame:
    """Primera evaluación en que la curva mediana alcanza el valor final mediano de la referencia."""
    med = _curves(df, metric).groupby(["M", "problem", "algo", "evals"])["value"].median().reset_index()
    sign = ORIENT[metric]
    rows = []
    for (M, problem), g in med.groupby(["M", "problem"]):
        ref = g[g["algo"] == reference]
        if ref.empty:
            continue
        target = sign * ref.loc[ref["evals"].idxmax(), "value"]
        for algo, h in g.groupby("algo"):
            reached = h[sign * h["value"] >= target - 1e-12]
            rows.append(dict(M=M, problem=problem, algo=algo,
                             evals=int(reached["evals"].min()) if len(reached) else np.nan))
    per_problem = pd.DataFrame(rows)
    agg = per_problem.groupby(["M", "algo"])["evals"].agg(
        reached="count", problems="size", median_evals="median").reset_index()
    return agg.sort_values(["M", "median_evals"])


def reference_coverage(df: pd.DataFrame, radii=(0.05, 0.1, 0.2)) -> pd.DataFrame:
    """Fracción de puntos del frente de referencia con una solución a distancia <= radio.

    En el espacio normalizado de los indicadores. Mediana sobre semillas por problema y
    promedio sobre problemas. Combina convergencia y cobertura: una solución cuenta si
    está cerca del punto, y un punto queda cubierto sólo si hay soluciones en su zona.
    """
    rows = []
    for r in df.itertuples():
        scale_m = scale(r.M)
        d, _ = cKDTree(np.vstack(r.front_F) / scale_m).query(reference_front(r.problem, r.M))
        rows += [dict(M=r.M, problem=r.problem, algo=r.algo, radius=radius, coverage=float((d <= radius).mean()))
                 for radius in radii]
    per_problem = pd.DataFrame(rows).groupby(["M", "problem", "algo", "radius"])["coverage"].median()
    return per_problem.groupby(["M", "algo", "radius"]).mean().reset_index()


def median_run_fronts(df: pd.DataFrame, M: int, problems, algos, metric: str = "hv") -> dict:
    """Frente de la corrida con el indicador mediano, por (problema, variante)."""
    out = {}
    for p in problems:
        for algo in algos:
            g = df[(df["M"] == M) & (df["problem"] == p) & (df["algo"] == algo)].sort_values(metric)
            if len(g):
                out[(p, algo)] = np.asarray(g.iloc[len(g) // 2]["front_F"].tolist(), dtype=float)
    return out


# ======================================================================
# Bucle (etapa B)
# ======================================================================

def loop_effects(df: pd.DataFrame, metric: str = "hv") -> pd.DataFrame:
    """Rango relativo de cada nivel de G y N_S dentro de cada variante (0 = mejor, 1 = peor).

    Las variantes tienen distinto número de configuraciones (ELM-ELM evalúa dos modelos),
    por eso el rango medio se reescala a [0, 1] con el número de configuraciones.
    """
    ranks = average_ranks(df, metric)
    cfg = df.drop_duplicates(["M", "variant", "config_id"]).set_index(["M", "variant", "config_id"])["config"]
    loops = [json.loads(cfg[(r.M, r.variant, r.config_id)])["loop"] for r in ranks.itertuples()]
    ranks["G"] = [lp["G"] for lp in loops]
    ranks["NS"] = [lp["NS"] for lp in loops]
    n_cfg = ranks.groupby(["M", "variant"])["config_id"].transform("count")
    ranks["relative_rank"] = (ranks["avg_rank"] - 1) / (n_cfg - 1)
    rows = []
    for factor in ("G", "NS"):
        g = ranks.groupby(["M", "variant", factor])["relative_rank"].mean().reset_index()
        rows += [dict(M=r.M, variant=r.variant, factor=factor, level=getattr(r, factor),
                      relative_rank=r.relative_rank) for r in g.itertuples()]
    return pd.DataFrame(rows)


# ======================================================================
# Factores del ELM (etapa A)
# ======================================================================

def factor_effects(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rango medio por nivel, tamaño de efecto por factor) para cada M."""
    ranks = average_ranks(df, metric)
    model = df.drop_duplicates("config_id").set_index("config_id")["config"].map(lambda c: json.loads(c)["model"])
    levels, effects = [], []
    for factor in ("act", "nh_factor", "C"):
        ranks[factor] = ranks["config_id"].map(lambda c: model[c][factor])
        by_level = ranks.groupby(["M", factor])["avg_rank"].mean().reset_index()
        for _, row in by_level.iterrows():
            levels.append(dict(M=row["M"], factor=factor, level=row[factor], avg_rank=row["avg_rank"]))
        span = by_level.groupby("M")["avg_rank"].agg(lambda s: s.max() - s.min())
        for M, value in span.items():
            effects.append(dict(M=M, factor=factor, metric=metric, effect=value))
    return pd.DataFrame(levels), pd.DataFrame(effects)


def friedman_configs(df: pd.DataFrame, metric: str) -> list[dict]:
    """Friedman global entre configuraciones (como la tabla del manuscrito)."""
    ok = df[df["status"] == "ok"]
    out = []
    for M, g in ok.groupby("M"):
        med = g.groupby(["problem", "config_id"])[metric].mean().unstack().dropna(axis=1)
        stat, p = friedmanchisquare(*[med[c] for c in med.columns])
        out.append(dict(M=M, metric=metric, k=med.shape[1], n=med.shape[0], chi2=stat, p=p))
    return out


# ======================================================================
# Salida
# ======================================================================

def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str, fmt: str = "{:.3f}") -> None:
    """Tabla booktabs mínima (sin depender de jinja2)."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "--" if np.isnan(v) else fmt.format(v)
        return str(v).replace("_", r"\_").replace("|", r"$|$")

    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 1 + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(cell(c) for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
