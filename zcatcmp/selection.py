"""Selección de configuraciones entre etapas por ranking promedio sobre problemas.

Para cada (M, variante): media del indicador sobre las semillas en cada
(configuración, problema), ranking de las configuraciones dentro de cada
problema (1 = mejor) y promedio sobre los 20 problemas.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config

HIGHER_IS_BETTER = {"hv": True, "igdp": False, "ep": False}


def average_ranks(df: pd.DataFrame, metric: str = "hv") -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    means = ok.groupby(["M", "variant", "config_id", "problem"])[metric].mean().reset_index()
    means["rank"] = means.groupby(["M", "variant", "problem"])[metric].rank(
        ascending=not HIGHER_IS_BETTER[metric], method="average")
    out = (means.groupby(["M", "variant", "config_id"])
           .agg(avg_rank=("rank", "mean"), n_problems=("problem", "nunique"))
           .reset_index().sort_values(["M", "variant", "avg_rank"]))
    return out


def _config_of(df: pd.DataFrame, M: int, variant: str, cid: str) -> dict:
    row = df[(df["M"] == M) & (df["variant"] == variant) & (df["config_id"] == cid)].iloc[0]
    return json.loads(row["config"])


def select(df: pd.DataFrame, stage: str, metric: str = "hv", top_k: int | None = None) -> dict:
    ranks = average_ranks(df, metric)
    if stage == "A":
        top_k = top_k or config.TOP_K_A
        return {str(M): [_config_of(df, M, "ELM-ELM", cid)["model"] for cid in g["config_id"].head(top_k)]
                for M, g in ranks[ranks["variant"] == "ELM-ELM"].groupby("M")}
    if stage == "B":
        out = {}
        for (M, variant), g in ranks.groupby(["M", "variant"]):
            out.setdefault(str(M), {})[variant] = _config_of(df, M, variant, g["config_id"].iloc[0])
        return out
    raise ValueError("sólo se selecciona tras las etapas A y B")
