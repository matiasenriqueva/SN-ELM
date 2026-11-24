"""Etapas de la campaña y construcción de la lista de trabajos.

pilot  1 semilla, 4 problemas, todas las variantes en la config. más barata y en
       la más cara de la grilla del bucle -> tiempos para dimensionar B y C.
A      modelo ELM (grilla ELM_GRID) con el bucle fijo, semillas de ajuste.
B      bucle (grilla LOOP_GRID) para todas las variantes con surrogado;
       usa los mejores modelos ELM de A (results/selection_A.json).
C      evaluación final con semillas nuevas: la mejor config. de cada variante
       (results/selection_B.json) + referencias.
"""
from __future__ import annotations

import itertools
import json

from . import config

STAGES = ("pilot", "A", "B", "C")


def grid(spec: dict) -> list[dict]:
    keys = list(spec)
    return [dict(zip(keys, values)) for values in itertools.product(*(spec[k] for k in keys))]


def is_kriging(variant: str) -> bool:
    return variant.endswith("-KRG") or variant.startswith("KRVEA")


def config_id(loop: dict | None, model: dict | None) -> str:
    parts = []
    if loop:
        parts.append(f"G{loop['G']}-NS{loop['NS']}")
    if model:
        parts.append(f"{model['act']}-nh{model['nh_factor']}-C{model['C']:g}")
    return "_".join(parts) or "default"


def selection_path(stage: str):
    return config.RESULTS_DIR / f"selection_{stage}.json"


def load_selection(stage: str) -> dict:
    path = selection_path(stage)
    if not path.exists():
        raise FileNotFoundError(f"falta {path}: ejecuta select_configs.py --stage {stage} primero")
    return json.loads(path.read_text())


def _specs_for_stage(stage: str, n_obj: int) -> list[tuple[str, dict | None, dict | None]]:
    """(variante, loop, modelo) para un M dado."""
    loops = grid(config.LOOP_GRID)
    no_loop_refs = [("NSGAII", None, None), ("LHS", None, None)]

    if stage == "pilot":
        cheap, costly = dict(G=10, NS=10), dict(G=max(config.LOOP_GRID["G"]), NS=min(config.LOOP_GRID["NS"]))
        specs = []
        for loop in (cheap, costly):
            for variant in config.TUNED_VARIANTS:
                model = config.ELM_DEFAULT if variant.endswith("-ELM") else None
                specs.append((variant, loop, model))
        specs += [("ELM-ELM", cheap, dict(config.ELM_DEFAULT, act="gaussian"))]
        specs += [("KRVEA-pysamoo", config.KRVEA_PAPER, None)] + no_loop_refs
        return specs

    if stage == "A":
        return [("ELM-ELM", config.LOOP_FIXED_A, model) for model in grid(config.ELM_GRID)]

    if stage == "B":
        best_models = load_selection("A")[str(n_obj)]
        specs = [("ELM-ELM", loop, model) for model in best_models for loop in loops]
        specs += [("SSA-ELM", loop, best_models[0]) for loop in loops]
        for variant in ("ELM-RBF", "ELM-KRG", "SSA-RBF", "SSA-KRG", "KRVEA"):
            specs += [(variant, loop, None) for loop in loops]
        return specs

    if stage == "C":
        chosen = load_selection("B")[str(n_obj)]
        specs = [(variant, c["loop"], c["model"]) for variant, c in chosen.items()]
        return specs + [("KRVEA-pysamoo", config.KRVEA_PAPER, None)] + no_loop_refs

    raise ValueError(f"etapa desconocida: {stage}")


def build_jobs(stage: str, objectives=config.OBJECTIVES, problems=None, seeds=None,
               variants=None, code_version: str = "unknown") -> list[dict]:
    if problems is None:
        problems = config.PILOT_PROBLEMS if stage == "pilot" else config.PROBLEMS
    if seeds is None:
        seeds = {"pilot": config.SEEDS_PILOT, "C": config.SEEDS_FINAL}.get(stage, config.SEEDS_TUNING)
        default_seeds = True
    else:
        default_seeds = False

    jobs = []
    for n_obj in objectives:
        for variant, loop, model in _specs_for_stage(stage, n_obj):
            if variants and variant not in variants:
                continue
            cid = config_id(loop, model)
            variant_seeds = (config.SEEDS_TUNING_KRIGING if default_seeds and stage == "B" and is_kriging(variant)
                             else seeds)
            for p in problems:
                for s in variant_seeds:
                    jobs.append({
                        "key": f"{stage}|{variant}|{cid}|ZCAT{p}|M{n_obj}|s{s}",
                        "stage": stage, "variant": variant, "config_id": cid,
                        "loop": loop, "model": model,
                        "problem": p, "M": n_obj, "seed": s,
                        "n_var": config.N_VAR, "budget": config.N_EVAL,
                        "n_doe": config.N_DOE, "n_train": config.N_TRAIN,
                        "checkpoints": config.CHECKPOINTS,
                        "store_X": stage == "C",
                        "max_seconds": config.MAX_JOB_SECONDS,
                        "krg_theta_every": config.KRG_THETA_EVERY,
                        "code_version": code_version,
                    })
    # Lo caro primero: evita que la campaña termine con pocos procesos ocupados.
    jobs.sort(key=lambda j: (0 if is_kriging(j["variant"]) else 1, -j["M"]))
    return jobs
