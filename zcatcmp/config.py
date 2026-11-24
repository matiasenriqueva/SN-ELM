"""Configuración central de la campaña de comparación.

Todo lo que define el protocolo experimental vive aquí: presupuesto, operadores,
grillas de hiperparámetros, semillas e indicadores. Los trabajos (``Job``) copian
los valores que necesitan, de modo que un resultado guardado es autocontenido.
"""
from __future__ import annotations

import os
from pathlib import Path

# ------------------------------------------------------------------
# Rutas
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # comparison/
REPO = ROOT.parent                                  # smoea-zcat/
REFERENCE_FRONTS = ROOT / "data" / "reference_fronts"     # ZCAT1-20, 2D y 3D (de code/resources)
# ZCATCMP_RESULTS / ZCATCMP_LOGS permiten desviar una prueba a otra carpeta
RESULTS_DIR = Path(os.environ.get("ZCATCMP_RESULTS", ROOT / "results"))
LOGS_DIR = Path(os.environ.get("ZCATCMP_LOGS", ROOT / "logs"))

# ------------------------------------------------------------------
# Problema: ZCAT nivel 1, sin bias / imbalance / PS complicado
# ------------------------------------------------------------------
N_VAR = 30
PROBLEMS = tuple(range(1, 21))
OBJECTIVES = (2, 3)
ZCAT_KWARGS = dict(level=1, bias=False, imbalance=False, complicated_pareto_set=False)

# ------------------------------------------------------------------
# Presupuesto (igual para todos los algoritmos)
# ------------------------------------------------------------------
N_EVAL = 2000                 # N_E: evaluaciones reales totales
N_DOE = 20 * N_VAR            # LHS inicial compartido (600)
N_TRAIN = N_DOE               # tope del archivo de entrenamiento de TODOS los surrogados
CHECKPOINTS = tuple(range(100, N_EVAL + 1, 100))   # para las curvas anytime

# ------------------------------------------------------------------
# Operadores: los mismos para NSGA-II base y para los NSGA-II internos
# ------------------------------------------------------------------
POP_SIZE = 50
SBX_PROB = 0.9
SBX_ETA = 20.0
PM_FACTOR = 2.0               # prob. de mutación = PM_FACTOR / n
PM_ETA = 20.0
EPSILON = 1e-3                # distancia mínima del filtro de novedad (bucle ELM)
MAX_EMPTY_ITERATIONS = 50     # salvaguarda: iteraciones seguidas sin candidatos
MAX_JOB_SECONDS = 4 * 3600    # salvaguarda: tiempo máximo por corrida (sólo Linux, SIGALRM)
KRG_THETA_EVERY = 5           # Kriging: θ por máxima verosimilitud cada 5 ajustes, fijo entre medio

# ------------------------------------------------------------------
# Indicadores: objetivos normalizados por el rango del frente f_i in [0, i^2]
# ------------------------------------------------------------------
HV_REF = 1.1                  # punto de referencia del HV en el espacio normalizado

# ------------------------------------------------------------------
# Semillas: ajuste (etapas A/B) y evaluación final (etapa C) NO se solapan
# ------------------------------------------------------------------
SEEDS_TUNING = tuple(range(0, 10))
# Etapa B, variantes Kriging (ELM-KRG, SSA-KRG, KRVEA): ~89% del costo de B. B sólo elige la
# configuración de cada variante (se compara dentro de la variante); C usa 30 semillas para todas.
SEEDS_TUNING_KRIGING = tuple(range(0, 5))
SEEDS_FINAL = tuple(range(100, 130))
SEEDS_PILOT = (0,)
PILOT_PROBLEMS = (1, 11, 14, 17)   # uno por clase de frente

# ------------------------------------------------------------------
# Grillas
# ------------------------------------------------------------------
# Etapa A: modelo ELM con el bucle fijo
ELM_GRID = dict(
    act=("sigmoid", "gaussian", "multiquadric"),
    nh_factor=(2, 4, 8),              # N_H = nh_factor * n + 1
    C=(2.0 ** 0, 2.0 ** 5, 2.0 ** 10),
)
LOOP_FIXED_A = dict(G=10, NS=10)

# Etapa B: bucle, la misma grilla para todos los algoritmos con surrogado
LOOP_GRID = dict(G=(5, 10, 20), NS=(5, 10))

# Configuraciones por defecto (piloto y referencias)
ELM_DEFAULT = dict(act="multiquadric", nh_factor=4, C=2.0 ** 5)   # mejor config. del paper
KRVEA_PAPER = dict(G=20, NS=5)                                    # u y w_max de Chugh et al. (2018)

# Cuántas configuraciones pasan de una etapa a la siguiente
TOP_K_A = 2      # modelos ELM que pasan a la etapa B
TOP_K_B = 1      # configuración final por variante

# ------------------------------------------------------------------
# Variantes
# ------------------------------------------------------------------
# bucle x surrogado (ablación 2x3) + KRVEA + referencias
SURROGATE_LOOPS = ("ELM", "SSA")
SURROGATE_MODELS = ("ELM", "RBF", "KRG")
ABLATION_VARIANTS = tuple(f"{loop}-{model}" for loop in SURROGATE_LOOPS for model in SURROGATE_MODELS)
TUNED_VARIANTS = ABLATION_VARIANTS + ("KRVEA",)
REFERENCE_VARIANTS = ("NSGAII", "LHS", "KRVEA-pysamoo")
# SSANSGA2 de pysamoo sin cambios no se incluye: en el commit fijado su conjunto de
# modelos por defecto es incompatible con ezmodel (y sería prohibitivo en costo).
