# SN-ELM

Code and results for the manuscript

> M. Vergara, Y. Salgueiro. *SN-ELM: an extreme learning machine-assisted NSGA-II for expensive
> multi-objective optimization*. Submitted to *Evolutionary Intelligence*.

SN-ELM is a surrogate-assisted NSGA-II that models all objectives with a single regularized
extreme learning machine (ELM), retrains it after every batch of true evaluations, searches it
with a short NSGA-II run started from the non-dominated solutions found so far, and evaluates the
most spread and novel solutions of the predicted front. This repository contains its
implementation, the six algorithms it is compared with, the experimental protocol of the three
stages, and the statistical analysis that produces the tables and figures of the manuscript.

## Algorithms

The code uses internal names; the manuscript uses the names on the left.

| Manuscript | Code | Loop | Surrogate |
|---|---|---|---|
| SN-ELM | `ELM-ELM` | SN loop (`zcatcmp/elm_loop.py`) | ELM |
| SN-RBF | `ELM-RBF` | SN loop | RBF |
| SN-KRG | `ELM-KRG` | SN loop | Kriging |
| SSA-ELM | `SSA-ELM` | surrogate-assisted NSGA-II of pysamoo, same operators | ELM |
| SSA-KRG | `SSA-KRG` | surrogate-assisted NSGA-II of pysamoo, same operators | Kriging |
| KRVEA | `KRVEA` | K-RVEA of pysamoo, RVEA population kept across updates | Kriging |
| NSGA-II | `NSGAII` | no surrogate, same operators | — |

The code also defines `SSA-RBF`, `KRVEA-pysamoo` (K-RVEA of pysamoo without changes) and `LHS`
(2000 Latin hypercube points). They were run in the campaign but are not part of the manuscript;
the commands below leave them out.

Shared by all algorithms (`zcatcmp/config.py`): ZCAT1–20 with n = 30 and M ∈ {2, 3}; 2000 true
evaluations, of which 600 form a Latin hypercube design shared per problem, M and seed; a training
archive of at most 600 solutions with the same replacement rule; an inner population of 50 (RVEA:
one per reference vector), SBX (0.9, 20) and polynomial mutation (2/n, 20). Hypervolume (reference
1.1), IGD+ and additive epsilon are computed on objectives normalized by f_i / i², against the
reference fronts in `data/reference_fronts/`.

## Installation

Python 3.12 and git are required (three packages of the pysamoo ecosystem are installed from
GitHub at fixed commits, because the needed versions are not on PyPI):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

ZCAT is included in `zcatcmp/zcat_jmetal.py` (a copy of the jMetalPy implementation, checked
against stored values in `tests/data/zcat_golden.npz`), because jmetalpy 1.9.0 on PyPI does not
include it. The tests that compare the rewritten loop with the authors' earlier code are skipped,
since that code is not part of this repository.

## Reproducing the experiments

The experiment has three stages (`zcatcmp/config.py`); tuning (A, B) and evaluation (C) use
disjoint seeds.

| Stage | Tunes | Seeds | Runs* | CPU time* |
|---|---|---|---|---|
| A | ELM model of SN-ELM: activation × N_H × C (27), loop fixed at G = 10, N_S = 10 | 0–9 | 10 800 | 42 h |
| B | loop G ∈ {5, 10, 20} × N_S ∈ {5, 10} of the six surrogate-assisted algorithms | 0–9 (Kriging: 0–4) | 13 200 | 2 911 h |
| C | final comparison of the selected configurations | 100–129 | 8 400 | 2 511 h |

\* For the seven algorithms of the manuscript, measured on a 48-core Linux server.

```bash
SURROGATE="ELM-ELM ELM-RBF ELM-KRG SSA-ELM SSA-KRG KRVEA"
python run_experiments.py --stage A --workers 47
python select_configs.py --stage A                      # -> results/selection_A.json
python run_experiments.py --stage B --workers 47 --variants $SURROGATE
python select_configs.py --stage B                      # -> results/selection_B.json
python run_experiments.py --stage C --workers 47 --variants $SURROGATE NSGAII
```

`results/selection_A.json` and `results/selection_B.json` contain the configurations selected in
the campaign, so stage C can be run directly. Each run can also be reproduced on its own:
`--problems`, `--M` and `--seeds` restrict a stage, `--dry-run` counts the pending runs, and
finished runs are skipped when a command is repeated. Results are written to
`results/<stage>/part-*.parquet`, one row per run with the indicators, the anytime curves, the
final non-dominated set and the CPU time; `zcatcmp.storage.load_results` reads them into a
DataFrame. `python status.py --stage C` reports the progress of a stage.

## Analysis, tables and figures

```bash
python analyze.py --stage A                              # effect of the ELM hyperparameters
python ei/make_assets.py                                 # stages B and C, figures and tables
```

`ei/make_assets.py` runs the analysis of stages B and C on the seven algorithms (Friedman test
with Holm post-hoc against SN-ELM, Mann–Whitney U with Holm correction and Vargha–Delaney Â12 per
problem, critical-difference diagram, cost–quality, anytime curves, approximated fronts and
coverage of the reference front) and writes the figures to `ei/paper/` and the data of the tables
to `ei/tables/`. Both scripts need the raw results of the stages.

The outputs of the campaign are included:

| File | Content |
|---|---|
| `ei/paper/ESM_2.csv` | Online Resource 2: per-problem median, quartiles and comparison with SN-ELM for HV, IGD+ and EP |
| `ei/tables/factor_effects_A.csv` | effect of each ELM hyperparameter (stage A) |
| `ei/tables/selection_B.csv` | configuration selected for each algorithm (stage B) |
| `ei/tables/ranks_cpu.csv`, `friedman_stats.csv` | Friedman mean ranks, test statistics and CPU time per run (stage C) |
| `ei/tables/wtl.csv`, `posthoc_vs_sn_elm.csv` | wins/ties/losses and Holm post-hoc of SN-ELM against each algorithm |
| `ei/tables/coverage_reference_front.csv`, `evals_to_reach_nsgaii.csv` | coverage of the reference front and evaluations needed to reach the final HV of NSGA-II |

The raw results (about 570 MB of Parquet files) are available from the corresponding author on
reasonable request.

## Repository layout

```
zcatcmp/                 algorithms, surrogates, problems, indicators, stages and analysis
  elm_loop.py            SN loop (SN-ELM, SN-RBF, SN-KRG)
  surrogates.py          ELM, RBF and Kriging surrogates and the training archive
  algorithms.py          SSA-*, KRVEA and NSGA-II on pysamoo/pymoo
  config.py              protocol: budget, operators, grids, seeds
run_experiments.py       runs a stage in parallel
select_configs.py        selects the configurations that pass to the next stage
analyze.py               statistical analysis of a stage
ei/make_assets.py        tables and figures of the manuscript
data/reference_fronts/   reference fronts of ZCAT1–20 with two and three objectives
tests/                   unit tests
```

Comments and messages in the code are written in Spanish.

## Contact

Matías Vergara, Universidad de Talca, Chile — mavergara19@utalca.cl
