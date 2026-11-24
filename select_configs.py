"""Elige las configuraciones que pasan a la siguiente etapa.

    python select_configs.py --stage A      # -> results/selection_A.json (mejores modelos ELM)
    python select_configs.py --stage B      # -> results/selection_B.json (mejor config. por variante)
"""
import argparse
import json

import pandas as pd

from zcatcmp import config
from zcatcmp.selection import average_ranks, select
from zcatcmp.stages import selection_path
from zcatcmp.storage import load_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("A", "B"))
    ap.add_argument("--metric", default="hv", choices=("hv", "igdp", "ep"))
    ap.add_argument("--top", type=int, help="cuántas configuraciones pasan (por defecto config.TOP_K_*)")
    args = ap.parse_args()

    df = load_results(config.RESULTS_DIR / args.stage,
                      columns=["variant", "config_id", "config", "problem", "M", "seed", "status", args.metric])
    if df.empty:
        raise SystemExit(f"no hay resultados en {config.RESULTS_DIR / args.stage}")

    with pd.option_context("display.max_rows", 200, "display.width", 160):
        for M, g in average_ranks(df, args.metric).groupby("M"):
            print(f"\n=== M={M} ({args.metric}) ===")
            print(g.groupby("variant").head(5).to_string(index=False))

    chosen = select(df, args.stage, args.metric, args.top)
    path = selection_path(args.stage)
    path.write_text(json.dumps(chosen, indent=2))
    print(f"\nguardado {path}\n{json.dumps(chosen, indent=2)}")


if __name__ == "__main__":
    main()
