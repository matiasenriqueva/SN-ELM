"""Lanza una etapa de la campaña en paralelo (reanudable).

Ejemplos (en el servidor, dentro de tmux y con el entorno ``zcat`` activo):

    python run_experiments.py --stage pilot --workers 47 --cpus 0-47
    python run_experiments.py --stage A --workers 47 --cpus 0-47
    python run_experiments.py --stage B --dry-run          # sólo cuenta trabajos

Las corridas ya terminadas con status 'ok' se saltan, así que basta volver a
ejecutar el mismo comando tras un corte.
"""
import os

# Un hilo de BLAS/OpenMP por proceso: el paralelismo es a nivel de proceso.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import logging
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from zcatcmp import config
from zcatcmp.runner import quiet_worker, run_job
from zcatcmp.stages import STAGES, build_jobs
from zcatcmp.storage import ResultWriter, completed_keys

HEARTBEAT_SECONDS = 600   # aviso en el log si en este lapso no terminó ninguna corrida


def parse_cpus(spec: str) -> set[int]:
    cpus = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def code_version() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=config.ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def setup_logging(stage: str) -> logging.Logger:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = logging.getLogger("campaign")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(config.LOGS_DIR / f"stage_{stage}.log", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--workers", type=int, default=max(1, min(47, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--cpus", help="CPUs permitidos, p. ej. 0-47 (Linux; lo heredan los procesos)")
    ap.add_argument("--M", type=int, nargs="+", default=list(config.OBJECTIVES))
    ap.add_argument("--problems", type=int, nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--limit", type=int, help="ejecuta sólo los primeros N trabajos pendientes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.cpus:
        os.sched_setaffinity(0, parse_cpus(args.cpus))

    log = setup_logging(args.stage)
    out_dir = config.RESULTS_DIR / args.stage
    filters = dict(objectives=args.M, problems=args.problems, seeds=args.seeds, variants=args.variants)
    jobs = build_jobs(args.stage, **filters, code_version=code_version())
    done = completed_keys(out_dir)
    pending = [j for j in jobs if j["key"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    by_variant = Counter(j["variant"] for j in pending)
    log.info(f"etapa {args.stage}: {len(jobs)} trabajos, {len(jobs) - len(pending)} ya hechos, "
             f"{len(pending)} pendientes | {args.workers} procesos | código {jobs[0]['code_version'] if jobs else '-'}")
    for variant, count in sorted(by_variant.items()):
        log.info(f"  {variant:12s} {count}")
    if args.dry_run or not pending:
        return

    writer = ResultWriter(out_dir, flush_every=args.flush_every)
    status = Counter()
    t0 = last_report = last_completion = time.monotonic()
    executor = ProcessPoolExecutor(max_workers=args.workers, initializer=quiet_worker)
    # pkill (SIGTERM) se trata como Ctrl+C: guarda lo terminado antes de salir.
    # Los procesos de cálculo restauran el comportamiento por defecto en quiet_worker.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        futures = {executor.submit(run_job, job): job for job in pending}
        waiting, i = set(futures), 0
        while waiting:
            # timeout: permite escribir un aviso aunque ninguna corrida termine (Kriging tarda mucho)
            finished, waiting = wait(waiting, timeout=60, return_when=FIRST_COMPLETED)
            for future in finished:
                i += 1
                job = futures[future]
                try:
                    row = future.result()
                except Exception as exc:                  # fallo del proceso, no del algoritmo
                    log.info(f"FALLO {job['key']}: {exc!r}")
                    status["crash"] += 1
                    continue
                writer.add(row)
                status[row["status"]] += 1
                if row["status"] != "ok":
                    first_line = (row["error"] or "").strip().splitlines()[-1:] or [""]
                    log.info(f"{row['status'].upper()} {job['key']} n_evals={row['n_evals']} {first_line[0]}")
            now = time.monotonic()
            elapsed = now - t0
            if finished:
                last_completion = now
                if now - last_report >= 60 or not waiting:
                    eta = elapsed / i * (len(pending) - i)
                    log.info(f"[{i}/{len(pending)}] {dict(status)} | transcurrido {fmt_duration(elapsed)} "
                             f"| restante ~{fmt_duration(eta)}")
                    last_report = now
            elif now - max(last_report, last_completion) >= HEARTBEAT_SECONDS:
                running = min(args.workers, len(waiting))
                log.info(f"[{i}/{len(pending)}] en curso: {running} corridas en ejecución; la última terminó "
                         f"hace {int((now - last_completion) // 60)} min | transcurrido {fmt_duration(elapsed)}")
                last_report = now
    except KeyboardInterrupt:
        # Ctrl+C o pkill: no esperar a que terminen las corridas en curso (Kriging tarda
        # mucho); se pierden sólo esas y al relanzar se vuelven a correr.
        log.info("interrumpido: guardando lo terminado y deteniendo los procesos...")
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()
    finally:
        path = writer.flush()
        if path:
            log.info(f"guardado {path.name}")
    executor.shutdown(wait=True)
    log.info(f"fin etapa {args.stage}: {dict(status)}")


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    main()
