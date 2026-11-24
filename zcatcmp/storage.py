"""Resultados en Parquet: un solo proceso escritor y pocos archivos grandes.

Pensado para un HDD: los procesos de cálculo no tocan el disco; el proceso
principal acumula filas y escribe un archivo ``part-*.parquet`` cada
``flush_every`` corridas (o cada ``flush_seconds``). Cada archivo se escribe con
un nombre temporal y se renombra al final, así un corte no deja archivos a medias.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_FLOAT_LIST = pa.list_(pa.float64())

SCHEMA = pa.schema([
    ("key", pa.string()), ("stage", pa.string()), ("variant", pa.string()),
    ("config_id", pa.string()), ("config", pa.string()),
    ("problem", pa.int16()), ("M", pa.int8()), ("seed", pa.int32()),
    ("status", pa.string()), ("error", pa.string()), ("n_evals", pa.int32()),
    ("cpu_time", pa.float64()), ("wall_time", pa.float64()),
    ("hv", pa.float64()), ("igdp", pa.float64()), ("ep", pa.float64()),
    ("curve_evals", pa.list_(pa.int32())), ("curve_hv", _FLOAT_LIST),
    ("curve_igdp", _FLOAT_LIST), ("curve_ep", _FLOAT_LIST),
    ("front_F", pa.list_(_FLOAT_LIST)), ("front_X", pa.list_(_FLOAT_LIST)),
    ("info", pa.string()), ("host", pa.string()), ("finished_at", pa.string()),
    ("code_version", pa.string()),
])


class ResultWriter:
    def __init__(self, out_dir: Path, flush_every: int = 200, flush_seconds: float = 600.0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.flush_seconds = flush_seconds
        self._buffer: list[dict] = []
        self._last_flush = time.monotonic()
        self._n_files = 0

    def add(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every or time.monotonic() - self._last_flush >= self.flush_seconds:
            self.flush()

    def flush(self) -> Path | None:
        self._last_flush = time.monotonic()
        if not self._buffer:
            return None
        table = pa.Table.from_pylist(self._buffer, schema=SCHEMA)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"part-{stamp}-{os.getpid()}-{self._n_files:05d}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = self.out_dir / f".{name}.tmp"
        pq.write_table(table, tmp, compression="zstd")
        final = self.out_dir / name
        os.replace(tmp, final)
        self._n_files += 1
        self._buffer.clear()
        return final


def part_files(out_dir: Path) -> list[Path]:
    return sorted(Path(out_dir).glob("part-*.parquet"))


def load_results(out_dir: Path, columns: list[str] | None = None):
    """DataFrame con todas las filas de una etapa (columnas opcionales)."""
    files = part_files(out_dir)
    if not files:
        return pa.Table.from_pylist([], schema=SCHEMA).select(columns or SCHEMA.names).to_pandas()
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files]).to_pandas()


def completed_keys(out_dir: Path) -> set[str]:
    """Claves con status 'ok' (los errores e incompletos se vuelven a correr)."""
    df = load_results(out_dir, columns=["key", "status"])
    return set(df.loc[df["status"] == "ok", "key"])
