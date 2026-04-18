from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from typing import Any


def _stable_json(obj: Any) -> str:
    def _default(o: Any):
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    return json.dumps(obj, sort_keys=True, default=_default, separators=(",", ":"))


def compute_config_hash(cfg) -> str:
    payload = _stable_json(asdict(cfg))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def append_run_to_ledger(
    *,
    ledger_path: str,
    run_id: str,
    timestamp: str,
    out_dir: str,
    cfg,
    performance: dict,
    extra: dict | None = None,
) -> None:
    """Append (or upsert) a run row into a SQLite ledger.

    This is best-effort and should never crash the pipeline.
    """

    if not ledger_path:
        return

    os.makedirs(os.path.dirname(os.path.abspath(ledger_path)), exist_ok=True)

    cfg_json = _stable_json(asdict(cfg))
    cfg_hash = compute_config_hash(cfg)
    perf_json = _stable_json(performance)
    extra_json = _stable_json(extra or {})

    conn = sqlite3.connect(ledger_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                out_dir TEXT PRIMARY KEY,
                run_id TEXT,
                timestamp TEXT,
                config_hash TEXT,
                config_json TEXT,
                performance_json TEXT,
                extra_json TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT OR REPLACE INTO runs (
                out_dir, run_id, timestamp, config_hash, config_json, performance_json, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (out_dir, run_id, timestamp, cfg_hash, cfg_json, perf_json, extra_json),
        )
        conn.commit()
    finally:
        conn.close()
