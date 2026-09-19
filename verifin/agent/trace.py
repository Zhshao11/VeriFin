"""执行轨迹入库。

为什么要存：Agent 的价值一半在于**可复盘**。
评测 V0/V1/V2 三版、做失败案例归档、解释「为什么这次拒答」，
都要能回放当时的每一步、每个工具调用的入参与返回值。

存的是单文件 SQLite，和检索索引同一个技术栈，零运维。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    question        TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    route           TEXT,
    decision        TEXT,
    reason          TEXT,
    steps           INTEGER DEFAULT 0,
    tool_calls      INTEGER DEFAULT 0,
    llm_calls       INTEGER DEFAULT 0,
    budget_exceeded INTEGER DEFAULT 0,
    planner_source  TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    node     TEXT NOT NULL,
    tool     TEXT,
    args     TEXT,
    ok       INTEGER,
    detail   TEXT,
    source   TEXT,
    PRIMARY KEY (run_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id);
"""

def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TraceStore:
    """追加写的轨迹库。"""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        self._con.executescript(_SCHEMA)
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    # ---------------------------------------------------------------- 写
    def start_run(self, run_id: str, question: str, *, route: str | None = None) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO runs(run_id, question, started_at, route)"
            " VALUES (?,?,?,?)",
            (run_id, question, utc_now(), route),
        )
        self._con.commit()

    def record_step(
        self,
        run_id: str,
        seq: int,
        *,
        node: str,
        tool: str | None,
        args: Mapping[str, Any] | None,
        ok: bool | None,
        detail: str,
        source: str,
    ) -> None:
        self._con.execute(
            "INSERT OR REPLACE INTO steps(run_id, seq, node, tool, args, ok, detail, source)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                seq,
                node,
                tool,
                json.dumps(args or {}, ensure_ascii=False),
                None if ok is None else int(ok),
                detail[:800],
                source,
            ),
        )

    def finish_run(
        self,
        run_id: str,
        *,
        decision: str,
        reason: str | None = None,
        steps: int = 0,
        tool_calls: int = 0,
        llm_calls: int = 0,
        budget_exceeded: bool = False,
        planner_source: str = "policy",
    ) -> None:
        self._con.execute(
            "UPDATE runs SET ended_at=?, decision=?, reason=?, steps=?, tool_calls=?,"
            " llm_calls=?, budget_exceeded=?, planner_source=? WHERE run_id=?",
            (
                utc_now(),
                decision,
                reason,
                steps,
                tool_calls,
                llm_calls,
                int(budget_exceeded),
                planner_source,
                run_id,
            ),
        )
        self._con.commit()

    def commit(self) -> None:
        self._con.commit()

    # ---------------------------------------------------------------- 读
    def runs(self) -> list[sqlite3.Row]:
        self._con.row_factory = sqlite3.Row
        return list(
            self._con.execute("SELECT * FROM runs ORDER BY started_at DESC")
        )

    def steps(self, run_id: str) -> list[sqlite3.Row]:
        self._con.row_factory = sqlite3.Row
        return list(
            self._con.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY seq", (run_id,)
            )
        )

    def abbreviated(self, run_id: str) -> str:
        """回放一条轨迹的可读形式，用于演示与失败案例归档。"""
        lines: list[str] = []
        for row in self.steps(run_id):
            mark = {1: "✓", 0: "✗", None: "·"}[row["ok"]]
            lines.append(
                f"  {row['seq']:>2}. [{mark}] {row['node']:<14}"
                f"{('→ ' + row['tool']) if row['tool'] else ''}"
                f"  ({row['source']}) {row['detail'][:60]}"
            )
        return "\n".join(lines)


def summarize_rows(rows: Iterable[Mapping[str, Any]], limit: int = 3) -> str:
    """把工具返回的多行结果压成一两句，供 LLM 调度时看（避免把全文塞进提示词）。"""
    items = list(rows)[:limit]
    return "; ".join(
        json.dumps({k: v for k, v in item.items() if k in ("label", "page", "value", "id")},
                   ensure_ascii=False)
        for item in items
    )
