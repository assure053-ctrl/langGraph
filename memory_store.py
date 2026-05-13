"""
장기 사실 메모리 (Long-term fact memory)

- SQLite에 chat_id별로 격리된 key-value 사실 저장
- ReAct 에이전트에 remember / recall / list_memories / forget 도구 노출
- chat_id는 contextvars로 도구에 전달 (텔레그램 핸들러에서 세팅)
"""
from __future__ import annotations

import contextvars
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


# 핸들러에서 매 메시지마다 세팅; 도구가 읽음
current_chat_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "current_chat_id", default=None
)


DB_PATH = Path(
    os.environ.get(
        "MEMORY_DB_PATH",
        str(Path(__file__).resolve().parent / "agent_memory.db"),
    )
)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, isolation_level=None)  # autocommit
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """앱 시작 시 1회 호출. memories 테이블이 없으면 생성."""
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                chat_id    INTEGER NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (chat_id, key)
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_chat ON memories(chat_id)"
        )


def _require_chat_id() -> int:
    cid = current_chat_id.get()
    if cid is None:
        # CLI 테스트용 fallback
        return 0
    return cid


# ==========================================
# Tool 함수들 (ReAct 에이전트가 호출)
# ==========================================
@tool
async def remember(key: str, value: str) -> str:
    """사용자에 관한 사실, 선호도, 컨텍스트를 장기 메모리에 저장합니다.
    같은 key로 다시 저장하면 덮어씁니다.

    사용 예:
      remember("관심종목", "SK하이닉스, 한미반도체")
      remember("작업 폴더", "C:\\\\Users\\\\lkh\\\\projects")
      remember("선호 모델", "qwen3:14b")

    사용자가 자신의 이름, 선호, 자주 쓰는 정보 등을 말하면 이 도구로 저장하세요.
    """
    cid = _require_chat_id()
    now = int(time.time())
    with _conn() as c:
        c.execute(
            """
            INSERT INTO memories(chat_id, key, value, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (cid, key, value, now, now),
        )
    return f"OK 기억함: {key} = {value}"


@tool
async def recall(query: str = "") -> str:
    """장기 메모리에서 query와 관련된 사실을 검색해 반환합니다.
    query를 비우면 모든 메모리를 반환합니다.
    query에 단어가 있으면 key 또는 value에 그 단어가 포함된 항목만 반환.

    작업 시작 전에 관련 컨텍스트가 있는지 확인할 때 호출하세요.
    예: 종목 분석 요청 → recall("종목") → 사용자의 관심 종목 확인
    """
    cid = _require_chat_id()
    with _conn() as c:
        if query.strip():
            like = f"%{query.strip()}%"
            rows = c.execute(
                """
                SELECT key, value FROM memories
                WHERE chat_id = ? AND (key LIKE ? OR value LIKE ?)
                ORDER BY updated_at DESC
                """,
                (cid, like, like),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT key, value FROM memories WHERE chat_id = ? "
                "ORDER BY updated_at DESC",
                (cid,),
            ).fetchall()

    if not rows:
        return "(관련 메모리 없음)"
    return "\n".join(f"- {r['key']}: {r['value']}" for r in rows)


@tool
async def list_memories() -> str:
    """저장된 모든 메모리의 key 목록을 반환합니다."""
    cid = _require_chat_id()
    with _conn() as c:
        rows = c.execute(
            "SELECT key, length(value) AS vlen, updated_at FROM memories "
            "WHERE chat_id = ? ORDER BY updated_at DESC",
            (cid,),
        ).fetchall()
    if not rows:
        return "(저장된 메모리 없음)"
    lines = ["저장된 메모리 키:"]
    for r in rows:
        lines.append(f"- {r['key']} ({r['vlen']}자)")
    return "\n".join(lines)


@tool
async def forget(key: str) -> str:
    """장기 메모리에서 지정한 key의 항목을 삭제합니다."""
    cid = _require_chat_id()
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM memories WHERE chat_id = ? AND key = ?",
            (cid, key),
        )
        deleted = cur.rowcount
    return f"OK 삭제 {deleted}개: {key}" if deleted else f"(키 '{key}' 없음)"


MEMORY_TOOLS = [remember, recall, list_memories, forget]
