"""
Telegram 봇 ↔ LangGraph ReAct 에이전트 (하이브리드 스킬 + 영구 메모리)

기능:
- 스킬 시스템: skills/*.md 파일 정의, 3가지 트리거 방식
    ① 키워드 자동 매칭   - 메시지에 trigger 단어 있으면 본문 자동 주입
    ② load_skill 도구    - 모델이 직접 호출
    ③ 명시적 지정        - /skill <이름> ... 또는 [스킬:이름] ...
- 영구 메모리:
    ① 대화 히스토리      - SqliteSaver (chat_id별 영구 보존)
    ② 사실 메모리        - remember/recall/list_memories/forget 도구
- 도구 카테고리:
    브라우저 12 + 윈도우 7 + 스킬 3 + 메모리 4 = 총 26개
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import os
import sqlite3
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import aiosqlite

# Windows cp949 출력 깨짐 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters,
)

from browser_tools import BROWSER_TOOLS, shutdown_browser
from windows_tools import WINDOWS_TOOLS
import memory_store
from memory_store import MEMORY_TOOLS, current_chat_id, init_db as init_memory_db
import skill_loader
from skill_loader import SKILL_TOOLS, registry as skill_registry


# ==========================================
# 설정
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://192.168.0.201:11435/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3:14b")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "300"))
MEMORY_DB_PATH = Path(
    os.environ.get(
        "MEMORY_DB_PATH",
        str(Path(__file__).resolve().parent / "agent_memory.db"),
    )
)
ALLOWED_CHAT_IDS = {
    int(x) for x in os.environ.get("ALLOWED_CHAT_IDS", "").split(",") if x.strip().isdigit()
}

# 로그 설정
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = Path(
    os.environ.get(
        "LOG_FILE",
        str(Path(__file__).resolve().parent / "agent.log"),
    )
)

if not TELEGRAM_TOKEN:
    print("❌ TELEGRAM_BOT_TOKEN 이 .env 에 설정되지 않았습니다.")
    print("   .env.example 을 참고하여 .env 파일을 만들어 주세요.")
    sys.exit(1)


def _setup_logging() -> logging.Logger:
    """콘솔 + 회전 파일 핸들러로 로그 출력."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    # 기존 핸들러 제거 (중복 출력 방지)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            str(LOG_FILE), maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except Exception as e:
        print(f"⚠️ 로그 파일 설정 실패: {e}")

    # httpx의 요청별 INFO 로그를 줄임 (너무 시끄러움)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger("telegram-agent")


log = _setup_logging()


# ==========================================
# 명시적 스킬 호출 추적용 contextvar
# ==========================================
current_explicit_skill: contextvars.ContextVar = contextvars.ContextVar(
    "current_explicit_skill", default=None
)


# ==========================================
# 시스템 프롬프트
# ==========================================
BASE_SYSTEM_PROMPT = """당신은 Windows PC를 제어하는 자율 에이전트입니다.
사용자가 한국어로 명령하면, 사용 가능한 도구들을 조합해 작업을 수행하세요.

핵심 원칙:
1. **도구를 적극 활용하세요.** 모르는 것은 추측하지 말고 도구로 직접 확인.
2. **사실 메모리 활용**: 작업 시작 전 recall로 관련 컨텍스트를 확인하세요.
   사용자가 자신에 관한 정보(이름, 선호, 자주 쓰는 경로/종목 등)를 말하면
   remember로 저장해 미래에 활용하세요.
3. **스킬 시스템**: 현재 메시지에 관련 스킬 절차가 자동 첨부됐으면 그것을 따르세요.
   첨부 안 됐는데 필요하면 load_skill("이름")으로 절차를 로드.
4. **브라우저 작업 순서**: browser_open → (browser_get_inputs/browser_get_links)
   → browser_type / browser_click_* → browser_get_text 로 결과 확인.
5. **Windows 작업**: PowerShell = run_powershell, 프로그램 = open_program,
   파일 = read_file/write_file/list_directory.
6. **응답은 간결하게**: 작업 결과만 한글로 깔끔히 보고. 도구 호출 과정 생략.
7. **위험한 작업** (시스템 종료, 디스크 포맷, 시스템 디렉터리 삭제 등)은 거부.
8. **한 번에 한 도구씩** 호출하고 결과를 확인한 뒤 다음을 결정.
"""


def _last_user_text(messages: list) -> str:
    """messages 리스트에서 마지막 사용자 메시지 텍스트를 추출."""
    for m in reversed(messages):
        # langchain_core HumanMessage
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
        # dict 형식 fallback
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
        if hasattr(m, "type") and m.type == "human":
            return getattr(m, "content", "") or ""
    return ""


def make_prompt(state) -> list:
    """매 LLM 호출 시 동적으로 system prompt 생성.
    - 명시적 스킬 (contextvar) → 본문 주입
    - 키워드 매칭 스킬 → 본문 주입
    - 스킬 목록 요약 항상 첨부 → 모델이 load_skill 호출 가능
    """
    messages = state.get("messages") if isinstance(state, dict) else state.messages
    last_text = _last_user_text(messages)

    skills_to_inject = []

    # ① 명시적
    explicit = current_explicit_skill.get()
    if explicit is not None:
        skills_to_inject.append(explicit)

    # ② 키워드 매칭
    for s in skill_registry.match_by_keyword(last_text):
        if s not in skills_to_inject:
            skills_to_inject.append(s)

    parts = [BASE_SYSTEM_PROMPT]

    if skills_to_inject:
        parts.append("\n=== 이번 메시지에 적용할 스킬 절차 ===")
        for s in skills_to_inject:
            parts.append(f"\n[스킬: {s.name}]\n설명: {s.description}\n절차:\n{s.body}")

    parts.append("\n=== 등록된 스킬 목록 (필요시 load_skill로 본문 호출) ===")
    parts.append(skill_registry.list_summary())

    system_content = "\n".join(parts)
    return [SystemMessage(content=system_content)] + list(messages)


# ==========================================
# LLM + 도구 + 체크포인터
# ==========================================
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key="ollama",
        model=MODEL_NAME,
        temperature=0,
        timeout=LLM_TIMEOUT,
    )


TOOLS = BROWSER_TOOLS + WINDOWS_TOOLS + SKILL_TOOLS + MEMORY_TOOLS

# 에이전트와 체크포인터는 비동기 환경에서 초기화 (AsyncSqliteSaver는 async 전용)
# 호환을 위해 모듈 레벨 placeholder + 비동기 빌더 제공
_state: dict = {"agent": None, "aio_conn": None}


async def init_agent() -> None:
    """비동기 컨텍스트에서 에이전트와 체크포인터를 초기화.
    텔레그램의 post_init에서 호출되며, CLI 테스트도 동일 함수 사용."""
    if _state["agent"] is not None:
        return
    conn = await aiosqlite.connect(str(MEMORY_DB_PATH))
    checkpointer = AsyncSqliteSaver(conn=conn)
    _state["aio_conn"] = conn
    _state["agent"] = create_react_agent(
        model=build_llm(),
        tools=TOOLS,
        prompt=make_prompt,
        checkpointer=checkpointer,
    )


def get_agent():
    """이미 초기화된 에이전트 반환. 미초기화 시 RuntimeError."""
    if _state["agent"] is None:
        raise RuntimeError("agent 미초기화 — init_agent()를 먼저 호출하세요.")
    return _state["agent"]


# ==========================================
# Telegram 핸들러
# ==========================================
def _authorized(chat_id: int) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    return chat_id in ALLOWED_CHAT_IDS


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    log.info(f"/start from chat_id={cid}")
    if not _authorized(cid):
        await update.message.reply_text(f"❌ 권한 없음. chat_id={cid}")
        return
    msg = (
        f"🤖 LangGraph ReAct 에이전트\n"
        f"모델: {MODEL_NAME}\n"
        f"도구 {len(TOOLS)}개 (브라우저 {len(BROWSER_TOOLS)} + 윈도우 {len(WINDOWS_TOOLS)} "
        f"+ 스킬 {len(SKILL_TOOLS)} + 메모리 {len(MEMORY_TOOLS)})\n"
        f"스킬 {len(skill_registry.skills)}개 로드됨\n\n"
        f"자연어로 명령하면 모델이 알아서 도구를 선택합니다.\n\n"
        f"명령:\n"
        f"/skills - 등록된 스킬 목록\n"
        f"/skill <이름> <내용> - 특정 스킬 강제 적용\n"
        f"/reload - 스킬 폴더 재로드\n"
        f"/memory - 저장된 사실 메모리 보기\n"
        f"/tools - 모든 도구 목록\n"
        f"/reset - 대화 히스토리 초기화"
    )
    await update.message.reply_text(msg)


async def cmd_skills(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if not _authorized(cid):
        return
    text = "🧩 등록된 스킬:\n\n" + skill_registry.list_summary()
    text += "\n\n사용법: /skill <이름> <요청내용>\n예: /skill 네이버주식분석 SK하이닉스"
    await update.message.reply_text(text[:4000])


async def cmd_reload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if not _authorized(cid):
        return
    n = skill_registry.reload()
    await update.message.reply_text(f"🔄 스킬 재로드: {n}개")


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if not _authorized(cid):
        return
    # 메모리는 contextvar 의존이므로 직접 SQL 조회
    with sqlite3.connect(str(MEMORY_DB_PATH)) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT key, value FROM memories WHERE chat_id = ? "
            "ORDER BY updated_at DESC",
            (cid,),
        ).fetchall()
    if not rows:
        await update.message.reply_text("📭 저장된 사실 메모리 없음")
        return
    lines = ["🧠 사실 메모리:"]
    for r in rows:
        v = r["value"]
        if len(v) > 100:
            v = v[:100] + "..."
        lines.append(f"• {r['key']}: {v}")
    await update.message.reply_text("\n".join(lines)[:4000])


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if not _authorized(cid):
        return
    # 새 thread_id로 분기 (이전 히스토리는 DB에 남지만 사용 안 함)
    context.chat_data["thread_suffix"] = context.chat_data.get("thread_suffix", 0) + 1
    await update.message.reply_text("🧹 대화 히스토리 초기화 (사실 메모리는 유지)")


async def cmd_tools(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if not _authorized(cid):
        return
    lines = ["🔧 도구 목록:"]
    for t in TOOLS:
        doc = (t.description or "").splitlines()[0]
        lines.append(f"• {t.name} — {doc}")
    msg = "\n".join(lines)
    for i in range(0, len(msg), 4000):
        await update.message.reply_text(msg[i:i + 4000])


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    if not _authorized(cid):
        log.warning(f"비인증 chat_id={cid} 메시지 무시")
        return

    user_msg = (update.message.text or "").strip()
    if not user_msg:
        return

    # 명시적 스킬 호출 파싱 ([스킬:이름] 또는 "/skill 이름 ..." 슬래시 명령은
    # 텔레그램이 잘라서 args만 넘기지만, 본문에도 [스킬:이름] 형식 허용)
    explicit_skill, remaining = skill_loader.parse_explicit(user_msg)
    if explicit_skill:
        log.info(f"[chat_id={cid}] 명시적 스킬: {explicit_skill.name}")
        user_msg = remaining or f"위 '{explicit_skill.name}' 스킬 절차를 시작해 주세요."

    # 키워드 매칭 사전 로그 (디버그용)
    keyword_matches = skill_registry.match_by_keyword(user_msg)
    if keyword_matches:
        log.info(f"[chat_id={cid}] 키워드 매칭 스킬: "
                 f"{[s.name for s in keyword_matches]}")

    log.info(f"[chat_id={cid}] 입력: {user_msg[:200]}")

    # contextvar 세팅 (도구가 chat_id와 명시적 스킬을 알 수 있도록)
    chat_id_token = current_chat_id.set(cid)
    explicit_token = current_explicit_skill.set(explicit_skill)

    await context.bot.send_chat_action(chat_id=cid, action=ChatAction.TYPING)

    suffix = context.chat_data.get("thread_suffix", 0)
    thread_id = f"chat-{cid}-{suffix}"

    try:
        # 이번 invoke 전 메시지 개수 (이번 턴 신규 메시지만 골라 로그)
        prev_count = 0
        try:
            snap = await get_agent().aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            prev_count = len(snap.values.get("messages", []) or [])
        except Exception:
            pass

        result = await get_agent().ainvoke(
            {"messages": [{"role": "user", "content": user_msg}]},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 30},
        )

        # 이번 턴에 추가된 메시지만 단계별 로그 (ReAct 루프 가시화)
        new_messages = result["messages"][prev_count:]
        log.info(f"[chat_id={cid}] === ReAct 단계 ({len(new_messages)}개 신규) ===")
        for i, m in enumerate(new_messages, 1):
            role = m.__class__.__name__
            tcs = getattr(m, "tool_calls", None)
            content_preview = getattr(m, "content", "") or ""
            if isinstance(content_preview, str):
                content_preview = content_preview.replace("\n", " ")[:200]
            if tcs:
                for tc in tcs:
                    args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)[:300]
                    log.info(f"  [{i}|{role}] 🔧 {tc['name']}({args_str})")
            elif role == "ToolMessage":
                tname = getattr(m, "name", "?")
                log.info(f"  [{i}|{role}] ↳ {tname} → {content_preview}")
            elif content_preview:
                log.info(f"  [{i}|{role}] 💬 {content_preview}")
        log.info(f"[chat_id={cid}] === 루프 종료 ===")

        last = result["messages"][-1]
        content = getattr(last, "content", str(last))
        if not content:
            content = "(빈 응답)"
    except Exception as e:
        log.exception("에이전트 실행 오류")
        content = f"⚠️ 에이전트 오류: {e}"
    finally:
        current_chat_id.reset(chat_id_token)
        current_explicit_skill.reset(explicit_token)

    # Telegram 메시지 4096자 제한 → 4000자 단위로 분할
    for i in range(0, len(content), 4000):
        await update.message.reply_text(content[i:i + 4000])


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    log.exception("Unhandled error", exc_info=context.error)


# ==========================================
# 메인
# ==========================================
def main() -> None:
    # 메모리 DB 초기화 (memories 테이블 생성)
    init_memory_db()

    print(f"🚀 Telegram ReAct 에이전트 시작 (하이브리드 스킬 + 영구 메모리)")
    print(f"   LLM:  {MODEL_NAME} @ {LLM_BASE_URL}")
    print(f"   DB:   {MEMORY_DB_PATH}")
    print(f"   도구: {len(TOOLS)}개 "
          f"(브라우저 {len(BROWSER_TOOLS)} + 윈도우 {len(WINDOWS_TOOLS)} "
          f"+ 스킬 {len(SKILL_TOOLS)} + 메모리 {len(MEMORY_TOOLS)})")
    print(f"   스킬: {len(skill_registry.skills)}개 로드됨")
    for s in skill_registry.skills.values():
        print(f"         - {s.name}: {s.description}")
    if ALLOWED_CHAT_IDS:
        print(f"   허용 chat_id: {ALLOWED_CHAT_IDS}")
    else:
        print(f"   ⚠️ ALLOWED_CHAT_IDS 미설정 — 누구든 봇과 대화 가능")

    async def _post_init(_app):
        await init_agent()
        log.info("✅ ReAct 에이전트 + AsyncSqliteSaver 초기화 완료")

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(_post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("skills", cmd_skills))
    app.add_handler(CommandHandler("reload", cmd_reload))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("tools", cmd_tools))
    app.add_handler(CommandHandler("reset", cmd_reset))
    # /skill <name> <text>  → 본문이 [스킬:name] text 로 변환되어 handle_text 로
    app.add_handler(
        MessageHandler(filters.Regex(r"^/skill\s+\S+"), handle_text)
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(on_error)

    async def _post_shutdown(_app):
        await shutdown_browser()
        try:
            if _state["aio_conn"] is not None:
                await _state["aio_conn"].close()
        except Exception:
            pass
    app.post_shutdown = _post_shutdown

    print("✅ 봇 폴링 시작. Ctrl+C 로 종료.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
