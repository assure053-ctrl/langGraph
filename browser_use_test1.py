"""
LangGraph + Playwright 하이브리드 주식 분석 에이전트

설계 원칙:
- 브라우저 컨트롤: Playwright 직접 사용 (LLM 미개입 → 빠르고 결정적)
- LLM 호출: 마지막 요약 단계 1회만 (tool calling 불필요 → 느린 로컬 모델도 OK)
- 오케스트레이션: LangGraph 노드 4개로 흐름 제어

흐름:
  START → search_stock → foreign_trading → news → summarize → END
"""
import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Optional, List, Dict

# Windows cp949 환경에서도 한글/이모지 출력이 깨지지 않도록 UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from playwright.async_api import async_playwright, Page
from langgraph.graph import StateGraph, START, END
from openai import AsyncOpenAI


# ==========================================
# 1. 환경 설정
# ==========================================
UBUNTU_SERVER_URL = "http://192.168.0.201:11435/v1"
MODEL_NAME = "gemma4:31b"
LLM_TIMEOUT_SEC = 600

# HEADLESS=1 환경변수면 브라우저 숨김 (CI/테스트용), 기본은 시각화 모드
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

# 결과 보고서 저장 경로 (스크립트와 같은 디렉터리)
RESULT_FILE = Path(__file__).resolve().parent / "browser_use_test1_result.txt"


class AgentState(TypedDict, total=False):
    stock_name: str
    stock_code: Optional[str]
    foreign_trading: Optional[List[Dict[str, str]]]
    news_titles: Optional[List[str]]
    result: Optional[str]
    error: Optional[str]


# 모든 노드가 같은 페이지를 공유 (각 노드가 다른 페이지를 띄우지 않음)
_ctx: Dict[str, object] = {}


# ==========================================
# 2. LangGraph 노드들
# ==========================================
async def search_stock_node(state: AgentState):
    """네이버 금융 메인 → 검색창에 종목명 입력 → 종목코드 추출"""
    stock_name = state["stock_name"]
    print(f"\n🌐 [1/4] '{stock_name}' 검색 중...")

    page: Page = _ctx["page"]
    await page.goto("https://finance.naver.com/", wait_until="domcontentloaded")
    await asyncio.sleep(0.8)

    # 검색창에 직접 입력 → 사용자가 시각적으로 확인 가능
    try:
        # 네이버 금융 상단 검색창의 input
        search_selectors = [
            'input[name="query"]',
            'input#stock_items',
            'input.snb_search_text',
        ]
        filled = False
        for sel in search_selectors:
            try:
                await page.fill(sel, stock_name, timeout=2000)
                await page.press(sel, "Enter")
                filled = True
                break
            except Exception:
                continue

        if not filled:
            raise RuntimeError("검색창을 찾지 못함")

        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(1.5)
    except Exception as e:
        print(f"   ⚠️ 검색창 사용 실패, 직접 검색 URL로 fallback ({e})")
        await page.goto(
            f"https://finance.naver.com/search/searchList.naver?query={stock_name}",
            wait_until="domcontentloaded",
        )
        await asyncio.sleep(1)

    # URL에서 종목코드 추출
    m = re.search(r"code=(\d{6})", page.url)
    if not m:
        # 검색결과 페이지에 있다면 첫 결과 클릭
        try:
            await page.locator("a[href*='code=']").first.click(timeout=3000)
            await page.wait_for_load_state("domcontentloaded")
            m = re.search(r"code=(\d{6})", page.url)
        except Exception:
            pass

    if not m:
        return {"error": f"종목코드 추출 실패: {page.url}"}

    code = m.group(1)
    print(f"   ✅ 종목코드: {code}")
    return {"stock_code": code}


async def foreign_trading_node(state: AgentState):
    """외국인/기관 매매 동향 페이지에서 최근 5거래일 데이터 추출"""
    code = state.get("stock_code")
    if not code:
        return {"error": "stock_code 없음"}

    print(f"\n📈 [2/4] 외국인·기관 매매 동향 수집 중...")
    page: Page = _ctx["page"]
    await page.goto(
        f"https://finance.naver.com/item/frgn.naver?code={code}",
        wait_until="domcontentloaded",
    )
    await asyncio.sleep(1.2)

    # 테이블에서 5일치 데이터 추출
    rows = await page.evaluate(
        r"""
        () => {
            const isDate = s => /^\d{4}\.\d{2}\.\d{2}/.test((s || '').trim());
            const out = [];
            const tables = document.querySelectorAll('table');
            for (const t of tables) {
                const text = t.innerText || '';
                if (!(text.includes('외국인') && text.includes('기관'))) continue;
                const trs = t.querySelectorAll('tr');
                for (const tr of trs) {
                    const tds = tr.querySelectorAll('td');
                    if (tds.length < 7) continue;
                    const cells = Array.from(tds).map(td => td.innerText.trim().replace(/\s+/g, ' '));
                    if (isDate(cells[0])) out.push(cells);
                    if (out.length >= 5) break;
                }
                if (out.length) break;
            }
            return out;
        }
        """
    )

    if not rows:
        return {"error": "외국인 매매 데이터 추출 실패"}

    # 컬럼 매핑: 날짜, 종가, 전일비, 등락률, 거래량, 기관순매수, 외국인순매수, 외국인보유주식, 보유비율
    cols = ["날짜", "종가", "전일비", "등락률", "거래량",
            "기관순매수", "외국인순매수", "외국인보유주식", "보유비율"]
    structured = []
    for row in rows:
        d = {cols[i]: row[i] for i in range(min(len(cols), len(row)))}
        structured.append(d)

    print(f"   ✅ {len(structured)}일치 데이터")
    for r in structured:
        print(f"      {r.get('날짜')} | 종가 {r.get('종가')} | "
              f"외국인 {r.get('외국인순매수')} | 기관 {r.get('기관순매수')}")

    return {"foreign_trading": structured}


async def news_node(state: AgentState):
    """뉴스·공시 페이지에서 최신 뉴스 제목 5개 추출 (iframe 내부 스크래핑)"""
    code = state.get("stock_code")
    if not code:
        return {"error": "stock_code 없음"}

    print(f"\n📰 [3/4] 최신 뉴스 수집 중...")
    page: Page = _ctx["page"]
    # 종목 메인 뉴스 페이지로 이동 (iframe 컨테이너)
    await page.goto(
        f"https://finance.naver.com/item/news.naver?code={code}",
        wait_until="domcontentloaded",
    )
    await asyncio.sleep(1.5)

    # news_frame iframe 안에서 뉴스 제목 추출
    target_frame = None
    for frame in page.frames:
        if "news_news.naver" in frame.url:
            target_frame = frame
            break

    if target_frame is None:
        return {"error": "news_frame iframe을 찾지 못함"}

    titles = await target_frame.evaluate(
        r"""
        () => {
            const seen = new Set();
            const out = [];
            for (const a of document.querySelectorAll('td.title a')) {
                const t = (a.innerText || '').trim().replace(/\s+/g, ' ');
                if (t && t.length > 3 && !seen.has(t)) {
                    seen.add(t);
                    out.push(t);
                }
                if (out.length >= 5) return out;
            }
            return out;
        }
        """
    )

    if not titles:
        return {"error": "뉴스 제목 추출 실패"}

    print(f"   ✅ 뉴스 {len(titles)}개")
    for i, t in enumerate(titles, 1):
        print(f"      {i}. {t}")

    return {"news_titles": titles}


async def summarize_node(state: AgentState):
    """수집한 데이터를 로컬 LLM으로 보고서 형식 요약 (LLM 호출 단 1회)"""
    print(f"\n🧠 [4/4] LLM 요약 생성 중 (모델: {MODEL_NAME})...")

    trading = state.get("foreign_trading") or []
    news = state.get("news_titles") or []

    if not trading and not news:
        return {"error": "요약할 데이터 없음"}

    trading_text = "\n".join(
        f"- {r.get('날짜')}: 종가 {r.get('종가')}, "
        f"외국인 순매수 {r.get('외국인순매수')}, 기관 순매수 {r.get('기관순매수')}"
        for r in trading
    ) or "(데이터 없음)"

    news_text = "\n".join(f"- {t}" for t in news) or "(뉴스 없음)"

    prompt = (
        f"다음은 '{state['stock_name']}'의 최근 거래일 외국인·기관 매매 데이터와 "
        f"최신 뉴스 제목입니다.\n\n"
        f"[매매 동향]\n{trading_text}\n\n"
        f"[최신 뉴스 제목]\n{news_text}\n\n"
        f"위 데이터를 바탕으로 한글 보고서를 두 섹션으로 작성:\n"
        f"1. 외국인·기관 매매 동향 요약 (3~4문장)\n"
        f"2. 뉴스 기반 시장 분위기 (긍정/부정 판단 + 근거, 3~4문장)\n"
    )

    client = AsyncOpenAI(
        base_url=UBUNTU_SERVER_URL,
        api_key="ollama",
        timeout=LLM_TIMEOUT_SEC,
    )
    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        summary = resp.choices[0].message.content
        return {"result": summary}
    except Exception as e:
        return {"error": f"LLM 호출 실패: {e}"}


# ==========================================
# 3. LangGraph 워크플로우
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("search", search_stock_node)
workflow.add_node("foreign_trading", foreign_trading_node)
workflow.add_node("news", news_node)
workflow.add_node("summarize", summarize_node)

workflow.add_edge(START, "search")
workflow.add_edge("search", "foreign_trading")
workflow.add_edge("foreign_trading", "news")
workflow.add_edge("news", "summarize")
workflow.add_edge("summarize", END)
app = workflow.compile()


# ==========================================
# 4. 결과 저장
# ==========================================
def save_report(stock_name: str, state: dict, file_path: Path) -> None:
    """수집 데이터 + LLM 요약을 사람이 읽기 좋은 텍스트로 저장"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trading = state.get("foreign_trading") or []
    news = state.get("news_titles") or []
    summary = state.get("result") or "(LLM 요약 없음)"
    error = state.get("error")

    lines = []
    lines.append("=" * 60)
    lines.append(f"📊 {stock_name} 분석 결과 보고서")
    lines.append(f"생성 시각: {now}")
    lines.append(f"모델: {MODEL_NAME} @ {UBUNTU_SERVER_URL}")
    lines.append(f"종목코드: {state.get('stock_code', '-')}")
    lines.append("=" * 60)

    lines.append("\n[1] 최근 거래일 외국인·기관 매매 동향")
    lines.append("-" * 60)
    if trading:
        header = (f"{'날짜':<12}{'종가':>12}{'등락률':>10}"
                  f"{'외국인순매수':>16}{'기관순매수':>14}")
        lines.append(header)
        for r in trading:
            lines.append(
                f"{r.get('날짜', '-'):<12}"
                f"{r.get('종가', '-'):>12}"
                f"{r.get('등락률', '-'):>10}"
                f"{r.get('외국인순매수', '-'):>16}"
                f"{r.get('기관순매수', '-'):>14}"
            )
    else:
        lines.append("(데이터 없음)")

    lines.append("\n[2] 최신 뉴스 제목")
    lines.append("-" * 60)
    if news:
        for i, t in enumerate(news, 1):
            lines.append(f"{i}. {t}")
    else:
        lines.append("(뉴스 없음)")

    lines.append("\n[3] LLM 요약 보고서")
    lines.append("-" * 60)
    lines.append(summary)

    if error:
        lines.append("\n[!] 처리 중 발생한 오류")
        lines.append("-" * 60)
        lines.append(str(error))

    lines.append("=" * 60)

    file_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n💾 결과 저장 완료: {file_path}")


# ==========================================
# 5. 메인
# ==========================================
async def main():
    print(f"🚀 LangGraph 에이전트 가동 (두뇌: {MODEL_NAME} @ {UBUNTU_SERVER_URL})")
    print(f"   브라우저 모드: {'headless' if HEADLESS else 'visible'}")

    inputs: AgentState = {"stock_name": "SK하이닉스"}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="ko-KR",
        )
        page = await context.new_page()
        _ctx["page"] = page

        final_state: dict = {}
        try:
            final_state = await app.ainvoke(inputs)

            if final_state.get("error"):
                print(f"\n⚠️ 실패: {final_state['error']}")
            else:
                print("\n" + "=" * 60)
                print(f"📊 {inputs['stock_name']} 분석 결과 보고서")
                print("=" * 60)
                print(final_state.get("result") or "(결과 없음)")
                print("=" * 60)
        finally:
            if final_state:
                save_report(inputs["stock_name"], final_state, RESULT_FILE)
            if not HEADLESS:
                await asyncio.sleep(3)  # 결과 페이지 확인 시간
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
