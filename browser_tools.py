"""
Playwright 기반 브라우저 도구 (ReAct 에이전트용)

- 모든 도구가 동일한 브라우저/페이지 인스턴스를 공유 (BrowserManager 싱글톤)
- LangChain의 @tool 데코레이터로 ReAct 에이전트가 사용 가능한 형태로 노출
- 모델이 도구를 선택해서 호출할 수 있도록 docstring은 자세히 작성
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from playwright.async_api import async_playwright, Browser, BrowserContext, Page


HEADLESS = os.environ.get("BROWSER_HEADLESS", "0") == "1"
SCREENSHOT_DIR = Path(__file__).resolve().parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)


class BrowserManager:
    """Playwright 브라우저 싱글톤. 모든 tool 호출이 같은 페이지를 공유."""

    def __init__(self) -> None:
        self._pw = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._lock = asyncio.Lock()

    async def ensure_page(self) -> Page:
        async with self._lock:
            if self._page is None or self._page.is_closed():
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=HEADLESS)
                self._context = await self._browser.new_context(
                    viewport={"width": 1280, "height": 900},
                    locale="ko-KR",
                )
                self._page = await self._context.new_page()
            return self._page

    async def shutdown(self) -> None:
        try:
            if self._browser:
                await self._browser.close()
            if self._pw:
                await self._pw.stop()
        finally:
            self._browser = None
            self._context = None
            self._page = None
            self._pw = None


_mgr = BrowserManager()


# ==========================================
# 헬퍼
# ==========================================
def _trunc(s: str, n: int = 3000) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n...(생략 {len(s) - n}자)"


# ==========================================
# Tool 함수들 (모두 ReAct 에이전트에게 노출됨)
# ==========================================
@tool
async def browser_open(url: str) -> str:
    """URL을 브라우저로 엽니다. URL은 http(s)://로 시작해야 합니다.
    페이지 제목과 현재 URL을 반환합니다.
    예: browser_open("https://finance.naver.com")"""
    page = await _mgr.ensure_page()
    if not re.match(r"^https?://", url):
        url = "https://" + url
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(0.8)
        title = await page.title()
        return f"OK. 현재 URL: {page.url}\n페이지 제목: {title}"
    except Exception as e:
        return f"ERROR 페이지 열기 실패: {e}"


@tool
async def browser_url() -> str:
    """현재 브라우저에서 열려있는 페이지의 URL과 제목을 반환합니다."""
    page = await _mgr.ensure_page()
    return f"URL: {page.url}\n제목: {await page.title()}"


@tool
async def browser_get_text(max_chars: int = 3000) -> str:
    """현재 페이지의 가시 텍스트(body innerText)를 반환합니다.
    내용이 너무 길 수 있으니 필요할 때만 호출하세요.
    max_chars(기본 3000)로 잘려서 반환됩니다."""
    page = await _mgr.ensure_page()
    text = await page.evaluate("() => document.body ? document.body.innerText : ''")
    return _trunc(text, max_chars)


@tool
async def browser_get_links(limit: int = 30) -> str:
    """현재 페이지의 클릭 가능한 링크 목록을 '번호 | 텍스트 | URL' 형식으로 반환합니다.
    이후 browser_click_link(번호)로 클릭할 수 있습니다."""
    page = await _mgr.ensure_page()
    links = await page.evaluate(
        r"""
        (lim) => {
            const out = [];
            const seen = new Set();
            for (const a of document.querySelectorAll('a[href]')) {
                const text = (a.innerText || '').trim().replace(/\s+/g, ' ');
                const href = a.href;
                if (!text || text.length < 2 || href.startsWith('javascript:')) continue;
                const key = text + '|' + href;
                if (seen.has(key)) continue;
                seen.add(key);
                out.push({ text: text.slice(0, 80), href });
                if (out.length >= lim) break;
            }
            return out;
        }
        """,
        limit,
    )
    if not links:
        return "(링크 없음)"
    return "\n".join(f"{i+1}. {l['text']}  →  {l['href']}" for i, l in enumerate(links))


@tool
async def browser_click_link(index: int) -> str:
    """browser_get_links로 얻은 번호의 링크를 클릭합니다 (1부터 시작)."""
    page = await _mgr.ensure_page()
    try:
        result = await page.evaluate(
            r"""
            (idx) => {
                const arr = [];
                const seen = new Set();
                for (const a of document.querySelectorAll('a[href]')) {
                    const text = (a.innerText || '').trim().replace(/\s+/g, ' ');
                    const href = a.href;
                    if (!text || text.length < 2 || href.startsWith('javascript:')) continue;
                    const key = text + '|' + href;
                    if (seen.has(key)) continue;
                    seen.add(key);
                    arr.push(a);
                    if (arr.length === idx) {
                        a.click();
                        return { ok: true, text: text.slice(0, 80), href };
                    }
                }
                return { ok: false, count: arr.length };
            }
            """,
            index,
        )
        if not result.get("ok"):
            return f"ERROR 해당 번호 링크 없음 (총 {result.get('count', 0)}개)"
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
        await asyncio.sleep(0.6)
        return f"OK 클릭: {result['text']}\n이동 후 URL: {page.url}"
    except Exception as e:
        return f"ERROR 링크 클릭 실패: {e}"


@tool
async def browser_click_text(text: str) -> str:
    """페이지에서 보이는 텍스트와 일치하는 요소(버튼/링크)를 클릭합니다.
    Playwright의 텍스트 셀렉터를 사용합니다.
    예: browser_click_text("로그인")"""
    page = await _mgr.ensure_page()
    try:
        await page.get_by_text(text, exact=False).first.click(timeout=5000)
        await asyncio.sleep(0.6)
        return f"OK '{text}' 클릭됨. 현재 URL: {page.url}"
    except Exception as e:
        return f"ERROR 클릭 실패: {e}"


@tool
async def browser_type(selector: str, text: str, press_enter: bool = False) -> str:
    """CSS 셀렉터로 input/textarea를 찾아 텍스트를 입력합니다.
    press_enter=True면 Enter도 누릅니다.
    예: browser_type("input[name='query']", "SK하이닉스", press_enter=True)"""
    page = await _mgr.ensure_page()
    try:
        await page.fill(selector, text, timeout=5000)
        if press_enter:
            await page.press(selector, "Enter")
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            await asyncio.sleep(0.6)
        return f"OK '{selector}'에 '{text}' 입력 완료"
    except Exception as e:
        return f"ERROR 입력 실패: {e}"


@tool
async def browser_get_inputs() -> str:
    """페이지의 input/textarea 필드를 셀렉터와 함께 나열합니다.
    이후 browser_type(셀렉터, 텍스트)로 입력할 수 있습니다."""
    page = await _mgr.ensure_page()
    inputs = await page.evaluate(
        r"""
        () => {
            const out = [];
            const els = document.querySelectorAll('input, textarea');
            for (const el of els) {
                const t = (el.tagName || '').toLowerCase();
                if (el.type === 'hidden') continue;
                const sel =
                    (el.id ? `#${el.id}` :
                     el.name ? `${t}[name="${el.name}"]` :
                     null);
                out.push({
                    selector: sel,
                    type: el.type || t,
                    name: el.name || '',
                    placeholder: el.placeholder || '',
                });
                if (out.length >= 20) break;
            }
            return out;
        }
        """
    )
    if not inputs:
        return "(입력 필드 없음)"
    return "\n".join(
        f"{i+1}. selector={x['selector']}  type={x['type']}  "
        f"name='{x['name']}'  placeholder='{x['placeholder']}'"
        for i, x in enumerate(inputs)
    )


@tool
async def browser_press(key: str) -> str:
    """현재 포커스된 요소에 키를 입력합니다.
    예: 'Enter', 'Tab', 'Escape', 'ArrowDown'"""
    page = await _mgr.ensure_page()
    try:
        await page.keyboard.press(key)
        await asyncio.sleep(0.3)
        return f"OK '{key}' 키 입력"
    except Exception as e:
        return f"ERROR 키 입력 실패: {e}"


@tool
async def browser_back() -> str:
    """브라우저 뒤로가기. 이전 페이지로 이동."""
    page = await _mgr.ensure_page()
    try:
        await page.go_back(wait_until="domcontentloaded", timeout=10000)
        return f"OK 뒤로가기. 현재 URL: {page.url}"
    except Exception as e:
        return f"ERROR 뒤로가기 실패: {e}"


@tool
async def browser_screenshot() -> str:
    """현재 페이지의 스크린샷을 PNG로 저장하고 파일 경로를 반환합니다."""
    page = await _mgr.ensure_page()
    fname = f"shot_{int(time.time())}.png"
    path = SCREENSHOT_DIR / fname
    await page.screenshot(path=str(path), full_page=False)
    return f"OK 스크린샷 저장: {path}"


@tool
async def browser_evaluate(js_expression: str) -> str:
    """페이지 컨텍스트에서 JavaScript 표현식을 실행하고 결과를 반환합니다.
    예: browser_evaluate("document.title")
    주의: 'return' 키워드 없이 표현식만 쓰세요."""
    page = await _mgr.ensure_page()
    try:
        result = await page.evaluate(f"() => {{ return ({js_expression}); }}")
        return _trunc(str(result), 2000)
    except Exception as e:
        return f"ERROR JS 실행 실패: {e}"


# 외부에서 import 할 도구 목록
BROWSER_TOOLS = [
    browser_open,
    browser_url,
    browser_get_text,
    browser_get_links,
    browser_click_link,
    browser_click_text,
    browser_type,
    browser_get_inputs,
    browser_press,
    browser_back,
    browser_screenshot,
    browser_evaluate,
]


async def shutdown_browser() -> None:
    """앱 종료 시 호출"""
    await _mgr.shutdown()
