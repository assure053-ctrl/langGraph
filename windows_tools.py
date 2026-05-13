"""
Windows 제어 도구 (ReAct 에이전트용)

- PowerShell 실행이 핵심. 모델이 자연어 요청 → PowerShell 명령으로 실행
- 일부 유틸리티 (앱 열기, 클립보드, 파일 목록) 별도 도구 제공
- 시스템 보호를 위해 일부 위험한 패턴은 거부
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
from pathlib import Path

from langchain_core.tools import tool


# 명확히 막을 패턴 (실수/모델 폭주 방지). 사용자가 시스템 망가뜨릴 가능성 있는 명령.
DANGEROUS_PATTERNS = [
    r"\bformat\s+[a-z]:",                   # format C:
    r"\bremove-item\b.*-recurse.*c:\\",     # 시스템 드라이브 재귀 삭제
    r"\brm\s+-rf\s+[/c]",                   # rm -rf /
    r"shutdown\s+/[sr]",                    # 시스템 종료/재부팅
    r"diskpart",
]


def _is_dangerous(cmd: str) -> bool:
    low = cmd.lower()
    return any(re.search(p, low) for p in DANGEROUS_PATTERNS)


def _trunc(s: str, n: int = 2500) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n...(생략 {len(s) - n}자)"


# ==========================================
# Tool 함수들
# ==========================================
@tool
async def run_powershell(command: str, timeout_sec: int = 60) -> str:
    """Windows PowerShell 명령을 실행하고 표준출력+표준에러를 반환합니다.

    예시:
      - 디렉터리 목록: 'Get-ChildItem C:\\Users'
      - 현재 시각: 'Get-Date'
      - 프로세스 목록: 'Get-Process | Select-Object -First 10'
      - 파일 내용 읽기: 'Get-Content C:\\path\\to\\file.txt -TotalCount 20'

    파괴적 명령(format, shutdown, 시스템 드라이브 재귀삭제 등)은 거부됩니다.
    timeout_sec: 명령 최대 실행 시간 (기본 60초)
    """
    if _is_dangerous(command):
        return "ERROR 위험한 명령으로 판단되어 거부되었습니다."

    def _run() -> str:
        try:
            r = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-Command", command],
                capture_output=True, text=True, timeout=timeout_sec,
                encoding="utf-8", errors="replace",
            )
            out = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            parts = []
            if out:
                parts.append("[stdout]\n" + out)
            if err:
                parts.append("[stderr]\n" + err)
            parts.append(f"[exit_code] {r.returncode}")
            return _trunc("\n".join(parts))
        except subprocess.TimeoutExpired:
            return f"ERROR 명령이 {timeout_sec}초를 초과해 종료되었습니다."
        except Exception as e:
            return f"ERROR PowerShell 실행 실패: {e}"

    return await asyncio.to_thread(_run)


@tool
async def open_program(name: str) -> str:
    """이름으로 Windows 프로그램을 실행합니다.
    예: 'notepad', 'calc', 'explorer', 'chrome', 'cmd'
    PATH에 등록된 실행파일이거나 start에서 찾을 수 있어야 합니다."""

    def _run() -> str:
        try:
            # start 명령으로 실행 (비동기, 셸 통해)
            subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", f"Start-Process {name}"],
                shell=False,
            )
            return f"OK '{name}' 실행 요청 완료"
        except Exception as e:
            return f"ERROR 실행 실패: {e}"

    return await asyncio.to_thread(_run)


@tool
async def list_directory(path: str = "C:\\Users") -> str:
    """지정한 디렉터리의 파일/폴더 목록을 반환합니다 (최대 50개)."""

    def _run() -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"ERROR 경로 없음: {path}"
            if not p.is_dir():
                return f"ERROR 디렉터리가 아님: {path}"
            items = []
            for i, child in enumerate(sorted(p.iterdir())):
                if i >= 50:
                    items.append("... (50개 초과 생략)")
                    break
                kind = "DIR " if child.is_dir() else "FILE"
                items.append(f"{kind}  {child.name}")
            return f"경로: {p}\n" + "\n".join(items)
        except Exception as e:
            return f"ERROR: {e}"

    return await asyncio.to_thread(_run)


@tool
async def read_file(path: str, max_lines: int = 100) -> str:
    """텍스트 파일의 내용을 읽어 반환합니다 (최대 max_lines줄)."""

    def _run() -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"ERROR 파일 없음: {path}"
            if not p.is_file():
                return f"ERROR 파일이 아님: {path}"
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"... (이상 {max_lines}줄 초과 생략)")
                        break
                    lines.append(line.rstrip("\n"))
            return f"파일: {p}\n--- 내용 ({len(lines)}줄) ---\n" + "\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"

    return await asyncio.to_thread(_run)


@tool
async def write_file(path: str, content: str, append: bool = False) -> str:
    """텍스트 파일에 내용을 씁니다. append=True면 이어쓰기, 기본은 덮어쓰기.
    상위 디렉터리가 없으면 자동 생성됩니다."""

    def _run() -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(p, mode, encoding="utf-8") as f:
                f.write(content)
            return f"OK 파일 {'추가 저장' if append else '저장'}: {p} ({len(content)}자)"
        except Exception as e:
            return f"ERROR: {e}"

    return await asyncio.to_thread(_run)


@tool
async def get_clipboard() -> str:
    """현재 Windows 클립보드의 텍스트를 반환합니다."""

    def _run() -> str:
        try:
            r = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            return _trunc((r.stdout or "").strip() or "(클립보드 비어있음)")
        except Exception as e:
            return f"ERROR: {e}"

    return await asyncio.to_thread(_run)


@tool
async def set_clipboard(text: str) -> str:
    """주어진 텍스트를 Windows 클립보드에 복사합니다."""

    def _run() -> str:
        try:
            # 파이프로 텍스트 전달
            p = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-Command", "Set-Clipboard"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            p.communicate(input=text.encode("utf-8"), timeout=10)
            return f"OK 클립보드에 {len(text)}자 복사됨"
        except Exception as e:
            return f"ERROR: {e}"

    return await asyncio.to_thread(_run)


# 외부에서 import 할 도구 목록
WINDOWS_TOOLS = [
    run_powershell,
    open_program,
    list_directory,
    read_file,
    write_file,
    get_clipboard,
    set_clipboard,
]
