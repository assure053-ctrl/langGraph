"""
스킬 시스템: MD 파일을 읽어 ReAct 에이전트에 동적으로 절차를 주입

지원 트리거 방식 (하이브리드):
  ① 키워드 자동 매칭     - 메시지에 trigger 단어가 있으면 본문 자동 주입
  ② load_skill 도구       - 모델이 직접 본문 로드
  ③ 명시적 호출           - 사용자가 "/skill <이름>" 또는 "[스킬:이름]"로 지정

MD 형식:
  ---
  name: 스킬 이름
  description: 한 줄 설명
  trigger: [키워드1, 키워드2, ...]
  ---

  # 본문 (Markdown 자유 형식)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


SKILLS_DIR = Path(
    os.environ.get(
        "SKILLS_DIR",
        str(Path(__file__).resolve().parent / "skills"),
    )
)
SKILLS_DIR.mkdir(exist_ok=True)


# ==========================================
# 데이터 모델
# ==========================================
@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    body: str = ""
    path: Optional[Path] = None


# ==========================================
# Frontmatter 파서 (PyYAML 안 쓰고 가볍게)
# ==========================================
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """간단한 YAML 부분집합 파서 (name/description/trigger 만 지원)."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_raw, body = m.group(1), m.group(2)
    fm: dict = {}
    for line in fm_raw.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # 리스트 [a, b, c] 형식
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            items = [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
            fm[key] = items
        else:
            # 따옴표 제거
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            fm[key] = val
    return fm, body


# ==========================================
# 레지스트리
# ==========================================
class SkillRegistry:
    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir
        self.skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> int:
        """skills/*.md 전체를 다시 로드. 봇 실행 중 핫리로드 가능."""
        self.skills.clear()
        for path in sorted(self.skills_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                fm, body = _parse_frontmatter(text)
                name = (fm.get("name") or path.stem).strip()
                desc = (fm.get("description") or "").strip()
                triggers = fm.get("trigger") or fm.get("triggers") or []
                if isinstance(triggers, str):
                    triggers = [t.strip() for t in triggers.split(",") if t.strip()]
                skill = Skill(
                    name=name,
                    description=desc,
                    triggers=[t.lower() for t in triggers],
                    body=body.strip(),
                    path=path,
                )
                # 같은 name이 두 번 등록되면 마지막 파일이 이김
                self.skills[name] = skill
            except Exception as e:
                print(f"⚠️ 스킬 로드 실패 {path}: {e}")
        return len(self.skills)

    def get(self, name: str) -> Optional[Skill]:
        # 정확 일치 → 부분 일치 순으로 탐색
        if name in self.skills:
            return self.skills[name]
        # 공백/대소문자 무시
        norm = re.sub(r"\s+", "", name).lower()
        for k, v in self.skills.items():
            if re.sub(r"\s+", "", k).lower() == norm:
                return v
        for k, v in self.skills.items():
            if norm in re.sub(r"\s+", "", k).lower():
                return v
        return None

    def match_by_keyword(self, text: str) -> list[Skill]:
        """메시지에 trigger 키워드가 하나라도 들어있는 스킬을 반환 (중복 제거)."""
        if not text:
            return []
        low = text.lower()
        matched: list[Skill] = []
        seen: set[str] = set()
        for skill in self.skills.values():
            if skill.name in seen:
                continue
            for kw in skill.triggers:
                if kw and kw in low:
                    matched.append(skill)
                    seen.add(skill.name)
                    break
        return matched

    def list_summary(self) -> str:
        """system prompt에 넣을 한 줄 요약 목록."""
        if not self.skills:
            return "(등록된 스킬 없음)"
        lines = []
        for s in self.skills.values():
            trig = ", ".join(s.triggers[:5]) if s.triggers else "(키워드 없음)"
            lines.append(f"- {s.name}: {s.description} [키워드: {trig}]")
        return "\n".join(lines)


# 전역 레지스트리 (telegram_agent에서 import)
registry = SkillRegistry()


# ==========================================
# Tool 함수들
# ==========================================
@tool
async def load_skill(name: str) -> str:
    """이름으로 스킬의 절차 본문 전체를 반환합니다.
    list_skills로 사용 가능한 이름을 먼저 확인할 수 있습니다.
    예: load_skill("네이버 주식 분석")
    """
    skill = registry.get(name)
    if not skill:
        return f"ERROR '{name}' 스킬 없음. list_skills로 확인하세요."
    return (
        f"[스킬: {skill.name}]\n"
        f"설명: {skill.description}\n"
        f"--- 절차 본문 ---\n{skill.body}"
    )


@tool
async def list_skills() -> str:
    """사용 가능한 모든 스킬의 이름·설명·트리거 키워드 목록을 반환합니다."""
    return registry.list_summary()


@tool
async def reload_skills() -> str:
    """skills/ 폴더의 MD 파일을 다시 로드합니다 (스킬 추가/수정 후 사용)."""
    n = registry.reload()
    return f"OK {n}개 스킬 재로드됨"


SKILL_TOOLS = [load_skill, list_skills, reload_skills]


# ==========================================
# 명시적 호출 파서: "/skill 이름" 또는 "[스킬:이름] 본문"
# ==========================================
EXPLICIT_PATTERNS = [
    re.compile(r"^/skill\s+(.+?)(?:\s+(.+))?$", re.DOTALL),
    re.compile(r"^\[스킬:\s*(.+?)\]\s*(.*)$", re.DOTALL),
]


def parse_explicit(text: str) -> tuple[Optional[Skill], str]:
    """사용자가 명시적으로 스킬을 지정했는지 파싱.
    반환: (Skill or None, 남은 본문)
    """
    text = text.strip()
    for pat in EXPLICIT_PATTERNS:
        m = pat.match(text)
        if m:
            name = m.group(1).strip()
            rest = (m.group(2) or "").strip()
            skill = registry.get(name)
            return skill, rest
    return None, text
