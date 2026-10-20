"""답변 수치 근거 검증 (Groundedness Check) — LLM이 지어낸 숫자를 코드로 잡아낸다.

왜 필요한가
  운영 장애 보고에서 가장 위험한 실패는 '그럴듯하게 지어낸 숫자'다. 원본 코드의 주석에도
  실측 사례가 남아 있다: cache-service를 조사하지 않았는데 답변에 "CPU 53.5%"가 등장.
  그런데 기존 validate 노드는 "30자 이상 + '1)' 포함"이라는 형식만 검사했다.

방식 (LLM에게 다시 묻지 않고 결정론적 코드로 판정 — gate·replan과 같은 설계 원칙)
  1) 답변에서 '주장성 수치'만 뽑는다
       - 소수점이 있거나(2.4), 단위가 붙었거나(92%, 1.0s, 5분), 10보다 큰 수(504)
       - 절차 번호 "1)", 시스템명 "web-01"의 01 같은 10 이하 맨숫자는 주장이 아니므로 제외
  2) LLM에게 준 사실(사용자 질문 + 모니터링 값·알림 + runbook + 분석)의 모든 숫자와 대조
       - 반올림 인용은 허용: 85.2 → "85%" / "85.2%" 모두 근거 있음
  3) 근거 없는 수치가 있으면 그 숫자를 짚어 반려 → 재작성

주의: 파이썬 정규식에서 한글은 \\w로 취급된다(개선 1에서 겪은 문제). "약85%"처럼 한글 뒤에
숫자가 붙어도 뽑히도록 앞쪽 경계는 영문·숫자만 검사한다.
"""
import json
import os
import re

CLAIM_UNITS = {"%", "ms", "s", "초", "분", "시간", "gb", "mb", "건", "회", "배"}

_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])"                      # p95의 95, v1.2의 2 같은 부분 숫자 제외
    r"(\d+(?:\.\d+)?)"
    r"\s?(%|ms|s(?![A-Za-z])|초|분|시간|GB|MB|gb|mb|건|회|배)?")


def enabled() -> bool:
    """GROUNDING_CHECK=off 로 기동하면 개선 전(형식만 검사) 동작을 재현한다."""
    return os.environ.get("GROUNDING_CHECK", "on") != "off"


def extract_claims(text: str) -> list[tuple[str, float, int]]:
    """(원문 표기, 값, 소수 자릿수) 목록. 주장성 수치만 반환."""
    out = []
    for m in _NUMBER.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        value = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if decimals == 0 and not unit and value <= 10:
            continue                            # 절차 번호·작은 맨숫자는 주장이 아님
        out.append((m.group(0).strip(), value, decimals))
    return out


def fact_numbers(fact_text: str) -> set[float]:
    return {float(m.group(1)) for m in _NUMBER.finditer(fact_text)}


def _grounded(value: float, decimals: int, facts: set[float]) -> bool:
    for v in facts:
        if abs(value - v) < 1e-9:
            return True
        if decimals == 0 and round(v) == value:        # 85.2 → "85%"
            return True
        if decimals == 1 and round(v, 1) == value:     # 85.24 → "85.2%"
            return True
    return False


def build_fact_text(query: str, collected: list[dict], writer_prompt: str | None = None) -> str:
    """LLM이 답변을 쓸 때 받은 '근거'를 하나의 텍스트로 만든다.

    writer_prompt(답변 작성 LLM이 실제로 받은 프롬프트)가 있으면 반드시 포함한다.
    실측 교훈: 작성 프롬프트에는 알림 목록이 "알림 0건"처럼 '개수'로 가공돼 들어가는데,
    원본 데이터(빈 목록 [])만 보고 대조했더니 "0건"을 근거 없는 수치로 잘못 잡았다
    → 멀쩡한 답변이 2번 반려되고 응답이 100초로 늘어난 거짓 양성. 검증 기준은
    '모델이 실제로 본 것'과 같아야 한다."""
    parts = [query]
    for c in collected:
        data = c.get("data")
        parts.append(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False))
        if isinstance(data, dict) and isinstance(data.get("alerts"), list):
            parts.append(f"알림 {len(data['alerts'])}건")   # 프롬프트 없이 검증할 때의 안전망
    if writer_prompt:
        parts.append(writer_prompt)
    return "\n".join(parts)


def find_ungrounded(answer: str, fact_text: str) -> list[str]:
    facts = fact_numbers(fact_text)
    seen, out = set(), []
    for raw, value, decimals in extract_claims(answer):
        if not _grounded(value, decimals, facts) and raw not in seen:
            seen.add(raw)
            out.append(raw)
    return out


WARNING_PREFIX = "\n\n⚠️ 확인 필요:"


def warning_line(ungrounded: list[str]) -> str:
    return (f"{WARNING_PREFIX} 다음 수치는 수집된 모니터링·runbook 데이터에서 확인되지 "
            f"않았습니다 — {', '.join(ungrounded)}")


def strip_warning(answer: str) -> str:
    return answer.split(WARNING_PREFIX)[0]
