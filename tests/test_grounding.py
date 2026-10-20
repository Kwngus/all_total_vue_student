"""답변 수치 근거 검증 단위 테스트."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import grounding  # noqa: E402
from agent.graph import finalize_node, validate_node  # noqa: E402

COLLECTED = [
    {"source": "monitoring",
     "data": {"system_name": "auth-service", "status": "degraded", "cpu_percent": 85.2,
              "memory_percent": 64.6,
              "alerts": ["p95 latency 2.4s (threshold 1.0s)", "connection pool 92% used"]}},
    {"source": "runbook", "data": "2. 커넥션 풀 사용률이 90%를 넘으면 풀 크기를 늘린다.\n"
                                  "4. 5분 내 p95가 임계치(1.0s) 아래로 내려오지 않으면 ..."},
]
QUERY = "auth-service 로그인 응답이 느려졌어요"
FACTS = grounding.build_fact_text(QUERY, COLLECTED)


def test_fabricated_numbers_are_caught():
    """원본 코드 주석에 기록된 실측 사례: 조사하지 않은 값 'CPU 53.5%'를 지어냄."""
    answer = "1) CPU 53.5%, 응답 3.7s로 과부하 2) 커넥션 풀 92% 3) 정상화 확인"
    assert grounding.find_ungrounded(answer, FACTS) == ["53.5%", "3.7s"]


@pytest.mark.parametrize("answer", [
    "1) CPU 85.2%, 메모리 64.6% 2) p95 2.4s (임계 1.0s) 3) 풀 92%",
    "1) CPU 약 85% 2) 메모리 65% 수준 3) 풀 사용률 92%",            # 반올림 인용 허용
    "1) 커넥션 풀이 90%를 넘었으므로 2) 5분 내 개선 없으면 3) 캐시로 분산",  # runbook 수치
])
def test_grounded_answers_pass(answer):
    assert grounding.find_ungrounded(answer, FACTS) == []


def test_step_numbers_and_system_names_are_not_claims():
    claims = [c[0] for c in grounding.extract_claims("1) web-01 점검 2) 3단계 진행 3) p95 확인")]
    assert claims == []


def test_korean_prefix_does_not_hide_numbers():
    """'약85%'처럼 한글 바로 뒤의 숫자도 뽑혀야 한다 (\\w 경계 함정)."""
    assert grounding.find_ungrounded("CPU가 약77%입니다", FACTS) == ["77%"]


def test_derived_alert_count_is_not_a_false_positive():
    """실측 거짓 양성 회귀 테스트: 정상 시스템(알림 [])에 대해 '알림 0건'이라 쓴 답변."""
    healthy = [{"source": "monitoring",
                "data": {"system_name": "web-01", "status": "healthy", "alerts": []}}]
    answer = "1) web-01은 알림 0건으로 정상 2) 추가 조치 불필요 3) 모니터링 유지"
    assert grounding.find_ungrounded(answer, grounding.build_fact_text("web-01 점검", healthy)) == []


def test_writer_prompt_is_part_of_evidence():
    prompt = "## 수집된 사실\n- 모니터링 x: CPU 41.9%, 알림 0건"
    assert grounding.find_ungrounded("1) CPU 41.9% 2) 알림 0건 3) 정상",
                                     grounding.build_fact_text("q", [], prompt)) == []


def test_numbers_from_user_query_are_grounded():
    facts = grounding.build_fact_text("log-service 디스크 95% 찼어요", [])
    assert grounding.find_ungrounded("1) 디스크 95% 사용 중", facts) == []


# ── 그래프 노드 동작 ───────────────────────────────────────────────────
def _state(draft, attempts=0):
    return {"query": QUERY, "collected": COLLECTED, "draft": draft,
            "validation_attempts": attempts, "events": []}


def test_validate_rejects_with_specific_feedback():
    out = asyncio.run(validate_node(_state("1) CPU 53.5%로 과부하 2) 풀 확장 3) 문서화"), None))
    assert out["validation_feedback"] and "53.5%" in out["validation_feedback"]
    assert out["events"][-1]["ungrounded"] == ["53.5%"]


def test_after_max_attempts_answer_is_flagged_not_silently_passed():
    out = asyncio.run(validate_node(_state("1) CPU 53.5%로 과부하 2) 풀 확장 3) 문서화", 2), None))
    assert out["validation_feedback"] is None                 # 3회째 → 통과 처리
    fin = asyncio.run(finalize_node({**_state("1) CPU 53.5%로 과부하 2) 풀 확장 3) 문서화"),
                                     "ungrounded": out["ungrounded"]}, None))
    assert "⚠️ 확인 필요" in fin["final"] and "53.5%" in fin["final"]


def test_check_can_be_disabled_for_before_comparison(monkeypatch):
    monkeypatch.setenv("GROUNDING_CHECK", "off")
    out = asyncio.run(validate_node(_state("1) CPU 53.5%로 과부하 2) 풀 확장 3) 문서화"), None))
    assert out["validation_feedback"] is None and out["ungrounded"] == []
