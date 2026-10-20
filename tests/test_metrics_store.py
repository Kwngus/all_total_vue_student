"""관리자 대시보드 지표 저장소 단위 테스트."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.metrics_store import MetricsStore  # noqa: E402


def make(tmp_path):
    m = MetricsStore(tmp_path / "m.db")
    base = dict(channel="chat", thread_id="t", backend="fake")
    m.record(**base, query="auth-service 장애", status="ok", latency_ms=1000, route="incident",
             cost_usd=0.01, runbooks=["high-latency"], runbook_method="hybrid", validation_attempts=2)
    m.record(**base, query="비번은 [MASKED_PASSWORD]", status="ok", latency_ms=3000, route="incident",
             cost_usd=0.02, masked=["PASSWORD"], runbooks=["disk-full", "network-timeout"],
             runbook_method="hybrid", validation_attempts=1, ungrounded=1)
    m.record(**base, query="안녕", status="ok", latency_ms=200, route="chat", cost_usd=0.001)
    m.record(**base, query="", status="rejected", latency_ms=1, code="PROMPT_INJECTION")
    m.record(**base, query="x", status="error", latency_ms=50, code="LLM_UNAVAILABLE", error_id="abcd1234")
    return m


def test_summary_counts_and_rates(tmp_path):
    s = make(tmp_path).summary(24)
    assert (s["total"], s["ok"], s["rejected"], s["error"]) == (5, 3, 1, 1)
    assert s["success_rate"] == 0.75            # 3 / (3 + 1) — 정책 거절은 분모에서 제외
    assert s["cost_usd"] == 0.031
    assert s["latency_p95_ms"] == 3000
    assert s["codes"] == {"PROMPT_INJECTION": 1, "LLM_UNAVAILABLE": 1}
    assert s["routes"] == {"incident": 2, "chat": 1}
    assert s["runbooks"]["high-latency"] == 1 and s["runbooks"]["network-timeout"] == 1
    assert s["masked_requests"] == 1 and s["masked_types"] == {"PASSWORD": 1}
    assert s["grounding_warnings"] == 1 and s["avg_validation_attempts"] == 1.5
    assert sum(b["ok"] + b["rejected"] + b["error"] for b in s["hourly"]) == 5


def test_recent_is_newest_first(tmp_path):
    rows = make(tmp_path).recent(2)
    assert [r["status"] for r in rows] == ["error", "rejected"]
    assert rows[0]["error_id"] == "abcd1234"


def test_record_never_raises(tmp_path):
    m = MetricsStore(tmp_path / "m.db")
    m.close()
    m.record(channel="chat", thread_id="t", query="q", status="ok", latency_ms=1, backend="fake")
