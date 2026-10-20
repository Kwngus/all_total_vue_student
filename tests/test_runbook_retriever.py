"""runbook 검색기 단위 테스트 — 임베딩은 가짜(결정론) 객체로 대체해 네트워크 없이 검증한다."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import select_runbooks  # noqa: E402
from agent.runbook_retriever import (RunbookRetriever, alerts_from_collected,  # noqa: E402
                                     split_by_system)


def run(coro):
    return asyncio.run(coro)


def mon(name, status, alerts):
    return {"source": "monitoring",
            "data": {"system_name": name, "status": status, "alerts": alerts}}


class KeywordEmbedder:
    """runbook 이름에 해당하는 축에 1을 찍는 가짜 임베딩 — '의미 점수'가 섞이는지만 검증."""
    name = "fake:keyword"
    AXES = ["CPU", "DB", "디스크", "지연", "메모리", "네트워크"]

    async def embed(self, texts):
        return [[1.0 if a in t else 0.0 for a in self.AXES] + [0.01] for t in texts]


class BrokenEmbedder:
    name = "fake:broken"

    async def embed(self, texts):
        raise ConnectionError("ollama not running")


# ── 어휘 검색 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("query, expected", [
    ("web-01 디스크가 꽉 찼어요", "disk-full"),            # 키워드 방식은 db-down으로 오분류
    ("web-01 CPU 급증 원인 진단해줘", "cpu-spike"),        # 키워드 방식은 db-down으로 오분류
    ("auth-service 메모리 누수가 의심돼요", "memory-leak"),  # 키워드 방식은 알림 때문에 high-latency
])
def test_user_query_drives_selection(query, expected):
    r = run(RunbookRetriever().search(query, []))
    assert r.selected[0] == expected
    assert r.method == "lexical"


def test_english_fragments_do_not_create_noise():
    """'log'·'failover'를 글자 단위로 쪼개던 버그 회귀 테스트 (db-down 오분류)."""
    r = run(RunbookRetriever().search("log-service 로그가 쌓여서 /var/log 용량이 95%예요", []))
    assert r.selected[0] == "disk-full"


# ── 복합 장애 ──────────────────────────────────────────────────────────
def test_split_by_system():
    seg = split_by_system("log-service 디스크 부족이랑 gateway-service 타임아웃",
                          [{"system_name": "gateway-service", "alerts": ["a"]},
                           {"system_name": "log-service", "alerts": ["b"]}])
    assert seg["log-service"][0] == "log-service 디스크 부족이랑 "
    assert seg["gateway-service"] == ("gateway-service 타임아웃", ["a"])


def test_multi_incident_selects_two_runbooks():
    col = [mon("payments-service", "degraded", ["CPU 91% (threshold 80%)"]),
           mon("cache-service", "degraded", ["memory usage climbing steadily over 6h"])]
    r = run(RunbookRetriever().search_collected(
        "payments-service CPU 급증, cache-service는 메모리가 계속 올라가요", col))
    assert set(r.selected) == {"cpu-spike", "memory-leak"}


def test_healthy_and_unknown_systems_are_not_evidence():
    col = [mon("web-01", "healthy", []),
           mon("billing", "unknown", ["System 'billing' not found in monitoring"]),
           mon("log-service", "degraded", ["disk usage 95% on /var/log"])]
    assert alerts_from_collected(col) == ["disk usage 95% on /var/log"]


# ── 하이브리드 / 장애 시 대체 동작 ─────────────────────────────────────
def test_hybrid_mode_when_embedding_available():
    r = RunbookRetriever(KeywordEmbedder())
    run(r.build())
    assert r.method == "hybrid"
    res = run(r.search("메모리가 계속 올라가요", []))
    assert res.method == "hybrid" and res.selected[0] == "memory-leak"


def test_falls_back_to_lexical_when_embedding_fails():
    """Ollama가 꺼져 있어도 검색이 멈추지 않는다 (graceful degradation)."""
    r = RunbookRetriever(BrokenEmbedder())
    run(r.build())
    assert r.method == "lexical"
    assert run(r.search("web-01 디스크가 꽉 찼어요", [])).selected[0] == "disk-full"


def test_graph_uses_keyword_rules_when_no_retriever():
    """RUNBOOK_RETRIEVER=keyword (개선 전) 경로가 그대로 살아 있다."""
    out = run(select_runbooks("web-01 디스크가 꽉 찼어요", [mon("web-01", "healthy", [])], None))
    assert out == {"selected": ["db-down"], "method": "keyword", "scores": {}}
