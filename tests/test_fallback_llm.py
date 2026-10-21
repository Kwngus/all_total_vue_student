"""개선 7 — LLM 장애 자동 대체 + 서킷 브레이커 단위 테스트 (네트워크 없이 가짜 LLM으로)."""
import asyncio
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import fallback_llm  # noqa: E402
from agent.fallback_llm import CircuitBreaker, FallbackLLM, classify  # noqa: E402


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.HTTPStatusError("x", request=req, response=httpx.Response(status, request=req))


class StubLLM:
    def __init__(self, name, fail_with=None):
        self.name, self.fail_with, self.calls = name, fail_with, 0

    async def route(self, query):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return {"route": "incident", "needs_analysis": False}, {"model": self.name, "usd": 0.0}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _pair(primary_exc=None, threshold=3, cooldown=30.0, clock=None):
    primary, backup = StubLLM("openai:gpt-4o-mini", primary_exc), StubLLM("ollama:qwen3:0.6b")
    breaker = CircuitBreaker(threshold, cooldown, clock or Clock())
    return FallbackLLM(primary, backup, breaker, "fast"), primary, backup, breaker


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("exc,code", [
    (_http_error(503), "LLM_HTTP_503"), (_http_error(429), "LLM_RATE_LIMIT"),
    (_http_error(401), "LLM_AUTH"), (httpx.ConnectError("refused"), "LLM_UNREACHABLE"),
    (httpx.ReadTimeout("slow"), "TIMEOUT"), (RuntimeError("OPENAI_API_KEY 미설정"), "LLM_AUTH"),
])
def test_transient_failures_are_classified(exc, code):
    assert classify(exc) == code


@pytest.mark.parametrize("exc", [_http_error(400), ValueError("bug"), KeyError("x")])
def test_our_own_errors_are_not_fallen_back(exc):
    llm, primary, backup, breaker = _pair(exc)
    with pytest.raises(type(exc)):
        run(llm.route("q"))
    assert backup.calls == 0 and breaker.failures == 0   # 버그를 보조 모델로 덮지 않는다


def test_primary_ok_no_fallback():
    llm, primary, backup, _ = _pair()
    log = []
    fallback_llm.FALLBACK_LOG.set(log)
    result, usage = run(llm.route("q"))
    assert usage["model"].startswith("openai") and backup.calls == 0 and log == []


def test_primary_down_answers_from_backup_and_logs():
    llm, primary, backup, _ = _pair(httpx.ConnectError("refused"))
    log = []
    fallback_llm.FALLBACK_LOG.set(log)
    result, usage = run(llm.route("q"))
    assert usage["model"].startswith("ollama") and result["route"] == "incident"
    assert log[0]["reason"] == "LLM_UNREACHABLE" and log[0]["to"] == "ollama:qwen3:0.6b"


def test_breaker_opens_after_threshold_and_skips_primary():
    llm, primary, backup, breaker = _pair(_http_error(503), threshold=3)
    for _ in range(3):
        run(llm.route("q"))
    assert breaker.state == "open" and primary.calls == 3
    run(llm.route("q"))
    run(llm.route("q"))
    assert primary.calls == 3 and backup.calls == 5   # 열린 동안은 죽은 서버를 기다리지 않는다


def test_half_open_trial_recovers():
    clock = Clock()
    llm, primary, backup, breaker = _pair(_http_error(503), threshold=2, cooldown=30, clock=clock)
    run(llm.route("q")); run(llm.route("q"))
    assert breaker.state == "open"
    clock.t = 31
    assert breaker.state == "half_open"
    primary.fail_with = None   # 주 LLM 복구
    _, usage = run(llm.route("q"))
    assert usage["model"].startswith("openai") and breaker.state == "closed"


def test_half_open_trial_failure_reopens():
    clock = Clock()
    llm, primary, backup, breaker = _pair(_http_error(503), threshold=2, cooldown=30, clock=clock)
    run(llm.route("q")); run(llm.route("q"))
    clock.t = 31
    run(llm.route("q"))   # 시험 1건도 실패
    assert breaker.state == "open" and breaker.opened_at == 31


def test_both_down_raises_primary_cause():
    llm, primary, backup, _ = _pair(_http_error(503))
    backup.fail_with = httpx.ConnectError("ollama not running")
    with pytest.raises(httpx.HTTPStatusError):   # 기존 오류 처리(error_id)가 원래 원인으로 분류
        run(llm.route("q"))


def test_fast_and_quality_share_one_breaker():
    a, b = StubLLM("openai:mini", _http_error(503)), StubLLM("openai:4o")
    fast, quality, breaker = fallback_llm.wrap(a, b, StubLLM("ollama:s"), StubLLM("ollama:l"),
                                               threshold=2, cooldown_s=30)
    run(fast.route("q")); run(fast.route("q"))
    assert breaker.state == "open"
    run(quality.route("q"))
    assert b.calls == 0   # fast 쪽 연속 장애로 quality도 바로 우회
