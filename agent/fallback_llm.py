"""개선 7 — LLM 장애 자동 대체 (OpenAI → Ollama) + 서킷 브레이커.

개선 전 문제
  LLM_BACKEND=openai로 운영하는데 OpenAI가 장애(5xx)·과부하(429)·네트워크 단절·타임아웃이면
  모든 요청이 그대로 실패한다. 장애 대응 에이전트가 정작 장애 상황에서 멈추는 셈이다.
  게다가 OpenAI가 느리게 실패하는 동안(타임아웃) 요청마다 그 대기 시간을 매번 다 치른다.

개선
  1) 대체(fallback): 주 LLM 호출이 '일시적 장애'로 실패하면 같은 단계(route/plan/analyze/
     chat/write)를 보조 LLM(로컬 Ollama)으로 다시 호출한다. 사용자는 답을 받는다.
  2) 서킷 브레이커: 주 LLM이 연속 N번(기본 3) 실패하면 일정 시간(기본 30초) 동안 주 LLM을
     아예 부르지 않고 바로 보조로 보낸다 → 죽은 서버를 매번 기다리는 지연이 사라진다.
     시간이 지나면 1건만 주 LLM으로 시험 호출(half-open)해 복구됐으면 원래대로 돌아간다.
  3) 기록: 대체가 일어날 때마다 이번 요청의 대체 이력(contextvar)에 남긴다 → 서버가 화면
     이벤트(LLMFallback)와 관리자 지표로 내보낸다. 어떤 답이 보조 모델로 나갔는지 숨기지 않는다.

대체하지 않는 실패
  - 400(잘못된 요청), 프로그래밍 오류(ValueError 등): 보조로 다시 해도 똑같이 실패하거나
    버그를 가린다 → 그대로 올려 기존 오류 처리(error_id)로 보낸다.
  - 401/403(키·권한 문제)은 대체한다 — 서비스는 살리되, 지표의 대체 사유(LLM_AUTH)로
    관리자가 키 문제를 바로 알 수 있다.

LLM_FALLBACK=off 로 기동하면 개선 전 동작(대체 없음)을 그대로 재현한다.
"""
import asyncio
import contextvars
import logging
import os
import time

import httpx

logger = logging.getLogger("agent.fallback")

# 요청 1건 동안의 대체 이력 — server.py가 요청 시작 시 빈 리스트로 설정하고 끝나면 읽는다.
# LangGraph 노드는 이 컨텍스트를 복사한 태스크에서 돌지만, 리스트 객체는 공유되므로 append가 보인다.
FALLBACK_LOG: contextvars.ContextVar[list | None] = contextvars.ContextVar("fallback_log", default=None)

_METHODS = ("route", "plan", "analyze", "chat", "write")


def classify(exc: BaseException) -> str | None:
    """대체할 만한 '일시적·외부' 장애면 사유 코드를, 아니면 None을 돌려준다."""
    if isinstance(exc, httpx.HTTPStatusError):
        s = exc.response.status_code
        if s == 429:
            return "LLM_RATE_LIMIT"
        if s in (401, 403):
            return "LLM_AUTH"
        if s >= 500:
            return f"LLM_HTTP_{s}"
        return None                       # 4xx(요청 자체가 잘못됨)는 대체해도 소용없다
    if isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError)):
        return "TIMEOUT"
    if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)):
        return "LLM_UNREACHABLE"
    if isinstance(exc, RuntimeError) and "OPENAI_API_KEY" in str(exc):
        return "LLM_AUTH"                 # 키 미설정 (openai_llm._chat이 호출 전에 던짐)
    return None


class CircuitBreaker:
    """closed(정상) → 연속 실패 threshold번 → open(주 LLM 차단, cooldown초) → half-open(1건 시험)."""

    def __init__(self, threshold: int = 3, cooldown_s: float = 30.0, clock=time.monotonic):
        self.threshold, self.cooldown_s, self._clock = threshold, cooldown_s, clock
        self.failures = 0
        self.opened_at: float | None = None
        self._trial_in_flight = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self.opened_at >= self.cooldown_s else "open"

    def allow_primary(self) -> bool:
        st = self.state
        if st == "closed":
            return True
        if st == "half_open" and not self._trial_in_flight:
            self._trial_in_flight = True   # 시험 호출은 동시에 1건만
            return True
        return False

    def success(self) -> None:
        if self.opened_at is not None:
            logger.info("circuit closed — 주 LLM 복구 확인")
        self.failures, self.opened_at, self._trial_in_flight = 0, None, False

    def failure(self) -> None:
        self.failures += 1
        was_trial = self._trial_in_flight
        self._trial_in_flight = False
        if was_trial or self.failures >= self.threshold:
            self.opened_at = self._clock()   # 시험 실패면 다시 cooldown
            logger.warning("circuit open — 주 LLM %d회 연속 실패, %.0f초간 보조 LLM 사용",
                           self.failures, self.cooldown_s)

    def release_trial(self) -> None:
        """시험 호출이 '장애가 아닌' 이유로 끝났을 때 — 다음 요청이 다시 시험할 수 있게 풀어준다."""
        self._trial_in_flight = False

    def snapshot(self) -> dict:
        return {"state": self.state, "consecutive_failures": self.failures,
                "threshold": self.threshold, "cooldown_s": self.cooldown_s}


class FallbackLLM:
    """주 LLM과 같은 인터페이스(route/plan/analyze/chat/write)를 그대로 노출하는 래퍼.
    그래프 노드는 대체가 있었는지 몰라도 된다 — SupervisorDeps에 이걸 꽂기만 하면 된다."""

    def __init__(self, primary, backup, breaker: CircuitBreaker, role: str):
        self.primary, self.backup, self.breaker, self.role = primary, backup, breaker, role
        self.name = primary.name   # /health 등에는 평소 쓰는 모델명을 그대로 보여준다
        self.backup_name = backup.name

    def __getattr__(self, attr):
        if attr not in _METHODS:
            raise AttributeError(attr)
        return lambda *a, **kw: self._call(attr, *a, **kw)

    async def _call(self, method: str, *args, **kwargs):
        reason = "CIRCUIT_OPEN"
        if self.breaker.allow_primary():
            try:
                result = await getattr(self.primary, method)(*args, **kwargs)
                self.breaker.success()
                return result
            except Exception as exc:
                reason = classify(exc)
                if reason is None:   # 우리 쪽 문제(요청·코드) — 브레이커 판단에 넣지 않고 그대로 올린다
                    self.breaker.release_trial()
                    raise
                self.breaker.failure()
                primary_exc = exc
        else:
            primary_exc = None
        try:
            result = await getattr(self.backup, method)(*args, **kwargs)
        except Exception:
            # 보조도 실패하면 원래 원인(주 LLM 오류)을 올려 기존 오류 처리로 보낸다
            if primary_exc is not None:
                raise primary_exc
            raise
        self._log(method, reason)
        return result

    def _log(self, method: str, reason: str) -> None:
        entry = {"type": "LLMFallback", "role": self.role, "step": method, "reason": reason,
                 "from": self.primary.name, "to": self.backup.name,
                 "circuit": self.breaker.state}
        logger.warning("LLM fallback %s.%s %s → %s (%s)", self.role, method,
                       self.primary.name, self.backup.name, reason)
        log = FALLBACK_LOG.get()
        if log is not None:
            log.append(entry)
        try:   # 스트리밍 중이면 화면에 즉시 알림 (그래프 밖에서 호출되면 조용히 무시)
            from langgraph.config import get_stream_writer
            get_stream_writer()(entry)
        except Exception:
            pass


def wrap(primary_fast, primary_quality, backup_fast, backup_quality,
         threshold: int | None = None, cooldown_s: float | None = None):
    """fast/quality가 '하나의' 브레이커를 공유한다 — 같은 OpenAI 계정·네트워크라
    한쪽이 죽었으면 다른 쪽도 죽었다고 보는 게 맞다 (fast가 연속 실패하면 quality도 바로 우회)."""
    breaker = CircuitBreaker(
        threshold=threshold or int(os.environ.get("LLM_CB_THRESHOLD", "3")),
        cooldown_s=cooldown_s or float(os.environ.get("LLM_CB_COOLDOWN", "30")))
    return (FallbackLLM(primary_fast, backup_fast, breaker, "fast"),
            FallbackLLM(primary_quality, backup_quality, breaker, "quality"),
            breaker)
