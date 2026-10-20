"""오류 처리 표준화 — 내부 예외를 '사용자에게 보여줄 메시지'와 '관리자용 로그'로 분리한다.

원칙:
  - 사용자에게는 원인 범주와 다음 행동만 알려준다 ("잠시 후 다시 시도" 등).
    API 키 누락, 요금 한도, 스택트레이스 같은 내부 사정은 노출하지 않는다.
  - 대신 error_id를 함께 내려준다. 사용자가 이 번호로 문의하면 관리자는 서버 로그에서
    같은 error_id로 전체 스택트레이스를 찾을 수 있다.
  - 응답 포맷은 항상 {"error": {"code", "message", "error_id"}}로 통일한다.
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass

import httpx

from .guardrails import GuardrailViolation

logger = logging.getLogger("agent.errors")


@dataclass
class UserFacingError:
    status: int        # HTTP 상태 코드 (/chat용)
    code: str          # 프론트·대시보드가 분기할 수 있는 고정 코드
    message: str       # 사용자에게 그대로 보여줄 한국어 문구
    error_id: str      # 서버 로그와 연결하는 추적 번호

    def to_body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message,
                          "error_id": self.error_id}}

    def to_event(self) -> dict:
        """SSE 스트림용 이벤트 형태."""
        return {"type": "Error", **self.to_body()["error"]}


def _find(exc: BaseException, types) -> BaseException | None:
    """LangGraph·asyncio.gather를 거치며 예외가 감싸질 수 있어 __cause__/__context__까지 따라간다."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, types):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return None


def to_user_error(exc: BaseException) -> UserFacingError:
    error_id = uuid.uuid4().hex[:8]

    if isinstance(exc, GuardrailViolation):   # 정책 위반은 '오류'가 아니라 정상적인 거절
        return UserFacingError(400, exc.code, exc.user_message, error_id)

    if (e := _find(exc, httpx.HTTPStatusError)) is not None:
        status = e.response.status_code
        if status in (401, 403):
            code, msg, http = ("LLM_AUTH",
                               "AI 서비스 연결 설정에 문제가 있습니다. 관리자에게 문의해 주세요.", 503)
        elif status == 429:
            code, msg, http = ("LLM_RATE_LIMIT",
                               "요청이 많아 잠시 처리할 수 없습니다. 1분 후 다시 시도해 주세요.", 429)
        else:
            code, msg, http = ("LLM_UNAVAILABLE",
                               "AI 서비스가 일시적으로 응답하지 않습니다. 잠시 후 다시 시도해 주세요.", 503)
    elif _find(exc, (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError)) is not None:
        code, msg, http = ("TIMEOUT",
                           "응답 시간이 초과되었습니다. 질문을 짧게 줄이거나 잠시 후 다시 시도해 주세요.", 504)
    elif _find(exc, httpx.ConnectError) is not None:
        code, msg, http = ("LLM_UNAVAILABLE",
                           "AI 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.", 503)
    else:
        code, msg, http = ("INTERNAL_ERROR",
                           "요청을 처리하는 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.", 500)

    # 사용자에게는 숨긴 진짜 원인을 관리자 로그에만 남긴다.
    logger.error("error_id=%s code=%s %s: %s", error_id, code,
                 type(exc).__name__, exc, exc_info=exc)
    return UserFacingError(http, code, msg, error_id)
