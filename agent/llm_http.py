"""LLM HTTP 호출 공용 모듈 — 이벤트 루프를 막지 않는 비동기 호출 + 연결 재사용.

개선 전 문제
  openai_llm.py / ollama_llm.py가 `async def` 노드 안에서 동기 `httpx.post()`를 호출했다.
  FastAPI(uvicorn)는 스레드 하나의 이벤트 루프로 모든 요청을 번갈아 처리하는데, 동기 호출은
  응답이 올 때까지 그 루프를 붙잡고 놓지 않는다. 그 결과
    - LLM이 답을 쓰는 몇 초 동안 다른 사용자의 요청은 시작조차 못 하고 줄을 선다
    - 심지어 /health 같은 즉시 응답 API도 그동안 멈춘다
  즉 '동시 사용자 N명 = 대기 시간 N배'가 된다.

개선
  - httpx.AsyncClient로 `await` 하면 LLM 응답을 기다리는 동안 루프가 다른 요청을 처리한다
  - 클라이언트를 프로세스당 하나만 만들어 재사용한다(Wave 1 싱글턴 원칙) — 요청마다 TCP/TLS
    연결을 새로 맺는 비용도 사라진다

LLM_HTTP_MODE=sync 로 기동하면 개선 전 동작(동기 호출)을 그대로 재현한다 — Before/After 실측용.
"""
import os

import httpx

_client: httpx.AsyncClient | None = None


def mode() -> str:
    return os.environ.get("LLM_HTTP_MODE", "async")


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
    return _client


async def post_json(url: str, payload: dict, headers: dict | None = None,
                    timeout: float = 300.0) -> httpx.Response:
    if mode() == "sync":
        # 개선 전 재현용: 응답이 올 때까지 이벤트 루프 전체가 멈춘다
        return httpx.post(url, json=payload, headers=headers, timeout=timeout)
    return await _get_client().post(url, json=payload, headers=headers, timeout=timeout)


async def aclose() -> None:
    """서버 종료(lifespan) 시 연결 풀 정리."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
