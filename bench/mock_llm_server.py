"""OpenAI 호환 가짜 LLM 서버 — 과금 없이 '응답이 느린 LLM'을 재현한다 (벤치마크 전용).

모든 요청은 MOCK_LATENCY초(기본 2.0) 기다린 뒤 응답한다. 실제 모델처럼 응답 속도가
흔들리지 않으므로, 서버 구조(동기 vs 비동기 호출)의 차이만 깨끗하게 측정할 수 있다.

실행:
    python bench/mock_llm_server.py                  # http://127.0.0.1:8700/v1
그 다음 에이전트를 이 서버로 연결:
    LLM_BACKEND=openai OPENAI_API_KEY=mock OPENAI_BASE_URL=http://127.0.0.1:8700/v1 python server.py

개선 7(LLM 장애 자동 대체) 실험용 옵션
    MOCK_FAIL=503      모든 OpenAI 호출을 MOCK_LATENCY초 기다린 뒤 503으로 실패 ('느리게 죽는' 장애)
    MOCK_PORT=8701     다른 포트로 하나 더 띄워 Ollama 대역으로 쓴다 (/api/chat, Ollama 응답 형식)
"""
import asyncio
import hashlib
import json
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

LATENCY = float(os.environ.get("MOCK_LATENCY", "2.0"))
# MOCK_HALLUCINATE=1 이면 첫 보고서 초안에 수집되지 않은 수치(CPU 53.5%, 응답 3.7s)를 섞는다.
# 검증 반려 사유가 프롬프트에 들어오면 수집값만 쓴 답변으로 고쳐 쓴다 — 개선 5 동작 확인용.
HALLUCINATE = os.environ.get("MOCK_HALLUCINATE") == "1"
FAIL = os.environ.get("MOCK_FAIL")          # 예: "503" — 개선 7 실험용 장애 주입
PORT = int(os.environ.get("MOCK_PORT", "8700"))
CALLS = {"openai": 0, "ollama": 0}
app = FastAPI()


def _reply_for(system: str, user: str = "") -> str:
    """시스템 프롬프트로 어떤 단계의 호출인지 구분해 그럴듯한 응답을 돌려준다."""
    if "라우터" in system:
        return json.dumps({"route": "incident", "needs_analysis": False})
    if "planning" in system:
        return json.dumps({"needs_analysis": False,
                           "tasks": [{"agent": "research", "target": "auth-service"}]})
    if "패턴 분석가" in system:
        return "특이 패턴 없음"
    if "보고 작성자" in system:
        if HALLUCINATE and "반려 사유" not in user:
            return ("1) auth-service CPU 53.5%, 평균 응답 3.7s로 과부하 상태 "
                    "2) 커넥션 풀 92% — 풀 크기 확장 3) 정상화 후 원인 문서화")
        return ("1) auth-service 상태 degraded, p95 지연 2.4s 확인 "
                "2) 커넥션 풀 92% — 풀 크기 확장 및 느린 쿼리 타임아웃 적용 "
                "3) 정상화 후 느린 쿼리 로그 분석해 원인 문서화")
    return "안녕하세요! 무엇을 도와드릴까요?"


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
    user = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
    CALLS["openai"] += 1
    await asyncio.sleep(LATENCY)   # 비동기 대기 — 가짜 서버 자신은 동시 요청을 병렬 처리
    if FAIL:
        return JSONResponse({"error": {"message": "mock outage"}}, status_code=int(FAIL))
    return {"choices": [{"message": {"role": "assistant", "content": _reply_for(system, user)}}],
            "usage": {"total_tokens": 200}}


@app.post("/v1/embeddings")
async def embeddings(req: Request):
    body = await req.json()
    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]

    def vec(t: str) -> list[float]:   # 결정론적 가짜 벡터
        h = hashlib.sha256(t.encode()).digest()
        return [b / 255 for b in h[:16]]
    return {"data": [{"embedding": vec(t)} for t in texts]}


@app.post("/api/chat")
async def ollama_chat(req: Request):
    """Ollama 호환 (/api/chat) — stream=true면 NDJSON 두 줄(내용, done)로 응답."""
    body = await req.json()
    system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
    user = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
    CALLS["ollama"] += 1
    await asyncio.sleep(LATENCY)
    text = _reply_for(system, user)
    if body.get("stream"):
        async def gen():
            yield json.dumps({"message": {"content": text}, "done": False}) + "\n"
            yield json.dumps({"message": {"content": ""}, "done": True,
                              "prompt_eval_count": 150, "eval_count": 50}) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    return {"message": {"role": "assistant", "content": text}, "done": True,
            "prompt_eval_count": 150, "eval_count": 50}


@app.get("/calls")
async def calls():
    """벤치마크가 '주 LLM을 몇 번 불렀나'를 확인하는 용도."""
    return CALLS


if __name__ == "__main__":
    print(f"mock LLM: http://127.0.0.1:{PORT} (응답 지연 {LATENCY}s, 장애 주입 {FAIL or '없음'})")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
