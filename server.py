"""all_total 통합 실습 — FastAPI 백엔드 (Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : mcp_server.py를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : GET /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 legacy/신형 파이프라인 즉시 전환
- LangSmith: 요청당 chat_request 루트 run 아래로 노드·LLM·MCP 호출이 계층 기록됨
"""
import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()   # LANGCHAIN_TRACING_V2 / API_KEY / PROJECT — import보다 먼저

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from agent.deps import SupervisorDeps
from agent.errors import to_user_error
from agent.guardrails import filter_output, guard_input, mask_sensitive
from agent.fake_llm import FastLLM, QualityLLM
from agent import llm_http
from agent.graph import build_graph
from agent.mcp_client import MCPClient
from agent.runbook_retriever import RunbookRetriever, make_embedder
from agent.session_store import SqliteSessionStore
from agent.supervisor_v1 import run_v1

CREATION_COUNT = 0   # Wave 1 — 프로세스 내 Agent 번들 생성 횟수 (항상 1이어야 정상)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # 요청마다 찍히는 HTTP 로그는 숨김

# thread_id는 DB 조회 키로 쓰이므로 형식을 제한한다 (영문·숫자·_·- 만, 64자 이내).
THREAD_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"


class ChatRequest(BaseModel):   # Wave 1 — Pydantic 요청 검증
    message: str   # 빈 값·길이·인젝션 검사는 guardrails가 한국어 메시지로 처리
    thread_id: str = Field(default="default", pattern=THREAD_ID_PATTERN)


def _build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API)."""
    backend = os.environ.get("LLM_BACKEND", "fake")
    if backend == "ollama":
        from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
        fast = OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"),
                             usd_per_1k_tokens=0.00015)
        quality = OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return backend, fast, quality
    fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
    quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
    return backend, fast, quality


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CREATION_COUNT
    CREATION_COUNT += 1
    mcp = MCPClient()
    await mcp.start()   # Wave 2 — MCP subprocess는 서버 수명 동안 딱 1개
    backend, fast_llm, quality_llm = _build_llms()
    app.state.llm_backend = backend
    # 개선 3 — runbook 검색기도 싱글턴: 기동 시 1회 인덱싱(runbook 임베딩)해 모든 요청이 공유
    # RUNBOOK_RETRIEVER=keyword 로 기동하면 기존 키워드 규칙(개선 전)으로 즉시 되돌아간다
    retriever = None
    if os.environ.get("RUNBOOK_RETRIEVER", "hybrid") != "keyword":
        retriever = RunbookRetriever(make_embedder(backend))
        await retriever.build()
    deps = SupervisorDeps(
        fast_llm=fast_llm,
        quality_llm=quality_llm,
        mcp=mcp,
        session_store=SqliteSessionStore(),
        version=os.environ.get("SUPERVISOR_VERSION", "v2"),
        runbook_retriever=retriever,
    )
    app.state.deps = deps
    app.state.graph = build_graph(deps) if deps.version == "v2" else None
    app.state.instance_id = id(deps)
    yield
    await mcp.stop()
    await llm_http.aclose()   # 개선 4: 공용 LLM HTTP 연결 풀 정리
    deps.session_store.close()
    try:   # 종료 전 LangSmith 배치 전송 플러시 — kill 시 run이 pending으로 남는 것 방지
        from langchain_core.tracers.langchain import wait_for_all_tracers
        wait_for_all_tracers()
    except Exception:
        pass


app = FastAPI(lifespan=lifespan)


# ── 오류 응답 포맷 통일 ────────────────────────────────────────────────
# 어떤 경로로 실패하든 {"error": {"code", "message", "error_id"}} 하나의 형태로 내려간다.
@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {
        "code": "INVALID_REQUEST",
        "message": "요청 형식이 올바르지 않습니다. (thread_id는 영문·숫자·_·- 64자 이내)",
        "error_id": None}})


@app.exception_handler(Exception)
async def on_unhandled_error(request: Request, exc: Exception):
    err = to_user_error(exc)   # 스택트레이스는 서버 로그에만, 사용자에겐 안내 문구만
    return JSONResponse(status_code=err.status, content=err.to_body())


@app.get("/health")
async def health():
    return {"status": "ok", "instance_id": app.state.instance_id,
            "creation_count": CREATION_COUNT,
            "supervisor_version": app.state.deps.version,
            "llm_backend": app.state.llm_backend,
            "fast_llm": app.state.deps.fast_llm.name,
            "quality_llm": app.state.deps.quality_llm.name,
            "llm_http_mode": llm_http.mode(),
            "runbook_retriever": (app.state.deps.runbook_retriever.method
                                  if app.state.deps.runbook_retriever else "keyword")}


@app.get("/version")
async def version():
    return {"supervisor_version": app.state.deps.version}


def _metadata(result: dict, run_id: str | None) -> dict:
    return {
        "route": result.get("route"),
        "task_count": len(result.get("tasks", [])),
        "gate_removed": result.get("gate_removed", 0),
        "replan_rounds": result.get("replan_rounds", 0),
        "retry_count": result.get("retry_count", 0),
        "parallel_elapsed": result.get("parallel_elapsed"),
        "validation_attempts": result.get("validation_attempts", 0),
        "cost_usd": result.get("cost_usd", 0.0),
        "llm_backend": app.state.llm_backend,
        "supervisor_version": app.state.deps.version,
        "instance_id": app.state.instance_id,
        "langsmith_run_id": run_id,
    }


def _guard_event(input_types: list[str], output_types: list[str]) -> dict:
    """어떤 '종류'의 민감정보가 가려졌는지만 기록한다 (값 자체는 어디에도 남기지 않음)."""
    return {"type": "GuardrailApplied", "input_masked": input_types,
            "output_masked": output_types}


@traceable(name="chat_request")
async def _handle_chat(message: str, thread_id: str) -> tuple[dict, str | None]:
    """message는 이미 guard_input을 통과한 '마스킹된' 텍스트다 — LangSmith 트레이스에도
    원문이 아니라 마스킹본만 기록된다."""
    deps = app.state.deps
    history = deps.session_store.load(thread_id)   # Wave 12
    if deps.version == "v2":
        result = await app.state.graph.ainvoke(
            {"query": message, "thread_id": thread_id, "history": history,
             "events": [], "cost_usd": 0.0})
    else:
        result = await run_v1(deps, message, thread_id, history)   # Wave 14
    out = filter_output(result["final"])   # 출력 가드레일
    result["final"] = out.text
    result["output_masked"] = out.masked_types
    # DB에는 마스킹된 질문/답변만 저장 — 실패한 요청은 이력에 남기지 않는다
    deps.session_store.append(thread_id, "user", message)
    deps.session_store.append(thread_id, "assistant", result["final"])
    rt = get_current_run_tree()
    return result, (str(rt.id) if rt else None)


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        guarded = guard_input(req.message)   # 1) 검증 + 마스킹 (위반 시 LLM 호출 없이 거절)
        result, run_id = await _handle_chat(guarded.text, req.thread_id)
    except Exception as exc:                 # 2) 어떤 실패든 사용자용 메시지로 변환
        err = to_user_error(exc)
        return JSONResponse(status_code=err.status, content=err.to_body())
    events = [_guard_event(guarded.masked_types, result["output_masked"])] \
        + result.get("events", [])
    meta = _metadata(result, run_id)
    meta["masked_input_types"] = guarded.masked_types
    return {"reply": result["final"], "events": events, "metadata": meta}


@app.get("/chat/stream")
async def chat_stream(message: str,
                      thread_id: str = Query("default", pattern=THREAD_ID_PATTERN)):
    """Wave 3 — 그래프의 노드별 진행 상황을 SSE로 스트리밍.

    이벤트마다 'data: {json}\\n\\n' 형태로 내려보내고 마지막에 data: [DONE] 센티널을
    보낸다 — 프론트엔드는 \\n\\n 경계로 이벤트를 분리해 파싱해야 한다(Wave 3b).
    """
    deps = app.state.deps

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def gen():
        # EventSource는 HTTP 4xx/5xx 응답의 본문을 읽지 못한다. 그래서 스트림에서는
        # 상태 코드 대신 200으로 열어 두고 {"type": "Error"} 이벤트로 실패를 알린다.
        try:
            guarded = guard_input(message)
        except Exception as exc:
            yield sse(to_user_error(exc).to_event())
            yield "data: [DONE]\n\n"
            return
        yield sse(_guard_event(guarded.masked_types, []))
        try:
            async for chunk in _run_stream(guarded.text):
                yield chunk
        except Exception as exc:
            yield sse(to_user_error(exc).to_event())
        yield "data: [DONE]\n\n"

    async def _run_stream(message: str):
        history = deps.session_store.load(thread_id)
        if deps.version != "v2":
            result = await run_v1(deps, message, thread_id, history)
            for ev in result["events"]:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            final = filter_output(result["final"]).text
            yield f"data: {json.dumps({'type': 'FinalAnswer', 'reply': final}, ensure_ascii=False)}\n\n"
            deps.session_store.append(thread_id, "user", message)
            deps.session_store.append(thread_id, "assistant", final)
        else:
            seen = 0
            final = ""
            last_prompt = None   # finalize까지 살아남은 최종 write/chat 호출의 실제 프롬프트
            # stream_mode를 리스트로 주면 (mode, payload) 튜플이 나온다 — "updates"는
            # 노드가 끝날 때마다의 상태 델타, "custom"은 write_node/chat_answer_node가
            # get_stream_writer()로 실행 중에 즉시 흘려보내는 토큰(AnswerToken) 이벤트다.
            async for mode, payload in app.state.graph.astream(
                    {"query": message, "thread_id": thread_id, "history": history,
                     "events": [], "cost_usd": 0.0}, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    if payload.get("type") == "AnswerToken":   # 토큰 조각에도 출력 필터 적용
                        payload = {**payload, "delta": mask_sensitive(payload["delta"]).text}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    continue
                for node_name, delta in payload.items():
                    for ev in (delta.get("events") or [])[seen:]:
                        ev = {**ev, "node": node_name}
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if delta.get("events"):
                        seen = len(delta["events"])
                    if delta.get("final"):
                        final = delta["final"]
                    if delta.get("last_prompt"):
                        last_prompt = delta["last_prompt"]
            final = filter_output(final).text   # 최종본으로 말풍선을 덮어써 토큰 경계에 걸친 누락도 보정
            yield f"data: {json.dumps({'type': 'FinalAnswer', 'reply': final, 'prompt': last_prompt}, ensure_ascii=False)}\n\n"
            deps.session_store.append(thread_id, "user", message)
            deps.session_store.append(thread_id, "assistant", final)

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
