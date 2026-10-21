"""all_total 통합 실습 — FastAPI 백엔드 (Wave 1·3·12·14 축 + 전체 조립).

- Wave 1  : lifespan에서 Agent 번들(MCP 연결·LLM·그래프)을 프로세스당 정확히 1회 생성,
            app.state로 모든 요청이 공유 (creation_count/instance_id로 증명)
- Wave 2  : mcp_server.py를 subprocess(stdio)로 기동해 연결 유지
- Wave 3  : GET /chat/stream — 그래프 실행 과정을 SSE 이벤트로 스트리밍
- Wave 12 : 대화 이력을 SQLite(session.db)에 저장 — 재시작해도 유지
- Wave 14 : SUPERVISOR_VERSION(v1/v2) 환경변수로 legacy/신형 파이프라인 즉시 전환
- LangSmith: 요청당 chat_request 루트 run 아래로 노드·LLM·MCP 호출이 계층 기록됨
- 개선 7  : LLM 장애 자동 대체 (OpenAI → Ollama, 서킷 브레이커) — LLM_FALLBACK=off로 끔
- 개선 8  : 조치 제안 → 사람 승인(interrupt) → 실행·재확인 — REMEDIATION=off로 끔
"""
import json
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()   # LANGCHAIN_TRACING_V2 / API_KEY / PROJECT — import보다 먼저

from fastapi import FastAPI, Header, Query, Request
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, Field

from agent.deps import SupervisorDeps
from agent.errors import to_user_error
from agent import fallback_llm, grounding, remediation
from agent.guardrails import GuardrailViolation, filter_output, guard_input, mask_sensitive
from agent.metrics_store import MetricsStore
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


def _ollama_pair():
    from agent.ollama_llm import OllamaFastLLM, OllamaQualityLLM
    return (OllamaFastLLM(os.environ.get("OLLAMA_FAST_MODEL", "qwen3:0.6b"), usd_per_1k_tokens=0.00015),
            OllamaQualityLLM(os.environ.get("OLLAMA_QUALITY_MODEL", "qwen3:8b"), usd_per_1k_tokens=0.005))


def _build_llms():
    """LLM_BACKEND=fake(기본, 결정론) | ollama(로컬 실모델) | openai(실제 OpenAI API).
    개선 7 — openai일 때 LLM_FALLBACK=ollama(기본)면 장애 시 로컬 Ollama로 자동 대체한다."""
    fast, quality, backend = _build_primary()
    breaker = None
    if backend == "openai" and os.environ.get("LLM_FALLBACK", "ollama") == "ollama":
        backup_fast, backup_quality = _ollama_pair()
        fast, quality, breaker = fallback_llm.wrap(fast, quality, backup_fast, backup_quality)
    return backend, fast, quality, breaker


def _build_primary():
    backend = os.environ.get("LLM_BACKEND", "fake")
    if backend == "ollama":
        fast, quality = _ollama_pair()
        return fast, quality, backend
    if backend == "openai":
        from agent.openai_llm import OpenAIFastLLM, OpenAIQualityLLM
        fast = OpenAIFastLLM(os.environ.get("OPENAI_FAST_MODEL", "gpt-4o-mini"),
                             usd_per_1k_tokens=0.00015)
        quality = OpenAIQualityLLM(os.environ.get("OPENAI_QUALITY_MODEL", "gpt-4o"),
                                   usd_per_1k_tokens=0.005)
        return fast, quality, backend
    fast = FastLLM("fast-model-stub", usd_per_1k_tokens=0.00015)
    quality = QualityLLM("quality-model-stub", usd_per_1k_tokens=0.005)
    return fast, quality, backend


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CREATION_COUNT
    CREATION_COUNT += 1
    mcp = MCPClient()
    await mcp.start()   # Wave 2 — MCP subprocess는 서버 수명 동안 딱 1개
    backend, fast_llm, quality_llm, breaker = _build_llms()
    app.state.llm_backend = backend
    app.state.breaker = breaker   # 개선 7 — None이면 대체 기능 꺼짐
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
    app.state.metrics = MetricsStore()   # 개선 6 — 관리자 대시보드용 요청 단위 지표
    # 개선 8 — 승인 대기 중인 실행 상태를 보관할 체크포인터 (프로세스 메모리, 재시작 시 사라짐)
    app.state.checkpointer = InMemorySaver() if remediation.enabled() else None
    app.state.approvals = {}   # approval_id → 승인 대기 정보
    app.state.graph = (build_graph(deps, app.state.checkpointer)
                       if deps.version == "v2" else None)
    app.state.instance_id = id(deps)
    yield
    await mcp.stop()
    await llm_http.aclose()   # 개선 4: 공용 LLM HTTP 연결 풀 정리
    deps.session_store.close()
    app.state.metrics.close()
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
                                  if app.state.deps.runbook_retriever else "keyword"),
            "llm_fallback": _fallback_config(),
            "remediation": remediation.enabled()}


def _fallback_config() -> dict | None:
    b = app.state.breaker
    if b is None:
        return None
    return {"backup_fast": app.state.deps.fast_llm.backup_name,
            "backup_quality": app.state.deps.quality_llm.backup_name, "circuit": b.snapshot()}


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


def _result_summary(result: dict) -> dict:
    """그래프 결과(또는 스트림에서 모은 상태)에서 대시보드에 필요한 값만 뽑는다."""
    events = result.get("events") or []
    rc = next((e for e in events if e.get("type") == "ReplanCheck"
               and not e.get("is_sufficient")), None)
    return {"route": result.get("route"), "cost_usd": result.get("cost_usd") or 0.0,
            "runbooks": rc["target"].split(", ") if rc and rc.get("target") else [],
            "runbook_method": rc.get("method") if rc else None,
            "validation_attempts": result.get("validation_attempts") or 0,
            "ungrounded": len(result.get("ungrounded") or [])}


def _record(channel: str, thread_id: str, t0: float, *, query: str = "",
            masked: list[str] | None = None, result: dict | None = None,
            exc: Exception | None = None, error_id: str | None = None,
            code: str | None = None, run_id: str | None = None,
            fallbacks: list[dict] | None = None) -> None:
    """요청 1건을 지표 저장소에 남긴다. query는 반드시 마스킹본(가드레일 통과 후)만 넘긴다."""
    status = "ok" if exc is None else (
        "rejected" if isinstance(exc, GuardrailViolation) else "error")
    app.state.metrics.record(
        channel=channel, thread_id=thread_id, query=query, status=status, code=code,
        error_id=error_id, latency_ms=(time.perf_counter() - t0) * 1000,
        backend=app.state.llm_backend, masked=masked, run_id=run_id,
        fallbacks=[f["reason"] for f in (fallbacks or [])],
        **(_result_summary(result) if result else {}))


# ── 개선 8: 승인 대기 관리 ────────────────────────────────────────────
def _run_config() -> dict:
    """요청마다 별도의 체크포인트 스레드 — 대화 세션(thread_id)과 섞이지 않게 한다."""
    return {"configurable": {"thread_id": f"run-{uuid.uuid4().hex}"}}


def _expire_stale() -> None:
    """결정 없이 만료된 승인 요청 정리 — 실행 상태(체크포인트)를 지우고 감사 기록에 '만료'로 남긴다."""
    now = time.time()
    for aid, item in list(app.state.approvals.items()):
        if item["status"] == "pending" and now - item["created"] > remediation.APPROVAL_TTL_S:
            item["status"] = "expired"
            _forget_run(item["config"])
            app.state.metrics.record_approval(approval_id=aid, thread_id=item["thread_id"],
                                              actions=item["actions"], decision="expired",
                                              approver="-", results=[])
        elif item["status"] != "pending" and now - item["created"] > 24 * 3600:
            del app.state.approvals[aid]   # 하루 지난 처리 완료 건은 메모리에서 제거 (감사 기록은 DB에 남아 있음)


def _register_approval(payload: dict, config: dict, thread_id: str) -> dict:
    _expire_stale()
    app.state.approvals[payload["approval_id"]] = {
        "config": config, "thread_id": thread_id, "actions": payload["actions"],
        "created": time.time(), "status": "pending"}
    return payload


def _interrupt_payload(result: dict) -> dict | None:
    intr = result.get("__interrupt__")
    return intr[0].value if intr else None


def _forget_run(config: dict) -> None:
    """끝난 실행의 체크포인트 삭제 — 메모리에 쌓이지 않게."""
    cp = app.state.checkpointer
    if cp is not None:
        try:
            cp.delete_thread(config["configurable"]["thread_id"])
        except Exception:
            pass


@traceable(name="chat_request")
async def _handle_chat(message: str, thread_id: str) -> tuple[dict, str | None]:
    """message는 이미 guard_input을 통과한 '마스킹된' 텍스트다 — LangSmith 트레이스에도
    원문이 아니라 마스킹본만 기록된다."""
    deps = app.state.deps
    history = deps.session_store.load(thread_id)   # Wave 12
    if deps.version == "v2":
        config = _run_config()
        result = await app.state.graph.ainvoke(
            {"query": message, "thread_id": thread_id, "history": history,
             "events": [], "cost_usd": 0.0}, config)
        payload = _interrupt_payload(result)
        if payload:   # 개선 8 — 답변은 바로 주고, 조치는 승인 대기로 남긴다
            result["approval"] = _register_approval(payload, config, thread_id)
        else:
            _forget_run(config)
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
    t0, guarded = time.perf_counter(), None
    fallbacks: list[dict] = []
    fallback_llm.FALLBACK_LOG.set(fallbacks)   # 개선 7 — 이 요청의 LLM 대체 이력
    try:
        guarded = guard_input(req.message)   # 1) 검증 + 마스킹 (위반 시 LLM 호출 없이 거절)
        result, run_id = await _handle_chat(guarded.text, req.thread_id)
    except Exception as exc:                 # 2) 어떤 실패든 사용자용 메시지로 변환
        err = to_user_error(exc)
        # 거절된 원문은 저장하지 않는다(민감정보·인젝션 문구일 수 있음) — 마스킹본만
        _record("chat", req.thread_id, t0, query=guarded.text if guarded else "",
                masked=guarded.masked_types if guarded else None, exc=exc,
                code=err.code, error_id=err.error_id, fallbacks=fallbacks)
        return JSONResponse(status_code=err.status, content=err.to_body())
    _record("chat", req.thread_id, t0, query=guarded.text, masked=guarded.masked_types,
            result=result, run_id=run_id, fallbacks=fallbacks)
    events = [_guard_event(guarded.masked_types, result["output_masked"])] \
        + result.get("events", []) + fallbacks
    meta = _metadata(result, run_id)
    meta["masked_input_types"] = guarded.masked_types
    meta["llm_fallbacks"] = len(fallbacks)
    return {"reply": result["final"], "events": events, "metadata": meta,
            "approval": result.get("approval")}


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
        t0 = time.perf_counter()
        try:
            guarded = guard_input(message)
        except Exception as exc:
            err = to_user_error(exc)
            _record("stream", thread_id, t0, exc=exc, code=err.code, error_id=err.error_id)
            yield sse(err.to_event())
            yield "data: [DONE]\n\n"
            return
        yield sse(_guard_event(guarded.masked_types, []))
        snap: dict = {}   # 스트림 중 노드 델타를 모아 최종 상태를 재구성 (지표 기록용)
        fallbacks: list[dict] = []
        fallback_llm.FALLBACK_LOG.set(fallbacks)   # 개선 7 — 이 요청의 LLM 대체 이력
        try:
            async for chunk in _run_stream(guarded.text, snap):
                yield chunk
            _record("stream", thread_id, t0, query=guarded.text,
                    masked=guarded.masked_types, result=snap, fallbacks=fallbacks)
        except Exception as exc:
            err = to_user_error(exc)
            _record("stream", thread_id, t0, query=guarded.text, masked=guarded.masked_types,
                    exc=exc, code=err.code, error_id=err.error_id, fallbacks=fallbacks)
            yield sse(err.to_event())
        yield "data: [DONE]\n\n"

    async def _run_stream(message: str, snap: dict):
        history = deps.session_store.load(thread_id)
        if deps.version != "v2":
            result = await run_v1(deps, message, thread_id, history)
            snap.update(result)
            for ev in result["events"] + (fallback_llm.FALLBACK_LOG.get() or []):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            final = filter_output(result["final"]).text
            yield f"data: {json.dumps({'type': 'FinalAnswer', 'reply': final}, ensure_ascii=False)}\n\n"
            deps.session_store.append(thread_id, "user", message)
            deps.session_store.append(thread_id, "assistant", final)
        else:
            seen = 0
            final = ""
            last_prompt = None   # finalize까지 살아남은 최종 write/chat 호출의 실제 프롬프트
            config = _run_config()
            approval = None
            # stream_mode를 리스트로 주면 (mode, payload) 튜플이 나온다 — "updates"는
            # 노드가 끝날 때마다의 상태 델타, "custom"은 write_node/chat_answer_node가
            # get_stream_writer()로 실행 중에 즉시 흘려보내는 토큰(AnswerToken) 이벤트다.
            async for mode, payload in app.state.graph.astream(
                    {"query": message, "thread_id": thread_id, "history": history,
                     "events": [], "cost_usd": 0.0}, config, stream_mode=["updates", "custom"]):
                if mode == "updates" and "__interrupt__" in payload:   # 개선 8 — 승인 대기
                    approval = _register_approval(payload["__interrupt__"][0].value, config, thread_id)
                    continue
                if mode == "custom":
                    if payload.get("type") == "AnswerToken":   # 토큰 조각에도 출력 필터 적용
                        payload = {**payload, "delta": mask_sensitive(payload["delta"]).text}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    continue
                for node_name, delta in payload.items():
                    snap.update(delta or {})
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
            if approval:   # 답변 뒤에 승인 카드를 띄우도록 알린다
                yield f"data: {json.dumps(approval, ensure_ascii=False)}\n\n"
            else:
                _forget_run(config)

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── 개선 8: 조치 승인 API ─────────────────────────────────────────────
class ApprovalDecision(BaseModel):
    approve: bool
    approver: str = Field(default="운영자", max_length=40)


@app.post("/approvals/{approval_id}")
async def decide_approval(approval_id: str, body: ApprovalDecision,
                          x_admin_token: str | None = Header(None)):
    """사람의 결정으로 멈춰 있던 그래프를 이어서 실행한다.
    관리자 토큰이 있어야 한다 — 채팅으로 질문할 수 있는 사람과 운영 조치를 승인할 수 있는 사람을 분리."""
    if (denied := _admin_denied(x_admin_token)):
        return denied
    item = app.state.approvals.get(approval_id)
    if item is None:
        return JSONResponse(status_code=404, content={"error": {
            "code": "APPROVAL_NOT_FOUND", "error_id": None,
            "message": "승인 요청을 찾을 수 없습니다 (서버 재시작 시 대기 중인 승인은 사라집니다)."}})
    if item["status"] != "pending":   # 중복 클릭·재전송으로 두 번 실행되지 않게
        return JSONResponse(status_code=409, content={"error": {
            "code": "APPROVAL_ALREADY_DECIDED", "error_id": None,
            "message": f"이미 처리된 승인 요청입니다 ({item['status']})."}})
    expired = time.time() - item["created"] > remediation.APPROVAL_TTL_S
    decision = {"approved": body.approve and not expired, "approver": body.approver,
                "expired": expired}
    item["status"] = "running"   # await 전에 바꿔 둬야 동시 요청이 409로 막힌다
    result = await app.state.graph.ainvoke(Command(resume=decision), item["config"])
    item["status"] = "expired" if expired else ("approved" if body.approve else "rejected")
    _forget_run(item["config"])
    report = filter_output(result.get("action_report", "")).text
    app.state.deps.session_store.append(item["thread_id"], "assistant", report)
    app.state.metrics.record_approval(
        approval_id=approval_id, thread_id=item["thread_id"], actions=item["actions"],
        decision=item["status"], approver=body.approver,
        results=[{"system_name": r["system_name"], "ok": r["ok"],
                  "after": (r["after"] or {}).get("status")} for r in result.get("action_results", [])])
    events = [e for e in result.get("events", [])
              if e.get("type") in ("ActionExecuted", "ActionRejected")]
    body_out = {"status": item["status"], "report": report, "events": events}
    return JSONResponse(status_code=410 if expired else 200, content=body_out)


@app.get("/approvals")
async def list_approvals(x_admin_token: str | None = Header(None)):
    if (denied := _admin_denied(x_admin_token)):
        return denied
    now = time.time()
    return {"approvals": [
        {"approval_id": k, "status": v["status"], "actions": v["actions"],
         "age_s": round(now - v["created"])}
        for k, v in sorted(app.state.approvals.items(), key=lambda kv: -kv[1]["created"])][:50]}


# ── 관리자 API (개선 6) ────────────────────────────────────────────────
# 비용과 사용자 질문(마스킹본)이 보이는 화면이므로 관리자 토큰이 있어야 연다.
# 토큰은 .env의 ADMIN_TOKEN — 설정하지 않으면 관리자 API 자체를 열지 않는다(안전한 기본값).
def _admin_denied(token: str | None) -> JSONResponse | None:
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        return JSONResponse(status_code=503, content={"error": {
            "code": "ADMIN_DISABLED", "error_id": None,
            "message": "관리자 기능이 비활성화되어 있습니다. .env에 ADMIN_TOKEN을 설정하세요."}})
    if not secrets.compare_digest(token or "", expected):   # 타이밍 공격에 안전한 비교
        return JSONResponse(status_code=401, content={"error": {
            "code": "ADMIN_UNAUTHORIZED", "error_id": None,
            "message": "관리자 토큰이 올바르지 않습니다."}})
    return None


@app.get("/admin/metrics")
async def admin_metrics(hours: int = Query(24, ge=1, le=24 * 30),
                        x_admin_token: str | None = Header(None)):
    if (denied := _admin_denied(x_admin_token)):
        return denied
    deps = app.state.deps
    from datetime import datetime, timedelta
    return {"summary": app.state.metrics.summary(hours),
            "approvals": app.state.metrics.approvals(datetime.now() - timedelta(hours=hours)),
            "config": {"llm_backend": app.state.llm_backend,
                       "fast_llm": deps.fast_llm.name, "quality_llm": deps.quality_llm.name,
                       "supervisor_version": deps.version,
                       "runbook_retriever": (deps.runbook_retriever.method
                                             if deps.runbook_retriever else "keyword"),
                       "llm_http_mode": llm_http.mode(),
                       "grounding_check": grounding.enabled(),
                       "llm_fallback": _fallback_config(),
                       "remediation": remediation.enabled()}}


@app.get("/admin/requests")
async def admin_requests(limit: int = Query(50, ge=1, le=500),
                         x_admin_token: str | None = Header(None)):
    if (denied := _admin_denied(x_admin_token)):
        return denied
    return {"requests": app.state.metrics.recent(limit)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
