"""v2 Supervisor — 실제 langgraph.StateGraph (Wave 4·5·6·7·8·10b·15 통합).

route → (chat 직행 | incident 파이프라인)
incident: plan(MCP Prompt 절차 반영) → gate(권위 게이트) → research(가드+병렬+재시도)
          → replan_check(부족 시 gap task 후 재조사) → analysis(needs_analysis=True일 때만
          실제 실행) → write → validate(반려 시 재작성) → finalize
          → (개선 8) propose_action → approval(interrupt, 사람 승인 대기) → execute_action

모든 노드는 SupervisorDeps를 functools.partial로 주입받는 순수 함수다(Wave 13).
각 노드는 @traceable로 LangSmith에 계층 트리로 기록된다.
"""
import asyncio
import json
import time
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from langsmith import traceable

from . import grounding, remediation
from .deps import SupervisorDeps
from .retry import with_retry


def _load_known_systems() -> set[str]:
    """ollama_llm.py/openai_llm.py와 동일한 소스(data/monitoring.json)에서
    시스템 목록을 읽는다 — plan_node가 MCP Prompt에 넘길 system_name을
    쿼리에서 멤버십 매칭으로 뽑아낼 때 쓴다."""
    path = Path(__file__).resolve().parent.parent / "data" / "monitoring.json"
    try:
        return set(json.loads(path.read_text()).keys())
    except (OSError, json.JSONDecodeError):
        return {"auth-service", "web-01"}


KNOWN_SYSTEMS = _load_known_systems()


def _extract_system_name(query: str) -> str:
    """정규식 대신 실제 KNOWN_SYSTEMS 멤버십 매칭 — MCP Prompt(incident_analysis_
    prompt)의 system_name은 "시스템: X" 한 줄짜리 컨텍스트라 항상 정해진 이름
    중 하나로만 고정한다(여러 개 언급돼도 join하지 않음 — 실제 조사 대상 목록은
    이후 plan()의 tasks가 별도로 결정). 쿼리에서 가장 먼저 등장하는 시스템을
    고르고, 하나도 안 걸리면 가장 흔한 사례인 auth-service로 폴백."""
    matches = [(query.index(s), s) for s in KNOWN_SYSTEMS if s in query]
    return min(matches)[1] if matches else "auth-service"


class AgentState(TypedDict, total=False):
    query: str
    thread_id: str
    history: list[dict]
    route: str
    needs_analysis: bool
    tasks: list[dict]
    gate_removed: int
    collected: list[dict]
    replan_rounds: int
    retry_count: int
    parallel_elapsed: float
    draft: str
    validation_attempts: int
    validation_feedback: str
    final: str
    last_prompt: str
    ungrounded: list[str]   # 개선 5 — 수집 데이터로 확인되지 않은 답변 속 수치
    cost_usd: float
    events: list[dict]
    actions: list[dict]       # 개선 8 — 승인 대기 중인 조치 제안
    approval_id: str
    decision: dict            # 사람의 결정 {approved, approver, expired}
    action_results: list[dict]
    action_report: str


def _cost(state: AgentState, usage: dict) -> float:
    return round(state.get("cost_usd", 0.0) + usage["usd"], 6)


@traceable(name="route_node")
async def route_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 — 모든 쿼리가 무거운 파이프라인을 탈 필요는 없다."""
    result, usage = await deps.fast_llm.route(state["query"])
    event = {"type": "RouteDecided", "route": result["route"]}
    return {"route": result["route"], "needs_analysis": result["needs_analysis"],
            "cost_usd": _cost(state, usage),
            "events": state.get("events", []) + [event]}


@traceable(name="chat_answer_node")
async def chat_answer_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 5 direct 경로 — 히스토리 기반 응답 (Wave 12 세션 증명 지점).

    토큰 단위 스트리밍 — get_stream_writer()는 /chat/stream처럼 custom 모드로 소비될
    때만 실제로 전송되고, /chat(ainvoke)처럼 스트리밍하지 않는 호출에서는 자동으로
    no-op이 된다."""
    writer = get_stream_writer()
    reply, usage = await deps.fast_llm.chat(
        state["query"], state.get("history", []),
        on_token=lambda piece: writer({"type": "AnswerToken", "delta": piece}))
    return {"final": reply, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "Finalize", "pipeline": "chat-direct"}]}


@traceable(name="plan_node")
async def plan_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 4 — Plan/Task 분해. Wave 2의 MCP Prompt가 표준 절차를 깔아준다.

    resource와 마찬가지로 이름을 어느 서버가 들고 있는지 미리 가정하지 않고,
    find_prompt()로 연결된 서버들을 먼저 조회해 찾은 뒤에만 get_prompt를 보낸다."""
    prompt_server = await deps.mcp.find_prompt("incident_analysis_prompt")
    procedure = await deps.mcp.get_prompt(
        "incident_analysis_prompt",
        {"system_name": _extract_system_name(state["query"]), "symptom": state["query"]},
        server=prompt_server,
    )
    result, usage = await deps.fast_llm.plan(state["query"], procedure)
    targets = [t["target"] for t in result["tasks"]
               if t["agent"] == "research" and t.get("target")]
    event = {"type": "PlanCreated", "task_count": len(result["tasks"]),
             "targets": targets}
    return {"tasks": result["tasks"], "needs_analysis": result["needs_analysis"],
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="gate_node")
async def gate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 6 — 권위 게이트. needs_analysis=False인데 섞여 들어온 analysis task를
    프롬프트가 아니라 코드로 강제 제거한다."""
    tasks = state["tasks"]
    if not state.get("needs_analysis", False):
        kept = [t for t in tasks if t["agent"] != "analysis"]
        removed = len(tasks) - len(kept)
    else:
        kept, removed = tasks, 0
    event = {"type": "GateApplied", "removed_analysis_tasks": removed}
    return {"tasks": kept, "gate_removed": removed,
            "events": state["events"] + [event]}


@traceable(name="research_node")
async def research_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 15 가드 + Wave 10b 병렬 + Wave 9 재시도 + Wave 2 Tool/Resource 실호출."""
    tasks = state["tasks"]
    research_indices = [i for i, t in enumerate(tasks)
                        if t["agent"] == "research" and t["status"] == "pending"]
    # Wave 15 — 객체가 아니라 '실행할 목록'을 검사한다 (not research_agent 버그의 수정판)
    if not research_indices:
        return {"events": state["events"] +
                [{"type": "ResearchSkipped", "reason": "pending research task 0개"}]}

    retry_total = state.get("retry_count", 0)

    # URI를 코드에 하드코딩하지 않는다. 여러 MCP 서버가 연결돼 있고 그중 어느 서버가
    # "get_incident_runbook" resource를 제공하는지 모른다고 가정하고, 먼저 연결된
    # 서버들을 조회(list)해 유효한 서버를 찾은 뒤에만 그 서버로 요청한다.
    needs_runbook = any(tasks[i]["kind"] != "monitoring" for i in research_indices)
    runbook_server = runbook_uri_template = None
    if needs_runbook:
        runbook_server, runbook_uri_template = await deps.mcp.find_resource_template(
            "get_incident_runbook")

    async def run_one(idx: int) -> dict:
        task = tasks[idx]
        await asyncio.sleep(0.35)   # 태스크당 고정 지연 — 병렬 효과를 실측하기 위함
        if task["kind"] == "monitoring":
            data, retries = await with_retry(
                deps.mcp.call_tool, "get_monitoring_status",
                {"system_name": task["target"]})
            return {"idx": idx, "source": "monitoring", "data": data, "retries": retries}
        uri = runbook_uri_template.format(incident_type=task["target"])
        data = await deps.mcp.read_resource(uri, server=runbook_server)
        return {"idx": idx, "source": "runbook", "data": data, "retries": 0}

    started = time.perf_counter()
    results = await asyncio.gather(*(run_one(i) for i in research_indices))
    elapsed = round(time.perf_counter() - started, 2)

    new_tasks = [dict(t) for t in tasks]
    collected = list(state.get("collected", []))
    events = list(state["events"])
    for r in results:
        new_tasks[r["idx"]]["status"] = "done"
        collected.append({"source": r["source"], "data": r["data"]})
        retry_total += r["retries"]
        events.append({"type": "TaskCompleted", "id": new_tasks[r["idx"]]["id"],
                       "source": r["source"], "target": new_tasks[r["idx"]]["target"],
                       "retries": r["retries"]})
    return {"tasks": new_tasks, "collected": collected, "retry_count": retry_total,
            "parallel_elapsed": elapsed, "events": events}


# 수집된 모니터링 결과(상태·알림 문구)를 보고 어떤 runbook이 맞는지 고른다. LLM에게
# 다시 묻는 대신 코드 레벨 규칙으로 판단한다 — gate(Wave 6)·replan(Wave 7)과 같은 설계
# 원칙: 모델의 판단을 그대로 믿지 않고, 실측 데이터로 검증 가능한 결정은 코드가 내린다.
_INCIDENT_TYPE_RULES = [
    ("latency", "high-latency"),
    ("pool", "high-latency"),
    ("cpu", "cpu-spike"),
    ("memory", "memory-leak"),
    ("heap", "memory-leak"),
    ("timeout", "network-timeout"),
    ("network", "network-timeout"),
    ("disk", "disk-full"),
    ("down", "db-down"),
]


def _classify_incident_type(collected: list[dict]) -> str:
    text = " ".join(
        f"{c['data'].get('status', '')} " + " ".join(c["data"].get("alerts", []))
        for c in collected if c.get("source") == "monitoring"
    ).lower()
    for keyword, incident_type in _INCIDENT_TYPE_RULES:
        if keyword in text:
            return incident_type
    return "db-down"   # 매칭되는 증상이 없으면 가장 일반적인 장애 대응 절차로 폴백


@traceable(name="select_runbooks")
async def select_runbooks(query: str, collected: list[dict], retriever) -> dict:
    """개선 3 — runbook 선택. retriever가 있으면 하이브리드 검색(사용자 질문 + 이상 알림),
    없으면 기존 키워드 규칙. LangSmith에 후보별 점수가 함께 기록돼 '왜 이 runbook인지' 추적된다."""
    if retriever is None:
        return {"selected": [_classify_incident_type(collected)], "method": "keyword",
                "scores": {}}
    r = await retriever.search_collected(query, collected)
    return {"selected": r.selected, "method": r.method, "scores": r.scores, **r.detail}


@traceable(name="replan_check_node")
async def replan_check_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 7 — 1차 수집 결과 평가. runbook이 없으면 gap task만 추가해 재조사.
    개선 3: 복합 장애면 runbook gap task를 최대 2개까지 만든다."""
    has_runbook = any(c["source"] == "runbook" for c in state.get("collected", []))
    rounds = state.get("replan_rounds", 0)
    if has_runbook or rounds >= 1:
        verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": True}
        return {"events": state["events"] + [verdict]}
    picked = await select_runbooks(state["query"], state.get("collected", []),
                                   deps.runbook_retriever)
    gap_tasks = [{"id": f"task-gap-runbook-{i}", "agent": "research", "kind": "runbook",
                  "target": t, "status": "pending"} for i, t in enumerate(picked["selected"])]
    verdict = {"type": "ReplanCheck", "round": rounds, "is_sufficient": False,
               "gap_task": [t["id"] for t in gap_tasks],
               "target": ", ".join(picked["selected"]),
               "method": picked["method"], "scores": picked["scores"]}
    return {"tasks": state["tasks"] + gap_tasks, "replan_rounds": rounds + 1,
            "events": state["events"] + [verdict]}


@traceable(name="analysis_node")
async def analysis_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 6 보완 — needs_analysis=True가 정당한 경우(모델의 자기모순이 아니라
    실제로 패턴 비교가 필요하다고 판단한 경우)를 위한 실행 경로. gate가 살려둔
    agent=='analysis' task가 있을 때만 수집된 monitoring 결과들을 비교 분석한다."""
    tasks = state["tasks"]
    pending = [t for t in tasks if t["agent"] == "analysis" and t["status"] == "pending"]
    if not pending:
        return {"events": state["events"] +
                [{"type": "AnalysisSkipped", "reason": "analysis task 0개"}]}
    summary, usage = await deps.fast_llm.analyze(state.get("collected", []))
    new_tasks = [dict(t) for t in tasks]
    for t in new_tasks:
        if t["agent"] == "analysis" and t["status"] == "pending":
            t["status"] = "done"
    collected = state.get("collected", []) + [{"source": "analysis", "data": summary}]
    event = {"type": "AnalysisCompleted", "length": len(summary)}
    return {"tasks": new_tasks, "collected": collected,
            "cost_usd": _cost(state, usage), "events": state["events"] + [event]}


@traceable(name="write_node")
async def write_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8/10a — 고품질 모델로 초안 작성 (1차는 의도적 저품질).

    토큰마다 attempt 번호를 같이 실어 보낸다 — validate가 반려해 재작성(attempt 증가)
    될 때 프론트가 새 말풍선으로 끊어서 보여줄 수 있게 하기 위함(Wave 8 재작성 루프를
    그대로 노출)."""
    attempt = state.get("validation_attempts", 0) + 1
    writer = get_stream_writer()
    draft, usage = await deps.quality_llm.write(
        state["query"], state.get("collected", []),
        attempt, state.get("validation_feedback"),
        on_token=lambda piece: writer(
            {"type": "AnswerToken", "attempt": attempt, "delta": piece}))
    return {"draft": draft, "last_prompt": usage.get("prompt"),
            "cost_usd": _cost(state, usage),
            "events": state["events"] + [{"type": "DraftWritten", "attempt": attempt,
                                          "length": len(draft)}]}


@traceable(name="validate_node")
async def validate_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """Wave 8 — 코드 기준 검증: 30자 이상 + 절차 번호 포함이어야 합격.
    개선 5 — 형식에 더해 '답변 속 수치가 수집된 데이터에 실제로 있는가'를 검사한다.
    근거 없는 수치가 있으면 그 숫자를 짚어서 반려 → write가 피드백을 반영해 재작성."""
    draft = state["draft"]
    attempts = state.get("validation_attempts", 0) + 1
    problems = []
    if not (len(draft) >= 30 and "1)" in draft):
        problems.append(f"응답이 너무 짧거나 형식이 맞지 않습니다({len(draft)}자, 최소 30자). "
                        f"절차를 1) 2) 3) 번호로 구체적으로 설명하세요.")
    ungrounded = []
    if grounding.enabled():
        fact_text = grounding.build_fact_text(state["query"], state.get("collected", []),
                                              state.get("last_prompt"))
        ungrounded = grounding.find_ungrounded(draft, fact_text)
        if ungrounded:
            problems.append(f"다음 수치는 수집된 데이터에 없습니다: {', '.join(ungrounded)}. "
                            f"수치를 추정하거나 계산하지 말고 '수집된 사실'에 있는 값만 그대로 "
                            f"인용하세요. 근거가 없으면 수치 없이 서술하세요.")
    ok = not problems
    verdict = "합격" if ok else "불합격"
    if not ok and attempts >= 3:   # 실모델 무한 재작성 방지 — 최대 3회 후 통과 처리
        ok, verdict = True, "합격(최대 시도 도달 — 통과 처리)"
    feedback = None if ok else " / ".join(problems)
    event = {"type": "Validation", "attempt": attempts, "verdict": verdict,
             "ungrounded": ungrounded}
    return {"validation_attempts": attempts, "validation_feedback": feedback,
            "ungrounded": ungrounded, "events": state["events"] + [event]}


@traceable(name="finalize_node")
async def finalize_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """개선 5 — 재작성 한도까지 써도 근거 없는 수치가 남으면, 조용히 내보내지 않고
    답변 끝에 '확인 필요' 경고를 붙여 사용자가 그 숫자를 그대로 믿지 않게 한다."""
    final = state["draft"]
    if state.get("ungrounded"):
        final += grounding.warning_line(state["ungrounded"])
    return {"final": final,
            "events": state["events"] + [{"type": "Finalize", "pipeline": "incident-v2"}]}


@traceable(name="propose_action_node")
async def propose_action_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """개선 8 — 이상 시스템 + runbook 유형으로 조치를 코드 규칙으로 제안 (LLM이 고르지 않음)."""
    if not remediation.enabled():
        return {"actions": []}
    actions = remediation.propose(state.get("collected", []), state["events"])
    if not actions:
        return {"actions": [], "events": state["events"] +
                [{"type": "ActionSkipped", "reason": "조치할 이상 시스템 없음"}]}
    approval_id = remediation.new_approval_id()
    return {"actions": actions, "approval_id": approval_id,
            "events": state["events"] + [{"type": "ActionProposed", "approval_id": approval_id,
                                          "actions": actions}]}


@traceable(name="approval_node")
async def approval_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """개선 8 — 여기서 그래프가 멈춘다. 사람이 /approvals/{id}로 결정을 보내면
    Command(resume=결정)으로 이 노드가 다시 실행되고, interrupt()가 그 결정을 돌려준다."""
    decision = interrupt({"type": "ApprovalRequired", "approval_id": state["approval_id"],
                          "actions": state["actions"],
                          "expires_in_s": remediation.APPROVAL_TTL_S})
    return {"decision": decision}


@traceable(name="execute_action_node")
async def execute_action_node(state: AgentState, deps: SupervisorDeps) -> dict:
    """개선 8 — 승인된 경우에만 실행하고, 같은 시스템을 다시 조회해 조치 전/후를 남긴다."""
    decision = state.get("decision") or {}
    events = list(state["events"])
    results = []
    if decision.get("approved"):
        for a in state["actions"]:
            r = await deps.mcp.call_tool("execute_remediation", {
                "system_name": a["system_name"], "action": a["action"],
                "approval_id": state["approval_id"]})
            after = None
            if r.get("ok"):
                after = await deps.mcp.call_tool("get_monitoring_status",
                                                 {"system_name": a["system_name"]})
            results.append({**a, "ok": bool(r.get("ok")), "message": r.get("message", ""),
                            "after": after})
            events.append({"type": "ActionExecuted", "system_name": a["system_name"],
                           "action": a["action"], "ok": bool(r.get("ok")),
                           "before_status": a["before"]["status"],
                           "after_status": after["status"] if after else None})
    else:
        events.append({"type": "ActionRejected", "approval_id": state["approval_id"],
                       "expired": bool(decision.get("expired"))})
    return {"action_results": results, "action_report": remediation.report(results, decision),
            "events": events}


def _after_propose(state: AgentState) -> str:
    return "approval" if state.get("actions") else END


def _after_route(state: AgentState) -> str:
    return "chat_answer" if state["route"] == "chat" else "plan"


def _after_replan_check(state: AgentState) -> str:
    last = state["events"][-1]
    return "analysis" if last.get("is_sufficient") else "research"


def _after_validate(state: AgentState) -> str:
    return "finalize" if state.get("validation_feedback") is None else "write"


def build_graph(deps: SupervisorDeps, checkpointer=None):
    """checkpointer — 개선 8의 승인 대기(interrupt) 동안 실행 상태를 보관한다.
    없으면(테스트·v1 호환) 조치 제안 단계까지만 가고 승인 대기 노드는 붙이지 않는다."""
    g = StateGraph(AgentState)
    node = lambda fn: partial(fn, deps=deps)   # Wave 13 — 명시적 의존성 주입
    g.add_node("route", node(route_node))
    g.add_node("chat_answer", node(chat_answer_node))
    g.add_node("plan", node(plan_node))
    g.add_node("gate", node(gate_node))
    g.add_node("research", node(research_node))
    g.add_node("replan_check", node(replan_check_node))
    g.add_node("analysis", node(analysis_node))
    g.add_node("write", node(write_node))
    g.add_node("validate", node(validate_node))
    g.add_node("finalize", node(finalize_node))
    hitl = checkpointer is not None
    if hitl:
        g.add_node("propose_action", node(propose_action_node))
        g.add_node("approval", node(approval_node))
        g.add_node("execute_action", node(execute_action_node))

    g.set_entry_point("route")
    g.add_conditional_edges("route", _after_route,
                            {"chat_answer": "chat_answer", "plan": "plan"})
    g.add_edge("chat_answer", END)
    g.add_edge("plan", "gate")
    g.add_edge("gate", "research")
    g.add_edge("research", "replan_check")
    g.add_conditional_edges("replan_check", _after_replan_check,
                            {"analysis": "analysis", "research": "research"})
    g.add_edge("analysis", "write")
    g.add_edge("write", "validate")
    g.add_conditional_edges("validate", _after_validate,
                            {"finalize": "finalize", "write": "write"})
    if hitl:
        g.add_edge("finalize", "propose_action")
        g.add_conditional_edges("propose_action", _after_propose,
                                {"approval": "approval", END: END})
        g.add_edge("approval", "execute_action")
        g.add_edge("execute_action", END)
    else:
        g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)
