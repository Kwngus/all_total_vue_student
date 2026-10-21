"""개선 8 — 조치 제안 → 사람 승인(interrupt) → 실행·재확인. 실제 그래프를 가짜 LLM·가짜 MCP로 돌린다."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agent import remediation  # noqa: E402
from agent.deps import SupervisorDeps  # noqa: E402
from agent.fake_llm import FastLLM, QualityLLM  # noqa: E402
from agent.graph import build_graph  # noqa: E402


class FakeMCP:
    """mcp_server.py와 같은 응답 형태. 조치가 실행된 시스템은 이후 healthy로 보고한다."""

    def __init__(self):
        self.executed = []

    async def find_prompt(self, name):
        return "ops"

    async def get_prompt(self, name, args, server="ops"):
        return "1. 모니터링 조회 2. runbook 확인 3. 원인·조치 제시"

    async def find_resource_template(self, name):
        return "ops", "runbook://incident-response/{incident_type}"

    async def read_resource(self, uri, server="ops"):
        return "1. p95 지연과 커넥션 풀을 확인한다 2. 풀 크기를 늘린다"

    async def call_tool(self, name, args, server="ops"):
        if name == "execute_remediation":
            self.executed.append((args["system_name"], args["action"], args["approval_id"]))
            return {"ok": True, "system_name": args["system_name"], "action": args["action"],
                    "message": "완료 (시뮬레이션)"}
        fixed = any(s == args["system_name"] for s, _, _ in self.executed)
        if args["system_name"] == "auth-service" and not fixed:
            return {"system_name": "auth-service", "status": "degraded", "cpu_percent": 81.0,
                    "memory_percent": 62.0, "alerts": ["p95 latency 2.4s (threshold 1.0s)"]}
        return {"system_name": args["system_name"], "status": "healthy", "cpu_percent": 34.0,
                "memory_percent": 48.0, "alerts": []}


def _graph(checkpointer=True):
    mcp = FakeMCP()
    deps = SupervisorDeps(fast_llm=FastLLM("f", 0.00015), quality_llm=QualityLLM("q", 0.005),
                          mcp=mcp, session_store=None, version="v2")
    return build_graph(deps, InMemorySaver() if checkpointer else None), mcp


QUERY = "auth-service 로그인 응답 지연 장애 진단해줘"
INPUT = {"query": QUERY, "thread_id": "t", "history": [], "events": [], "cost_usd": 0.0}
CFG = {"configurable": {"thread_id": "run-1"}}


def test_answer_first_then_waits_without_executing():
    graph, mcp = _graph()
    result = asyncio.run(graph.ainvoke(INPUT, CFG))
    assert result["final"]                                   # 답변은 바로 나온다
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "ApprovalRequired"
    assert payload["actions"][0] == {**payload["actions"][0], "system_name": "auth-service",
                                     "action": "scale_out"}
    assert mcp.executed == []                                # 승인 전에는 아무것도 실행하지 않는다


def test_approve_executes_and_rechecks():
    graph, mcp = _graph()
    first = asyncio.run(graph.ainvoke(INPUT, CFG))
    aid = first["__interrupt__"][0].value["approval_id"]
    done = asyncio.run(graph.ainvoke(Command(resume={"approved": True, "approver": "kim"}), CFG))
    assert mcp.executed == [("auth-service", "scale_out", aid)]
    r = done["action_results"][0]
    assert r["before"]["status"] == "degraded" and r["after"]["status"] == "healthy"
    assert "kim 승인" in done["action_report"] and "degraded → healthy" in done["action_report"]


def test_reject_executes_nothing():
    graph, mcp = _graph()
    asyncio.run(graph.ainvoke(INPUT, CFG))
    done = asyncio.run(graph.ainvoke(Command(resume={"approved": False, "approver": "kim"}), CFG))
    assert mcp.executed == [] and "실행하지 않았습니다" in done["action_report"]
    assert done["events"][-1]["type"] == "ActionRejected"


def test_expired_is_treated_as_rejection():
    graph, mcp = _graph()
    asyncio.run(graph.ainvoke(INPUT, CFG))
    done = asyncio.run(graph.ainvoke(Command(resume={"approved": False, "expired": True}), CFG))
    assert mcp.executed == [] and "만료" in done["action_report"]


def test_healthy_system_gets_no_proposal():
    graph, mcp = _graph()
    result = asyncio.run(graph.ainvoke({**INPUT, "query": "web-01 장애 진단해줘"},
                                       {"configurable": {"thread_id": "run-2"}}))
    # fake plan은 web-01만 조사 → 이상 시스템 없음 → 승인 대기 없이 끝난다
    assert "__interrupt__" not in result and mcp.executed == []


def test_without_checkpointer_graph_is_unchanged():
    graph, mcp = _graph(checkpointer=False)
    result = asyncio.run(graph.ainvoke(INPUT))
    assert result["final"] and "__interrupt__" not in result and "actions" not in result


def test_propose_uses_allowed_actions_only():
    collected = [{"source": "monitoring", "data": {"system_name": "log-service", "status": "degraded",
                                                    "cpu_percent": 20, "memory_percent": 35,
                                                    "alerts": ["disk usage 95%"]}}]
    events = [{"type": "ReplanCheck", "is_sufficient": False, "target": "disk-full"}]
    acts = remediation.propose(collected, events)
    assert [(a["system_name"], a["action"]) for a in acts] == [("log-service", "clear_old_logs")]
    assert set(remediation.ACTION_BY_INCIDENT.values()) <= set(remediation.ACTION_LABELS)


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("REMEDIATION", "off")
    graph, mcp = _graph()
    result = asyncio.run(graph.ainvoke(INPUT, {"configurable": {"thread_id": "run-3"}}))
    assert "__interrupt__" not in result
