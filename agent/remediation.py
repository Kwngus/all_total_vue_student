"""개선 8 — 조치 제안·승인·실행 (Human-in-the-loop).

개선 전 문제
  에이전트는 '원인과 권장 조치'를 글로 써 주는 데서 끝났다. 운영자는 답변을 읽고 직접 콘솔에
  들어가 조치하고, 다시 모니터링을 열어 정상화됐는지 확인해야 했다.
  그렇다고 에이전트가 조치를 마음대로 실행하게 두면, LLM의 오판이 곧 운영 사고가 된다.

개선 — '제안은 에이전트가, 결정은 사람이, 실행·확인은 다시 에이전트가'
  1) propose : 수집된 모니터링(이상 시스템)과 선택된 runbook 유형으로 조치를 '코드 규칙'으로 고른다.
               LLM이 고르게 하지 않는다 — 실행 가능한 조치는 허용 목록 4가지뿐이다.
  2) approval: LangGraph interrupt()로 그래프를 멈추고 승인 카드를 띄운다. 실행 상태는
               체크포인터에 저장돼 있어서, 사람이 승인 버튼을 누를 때 그 자리에서 이어서 실행된다.
               승인 API는 관리자 토큰이 있어야 호출할 수 있다 — 질문한 사람과 승인 권한자를 분리.
  3) execute : 승인된 경우에만 MCP Tool(execute_remediation)로 실행하고, 같은 시스템의 모니터링을
               다시 조회해 조치 전/후 상태를 보고한다. 거부·만료면 아무것도 실행하지 않는다.

REMEDIATION=off 로 기동하면 개선 전 동작(조치 제안 없음)을 그대로 재현한다.
"""
import os
import uuid

# runbook 유형 → 실행할 조치 (mcp_server.ALLOWED_ACTIONS와 같은 이름만 사용)
ACTION_BY_INCIDENT = {
    "high-latency": "scale_out",
    "cpu-spike": "scale_out",
    "memory-leak": "restart_service",
    "network-timeout": "restart_service",
    "disk-full": "clear_old_logs",
    "db-down": "failover_db",
}
ACTION_LABELS = {
    "scale_out": "인스턴스 1대 증설",
    "restart_service": "서비스 재시작 (롤링)",
    "clear_old_logs": "7일 지난 로그 정리",
    "failover_db": "DB 읽기 복제본으로 전환",
}
MAX_ACTIONS = 2
APPROVAL_TTL_S = float(os.environ.get("APPROVAL_TTL", "600"))


def enabled() -> bool:
    return os.environ.get("REMEDIATION", "on") != "off"


def _incident_types(events: list[dict]) -> list[str]:
    for e in events:
        if e.get("type") == "ReplanCheck" and not e.get("is_sufficient") and e.get("target"):
            return [t.strip() for t in e["target"].split(",")]
    return []


def propose(collected: list[dict], events: list[dict]) -> list[dict]:
    """이상 상태인 시스템마다 runbook 유형에 맞는 조치 1개 (최대 2건).
    시스템이 여러 개고 runbook도 여러 개면 순서대로 짝짓고, 모자라면 첫 runbook 조치를 쓴다."""
    abnormal = []
    for c in collected:
        if c.get("source") != "monitoring":
            continue
        d = c["data"]
        if d.get("status") in ("degraded", "down") and d["system_name"] not in abnormal:
            abnormal.append(d["system_name"])
    types = [t for t in _incident_types(events) if t in ACTION_BY_INCIDENT]
    if not abnormal or not types:
        return []
    proposals = []
    for i, system in enumerate(abnormal[:MAX_ACTIONS]):
        itype = types[i] if i < len(types) else types[0]
        action = ACTION_BY_INCIDENT[itype]
        status = next(c["data"] for c in collected
                      if c.get("source") == "monitoring" and c["data"]["system_name"] == system)
        proposals.append({"system_name": system, "action": action,
                          "label": ACTION_LABELS[action], "incident_type": itype,
                          "before": {"status": status["status"], "cpu": status["cpu_percent"],
                                     "memory": status["memory_percent"],
                                     "alerts": status.get("alerts", [])}})
    return proposals


def new_approval_id() -> str:
    return uuid.uuid4().hex   # 추측할 수 없는 번호 — 승인 URL에 그대로 쓰인다


def report(results: list[dict], decision: dict) -> str:
    """조치 결과를 사람이 읽는 문장으로 — 대화 이력에도 이 문장이 남는다."""
    who = decision.get("approver") or "운영자"
    if not decision.get("approved"):
        why = "만료" if decision.get("expired") else "거부"
        return f"제안한 조치를 실행하지 않았습니다 ({who} {why})."
    lines = [f"{who} 승인으로 조치를 실행했습니다."]
    for r in results:
        if not r["ok"]:
            lines.append(f"- {r['system_name']}: 실행 실패 — {r['message']}")
            continue
        b, a = r["before"], r["after"]
        lines.append(f"- {r['system_name']}: {r['label']} → 상태 {b['status']} → {a['status']}, "
                     f"CPU {b['cpu']}% → {a['cpu_percent']}%, 메모리 {b['memory']}% → {a['memory_percent']}%"
                     + ("" if a["status"] == "healthy" else " (아직 정상화되지 않음 — 추가 확인 필요)"))
    return "\n".join(lines)
