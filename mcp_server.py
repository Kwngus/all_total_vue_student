"""all_total 통합 실습 — MCP Server (Wave 2 축).

Tool / Resource / Prompt 3대 capability를 모두 노출하는 실제 fastmcp 서버.
FastAPI 백엔드(server.py)가 lifespan에서 이 파일을 subprocess(stdio)로 1회 기동해
프로세스가 살아 있는 동안 재사용한다 (Wave 1 싱글턴 + Wave 2 표준화의 결합).

- Tool    get_monitoring_status : 호출마다 값이 달라지는 동적 조회 → Tool이 맞다
- Resource runbook://incident-response/{incident_type} : URI로 식별되는 정적 문서
- Prompt  incident_analysis_prompt : 여러 Tool을 조합하는 표준 진단 절차를 강제
- Tool    execute_remediation : (개선 8) 사람이 승인한 조치만 실행 — 허용 목록 밖의 조치는 거부

조치 실행은 시뮬레이션이다: 실제 서버를 건드리지 않고 data/remediation_state.json에 '조치 적용'을
기록하고, 조치 후 일정 시간(REMEDIATION_TTL, 기본 600초) 동안 get_monitoring_status가 그 시스템을
정상 상태로 보고한다. 실제 운영이라면 이 함수 안이 쿠버네티스·클라우드 API 호출 자리다.
"""
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP(name="capstone-all-total-ops")

DATA_DIR = Path(__file__).parent / "data"
STATE_PATH = DATA_DIR / "remediation_state.json"
REMEDIATION_TTL = float(os.environ.get("REMEDIATION_TTL", "600"))

# 실행 가능한 조치의 허용 목록 — 에이전트(LLM)가 무엇을 제안하든 여기 없는 조치는 실행되지 않는다
ALLOWED_ACTIONS = {
    "scale_out": "인스턴스 1대 증설",
    "restart_service": "서비스 재시작 (롤링)",
    "clear_old_logs": "7일 지난 로그 정리",
    "failover_db": "DB 읽기 복제본으로 전환",
}


def _active_remediations() -> dict:
    try:
        state = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    now = time.time()
    return {k: v for k, v in state.items() if now - v.get("ts", 0) < REMEDIATION_TTL}


class SystemStatus(BaseModel):
    system_name: str = Field(description="시스템 이름")
    status: str = Field(description="상태: healthy / degraded / down")
    cpu_percent: float = Field(description="CPU 사용률 (%)")
    memory_percent: float = Field(description="메모리 사용률 (%)")
    last_checked: str = Field(description="마지막 체크 시각 (ISO 8601)")
    alerts: list[str] = Field(default_factory=list, description="활성 알림 목록")


@mcp.tool()
def get_monitoring_status(system_name: str) -> SystemStatus:
    """지정된 시스템의 현재 모니터링 상태를 조회합니다 (호출마다 값이 달라지는 동적 Tool)."""
    monitoring_data = json.loads((DATA_DIR / "monitoring.json").read_text())
    if system_name not in monitoring_data:
        return SystemStatus(
            system_name=system_name, status="unknown",
            cpu_percent=0.0, memory_percent=0.0,
            last_checked=datetime.now().isoformat(),
            alerts=[f"System '{system_name}' not found in monitoring"],
        )
    data = monitoring_data[system_name]
    jitter = random.uniform(-3.0, 8.0)
    applied = _active_remediations().get(system_name)
    if applied:   # 승인된 조치가 적용된 시스템 — 시뮬레이션상 정상화된 상태로 보고
        return SystemStatus(
            system_name=system_name, status="healthy",
            cpu_percent=round(min(data["cpu_base"], 45.0) + jitter / 2, 1),
            memory_percent=round(min(data["memory_base"], 50.0) + jitter / 4, 1),
            last_checked=datetime.now().isoformat(timespec="microseconds"),
            alerts=[],
        )
    return SystemStatus(
        system_name=system_name,
        status=data["status"],
        cpu_percent=round(data["cpu_base"] + jitter, 1),
        memory_percent=round(data["memory_base"] + jitter / 2, 1),
        last_checked=datetime.now().isoformat(timespec="microseconds"),
        alerts=data.get("alerts", []),
    )


class RemediationResult(BaseModel):
    ok: bool
    system_name: str
    action: str
    message: str


@mcp.tool()
def execute_remediation(system_name: str, action: str, approval_id: str) -> RemediationResult:
    """승인된 장애 조치를 실행합니다 (시뮬레이션). approval_id 없이는 실행하지 않습니다."""
    systems = json.loads((DATA_DIR / "monitoring.json").read_text())
    if not approval_id:
        return RemediationResult(ok=False, system_name=system_name, action=action,
                                 message="승인 번호가 없어 실행하지 않았습니다")
    if action not in ALLOWED_ACTIONS:
        return RemediationResult(ok=False, system_name=system_name, action=action,
                                 message=f"허용되지 않은 조치입니다: {action}")
    if system_name not in systems:
        return RemediationResult(ok=False, system_name=system_name, action=action,
                                 message=f"알 수 없는 시스템입니다: {system_name}")
    state = _active_remediations()
    state[system_name] = {"action": action, "approval_id": approval_id, "ts": time.time()}
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return RemediationResult(ok=True, system_name=system_name, action=action,
                             message=f"{ALLOWED_ACTIONS[action]} 완료 (시뮬레이션)")


@mcp.resource("runbook://incident-response/{incident_type}")
def get_incident_runbook(incident_type: str) -> str:
    """특정 incident 유형의 runbook(정적 문서)을 반환합니다."""
    runbook_path = DATA_DIR / "runbooks" / f"{incident_type}.md"
    if not runbook_path.exists():
        return f"# Runbook Not Found\n\nNo runbook for incident type: {incident_type}"
    return runbook_path.read_text()


@mcp.prompt()
def incident_analysis_prompt(system_name: str, symptom: str) -> str:
    """장애 진단 표준 절차 프롬프트 — planning 노드가 계획을 세울 때 이 절차를 따른다."""
    return f"""다음 장애에 대해 표준 진단 절차를 수행하세요.

## 장애 정보
- 시스템: {system_name}
- 증상: {symptom}

## 진단 절차 (반드시 순서대로 수행)
1. get_monitoring_status Tool로 관련 시스템의 현재 상태를 조회하세요.
2. 증상에 해당하는 runbook(runbook://incident-response/...)을 확인하세요.
3. 위 정보를 종합해 가능한 원인과 권장 조치를 제시하세요.
"""


if __name__ == "__main__":
    try:
        mcp.run(show_banner=False)
    except TypeError:   # 구버전 fastmcp 호환
        mcp.run()
