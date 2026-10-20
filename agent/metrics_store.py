"""요청 단위 운영 지표 저장소 — 관리자 대시보드의 데이터 원천 (개선 6).

LangSmith는 '한 요청 안에서 노드·LLM이 어떻게 오갔는가'(트레이스)를 보는 도구라면,
이 저장소는 '서비스 전체가 지금 어떤 상태인가'(운영 지표)를 본다:
요청 수·성공률·응답 시간·비용·오류 코드·마스킹·runbook 선택·근거 경고.

원칙
  - 질문은 가드레일을 통과한 '마스킹본'만 저장한다 — 관리자도 원문 민감정보를 볼 수 없다
  - 정책 거절(rejected)은 오류(error)와 구분한다 — 서비스 장애가 아니라 정상적인 방어 동작
  - 기록 실패가 사용자 응답을 막아서는 안 된다 (record는 예외를 삼킨다)
"""
import json
import logging
import sqlite3
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("agent.metrics")
DB_PATH = Path(__file__).parent.parent / "session.db"


class MetricsStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, channel TEXT, thread_id TEXT, query TEXT,
                status TEXT, code TEXT, error_id TEXT,
                route TEXT, latency_ms REAL, cost_usd REAL, backend TEXT,
                masked TEXT, runbooks TEXT, runbook_method TEXT,
                validation_attempts INTEGER, ungrounded INTEGER, run_id TEXT)""")
        self.conn.commit()

    def record(self, *, channel: str, thread_id: str, query: str, status: str,
               latency_ms: float, backend: str, code: str | None = None,
               error_id: str | None = None, route: str | None = None,
               cost_usd: float = 0.0, masked: list[str] | None = None,
               runbooks: list[str] | None = None, runbook_method: str | None = None,
               validation_attempts: int = 0, ungrounded: int = 0,
               run_id: str | None = None) -> None:
        try:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO requests (ts, channel, thread_id, query, status, code, error_id,"
                    " route, latency_ms, cost_usd, backend, masked, runbooks, runbook_method,"
                    " validation_attempts, ungrounded, run_id)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), channel, thread_id,
                     (query or "")[:200], status, code, error_id, route, round(latency_ms, 1),
                     cost_usd, backend, json.dumps(masked or []), json.dumps(runbooks or []),
                     runbook_method, validation_attempts, ungrounded, run_id))
                self.conn.commit()
        except Exception as exc:   # 지표 기록 실패가 서비스 응답을 막지 않게
            logger.warning("metrics record 실패: %s", exc)

    def _rows(self, since: datetime) -> list[dict]:
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM requests WHERE ts >= ? ORDER BY id", (since.isoformat(),))
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        for r in rows:
            r["masked"] = json.loads(r["masked"] or "[]")
            r["runbooks"] = json.loads(r["runbooks"] or "[]")
        return rows

    def recent(self, limit: int = 50) -> list[dict]:
        rows = self._rows(datetime.min)
        return list(reversed(rows[-limit:]))

    def summary(self, hours: int = 24) -> dict:
        now = datetime.now()
        rows = self._rows(now - timedelta(hours=hours))
        ok = [r for r in rows if r["status"] == "ok"]
        errors = [r for r in rows if r["status"] == "error"]
        rejected = [r for r in rows if r["status"] == "rejected"]
        lat = sorted(r["latency_ms"] for r in ok)
        incident = [r for r in ok if r["route"] == "incident"]

        def pct(p):
            return lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))] if lat else 0.0

        # 시간대별(1시간 단위) 상태별 요청 수 — 최근 12칸
        buckets = []
        for i in range(11, -1, -1):
            start = (now - timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
            inb = [r for r in rows if start.isoformat() <= r["ts"] < end.isoformat()]
            buckets.append({"hour": start.strftime("%H시"),
                            **{s: sum(1 for r in inb if r["status"] == s)
                               for s in ("ok", "rejected", "error")}})
        return {
            "window_hours": hours,
            "total": len(rows), "ok": len(ok), "rejected": len(rejected), "error": len(errors),
            # 성공률의 분모에서 정책 거절은 뺀다 — 거절은 서비스 실패가 아니다
            "success_rate": round(len(ok) / (len(ok) + len(errors)), 4) if ok or errors else None,
            "latency_avg_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
            "latency_p95_ms": pct(0.95),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in ok), 6),
            "masked_requests": sum(1 for r in rows if r["masked"]),
            "masked_types": Counter(t for r in rows for t in r["masked"]),
            "codes": Counter(r["code"] for r in rejected + errors if r["code"]),
            "routes": Counter(r["route"] for r in ok if r["route"]),
            "runbooks": Counter(t for r in incident for t in r["runbooks"]),
            "runbook_methods": Counter(r["runbook_method"] for r in incident if r["runbook_method"]),
            "avg_validation_attempts": round(sum(r["validation_attempts"] or 0 for r in incident)
                                             / len(incident), 2) if incident else 0.0,
            "grounding_warnings": sum(1 for r in incident if r["ungrounded"]),
            "hourly": buckets,
        }

    def close(self) -> None:
        self.conn.close()
