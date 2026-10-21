"""개선 7 Before/After — 주 LLM(OpenAI)이 죽었을 때 서비스가 어떻게 되는가.

과금 없이 재현한다: 가짜 OpenAI(장애 주입: 1초 기다린 뒤 503) + 가짜 Ollama(0.5초 정상 응답)를 띄우고,
에이전트 서버를 설정만 바꿔 3번 기동해 같은 장애 질문 N건을 순서대로 보낸다.

    python bench/fallback_bench.py            # 기본 6건
    python bench/fallback_bench.py --n 10

시나리오
  ① 개선 전        LLM_FALLBACK=off                       — 주 LLM 장애 = 서비스 장애
  ② 대체만         LLM_FALLBACK=ollama, 브레이커 사실상 끔   — 답은 나오지만 매 호출마다 죽은 서버를 기다림
  ③ 대체+브레이커   LLM_FALLBACK=ollama, 3회 연속 실패 시 차단 — 처음 몇 번만 기다리고 이후엔 바로 보조로
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
AGENT_PORT, PRIMARY_PORT, BACKUP_PORT = 8199, 8710, 8711
QUERY = "auth-service 로그인 응답 지연 장애 진단해줘"
FAIL_LATENCY, BACKUP_LATENCY = "1.0", "0.5"

SCENARIOS = [
    ("① 개선 전 (대체 없음)", {"LLM_FALLBACK": "off"}),
    ("② 대체만 (브레이커 없음)", {"LLM_FALLBACK": "ollama", "LLM_CB_THRESHOLD": "100000"}),
    ("③ 대체 + 서킷 브레이커", {"LLM_FALLBACK": "ollama", "LLM_CB_THRESHOLD": "3", "LLM_CB_COOLDOWN": "300"}),
]


def start(args, env, log):
    return subprocess.Popen(args, cwd=ROOT, env={**os.environ, **env},
                            stdout=open(log, "w"), stderr=subprocess.STDOUT)


def wait_up(url, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=2).status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"{url} 기동 실패")


def run_scenario(name, extra, n, logdir):
    primary = start([PY, "bench/mock_llm_server.py"],
                    {"MOCK_PORT": str(PRIMARY_PORT), "MOCK_FAIL": "503", "MOCK_LATENCY": FAIL_LATENCY},
                    logdir / "primary.log")
    backup = start([PY, "bench/mock_llm_server.py"],
                   {"MOCK_PORT": str(BACKUP_PORT), "MOCK_LATENCY": BACKUP_LATENCY}, logdir / "backup.log")
    env = {"LLM_BACKEND": "openai", "OPENAI_API_KEY": "mock",
           "OPENAI_BASE_URL": f"http://127.0.0.1:{PRIMARY_PORT}/v1",
           "OLLAMA_URL": f"http://127.0.0.1:{BACKUP_PORT}/api/chat",
           "RUNBOOK_RETRIEVER": "keyword", "REMEDIATION": "off", "GROUNDING_CHECK": "off",
           "LANGCHAIN_TRACING_V2": "false", **extra}
    agent = start([PY, "-m", "uvicorn", "server:app", "--port", str(AGENT_PORT), "--log-level", "warning"],
                  env, logdir / "agent.log")
    rows = []
    try:
        wait_up(f"http://127.0.0.1:{PRIMARY_PORT}/calls")
        wait_up(f"http://127.0.0.1:{BACKUP_PORT}/calls")
        wait_up(f"http://127.0.0.1:{AGENT_PORT}/health")
        with httpx.Client(timeout=120) as c:
            for i in range(n):
                t0 = time.perf_counter()
                r = c.post(f"http://127.0.0.1:{AGENT_PORT}/chat",
                           json={"message": QUERY, "thread_id": f"bench-{i}"})
                ms = (time.perf_counter() - t0) * 1000
                body = r.json()
                fb = [e for e in body.get("events", []) if e.get("type") == "LLMFallback"]
                rows.append({"i": i + 1, "status": r.status_code, "ms": ms,
                             "code": (body.get("error") or {}).get("code"),
                             "fallbacks": len(fb),
                             "reasons": sorted({e["reason"] for e in fb})})
                print(f"  [{name}] #{i + 1} {r.status_code} {ms:7.0f}ms fallback={len(fb)} "
                      f"{rows[-1]['reasons'] or rows[-1]['code'] or ''}")
        calls = httpx.get(f"http://127.0.0.1:{PRIMARY_PORT}/calls").json()["openai"]
        breaker = httpx.get(f"http://127.0.0.1:{AGENT_PORT}/health").json().get("llm_fallback")
    finally:
        for p in (agent, primary, backup):
            p.terminate()
        for p in (agent, primary, backup):
            p.wait(timeout=10)
    return rows, calls, breaker


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 6
    logdir = ROOT / "bench" / "logs"
    logdir.mkdir(exist_ok=True)
    results = []
    for name, extra in SCENARIOS:
        print(name)
        results.append((name, *run_scenario(name, extra, n, logdir)))

    lines = [f"# LLM 장애 자동 대체 벤치마크 — {datetime.now():%Y-%m-%d %H:%M}", "",
             f"주 LLM(가짜 OpenAI)은 매 호출 {FAIL_LATENCY}초 뒤 503 · 보조 LLM(가짜 Ollama)은 {BACKUP_LATENCY}초 정상 응답 · "
             f"장애 질문 {n}건 순차", "",
             "| 시나리오 | 성공 | 평균 응답 | 첫 요청 | 마지막 요청 | 죽은 주 LLM 호출 수 |",
             "|---|---|---|---|---|---|"]
    for name, rows, calls, _ in results:
        ok = [r for r in rows if r["status"] == 200]
        avg = sum(r["ms"] for r in ok) / len(ok) / 1000 if ok else 0
        lines.append(f"| {name} | {len(ok)}/{len(rows)} | {avg:.1f}s | {rows[0]['ms'] / 1000:.1f}s | "
                     f"{rows[-1]['ms'] / 1000:.1f}s | {calls} |")
    lines += ["", "## 요청별", "", "| 시나리오 | # | HTTP | 응답 | 대체 횟수 | 사유·오류 코드 |", "|---|---|---|---|---|---|"]
    for name, rows, _, _ in results:
        for r in rows:
            lines.append(f"| {name} | {r['i']} | {r['status']} | {r['ms'] / 1000:.1f}s | {r['fallbacks']} | "
                         f"{', '.join(r['reasons']) or r['code'] or ''} |")
    out = ROOT / "eval" / f"fallback_results_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:8 + len(results)]))
    print(f"\n결과 저장: eval/{out.name}")


if __name__ == "__main__":
    main()
