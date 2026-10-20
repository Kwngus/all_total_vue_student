"""동시 요청 벤치마크 — 실행 중인 에이전트 서버(:8100)에 부하를 주고 응답 시간을 잰다.

측정 항목
  A) LLM 처리 중 /health 응답 시간
     장애 진단 요청 1건을 보내 놓고, 처리되는 동안 /health를 5번 호출한다.
     동기 호출이면 LLM 응답이 끝날 때까지 /health도 멈춘다 → '서버 전체가 얼어붙는가'를 본다.
  B) 동시 진단 요청 N건 (기본 3건)
     N명이 동시에 질문했을 때 각자가 기다린 시간과 전체 소요 시간.
     동기 호출이면 요청이 한 줄로 서서 마지막 사람은 N배를 기다린다.

실행 (서버를 먼저 띄운 뒤):
    python bench/concurrency_bench.py            # 기본 3건
    python bench/concurrency_bench.py 5          # 5건
결과는 터미널 표 + bench/results_<모드>_<시각>.md 로 저장된다 (모드는 서버 /health에서 읽음).
"""
import asyncio
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8100"
QUERY = "auth-service 로그인 응답이 느려졌어요. 원인 확인해주세요"


async def timed(client: httpx.AsyncClient, method: str, url: str, **kw) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = await client.request(method, url, **kw)
    return time.perf_counter() - t0, r.status_code


async def chat(client, i):
    return await timed(client, "POST", f"{BASE}/chat",
                       json={"message": QUERY, "thread_id": f"bench-{i}-{time.time_ns()}"})


async def scenario_health(client) -> list[float]:
    task = asyncio.create_task(chat(client, "h"))
    await asyncio.sleep(0.3)               # 진단 요청이 LLM 호출 단계에 들어갈 시간
    lat = []
    for _ in range(5):
        dt, _ = await timed(client, "GET", f"{BASE}/health")
        lat.append(dt)
        await asyncio.sleep(0.1)
    await task
    return lat


async def scenario_concurrent(client, n: int) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    results = await asyncio.gather(*(chat(client, i) for i in range(n)))
    total = time.perf_counter() - t0
    bad = [code for _, code in results if code != 200]
    if bad:
        print(f"⚠️  실패 응답: {bad}")
    return [dt for dt, _ in results], total


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    async with httpx.AsyncClient(timeout=600.0) as client:
        h = (await client.get(f"{BASE}/health")).json()
        mode = h.get("llm_http_mode", "sync(개선 전 코드)")
        label = f"{h['llm_backend']} · LLM_HTTP_MODE={mode}"
        print(f"대상: {label}\n워밍업 1건...")
        await chat(client, "warmup")                     # 모델 로딩 시간은 측정에서 제외

        print("A) LLM 처리 중 /health 응답 시간 측정...")
        health = await scenario_health(client)
        print(f"B) 동시 진단 요청 {n}건 측정...")
        each, total = await scenario_concurrent(client, n)

    lines = [f"## 동시성 벤치마크 — {label} ({datetime.now():%Y-%m-%d %H:%M})", "",
             "| 항목 | 값 |", "|---|---|",
             f"| A) LLM 처리 중 /health 평균 | {statistics.mean(health) * 1000:.0f} ms |",
             f"| A) LLM 처리 중 /health 최대 | {max(health) * 1000:.0f} ms |",
             f"| B) 동시 {n}건 전체 소요 | {total:.2f} s |",
             f"| B) 요청별 응답 시간 | {', '.join(f'{x:.2f}s' for x in sorted(each))} |",
             f"| B) 가장 늦게 받은 사람 | {max(each):.2f} s |"]
    report = "\n".join(lines)
    print("\n" + report)
    out = Path(__file__).parent / f"results_{mode.split('(')[0]}_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text(report + "\n")
    print(f"\n결과 저장: bench/{out.name}")


if __name__ == "__main__":
    asyncio.run(main())
