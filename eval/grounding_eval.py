"""답변 수치 근거 검증 Before/After 평가 — 실제 LLM으로 18문항 전체 파이프라인을 돌린다.

같은 질문을 GROUNDING_CHECK=off(개선 전: 형식만 검사) → on(개선 후) 순서로 실행하고,
최종 답변에 '수집 데이터로 확인되지 않는 수치'가 몇 개 남았는지 같은 검사기로 센다.

실행 (서버를 따로 띄울 필요 없음 — 내부에서 에이전트를 직접 기동):
    python eval/grounding_eval.py                  # Ollama, 18문항 (8b 기준 10~20분)
    python eval/grounding_eval.py --limit 6        # 앞 6문항만 (빠른 확인)
    LLM_BACKEND=openai python eval/grounding_eval.py   # OpenAI (과금)

지표
  - 근거 없는 수치가 포함된 답변 비율 (낮을수록 좋음, 경고 문구 자체는 제외하고 셈)
  - 근거 없는 수치 총 개수
  - 평균 검수 시도 횟수 / 평균 응답 시간 (검증 비용)
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("LLM_BACKEND", "ollama")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ["REMEDIATION"] = "off"   # 개선 8의 승인 단계(체크포인터)는 이 평가와 무관 — 끄고 잰다

from agent import grounding  # noqa: E402
import server  # noqa: E402

CASES = json.loads((ROOT / "eval" / "runbook_cases.json").read_text())


async def run_mode(app, mode: str, cases: list[dict]) -> list[dict]:
    os.environ["GROUNDING_CHECK"] = mode          # validate_node가 호출 시점에 읽는다
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        try:
            r = await app.state.graph.ainvoke(
                {"query": c["query"], "thread_id": f"g-{mode}-{c['id']}-{time.time_ns()}",
                 "history": [], "events": [], "cost_usd": 0.0})
        except Exception as exc:
            print(f"  #{c['id']} 실패: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0
        body = grounding.strip_warning(r["final"])
        facts = grounding.build_fact_text(c["query"], r.get("collected", []), r.get("last_prompt"))
        bad = grounding.find_ungrounded(body, facts)
        # 검수 도중 반려 사유가 된 수치 — 최종 답변에는 안 남아도 검증기가 실제로 막아낸 것
        caught = [n for e in r.get("events", []) if e.get("type") == "Validation"
                  and not e.get("verdict", "").startswith("합격") for n in e.get("ungrounded", [])]
        rows.append({"id": c["id"], "route": r.get("route"), "ungrounded": bad, "caught": caught,
                     "warned": r["final"] != body, "attempts": r.get("validation_attempts", 0),
                     "sec": dt})
        mark = ("❌ " + ", ".join(bad) if bad else "✅") + \
            (f"  (검수 중 차단: {', '.join(caught)})" if caught else "")
        print(f"  [{mode}] #{c['id']:>2} {dt:5.1f}s 검수 {rows[-1]['attempts']}회  {mark}"
              + ("  (⚠️ 경고 표시)" if rows[-1]["warned"] else ""))
    return rows


def summarize(rows):
    inc = [r for r in rows if r["route"] == "incident"]
    n = len(inc) or 1
    return {"n": len(inc),
            "bad_answers": sum(1 for r in inc if r["ungrounded"]),
            "bad_numbers": sum(len(r["ungrounded"]) for r in inc),
            "warned": sum(1 for r in inc if r["warned"]),
            "caught_answers": sum(1 for r in inc if r["caught"]),
            "caught": sum(len(r["caught"]) for r in inc),
            "attempts": sum(r["attempts"] for r in inc) / n,
            "sec": sum(r["sec"] for r in inc) / n}


async def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(CASES)
    cases = CASES[:limit]
    async with server.lifespan(server.app):
        app = server.app
        print(f"백엔드: {app.state.llm_backend} ({app.state.deps.quality_llm.name}) · {len(cases)}문항\n")
        results = {}
        for mode in ("off", "on"):
            print(f"GROUNDING_CHECK={mode}")
            results[mode] = await run_mode(app, mode, cases)
            print()

    s = {m: summarize(r) for m, r in results.items()}
    lines = [f"## 답변 수치 근거 검증 — {app.state.deps.quality_llm.name} "
             f"({datetime.now():%Y-%m-%d %H:%M}, 장애 답변 {s['on']['n']}건)", "",
             "| 항목 | 개선 전 (형식만 검사) | 개선 후 (수치 근거 검사) |", "|---|---|---|",
             f"| 근거 없는 수치가 남은 답변 | {s['off']['bad_answers']}/{s['off']['n']} "
             f"| {s['on']['bad_answers']}/{s['on']['n']} |",
             f"| 근거 없는 수치 총 개수 | {s['off']['bad_numbers']} | {s['on']['bad_numbers']} |",
             f"| 그중 사용자에게 경고 표시 | - | {s['on']['warned']} |",
             f"| 검수 중 근거 없는 수치를 잡아 재작성시킨 답변 | - | {s['on']['caught_answers']} "
             f"(수치 {s['on']['caught']}개) |",
             f"| 평균 검수 시도 | {s['off']['attempts']:.2f} | {s['on']['attempts']:.2f} |",
             f"| 평균 응답 시간 | {s['off']['sec']:.1f}s | {s['on']['sec']:.1f}s |", "",
             "### 개선 전 답변에서 발견된 근거 없는 수치", ""]
    lines += [f"- #{r['id']}: {', '.join(r['ungrounded'])}" for r in results["off"] if r["ungrounded"]] \
        or ["- (없음)"]
    lines += ["", "### 개선 후 검수 과정에서 차단된 수치 (재작성으로 제거됨)", ""]
    lines += [f"- #{r['id']}: {', '.join(r['caught'])}" for r in results["on"] if r["caught"]] \
        or ["- (없음)"]
    report = "\n".join(lines)
    print(report)
    out = ROOT / "eval" / f"grounding_results_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text(report + "\n")
    print(f"\n결과 저장: eval/{out.name}")


if __name__ == "__main__":
    asyncio.run(main())
