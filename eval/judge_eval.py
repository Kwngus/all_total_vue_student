"""LLM-as-a-Judge 평가 — 원본 설정 vs 개선 설정의 '최종 답변 품질'을 LLM이 채점한다.

지금까지의 평가는 코드로 셀 수 있는 것(정답 runbook 포함률, 근거 없는 수치 개수, 지연)만 쟀다.
하지만 "답변이 질문에 맞는 절차를 안내하는가", "조치가 실행 가능한가"는 규칙으로 채점하기 어렵다.
그래서 별도의 LLM(심판)에게 채점 기준을 주고 점수를 매기게 한다.

비교 대상 (같은 18문항, 같은 LLM으로 답변 생성)
  - 원본 설정: runbook 키워드 규칙 + 형식만 검수   (RUNBOOK_RETRIEVER=keyword, GROUNDING_CHECK=off)
  - 개선 설정: runbook 하이브리드 RAG + 수치 근거 검증

채점 방식
  1) 개별 채점 — 답변마다 1~5점 3항목
       relevance      질문 증상에 맞는 장애 유형·절차를 안내하는가 (정답 장애 유형을 참고로 준다)
       groundedness   답변 속 사실·수치가 '수집된 사실'과 일치하는가 (지어낸 내용이 없는가)
       actionability  조치가 구체적이고 순서대로 실행 가능한가
  2) 블라인드 쌍대 비교 — 두 답변을 A/B로 섞어 "어느 쪽이 나은가"를 묻고, 순서를 바꿔 한 번 더 묻는다.
     두 번 모두 같은 쪽을 고른 경우만 승리로 센다 (심판 LLM의 '앞에 나온 답 선호' 편향 제거)

실행 (서버를 따로 띄울 필요 없음)
    LLM_BACKEND=openai python eval/judge_eval.py          # 답변 생성 OpenAI (약 4분, 약 $0.1)
    python eval/judge_eval.py                             # 답변 생성 Ollama (무료, 15~20분)
    python eval/judge_eval.py --limit 6                   # 앞 6문항만
  심판: OPENAI_API_KEY가 있으면 gpt-4o-mini (JUDGE_MODEL로 변경), 없으면 Ollama qwen3:8b
"""
import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("LLM_BACKEND", "ollama")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ["REMEDIATION"] = "off"   # 답변 품질만 비교 — 조치 승인 단계는 끈다

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)   # 셸에 남은 예전 키보다 프로젝트 .env 우선

from agent import grounding, llm_http  # noqa: E402
import server  # noqa: E402

CASES = json.loads((ROOT / "eval" / "runbook_cases.json").read_text())
random.seed(7)

POINTWISE = """당신은 IT 운영 장애 대응 답변을 채점하는 엄격한 심사위원이다.
아래 [사용자 질문], [참고: 정답 장애 유형], [수집된 사실], [답변]을 보고 세 항목을 1~5점 정수로 채점하라.
- relevance: 질문의 증상에 맞는 장애 유형과 대응 절차를 안내하는가 (정답 장애 유형과 다른 절차면 2점 이하)
- groundedness: 답변 속 사실과 수치가 [수집된 사실]과 일치하는가 (수집되지 않은 수치·상태를 지어내면 2점 이하)
- actionability: 조치가 구체적이고 순서대로 바로 실행할 수 있는가
JSON으로만 답하라: {"relevance": n, "groundedness": n, "actionability": n, "reason": "한국어 한 문장"}"""

PAIRWISE = """당신은 IT 운영 장애 대응 답변을 비교하는 엄격한 심사위원이다.
[사용자 질문]과 [참고: 정답 장애 유형]을 보고, 운영자에게 더 정확하고 쓸모 있는 답변을 고르라.
판단 기준: 질문에 맞는 절차인가 > 수집된 사실과 일치하는가 > 조치가 구체적인가. 길이는 기준이 아니다.
JSON으로만 답하라: {"winner": "A" 또는 "B" 또는 "tie", "reason": "한국어 한 문장"}"""


class Judge:
    def __init__(self):
        if os.environ.get("OPENAI_API_KEY"):
            self.kind, self.model = "openai", os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        else:
            self.kind, self.model = "ollama", os.environ.get("JUDGE_MODEL", "qwen3:8b")
        self.name = f"{self.kind}:{self.model}"
        self.tokens = 0

    async def ask(self, system: str, user: str) -> dict:
        if self.kind == "openai":
            base = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            r = await llm_http.post_json(f"{base}/chat/completions", {
                "model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, timeout=120)
            r.raise_for_status()
            d = r.json()
            self.tokens += d.get("usage", {}).get("total_tokens", 0)
            text = d["choices"][0]["message"]["content"]
        else:
            r = await llm_http.post_json(os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"), {
                "model": self.model, "stream": False, "format": "json", "think": False,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                timeout=300)
            r.raise_for_status()
            text = r.json()["message"]["content"]
        try:
            return json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            return {}


def facts_of(result: dict) -> str:
    """심판에게 주는 '수집된 사실' — 모니터링 원본 값과 조회한 runbook 제목."""
    lines = []
    for c in result.get("collected", []):
        d = c["data"]
        if c["source"] == "monitoring":
            lines.append(f"- 모니터링 {d['system_name']}: 상태={d['status']}, CPU {d['cpu_percent']}%, "
                         f"메모리 {d['memory_percent']}%, 알림={d.get('alerts', [])}")
        elif c["source"] == "runbook":
            lines.append(f"- 조회한 runbook: {str(d).splitlines()[0][:80]}")
    return "\n".join(lines) or "- (수집된 사실 없음)"


async def generate(app, mode: str, cases: list[dict]) -> dict:
    """mode=before: 키워드 규칙 + 형식 검수 / after: 하이브리드 RAG + 근거 검증."""
    deps = app.state.deps
    saved = deps.runbook_retriever
    deps.runbook_retriever = None if mode == "before" else saved
    os.environ["GROUNDING_CHECK"] = "off" if mode == "before" else "on"
    out = {}
    try:
        for c in cases:
            t0 = time.perf_counter()
            try:
                r = await app.state.graph.ainvoke(
                    {"query": c["query"], "thread_id": f"j-{mode}-{c['id']}", "history": [],
                     "events": [], "cost_usd": 0.0},
                    {"configurable": {"thread_id": f"judge-{mode}-{c['id']}-{time.time_ns()}"}})
            except Exception as exc:
                print(f"  [{mode}] #{c['id']} 생성 실패: {type(exc).__name__}: {exc}")
                continue
            rb = next((e.get("target") for e in r.get("events", [])
                       if e.get("type") == "ReplanCheck" and not e.get("is_sufficient")), "")
            out[c["id"]] = {"answer": grounding.strip_warning(r.get("final", "")).strip(),
                            "facts": facts_of(r), "runbook": rb, "route": r.get("route"),
                            "sec": time.perf_counter() - t0}
            print(f"  [{mode}] #{c['id']:>2} {out[c['id']]['sec']:5.1f}s runbook={rb or '-'}")
    finally:
        deps.runbook_retriever = saved
    return out


def _clip(n) -> int:
    try:
        return max(1, min(5, int(n)))
    except (TypeError, ValueError):
        return 0


async def judge_all(judge: Judge, cases, gens):
    rows = []
    for c in cases:
        b, a = gens["before"].get(c["id"]), gens["after"].get(c["id"])
        if not b or not a:
            continue
        gold = ", ".join(c["expected"])
        scores = {}
        for mode, g in (("before", b), ("after", a)):
            s = await judge.ask(POINTWISE, f"[사용자 질문]\n{c['query']}\n\n[참고: 정답 장애 유형]\n{gold}\n\n"
                                           f"[수집된 사실]\n{g['facts']}\n\n[답변]\n{g['answer']}")
            scores[mode] = {k: _clip(s.get(k)) for k in ("relevance", "groundedness", "actionability")}
            scores[mode]["reason"] = s.get("reason", "")
        votes = []
        for first in ("before", "after"):   # 같은 쌍을 순서만 바꿔 두 번 묻는다
            second = "after" if first == "before" else "before"
            p = await judge.ask(PAIRWISE, f"[사용자 질문]\n{c['query']}\n\n[참고: 정답 장애 유형]\n{gold}\n\n"
                                          f"[답변 A]\n{gens[first][c['id']]['answer']}\n\n"
                                          f"[답변 B]\n{gens[second][c['id']]['answer']}")
            w = str(p.get("winner", "tie")).strip().upper()
            votes.append(first if w == "A" else second if w == "B" else "tie")
        winner = votes[0] if votes[0] == votes[1] else "tie"   # 두 번 모두 같은 쪽일 때만 승리
        rows.append({"id": c["id"], "query": c["query"], "gold": gold,
                     "rb_before": b["runbook"], "rb_after": a["runbook"],
                     "before": scores["before"], "after": scores["after"], "winner": winner})
        sb, sa = scores["before"], scores["after"]
        print(f"  #{c['id']:>2} 원본 {sb['relevance']}/{sb['groundedness']}/{sb['actionability']}  "
              f"개선 {sa['relevance']}/{sa['groundedness']}/{sa['actionability']}  쌍대: {winner}")
    return rows


def report(rows, judge, gen_name, gens) -> str:
    def avg(mode, k):
        v = [r[mode][k] for r in rows if r[mode][k]]
        return sum(v) / len(v) if v else 0.0
    n = len(rows)
    wins = {k: sum(1 for r in rows if r["winner"] == k) for k in ("after", "before", "tie")}
    sec = {m: sum(g["sec"] for g in gens[m].values()) / max(1, len(gens[m])) for m in gens}
    L = [f"# LLM-as-a-Judge 평가 — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"답변 생성: {gen_name} · 심판: {judge.name} · {n}문항 (runbook_cases.json)", "",
         "## 개별 채점 (1~5점 평균)", "",
         "| 항목 | 원본 설정 (키워드 규칙 + 형식 검수) | 개선 설정 (하이브리드 RAG + 근거 검증) |", "|---|---|---|"]
    for k, label in (("relevance", "질문 적합성 (relevance)"), ("groundedness", "근거 일치 (groundedness)"),
                     ("actionability", "실행 가능성 (actionability)")):
        L.append(f"| {label} | {avg('before', k):.2f} | {avg('after', k):.2f} |")
    L += [f"| 평균 응답 시간 | {sec['before']:.1f}s | {sec['after']:.1f}s |", "",
          "## 블라인드 쌍대 비교 (순서를 바꿔 두 번 물어 같은 쪽을 고른 경우만 승리)", "",
          "| 개선 설정 승 | 원본 설정 승 | 무승부·판정 불일치 |", "|---|---|---|",
          f"| {wins['after']} | {wins['before']} | {wins['tie']} |", "",
          "## 문항별", "",
          "| # | 정답 유형 | runbook 원본 → 개선 | 원본 R/G/A | 개선 R/G/A | 쌍대 승자 | 심판 메모 (개선 답변) |",
          "|---|---|---|---|---|---|---|"]
    name = {"after": "개선", "before": "원본", "tie": "무"}
    for r in rows:
        b, a = r["before"], r["after"]
        L.append(f"| {r['id']} | {r['gold']} | {r['rb_before'] or '-'} → {r['rb_after'] or '-'} | "
                 f"{b['relevance']}/{b['groundedness']}/{b['actionability']} | "
                 f"{a['relevance']}/{a['groundedness']}/{a['actionability']} | {name[r['winner']]} | "
                 f"{a['reason'].replace('|', '/')} |")
    L += ["", "R = relevance, G = groundedness, A = actionability",
          "주의: 심판도 LLM이라 절대 점수보다 같은 심판으로 잰 '차이'를 본다. "
          "정답 장애 유형을 참고로 주는 reference-guided 방식이다."]
    if judge.kind == "openai":
        L.append(f"심판 사용 토큰: {judge.tokens:,} (gpt-4o-mini 기준 약 ${judge.tokens * 0.3 / 1e6:.4f})")
    return "\n".join(L)


async def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else len(CASES)
    cases = CASES[:limit]
    judge = Judge()
    async with server.lifespan(server.app):
        app = server.app
        gen_name = app.state.deps.quality_llm.name
        print(f"답변 생성: {gen_name} · 심판: {judge.name} · {len(cases)}문항\n")
        gens = {}
        for mode in ("before", "after"):
            print(f"[{mode}] 답변 생성")
            gens[mode] = await generate(app, mode, cases)
        print("\n채점 (원본 R/G/A · 개선 R/G/A)")
        rows = await judge_all(judge, cases, gens)
    text = report(rows, judge, gen_name, gens)
    print("\n" + "\n".join(text.splitlines()[:16]))
    out = ROOT / "eval" / f"judge_results_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n결과 저장: eval/{out.name}")


if __name__ == "__main__":
    asyncio.run(main())
