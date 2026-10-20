"""runbook 선택 정확도 Before/After 평가 — LLM 호출 없이 '선택 로직'만 같은 조건으로 채점한다.

실행 예:
    python eval/runbook_eval.py                    # 키워드(개선 전) vs 어휘 검색 vs 하이브리드(Ollama bge-m3)
    python eval/runbook_eval.py --sweep            # 의미 점수 비중 0~1 탐색 (임베딩 1회 계산 후 재사용)
    EMBED_BACKEND=none python eval/runbook_eval.py # 임베딩 없이 (키워드 vs 어휘만)
    EMBED_BACKEND=openai python eval/runbook_eval.py

공정한 비교를 위해 모든 방식에 같은 입력을 준다:
  - 질문에 등장한 시스템의 모니터링 데이터를 data/monitoring.json에서 읽어 'collected'를 만든다
    (실제 파이프라인의 research 노드 결과와 같은 형태, 시스템 미언급 시 auth-service 폴백)
  - LLM의 계획 단계 변동을 빼고 runbook 선택 로직만 순수하게 비교한다

지표:
  - Top-1 정확도      : 1순위로 고른 runbook이 정답 중 하나인가
  - 정답 전부 포함률  : 정답 runbook(복합 장애면 2개)을 빠짐없이 골랐는가
  - 평균 선택 수      : 불필요하게 많이 고르지 않는지 (낮을수록 프롬프트가 짧다)
결과는 터미널 표 + eval/results_<날짜>.md 로 저장된다.
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.graph import KNOWN_SYSTEMS, _classify_incident_type  # noqa: E402
from agent.runbook_retriever import (OllamaEmbedder, OpenAIEmbedder,  # noqa: E402
                                     RunbookRetriever)

MONITORING = json.loads((ROOT / "data" / "monitoring.json").read_text())
CASES = json.loads((Path(__file__).parent / "runbook_cases.json").read_text())


def collected_for(query: str) -> list[dict]:
    systems = sorted(s for s in KNOWN_SYSTEMS if s in query) or ["auth-service"]
    return [{"source": "monitoring",
             "data": {"system_name": s, "status": MONITORING[s]["status"],
                      "alerts": MONITORING[s].get("alerts", [])}} for s in systems]


async def run_method(name: str, retriever) -> list[dict]:
    rows = []
    for case in CASES:
        col = collected_for(case["query"])
        if retriever is None:
            selected, scores = [_classify_incident_type(col)], {}
        else:
            r = await retriever.search_collected(case["query"], col)
            selected, scores = r.selected, r.scores
        exp = set(case["expected"])
        rows.append({"id": case["id"], "selected": selected, "scores": scores,
                     "top1_ok": selected[0] in exp, "all_ok": exp <= set(selected)})
    return rows


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {"top1": sum(r["top1_ok"] for r in rows) / n,
            "all": sum(r["all_ok"] for r in rows) / n,
            "avg_selected": sum(len(r["selected"]) for r in rows) / n}


class CachingEmbedder:
    """같은 문장은 한 번만 임베딩한다 — 가중치만 바꿔 여러 번 채점할 때 Ollama 호출을 재사용."""

    def __init__(self, inner):
        self.inner, self.name, self.cache = inner, inner.name, {}

    async def embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            self.cache.update(zip(missing, await self.inner.embed(missing)))
        return [self.cache[t] for t in texts]


async def sweep(inner_embedder):
    """의미 점수 비중(semantic_weight)을 0~1로 바꿔 가며 같은 18문항을 채점한다.
    0.0 = 어휘 검색만, 1.0 = 임베딩만. 주의: 같은 문항으로 고른 값이라 과적합 가능성이 있다."""
    cached = CachingEmbedder(inner_embedder)
    print(f"semantic_weight 탐색 ({cached.name})\n")
    print("| 의미 비중 | Top-1 정확도 | 정답 전부 포함률 | 평균 선택 수 | 1순위 high-latency 오답 |")
    print("|---|---|---|---|---|")
    for w in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0):
        r = RunbookRetriever(cached, semantic_weight=w)
        await r.build()
        if not r.doc_vectors:
            print("⚠️  임베딩을 쓸 수 없습니다 (ollama serve / ollama pull bge-m3 확인)")
            return
        rows = await run_method(f"w={w}", r)
        s = summary(rows)
        hub = sum(1 for row in rows if not row["top1_ok"] and row["selected"][0] == "high-latency")
        print(f"| {w:.2f} | {s['top1']:.0%} ({sum(x['top1_ok'] for x in rows)}/{len(rows)}) "
              f"| {s['all']:.0%} ({sum(x['all_ok'] for x in rows)}/{len(rows)}) "
              f"| {s['avg_selected']:.2f} | {hub} |")


async def main():
    if "--sweep" in sys.argv:
        choice = os.environ.get("EMBED_BACKEND", "ollama")
        await sweep(OllamaEmbedder() if choice == "ollama" else OpenAIEmbedder())
        return
    methods: dict[str, object] = {"keyword(개선 전)": None,
                                  "lexical(어휘)": RunbookRetriever(embedder=None)}
    choice = os.environ.get("EMBED_BACKEND", "ollama")
    if choice in ("ollama", "openai"):
        hybrid = RunbookRetriever(OllamaEmbedder() if choice == "ollama" else OpenAIEmbedder())
        await hybrid.build()
        if hybrid.doc_vectors:
            methods[f"hybrid({hybrid.embedder.name})"] = hybrid
        else:
            print(f"⚠️  {choice} 임베딩을 쓸 수 없어 hybrid는 건너뜁니다 "
                  f"(Ollama 기동/모델 설치 확인: ollama pull bge-m3)\n")

    results = {name: await run_method(name, r) for name, r in methods.items()}
    names = list(results)

    # ── 케이스별 표 ──
    lines = ["| # | 질문 | 정답 | " + " | ".join(names) + " |",
             "|---|---|---|" + "---|" * len(names)]
    for i, case in enumerate(CASES):
        cells = []
        for name in names:
            r = results[name][i]
            mark = "✅" if r["all_ok"] else ("🔶" if r["top1_ok"] else "❌")
            cells.append(f"{mark} {', '.join(r['selected'])}")
        lines.append(f"| {case['id']} | {case['query']} | {', '.join(case['expected'])} | "
                     + " | ".join(cells) + " |")
    lines += ["", "✅ 정답 전부 선택 · 🔶 1순위만 정답(복합 장애 일부 누락) · ❌ 1순위 오답", "",
              "| 방식 | Top-1 정확도 | 정답 전부 포함률 | 평균 선택 수 |", "|---|---|---|---|"]
    for name in names:
        s = summary(results[name])
        lines.append(f"| {name} | {s['top1']:.0%} ({round(s['top1'] * len(CASES))}/{len(CASES)}) "
                     f"| {s['all']:.0%} ({round(s['all'] * len(CASES))}/{len(CASES)}) "
                     f"| {s['avg_selected']:.2f} |")
    report = "\n".join(lines)
    print(report)

    out = Path(__file__).parent / f"results_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text(f"# runbook 선택 정확도 평가 ({datetime.now():%Y-%m-%d %H:%M})\n\n"
                   f"MULTI_RATIO={os.environ.get('RUNBOOK_MULTI_RATIO', '0.9')}\n\n"
                   + report + "\n\n## 후보 점수 상세\n\n```json\n"
                   + json.dumps({n: [{"id": r["id"], "scores": r["scores"]} for r in rows]
                                 for n, rows in results.items() if n != names[0]},
                                ensure_ascii=False, indent=1) + "\n```\n")
    print(f"\n결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
