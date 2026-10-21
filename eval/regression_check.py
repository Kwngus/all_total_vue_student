"""회귀 테스트 쿼리(doc/regression_test_queries.md) 10개를 실행 중인 서버에 보내 결과를 표로 정리한다.

개선 7(LLM 대체)·8(조치 승인)을 추가한 뒤에도 원본의 동작(라우팅·조사 대상·runbook·세션 기억)이
깨지지 않았는지 실모델로 확인하는 용도.

    bash sh/run_agent_ollma.sh            # 또는 run_agent_openai.sh — 서버(8100)를 먼저 띄운다
    python eval/regression_check.py       # 다른 터미널에서

결과는 eval/regression_results_<시각>.md 로도 저장된다.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8100"
QUERIES = [
    ("payments-service 결제 응답이 느려지고 CPU가 치솟았어요, 원인 확인해주세요", "cpu-spike"),
    ("cache-service 메모리 사용률이 계속 올라가고 있어요", "memory-leak"),
    ("log-service 디스크 공간이 부족하다는 알림이 왔어요", "disk-full"),
    ("gateway-service에서 payments-service로 가는 요청이 타임아웃 나고 있어요", "두 시스템 모두 조사"),
    ("web-01 CPU 급증, auth-service 커넥션 풀도 같이 확인해줘", "web-01·auth-service 조사"),
    ("payments-service랑 cache-service 둘 다 저하됐는데 공통 원인이 있는지 분석해줘", "비교 분석"),
    ("로그인 응답이 느려요, 원인 확인해주세요", "auth-service 폴백"),
    ("billing-service 응답이 느려요", "없는 시스템 처리"),
    ("안녕, 너는 누구야?", "chat 직행"),
    ("제 이름은 민수입니다", "세션 기억 (다음 질문)"),
]


def main():
    try:
        httpx.get(f"{BASE}/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        sys.exit("서버(8100)가 떠 있지 않습니다. 먼저 bash sh/run_agent_ollma.sh 로 기동하세요.")
    rows = []
    with httpx.Client(timeout=300) as c:
        for i, (q, expect) in enumerate(QUERIES, 1):
            thread = "reg-memory" if i == 10 else f"reg-{int(time.time())}-{i}"
            t0 = time.perf_counter()
            r = c.post(f"{BASE}/chat", json={"message": q, "thread_id": thread})
            sec = time.perf_counter() - t0
            d = r.json()
            if r.status_code != 200:
                rows.append((i, q, expect, f"{r.status_code} {d.get('error', {}).get('code')}", "", "", "", f"{sec:.1f}s", ""))
                print(f"#{i} {r.status_code} {d}")
                continue
            ev, meta = d.get("events", []), d.get("metadata", {})
            targets = ", ".join(e["target"] for e in ev if e.get("type") == "TaskCompleted" and e.get("source") == "monitoring")
            rb = next((e.get("target") for e in ev if e.get("type") == "ReplanCheck" and not e.get("is_sufficient")), "")
            ap = d.get("approval")
            card = ", ".join(f"{a['system_name']}·{a['action']}" for a in ap["actions"]) if ap else ""
            fb = meta.get("llm_fallbacks", 0)
            rows.append((i, q, expect, meta.get("route"), targets, rb, card, f"{sec:.1f}s", fb))
            print(f"#{i} route={meta.get('route')} 조사=[{targets}] runbook={rb} 승인카드=[{card}] {sec:.1f}s 대체={fb}")
        r = c.post(f"{BASE}/chat", json={"message": "제 이름이 뭐였죠?", "thread_id": "reg-memory"})
        memory = r.json().get("reply", "")[:100].replace("\n", " ")
        print(f"#10 후속 '제 이름이 뭐였죠?' → {memory}")

    lines = [f"# 회귀 테스트 — {datetime.now():%Y-%m-%d %H:%M}", "",
             "| # | 쿼리 | 확인 목적 | route | 조사한 시스템 | runbook | 승인 카드 | 응답 | LLM 대체 |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    lines += ["", f"세션 기억: '제 이름이 뭐였죠?' → {memory}"]
    out = Path(__file__).parent / f"regression_results_{datetime.now():%Y%m%d_%H%M}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n결과 저장: eval/{out.name}")


if __name__ == "__main__":
    main()
