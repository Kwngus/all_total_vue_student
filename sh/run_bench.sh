#!/usr/bin/env bash
#
# 개선 4 Before/After 자동 측정 — 가짜 LLM(호출당 2초 고정, 과금 없음)에 연결해
# LLM_HTTP_MODE=sync(개선 전 재현) → async(개선 후) 순서로 같은 벤치마크를 돌린다.
#
#   bash sh/run_bench.sh          # 동시 3건
#   bash sh/run_bench.sh 5        # 동시 5건
#
# 결과: 터미널 표 2개 + bench/results_sync_*.md, bench/results_async_*.md
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x ".venv/bin/python3" ] && export PATH="$PWD/.venv/bin:$PATH"
N="${1:-3}"

export LANGCHAIN_TRACING_V2=false          # 벤치마크 트레이스로 LangSmith를 채우지 않음
export LLM_BACKEND=openai OPENAI_API_KEY=mock OPENAI_BASE_URL=http://127.0.0.1:8700/v1

cleanup() { lsof -ti:8100,8700 | xargs kill -9 2>/dev/null || true; }
trap cleanup EXIT
cleanup

python3 bench/mock_llm_server.py > /tmp/mock_llm.log 2>&1 &
disown   # 종료 시 "Killed" 메시지가 결과 화면에 섞이지 않게
sleep 2

for MODE in sync async; do
    lsof -ti:8100 | xargs kill -9 2>/dev/null || true
    rm -f session.db
    LLM_HTTP_MODE=$MODE python3 server.py > "/tmp/bench_server_$MODE.log" 2>&1 &
    disown
    for _ in $(seq 1 40); do curl -s localhost:8100/health >/dev/null 2>&1 && break; sleep 0.5; done
    echo
    echo "################ LLM_HTTP_MODE=$MODE ################"
    python3 bench/concurrency_bench.py "$N" | sed -n '/^## /,$p'
done
