#!/usr/bin/env bash
#
# 최종 결과물 검증 — 실제 OpenAI API로 개선 사항 전체를 한 번에 점검한다 (과금 발생).
#
#   bash sh/verify_openai.sh
#
# 순서: ① 단위 테스트 ② runbook 정확도(OpenAI 임베딩) ③ 서버 기동 후 대표 시나리오 6개
#       ④ 관리자 지표 ⑤ 답변 수치 근거 검증 6문항
# 예상 비용: 수십~수백 원 수준 (gpt-4o-mini 분류·계획 + gpt-4o 답변 작성 약 20회 + 임베딩)
# 결과: eval/ 아래 md 파일 3개 (runbook·scenario·grounding) — 보고서에 그대로 첨부
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x ".venv/bin/python3" ] && export PATH="$PWD/.venv/bin:$PATH"

[ -f .env ] || { echo "ERROR: .env가 없습니다 (cp .env.example .env 후 키 입력)"; exit 1; }
set -a; source .env; set +a
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "sk-..." ]; then
    echo "ERROR: .env의 OPENAI_API_KEY가 비어 있습니다."; exit 1
fi
export LLM_BACKEND=openai EMBED_BACKEND=openai
export LANGCHAIN_PROJECT="capstone-all-total-openai"
unset OPENAI_BASE_URL || true            # 벤치마크용 가짜 서버 주소가 남아 있지 않게

echo "실제 OpenAI API를 호출합니다 (fast=${OPENAI_FAST_MODEL:-gpt-4o-mini}, quality=${OPENAI_QUALITY_MODEL:-gpt-4o})."
read -r -p "계속하시겠습니까? [y/N] " CONFIRM
[[ "$CONFIRM" =~ ^[Yy]$ ]] || { echo "취소했습니다."; exit 0; }

echo; echo "################ ① 단위 테스트 ################"
python3 -m pytest tests -q

echo; echo "################ ② runbook 정확도 (OpenAI 임베딩) ################"
python3 eval/runbook_eval.py | sed -n '/^| 방식/,$p'

echo; echo "################ ③④ 대표 시나리오 + 관리자 지표 ################"
lsof -ti:8100 | xargs kill -9 2>/dev/null || true
rm -f session.db
python3 server.py > /tmp/verify_openai_server.log 2>&1 &
disown
trap 'lsof -ti:8100 | xargs kill -9 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do curl -s localhost:8100/health >/dev/null 2>&1 && break; sleep 0.5; done
python3 eval/scenario_check.py

echo; echo "################ ⑤ 답변 수치 근거 검증 (6문항) ################"
python3 eval/grounding_eval.py --limit 6 | sed -n '/^## /,$p'

echo
echo "완료. LangSmith 프로젝트 '$LANGCHAIN_PROJECT'에서 트레이스도 확인할 수 있습니다."
