"""Runbook 검색기 — 키워드 규칙 대신 '의미 + 어휘' 하이브리드 검색으로 runbook을 고른다.

기존 방식(graph._classify_incident_type)의 문제
  - 모니터링 알림 문구만 보고 규칙 순서(latency → pool → cpu → ...)대로 '처음 걸린 것'을 고른다.
    → 사용자가 무엇을 물었는지는 무시된다 ("web-01 디스크가 꽉 찼어요" → web-01은 알림이 없어 db-down)
    → 여러 시스템이 섞이면 규칙 순서가 이긴다 (gateway 타임아웃 질문인데 payments의 CPU 알림 때문에 cpu-spike)
    → runbook은 항상 1개만 고른다

이 모듈의 방식
  1) 인덱싱 (서버 기동 시 1회): runbook 6종의 본문을 임베딩해 메모리에 보관 (Wave 1 싱글턴과 같은 원칙)
  2) 검색 질의 2개를 만든다
       - 사용자 질문 (가중치 0.6) : 사람이 실제로 호소한 증상
       - 이상 시스템의 알림 문구 (가중치 0.4) : 모니터링이 보여주는 객관적 근거
  3) 각 질의에 대해 두 가지 점수를 계산해 섞는다 (하이브리드)
       - 의미 점수 (비중 0.4) : 임베딩 코사인 유사도 ("느려요" ≈ "응답 지연"처럼 단어가 달라도 잡힘)
       - 어휘 점수 (비중 0.6) : 한글 2~3-gram + 영문 단어 TF-IDF 코사인 ("p95", "/var/log" 같은 고유 표기를 정확히 잡음)
  4) 1위는 항상 선택. 복합 장애 대응(최대 2개):
       - 이상 시스템이 2개 이상 언급되면 시스템별로 쪼개 다시 검색해 다른 runbook을 추가
       - 아니면 2위가 1위 점수의 MULTI_RATIO 이상일 때만 함께 선택

임베딩을 쓸 수 없으면(Ollama 미기동, 모델 미설치, fake 백엔드) 어휘 점수만으로 동작한다 —
검색이 아예 멈추지 않도록 하는 graceful degradation.
"""
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger("agent.runbook_retriever")

RUNBOOK_DIR = Path(__file__).resolve().parent.parent / "data" / "runbooks"

QUERY_WEIGHT, ALERT_WEIGHT = 0.6, 0.4        # 사용자 질문 vs 모니터링 알림
# 임베딩 점수 비중 (나머지는 어휘 점수). eval/runbook_eval.py --sweep 실측(bge-m3, 18문항):
#   0.0(어휘만) 정답포함 83% · 0.2~0.5 94% (평탄 최고 구간) · 0.6~0.7 89% · 0.85~1.0 83%
#   0.6 이상에서는 'high-latency' runbook이 여러 질문에 두루 비슷하게 잡히는 쏠림이 커진다.
#   → 평탄 구간의 가운데이자 평균 선택 수가 가장 적은 0.4를 기본값으로 쓴다.
SEMANTIC_WEIGHT = float(os.environ.get("RUNBOOK_SEMANTIC_WEIGHT", "0.4"))
MULTI_RATIO = float(os.environ.get("RUNBOOK_MULTI_RATIO", "0.9"))
MAX_RUNBOOKS = 2


# ── 어휘 점수: 한글 글자 n-gram + 영문 단어 TF-IDF ──────────────────────
# 한국어는 띄어쓰기·조사 때문에 단어 단위 매칭이 잘 안 된다("디스크가" ≠ "디스크").
# 한글은 글자 2~3-gram으로 쪼개면 "디스" "스크"가 겹쳐 조사와 무관하게 매칭된다.
# 반면 영문까지 글자 단위로 쪼개면 "log"·"failover"의 "lo"·"er"처럼 의미 없는 조각이
# 점수를 만든다(실측: log-service 질문이 db-down으로 분류됨) → 영문·숫자는 단어 그대로 쓴다.
#
# 어느 장애 질문·runbook에나 나오는 말("상태 확인해주세요", "원인")은 변별력이 없어 제거한다.
_STOPWORDS = re.compile(r"확인|상태|원인|정상화|진단|대응|절차|서비스|문제|해주세요|해줘|있어요|같이")
_STOP_EN = {"service", "get_monitoring_status", "the", "to", "on", "of", "in"}


def _ngrams(text: str, ns=(2, 3)) -> Counter:
    grams = Counter()
    for token in re.findall(r"[가-힣]+|[0-9a-z_%./]+", _STOPWORDS.sub(" ", text.lower())):
        if re.match(r"[가-힣]", token):
            for n in ns:
                grams.update(token[i:i + n] for i in range(len(token) - n + 1))
        else:
            token = token.strip("./")
            if len(token) >= 2 and token not in _STOP_EN:
                grams[token] += 1
    return grams


def _cosine_sparse(a: dict, b: dict) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _cosine_dense(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class LexicalIndex:
    def __init__(self, docs: dict[str, str]):
        grams = {k: _ngrams(v) for k, v in docs.items()}
        df = Counter(g for c in grams.values() for g in c)
        n = len(docs)
        # 모든 runbook에 나오는 n-gram("get_monitoring_status" 등)은 변별력이 없으므로 idf로 깎는다
        self.idf = {g: math.log((n + 1) / (d + 1)) + 1.0 for g, d in df.items()}
        self.vectors = {k: self._weigh(c) for k, c in grams.items()}

    def _weigh(self, counts: Counter) -> dict:
        return {g: (1 + math.log(c)) * self.idf.get(g, 0.0) for g, c in counts.items()}

    def scores(self, text: str) -> dict[str, float]:
        q = self._weigh(_ngrams(text))
        return {k: _cosine_sparse(q, v) for k, v in self.vectors.items()}


# ── 의미 점수: 임베딩 ──────────────────────────────────────────────────
class OllamaEmbedder:
    """로컬 Ollama 임베딩. 한국어 성능 때문에 다국어 모델 bge-m3를 기본으로 쓴다
    (nomic-embed-text 같은 영어 중심 모델은 한국어 runbook 구분력이 떨어진다)."""

    def __init__(self, model: str | None = None, url: str = "http://127.0.0.1:11434/api/embed"):
        self.model = model or os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
        self.url = url
        self.name = f"ollama:{self.model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(self.url, json={"model": self.model, "input": texts})
            r.raise_for_status()
            return r.json()["embeddings"]


class OpenAIEmbedder:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.name = f"openai:{self.model}"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        key = os.environ.get("OPENAI_API_KEY", "")
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post("https://api.openai.com/v1/embeddings",
                                  headers={"Authorization": f"Bearer {key}"},
                                  json={"model": self.model, "input": texts})
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]


def make_embedder(llm_backend: str):
    """EMBED_BACKEND가 없으면 LLM 백엔드를 따라간다: ollama→Ollama, openai→OpenAI, fake→없음."""
    choice = os.environ.get("EMBED_BACKEND", llm_backend)
    if choice == "ollama":
        return OllamaEmbedder()
    if choice == "openai":
        return OpenAIEmbedder()
    return None


# ── 검색기 본체 ────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    selected: list[str]                     # 선택된 incident_type들 (runbook 파일명)
    scores: dict[str, float]                # 전체 후보 점수 (발표·디버깅용)
    method: str                             # "hybrid" | "lexical"
    detail: dict = field(default_factory=dict)


class RunbookRetriever:
    def __init__(self, embedder=None, runbook_dir: Path = RUNBOOK_DIR,
                 semantic_weight: float | None = None):
        self.semantic_weight = SEMANTIC_WEIGHT if semantic_weight is None else semantic_weight
        self.docs = {p.stem: p.read_text() for p in sorted(runbook_dir.glob("*.md"))}
        self.lexical = LexicalIndex(self.docs)
        self.embedder = embedder
        self.doc_vectors: dict[str, list[float]] | None = None

    async def build(self) -> None:
        """서버 기동 시 1회. 임베딩 실패는 치명적이지 않다 — 어휘 검색으로 계속 동작한다."""
        if self.embedder is None:
            logger.info("runbook retriever: 임베딩 없음 → 어휘(n-gram) 검색만 사용")
            return
        try:
            keys = list(self.docs)
            vecs = await self.embedder.embed([self.docs[k] for k in keys])
            self.doc_vectors = dict(zip(keys, vecs))
            logger.info("runbook retriever: %s로 runbook %d건 인덱싱 완료",
                        self.embedder.name, len(keys))
        except Exception as exc:
            logger.warning("runbook retriever: 임베딩 인덱싱 실패(%s: %s) → 어휘 검색으로 대체",
                           type(exc).__name__, exc)
            self.doc_vectors = None

    @property
    def method(self) -> str:
        return "hybrid" if self.doc_vectors else "lexical"

    async def _score_one(self, text: str) -> tuple[dict[str, float], bool]:
        lex = self.lexical.scores(text)
        if not self.doc_vectors:
            return lex, False
        try:
            qv = (await self.embedder.embed([text]))[0]
        except Exception as exc:   # 질의 시점 실패도 어휘 검색으로 계속 진행
            logger.warning("runbook retriever: 질의 임베딩 실패(%s) → 어휘 점수만 사용", exc)
            return lex, False
        sem = {k: _cosine_dense(qv, v) for k, v in self.doc_vectors.items()}
        w = self.semantic_weight
        return {k: w * sem[k] + (1 - w) * lex[k] for k in self.docs}, True

    async def search(self, query: str, alerts: list[str]) -> RetrievalResult:
        q_scores, q_sem = await self._score_one(query)
        if alerts:
            a_scores, a_sem = await self._score_one(" ".join(alerts))
            scores = {k: QUERY_WEIGHT * q_scores[k] + ALERT_WEIGHT * a_scores[k] for k in self.docs}
        else:
            scores, a_sem = q_scores, q_sem
        ranked = sorted(scores, key=scores.get, reverse=True)
        top = scores[ranked[0]]
        selected = [ranked[0]] + [k for k in ranked[1:MAX_RUNBOOKS]
                                  if top > 0 and scores[k] >= top * MULTI_RATIO]
        method = "hybrid" if (q_sem and a_sem) else "lexical"
        return RetrievalResult(selected=selected,
                               scores={k: round(scores[k], 4) for k in ranked},
                               method=method)


    async def search_collected(self, query: str, collected: list[dict]) -> RetrievalResult:
        """파이프라인용 진입점. 복합 장애를 시스템 단위로 쪼개서 본다.

        질문에 이상 시스템이 2개 이상 언급되면("payments-service CPU 급증, cache-service는
        메모리가 …") 전체를 한 번에 검색할 때 한쪽 증상이 다른 쪽을 덮어버린다.
        그래서 ① 전체 질문으로 1순위를 고르고 ② 시스템별로 '그 시스템이 언급된 문장 조각 +
        그 시스템의 알림'으로 다시 검색해, 1순위와 다른 runbook이 나오면 함께 선택한다.
        """
        overall = await self.search(query, alerts_from_collected(collected))
        segments = split_by_system(query, [c["data"] for c in collected
                                           if c.get("source") == "monitoring"
                                           and c["data"].get("status") not in ("healthy", "unknown")])
        if len(segments) < 2:
            return overall
        selected = [overall.selected[0]]
        per_system = {}
        for system, (segment, alerts) in segments.items():
            r = await self.search(segment, alerts)
            per_system[system] = r.selected[0]
            if r.selected[0] not in selected and len(selected) < MAX_RUNBOOKS:
                selected.append(r.selected[0])
        return RetrievalResult(selected=selected, scores=overall.scores, method=overall.method,
                               detail={"per_system": per_system})


def split_by_system(query: str, systems: list[dict]) -> dict[str, tuple[str, list[str]]]:
    """질문을 시스템 이름 위치 기준으로 자른다.
    "log-service 디스크 부족이랑 gateway-service 타임아웃" →
        {"log-service": ("log-service 디스크 부족이랑", [...]), "gateway-service": ("gateway-service 타임아웃", [...])}
    """
    found = sorted((query.index(d["system_name"]), d) for d in systems
                   if d.get("system_name") and d["system_name"] in query)
    out = {}
    for i, (pos, d) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(query)
        out[d["system_name"]] = (query[pos:end], list(d.get("alerts", [])))
    return out


def alerts_from_collected(collected: list[dict]) -> list[str]:
    """정상(healthy) 시스템은 근거가 아니므로 제외하고, 이상 시스템의 알림만 모은다."""
    out = []
    for c in collected:
        if c.get("source") != "monitoring":
            continue
        d = c["data"]
        if d.get("status") in ("healthy", "unknown"):   # unknown = 모니터링에 없는 시스템
            continue
        # 시스템 이름(…-service)은 모든 문서와 무관한 잡음이라 알림 문구만 쓴다
        out += list(d.get("alerts", []))
    return out
