"""입력/출력 가드레일 — LLM을 호출하기 '전'과 사용자에게 보여주기 '전'에 거는 방어선.

왜 코드로 거는가:
  프롬프트에 "민감정보를 출력하지 마라"라고 적어도 LLM은 지키지 않을 수 있다(gate 노드와
  같은 설계 원칙). 그리고 일단 LLM API로 넘어간 데이터는 외부 서버에 남는다. 그래서
  검증·마스킹은 LLM 앞단에서 결정론적인 코드로 처리한다.

세 단계:
  1) validate_input  — 빈 값·과도한 길이·프롬프트 인젝션 시도는 LLM 호출 없이 즉시 차단
                       (불필요한 과금 방지)
  2) mask_sensitive  — 비밀번호·API 키·토큰·이메일·IP·전화번호·주민번호·카드번호를
                       [MASKED_*]로 치환한 뒤 LLM에 전달, session.db에도 마스킹본만 저장
  3) filter_output   — LLM 응답에 민감정보 패턴이 섞여 나오면 사용자에게 보내기 전에 마스킹
"""
import re
from dataclasses import dataclass, field

MAX_MESSAGE_LENGTH = 1000   # 장애 문의 한 건으로 충분한 길이. 초과 시 토큰 과금 폭주 방지


class GuardrailViolation(Exception):
    """입력이 정책에 맞지 않아 LLM을 호출하지 않고 거절하는 경우."""

    def __init__(self, code: str, user_message: str):
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message


# (이름, 정규식, 치환 문자열) — 위에서부터 순서대로 적용한다.
# 순서가 중요하다: 구체적인 패턴(API 키, JWT)을 먼저 지워야 일반 패턴이 일부만 먹지 않는다.
#
# 주의: 파이썬 정규식의 \b는 한글도 '단어 문자(\w)'로 본다. 그래서 "10.0.0.5인데"처럼
# 값 바로 뒤에 조사가 붙으면 \b 경계가 성립하지 않아 매칭이 실패한다. 한국어 입력에서는
# \b 대신 '앞뒤가 숫자/영문이 아님'을 lookaround로 직접 검사한다.
_NA = r"(?<![A-Za-z0-9])"      # 앞에 영문·숫자가 없음 (한글·공백·기호는 허용)
_NZ = r"(?![A-Za-z0-9])"       # 뒤에 영문·숫자가 없음
_ND = r"(?<![\d.])", r"(?![\d.]?\d)"   # 숫자 패턴용: 더 긴 숫자열의 일부가 아님

_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("API_KEY", re.compile(_NA + r"(?:sk-(?:proj-)?[A-Za-z0-9_\-]{16,}|lsv2_[A-Za-z0-9_]{16,}"
                           r"|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,})"), "[MASKED_API_KEY]"),
    ("JWT", re.compile(_NA + r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
     "[MASKED_TOKEN]"),
    ("BEARER", re.compile(r"(?i)" + _NA + r"bearer\s+[A-Za-z0-9._\-]{10,}"), "Bearer [MASKED_TOKEN]"),
    # "비번은 hunter2", "password: abc", "pw=1234" → 키워드는 남기고 값만 가린다
    ("PASSWORD", re.compile(
        r"(?i)((?:비밀번호|비번|패스워드|password|passwd|pwd|pw)\s*(?:은|는|이|가|:|=)?\s*)"
        r"([A-Za-z0-9!@#$%^&*_\-+=?~]+)"), r"\1[MASKED_PASSWORD]"),
    ("RRN", re.compile(_ND[0] + r"\d{6}-?[1-4]\d{6}" + _ND[1]), "[MASKED_RRN]"),       # 주민등록번호
    ("CARD", re.compile(_ND[0] + r"(?:\d{4}[- ]?){3}\d{4}" + _ND[1]), "[MASKED_CARD]"),  # 카드번호
    # 도메인은 ASCII로 제한 — \w를 쓰면 "a@b.com으로"의 '으로'까지 먹어버린다
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
     "[MASKED_EMAIL]"),
    ("PHONE", re.compile(_ND[0] + r"01[016789]-?\d{3,4}-?\d{4}" + _ND[1]), "[MASKED_PHONE]"),
    ("IP", re.compile(_ND[0] + r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"
                      + _ND[1]), "[MASKED_IP]"),
]

# 시스템 프롬프트를 무력화하려는 대표적인 문구. 완벽한 탐지는 아니지만, 명백한 시도는
# LLM에 도달하기 전에 걸러 비용과 위험을 줄인다.
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|above|prior)\s+instructions?"),
    re.compile(r"(?i)disregard\s+(?:all\s+)?(?:previous|above|prior)"),
    re.compile(r"(?i)(?:reveal|show|print)\s+(?:me\s+)?(?:your|the)\s+system\s+prompt"),
    re.compile(r"(이전|위의?|앞의?)\s*(지시|명령|지침|프롬프트)[를을은는]?\s*(모두\s*)?(무시|잊어)"),
    re.compile(r"시스템\s*프롬프트[를을]?\s*(보여|알려|출력|공개)"),
]


@dataclass
class MaskResult:
    text: str
    masked_types: list[str] = field(default_factory=list)   # 어떤 종류가 가려졌는지 (값은 기록 안 함)

    @property
    def masked(self) -> bool:
        return bool(self.masked_types)


def mask_sensitive(text: str) -> MaskResult:
    masked_types = []
    for name, pattern, repl in _SENSITIVE_PATTERNS:
        text, n = pattern.subn(repl, text)
        if n:
            masked_types.append(name)
    return MaskResult(text=text, masked_types=masked_types)


def validate_input(message: str) -> str:
    """정책 위반이면 GuardrailViolation, 통과하면 앞뒤 공백을 정리한 메시지를 반환."""
    if message is None or not message.strip():
        raise GuardrailViolation("EMPTY_MESSAGE", "메시지를 입력해 주세요.")
    message = message.strip()
    if len(message) > MAX_MESSAGE_LENGTH:
        raise GuardrailViolation(
            "MESSAGE_TOO_LONG",
            f"메시지가 너무 깁니다({len(message)}자). {MAX_MESSAGE_LENGTH}자 이내로 핵심 증상만 적어 주세요.")
    if any(p.search(message) for p in _INJECTION_PATTERNS):
        raise GuardrailViolation(
            "PROMPT_INJECTION",
            "처리할 수 없는 요청입니다. 장애 증상이나 시스템 상태에 대해 질문해 주세요.")
    return message


def guard_input(message: str) -> MaskResult:
    """server.py가 호출하는 진입점: 검증 → 마스킹. 이 결과의 text만 LLM과 DB로 간다."""
    return mask_sensitive(validate_input(message))


def filter_output(reply: str) -> MaskResult:
    """LLM 응답을 사용자에게 보내기 전 마지막 필터 (LLM이 민감정보를 지어내거나 되풀이하는 경우 대비)."""
    return mask_sensitive(reply or "")
