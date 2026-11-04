"""Deterministic guardrails: prompt-injection detection, untrusted-input fencing, output grounding."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("override_instructions", r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instruction|rule|polic|guideline|prompt|directive)"),
        ("role_hijack", r"\b(you are now|act as|pretend (to be|you are)|from now on you|new persona)\b"),
        ("privileged_mode", r"\b(developer|admin|god|debug|maintenance|sudo|root)\s*mode\b|\bjailbreak\b|\bDAN\b"),
        ("system_prompt_probe", r"\b(system prompt|hidden instructions|reveal (your|the) (prompt|instructions))\b"),
        ("fake_authorization", r"\b(already|pre-?)\s*(been\s+)?(approved|authori[sz]ed)\b|\bapproved by (a |the |my )?(supervisor|manager|admin|ceo|billing)"),
        ("approval_bypass", r"\b(skip|bypass|circumvent|disable|without)\b[^.\n]{0,30}\b(approval|human|review|verification|supervisor|checks?)\b"),
        ("markup_injection", r"(<\s*/?\s*(system|assistant|instructions?)\s*>|\[/?INST\]|###\s*(system|instruction)|```\s*system)"),
        ("tool_coercion", r"\b(call|execute|invoke|run)\b[^.\n]{0,20}\b(tool|function|stripe|refund api|create_refund)\b"),
        # Text imitating the structure of a chat/system message: role keys in JSON/YAML/kwargs, chat
        # template tokens, bracketed role tags, and transcript-style "System:"/"Assistant:" turns.
        # Requires the role word to be directly followed by ":" so prose like "System error: ..." or
        # "my system shows ..." is not flagged.
        (
            "role_spoofing",
            r"""["']?\brole\b["']?\s*[:=]\s*["']?(system|assistant|developer|tool|function)\b"""
            r"""|["']\s*(system|developer)\s*["']\s*:"""
            r"""|<\|\s*(im_start|im_end|system|assistant|developer|endoftext)\s*\|>|<<\s*/?\s*sys\s*>>"""
            r"""|\[\s*/?\s*(system|assistant|developer)(\s+(message|prompt|note))?\s*\]"""
            r"""|(?:^|[\r\n]|[.!?]\s)\s*[#>*\-]*\s*(system|assistant|developer)\s*(prompt|message|note|override)?\s*:""",
        ),
        # --- non-English payloads (ES / FR / PT / DE) -------------------------
        (
            "override_instructions_i18n",
            r"\b(ignor[aáe]\w*|olvid[aáe]\w*|ignorez|oubliez|ignorer|esque[cç]a\w*|ignoriere\w*|vergiss\w*)\b[^.\n]{0,40}"
            r"\b(instruc\w*|regla\w*|reglas|pol[ií]tica\w*|r[eè]gle\w*|consigne\w*|regra\w*|anweisung\w*|richtlinie\w*|regeln)",
        ),
        (
            "fake_authorization_i18n",
            r"\b(ya\s+(est[aá]|fue|ha\s+sido)\s+(aprobad|autorizad)\w*|pre-?aprobad\w*|aprobad[oa]\s+por\s+(el|la|mi|un)\s+"
            r"(supervisor|gerente|jefe|manager)\w*|d[eé]j[aà]\s+(approuv|autoris|valid)\w*|pr[eé]-?approuv\w*|"
            r"j[aá]\s+(foi\s+)?(aprovad|autorizad)\w*|bereits\s+(genehmigt|freigegeben))",
        ),
        (
            "approval_bypass_i18n",
            r"\b(sin|saltea\w*|salta\w*|omit\w*|evit\w*|sans|contourn\w*|sem|pular|ohne|umgeh\w*)\b[^.\n]{0,30}"
            r"\b(aprobaci[oó]n|revisi[oó]n|verificaci[oó]n|supervisor\w*|human[oa]s?|validation|v[eé]rification|approbation|"
            r"humaine?|aprova[cç][aã]o|revis[aã]o|pr[uü]fung|freigabe|genehmigung)\b",
        ),
        # --- impersonating staff / internal systems to instruct the agent ------
        (
            "staff_impersonation",
            r"\[\s*internal\b|\binternal\s+(note|memo|instruction|directive)s?\b|"
            r"\b(this is|message from|on behalf of)\s+(the\s+)?(\w+\s+){0,2}(support|billing|security|compliance|finance)\s+"
            r"(team|department|desk|ops|operations)\b|"
            r"\b(identity|kyc|account)\s+verification\s+(is\s+|has\s+been\s+)?(complete|completed|done|passed)\b",
        ),
    )
)

_MONEY_RE = re.compile(
    r"(?:(?:US)?\$|USD\s?)\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?|(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?\s?(?:USD|dollars?|bucks)\b",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}")
_TXN_REF_RE = re.compile(r"\btxn[\s_\-#:]*[A-Za-z0-9][A-Za-z0-9_\-]{2,40}\b", re.IGNORECASE)
_STRICT_TXN_RE = re.compile(r"\btxn_[A-Za-z0-9]{4,32}\b")

_FORBIDDEN_DISCLOSURES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(p, re.IGNORECASE))
    for name, p in (
        ("policy_id_leak", r"\bPOL-[A-Z]+-\d{3}\b"),
        ("risk_score_leak", r"\brisk (score|level|rating)\b|\bfraud score\b"),
        ("internal_leak", r"\b(system prompt|langgraph|llm|internal heuristic|chroma|bm25)\b"),
    )
)


def detect_prompt_injection(text: str) -> list[str]:
    """Return the names of injection signatures found in untrusted text."""
    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]


def fence_untrusted(text: str, label: str = "CUSTOMER_MESSAGE") -> str:
    """Wrap untrusted input in explicit delimiters, neutralising delimiter spoofing."""
    safe = text.replace("<<<", "‹‹‹").replace(">>>", "›››")
    return f"<<<{label}>>>\n{safe}\n<<<END_{label}>>>"


def find_money_amounts(text: str) -> list[float]:
    amounts: list[float] = []
    for m in _MONEY_RE.finditer(text):
        whole = m.group(1) or m.group(3)
        cents = m.group(2) or m.group(4) or "0"
        if whole is None:
            continue
        try:
            amounts.append(round(float(whole.replace(",", "")) + float(f"0.{cents}"), 2))
        except ValueError:
            continue
    return amounts


def find_emails(text: str) -> list[str]:
    return [e.lower() for e in _EMAIL_RE.findall(text)]


def find_transaction_refs(text: str) -> list[str]:
    """Loose transaction references as written by the customer (may be malformed)."""
    seen: list[str] = []
    for m in _TXN_REF_RE.finditer(text):
        ref = m.group(0).strip()
        if ref not in seen:
            seen.append(ref)
    return seen[:10]


def find_strict_transaction_ids(text: str) -> list[str]:
    return list(dict.fromkeys(_STRICT_TXN_RE.findall(text)))


@dataclass(slots=True)
class GroundingReport:
    issues: list[str] = field(default_factory=list)
    critical: bool = False

    @property
    def score(self) -> float:
        if self.critical:
            return 0.0
        return max(0.0, 1.0 - 0.2 * len(self.issues))


def grounding_check(
    text: str,
    *,
    allowed_amounts: set[float],
    allowed_transaction_ids: set[str],
    required_amounts: set[float],
) -> GroundingReport:
    """Verify a customer-facing draft only states facts supported by the executed action."""
    report = GroundingReport()
    stated = set(find_money_amounts(text))
    unsupported = {a for a in stated if not any(abs(a - b) < 0.005 for b in allowed_amounts)}
    if unsupported:
        report.critical = True
        report.issues.append(f"states unsupported monetary amounts: {sorted(unsupported)}")
    for req in required_amounts:
        if not any(abs(req - s) < 0.005 for s in stated):
            report.issues.append(f"omits the executed refund amount {req:.2f}")
    bad_ids = set(find_strict_transaction_ids(text)) - allowed_transaction_ids
    if bad_ids:
        report.critical = True
        report.issues.append(f"references unknown transaction ids: {sorted(bad_ids)}")
    for name, pattern in _FORBIDDEN_DISCLOSURES:
        if pattern.search(text):
            report.issues.append(f"discloses internal information ({name})")
    return report
