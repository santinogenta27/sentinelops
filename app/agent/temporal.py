"""Deterministic reading of the dates customers mention (English and Spanish).

A date never authorises anything: it can only narrow down which ledger charge the customer means.
Windows are generous because people speak loosely ("a couple of months ago" for 75 days) and because
card statements post a few days after the charge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

_POSTING_LAG = timedelta(days=3)
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_ES_UNITS = {"dia": "day", "día": "day", "semana": "week", "mes": "month", "año": "year", "ano": "year"}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "un": 1, "una": 1, "uno": 1, "two": 2, "dos": 2, "couple of": 2, "a couple of": 2,
    "un par de": 2, "par de": 2, "three": 3, "tres": 3, "few": 3, "a few": 3, "unos pocos": 3, "unas pocas": 3,
    "unos": 3, "unas": 3, "four": 4, "cuatro": 4, "five": 5, "cinco": 5, "six": 6, "seis": 6, "several": 4,
    "varios": 4, "varias": 4,
}
_QUANTITY = r"\d{1,3}|" + "|".join(sorted((re.escape(w) for w in _NUMBER_WORDS), key=len, reverse=True))
_MORE_THAN = r"(?:more than|over|m[aá]s de)\s+"
_EN_AGO = re.compile(rf"\b({_MORE_THAN})?({_QUANTITY})\s+(day|week|month|year)s?\s+ago\b", re.IGNORECASE)
_ES_AGO = re.compile(
    rf"\bhace\s+({_MORE_THAN})?(?:como\s+|casi\s+)?({_QUANTITY})\s+(d[ií]a|semana|mes|a[nñ]o)(?:s|es)?\b", re.IGNORECASE
)
# (pattern, youngest age in days, oldest age in days)
_FIXED: tuple[tuple[re.Pattern[str], int, int], ...] = tuple(
    (re.compile(p, re.IGNORECASE), low, high)
    for p, low, high in (
        (r"\b(today|hoy)\b", 0, 2),
        (r"\b(yesterday|ayer)\b", 0, 3),
        (r"\b(this|esta) (week|semana)\b", 0, 9),
        (r"\b(last week|la semana pasada)\b", 3, 16),
        (r"\b(this month|este mes)\b", 0, 35),
        (r"\b(last month|el mes pasado)\b", 15, 65),
        (r"\b(last year|el a[nñ]o pasado)\b", 180, 550),
    )
)
_MONTHS = {
    "january": 1, "jan": 1, "enero": 1, "ene": 1, "february": 2, "feb": 2, "febrero": 2, "march": 3, "mar": 3,
    "marzo": 3, "april": 4, "apr": 4, "abril": 4, "abr": 4, "may": 5, "mayo": 5, "june": 6, "jun": 6, "junio": 6,
    "july": 7, "jul": 7, "julio": 7, "august": 8, "aug": 8, "agosto": 8, "september": 9, "sep": 9,
    "sept": 9, "septiembre": 9, "setiembre": 9, "october": 10, "oct": 10, "octubre": 10, "november": 11, "nov": 11,
    "noviembre": 11, "december": 12, "dec": 12, "diciembre": 12, "dic": 12,
}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_MONTH_DAY = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(20\d{{2}}))?", re.IGNORECASE)
_DAY_MONTH = re.compile(rf"(?<![$\d.,])\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:de\s+|of\s+)?({_MONTH})\b(?:\.?,?\s+(?:de\s+)?(20\d{{2}}))?", re.IGNORECASE)
# A bare month needs a preposition: "may" and "mar" are ordinary words.
_IN_MONTH = re.compile(r"\b(?:in|en|de|from|since)\s+(january|february|march|april|june|july|august|september|october|november|december|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DateHint:
    start: datetime
    end: datetime

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment <= self.end


def _quantity(word: str) -> int:
    key = re.sub(r"\s+", " ", word.strip().lower())
    return int(key) if key.isdigit() else _NUMBER_WORDS[key]


def _ago_window(now: datetime, days: int, more_than: bool) -> DateHint:
    if more_than:
        return DateHint(now - timedelta(days=366 * 3), now - timedelta(days=days * 0.9))
    youngest = 0.0 if days <= 7 else days * 0.6
    return DateHint(now - timedelta(days=days * 1.5) - _POSTING_LAG, now - timedelta(days=youngest))


def _latest(now: datetime, month: int, day: int, year: int | None) -> datetime | None:
    """That calendar day, in the given year or else the latest one that is not in the future."""
    for candidate_year in ([year] if year else [now.year, now.year - 1]):
        try:
            when = datetime(candidate_year, month, day, tzinfo=UTC)
        except ValueError:
            return None
        if when <= now + _POSTING_LAG:
            return when
    return None


def hint_for_day(day: date) -> DateHint:
    return _around(datetime(day.year, day.month, day.day, tzinfo=UTC))


def _around(when: datetime) -> DateHint:
    return DateHint(when - _POSTING_LAG, when + timedelta(days=1) + _POSTING_LAG)


def _month_window(now: datetime, month: int) -> DateHint | None:
    first = _latest(now, month, 1, None)
    if first is None:
        return None
    following = (first + timedelta(days=32)).replace(day=1)
    return DateHint(first - _POSTING_LAG, following + _POSTING_LAG)


def _windows(text: str, now: datetime) -> list[DateHint]:
    found: list[DateHint] = []
    for m in _EN_AGO.finditer(text):
        found.append(_ago_window(now, _quantity(m.group(2)) * _UNIT_DAYS[m.group(3).lower()], bool(m.group(1))))
    for m in _ES_AGO.finditer(text):
        unit = _ES_UNITS[m.group(3).lower()]
        found.append(_ago_window(now, _quantity(m.group(2)) * _UNIT_DAYS[unit], bool(m.group(1))))
    for pattern, low, high in _FIXED:
        if pattern.search(text):
            found.append(DateHint(now - timedelta(days=high), now - timedelta(days=low)))
    for m in _ISO_DATE.finditer(text):
        if (when := _latest(now, int(m.group(2)), int(m.group(3)), int(m.group(1)))) is not None:
            found.append(_around(when))
    for m in _MONTH_DAY.finditer(text):
        if m.group(1) == "may":  # the verb
            continue
        if (when := _latest(now, _MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3)) if m.group(3) else None)) is not None:
            found.append(_around(when))
    for m in _DAY_MONTH.finditer(text):
        if m.group(2) == "may":
            continue
        if (when := _latest(now, _MONTHS[m.group(2).lower()], int(m.group(1)), int(m.group(3)) if m.group(3) else None)) is not None:
            found.append(_around(when))
    if not found:
        for m in _IN_MONTH.finditer(text):
            if (window := _month_window(now, _MONTHS[m.group(1).lower()])) is not None:
                found.append(window)
    return found


def extract_date_hint(text: str, now: datetime) -> DateHint | None:
    """The period the message refers to; several mentions widen it. None when it mentions no date."""
    windows = _windows(text, now)
    if not windows:
        return None
    return DateHint(min(w.start for w in windows), max(w.end for w in windows))
