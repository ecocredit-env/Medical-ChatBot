"""Extract simple clinical details from a normalised message:
how long the problem has lasted, a temperature reading, and the age group.
"""
import re
from dataclasses import dataclass
from typing import Optional

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2, "few": 3, "several": 4,
}
_UNIT_DAYS = {"hour": 1 / 24, "hr": 1 / 24, "day": 1, "night": 1, "week": 7,
              "month": 30}

_DURATION = re.compile(
    r"\b(\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")\s*(?:of\s+)?"
    r"(hours?|hrs?|days?|nights?|weeks?|months?)\b")
_TEMPERATURE = re.compile(
    r"\b(\d{2,3}(?:\.\d+)?)\s*(?:degrees?|deg)?\s*(celsius|fahrenheit|c|f)?\b")
_INFANT = re.compile(r"\b(baby|infant|newborn|months? old)\b")
_CHILD = re.compile(r"\b(child|kid|son|daughter|toddler)\b")
_AGE_YEARS = re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?) old\b")


@dataclass
class Entities:
    duration_days: Optional[float] = None
    temperature_c: Optional[float] = None
    age_group: Optional[str] = None  # "infant" | "child" | None

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def __bool__(self):
        return bool(self.as_dict())


def extract_duration(text: str) -> Optional[float]:
    m = _DURATION.search(text)
    if m:
        raw, unit = m.group(1), m.group(2).rstrip("s")
        value = _NUMBER_WORDS.get(raw)
        if value is None:
            value = float(raw)
        return round(value * _UNIT_DAYS[unit], 2)
    if re.search(r"\b(since|from) yesterday\b|\blast night\b", text):
        return 1.0
    if re.search(r"\bsince (this )?(morning|today)\b|\bjust (now|started)\b", text):
        return 0.25
    return None


def extract_temperature(text: str) -> Optional[float]:
    """Return a body temperature in Celsius if the message contains one."""
    for m in _TEMPERATURE.finditer(text):
        value, unit = float(m.group(1)), m.group(2)
        if unit in ("f", "fahrenheit") or (unit is None and 94 <= value <= 108):
            celsius = (value - 32) * 5 / 9
        elif unit in ("c", "celsius") or (unit is None and 35 <= value <= 42.5):
            celsius = value
        else:
            continue
        if 34 <= celsius <= 43:  # plausible body temperature only
            return round(celsius, 1)
    return None


def extract_entities(text: str) -> Entities:
    """`text` must already be normalised."""
    years = _AGE_YEARS.search(text)
    if _INFANT.search(text):
        age = "infant"
    elif _CHILD.search(text) or (years and int(years.group(1)) <= 12):
        age = "child"
    else:
        age = None
    return Entities(duration_days=extract_duration(text),
                    temperature_c=extract_temperature(text),
                    age_group=age)
