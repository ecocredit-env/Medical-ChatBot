"""Safety rules that run before anything else.

If a message contains a red-flag phrase the bot stops normal processing and
tells the person to get emergency help. These rules err on the side of
caution: a false alarm is far better than a missed emergency.
"""
import os
import re

EMERGENCY_NUMBER = os.environ.get("CHATBOT_EMERGENCY_NUMBER", "112")
CRISIS_LINE = os.environ.get("CHATBOT_CRISIS_LINE", "Tele-MANAS on 14416")

_EMERGENCY_PHRASES = [
    r"chest pain", r"pain in (my |the )?chest", r"chest (is )?tight", r"tightness in (my |the )?chest",
    r"pressure in (my |the )?chest", r"heart attack",
    r"cannot breathe", r"difficulty (in )?breathing", r"trouble breathing",
    r"hard to breathe", r"struggling to breathe", r"short(ness)? of breath",
    r"breathless", r"choking", r"turning blue",
    r"unconscious", r"unresponsive", r"not responding", r"passed out",
    r"fainted", r"collapsed", r"not waking up",
    r"seizure", r"convulsions?", r"having fits", r"a fit",
    r"severe bleeding", r"heavy bleeding", r"bleeding heavily", r"bleeding a lot",
    r"(will|does) not stop bleeding", r"bleeding (will|does) not stop", r"losing a lot of blood",
    r"stroke", r"face (is )?drooping", r"slurred speech",
    r"cannot move (my |his |her |the )?(arm|leg|face|side)",
    r"sudden (weakness|numbness)",
    r"anaphyla\w+", r"throat (is )?(swelling|closing)", r"(lips|face|tongue) (is |are )?swelling",
    r"swelling of (the |my )?(lips|face|tongue|throat)",
    r"poison(ed|ing)?", r"overdosed?", r"swallowed (bleach|acid|poison|chemical|battery)",
    r"cough(ing|ed)? (up )?blood", r"vomiting blood", r"blood in (my |the )?vomit",
    r"snake ?bite", r"electric shock", r"electrocuted", r"drowning",
    r"severe burn", r"badly burned", r"head injury", r"hit (my|his|her) head hard",
]
_CRISIS_PHRASES = [
    r"suicid\w*", r"kill myself", r"end my life", r"take my (own )?life",
    r"want to die", r"wish i (was|were) dead", r"better off dead",
    r"self harm", r"self-harm", r"hurt myself", r"harm myself",
    r"cutting myself", r"cut myself on purpose",
    r"no reason to live", r"do not want to live",
]

_EMERGENCY = re.compile(r"\b(" + "|".join(_EMERGENCY_PHRASES) + r")\b")
_CRISIS = re.compile(r"\b(" + "|".join(_CRISIS_PHRASES) + r")\b")


def check(text: str):
    """Return 'crisis', 'emergency' or None for a normalised message."""
    if _CRISIS.search(text):
        return "crisis"
    if _EMERGENCY.search(text):
        return "emergency"
    return None


def emergency_message() -> str:
    return (f"This could be a medical emergency. Call {EMERGENCY_NUMBER} or go "
            "to the nearest emergency department now. Do not wait and do not "
            "drive yourself. If someone is with you, ask them to help.")


def crisis_message() -> str:
    return ("I'm really sorry you're going through this, and I'm glad you said "
            "it. You don't have to face it alone. Please reach out now: call "
            f"{CRISIS_LINE} (free, open 24 hours), or {EMERGENCY_NUMBER} if you "
            "are in immediate danger. If you can, tell someone you trust and "
            "stay with them. If you are outside India, call your local "
            "emergency number.")
