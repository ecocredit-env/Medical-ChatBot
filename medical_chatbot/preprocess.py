"""Text cleaning and normalisation applied to every user message."""
import re

# Longer phrases first so they are replaced before their parts.
_REPLACEMENTS = [
    (r"\bcan['’]?t\b", "cannot"),
    (r"\bcan not\b", "cannot"),
    (r"\bwon['’]?t\b", "will not"),
    (r"\bdon['’]?t\b", "do not"),
    (r"\bdoesn['’]?t\b", "does not"),
    (r"\bisn['’]?t\b", "is not"),
    (r"\bi['’]?m\b", "i am"),
    (r"\bi['’]?ve\b", "i have"),
    (r"\bit['’]?s\b", "it is"),
    (r"\bdiarrhea\b", "diarrhoea"),
    (r"\bdiarrhoea\b", "diarrhoea"),
    (r"\bloose motions?\b", "diarrhoea"),
    (r"\bthe runs\b", "diarrhoea"),
    (r"\bthrowing up\b", "vomiting"),
    (r"\bthrew up\b", "vomiting"),
    (r"\bthrow up\b", "vomiting"),
    (r"\bpuk(e|ing|ed)\b", "vomiting"),
    (r"\btummy\b", "stomach"),
    (r"\bbelly\b", "stomach"),
    (r"\btemp\b", "temperature"),
    (r"\bsugar levels?\b", "blood sugar"),
    (r"\bunable to\b", "cannot"),
    (r"\bnot able to\b", "cannot"),
    (r"\bmeds?\b", "medicine"),
    (r"\btablets?\b", "tablets"),
]
_COMPILED = [(re.compile(p), r) for p, r in _REPLACEMENTS]


def normalise(text: str) -> str:
    """Lower-case, expand contractions, unify spellings, strip punctuation."""
    text = text.lower().strip()
    text = text.replace("°", " degrees ")
    for pattern, repl in _COMPILED:
        text = pattern.sub(repl, text)
    text = re.sub(r"[^a-z0-9./ ]+", " ", text)   # keep digits, '.', '/'
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)  # drop dots that are not decimals
    return re.sub(r"\s+", " ", text).strip()


def tokenise(text: str) -> list[str]:
    return normalise(text).split()
