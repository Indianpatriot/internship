"""Step 2 - Bloom's L1-L6 verb bank + helpers + question validator."""
import json
import random
import re
from pathlib import Path

LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]
LEVEL_NAMES = {"L1": "Remember", "L2": "Understand", "L3": "Apply",
               "L4": "Analyze", "L5": "Evaluate", "L6": "Create"}
_NAME_TO_LEVEL = {v.lower(): k for k, v in LEVEL_NAMES.items()}
_PATH = Path(__file__).with_name("verbs.json")


def _load(path=_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {lvl: [v.lower() for v in data[lvl]] for lvl in LEVELS}


_BANK = _load()
_VERB_TO_LEVEL = {v: lvl for lvl, vs in _BANK.items() for v in vs}


def normalize_level(level) -> str:
    """Accept 3, 'L3', 'bt3', 'apply' -> 'L3'."""
    s = str(level).strip().lower()
    if s.isdigit() and 1 <= int(s) <= 6:
        return f"L{int(s)}"
    m = re.fullmatch(r"(?:l|bt)([1-6])", s)
    if m:
        return f"L{m.group(1)}"
    if s in _NAME_TO_LEVEL:
        return _NAME_TO_LEVEL[s]
    raise ValueError(f"Unknown Bloom's level: {level!r}")


def get_verbs(level):
    return list(_BANK[normalize_level(level)])


def sample_verbs(level, k=4, rng=None):
    rng = rng or random
    verbs = get_verbs(level)
    return rng.sample(verbs, min(k, len(verbs)))


def verb_level(verb):
    return _VERB_TO_LEVEL.get(verb.strip().lower())


def first_word(question: str) -> str:
    q = re.sub(r"^[\s\"'“”*_#\-•\d.)(]+", "", question or "")
    m = re.match(r"[A-Za-z]+", q)
    return m.group(0).lower() if m else ""


def validate_question(question, level, min_words=5, max_words=80):
    """Return (ok, reason). Checks verb-at-start, single question, sane length."""
    level = normalize_level(level)
    q = (question or "").strip()
    if not q:
        return False, "empty"
    if "\n" in q:
        return False, "multi-line"
    n = len(q.split())
    if n < min_words or n > max_words:
        return False, f"length {n}"
    if q.count("?") > 1:
        return False, "multiple questions"
    fw = first_word(q)
    if fw not in _BANK[level]:
        return False, f"first word '{fw}' not an {level} verb"
    return True, "ok"
