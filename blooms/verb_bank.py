"""Step 2 - Bloom's L1-L6 verb bank + helpers + question validator."""
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# -----------------------------------------------------------------------------
# Knowledge Dimension x Bloom's Verb Guide (Anderson & Krathwohl 2D Matrix)
# -----------------------------------------------------------------------------

KD_BLOOM_VERBS: dict[tuple[str, str], list[str]] = {
    # Factual Knowledge: basic elements, terminology, specific details
    ("L1", "Factual"): ["name", "define", "list", "state", "identify", "recall", "label", "enumerate", "mention", "match", "select"],
    ("L2", "Factual"): ["summarize", "describe", "paraphrase", "express", "outline"],
    ("L3", "Factual"): ["respond", "show", "provide"],
    ("L4", "Factual"): ["categorize", "relate"],
    ("L5", "Factual"): ["check", "determine"],
    ("L6", "Factual"): ["generate", "assemble"],

    # Conceptual Knowledge: interrelationships, classifications, models, theories
    ("L1", "Conceptual"): ["recognize", "identify", "select", "match"],
    ("L2", "Conceptual"): ["explain", "classify", "summarize", "describe", "discuss", "interpret", "paraphrase", "clarify"],
    ("L3", "Conceptual"): ["illustrate", "demonstrate", "show", "apply"],
    ("L4", "Conceptual"): ["analyze", "distinguish", "contrast", "compare", "differentiate", "examine", "categorize", "relate"],
    ("L5", "Conceptual"): ["evaluate", "determine", "judge", "assess", "appraise"],
    ("L6", "Conceptual"): ["formulate", "synthesize", "propose", "construct", "design"],

    # Procedural Knowledge: skills, algorithms, techniques, how-to methods
    ("L1", "Procedural"): ["recall", "list", "state", "identify"],
    ("L2", "Procedural"): ["outline", "clarify", "describe", "explain"],
    ("L3", "Procedural"): ["demonstrate", "implement", "apply", "modify", "execute", "calculate", "compute", "solve", "use", "illustrate"],
    ("L4", "Procedural"): ["integrate", "analyze", "deconstruct", "investigate"],
    ("L5", "Procedural"): ["validate", "assess", "check", "determine"],
    ("L6", "Procedural"): ["design", "construct", "develop", "plan", "compose", "devise"],

    # Metacognitive Knowledge: strategic knowledge, cognitive tasks, reflection
    ("L1", "Metacognitive"): ["identify", "recognize", "recall"],
    ("L2", "Metacognitive"): ["predict", "interpret", "clarify", "explain"],
    ("L3", "Metacognitive"): ["use", "apply", "demonstrate"],
    ("L4", "Metacognitive"): ["deconstruct", "analyze", "examine", "relate", "integrate"],
    ("L5", "Metacognitive"): ["justify", "recommend", "evaluate", "reflect", "defend", "critique", "prioritize", "argue"],
    ("L6", "Metacognitive"): ["devise", "plan", "formulate", "create", "propose"],
}


def normalize_knowledge_dimension(kd: Optional[str]) -> str:
    """Normalize KD name to canonical title-case (Factual, Conceptual, Procedural, Metacognitive)."""
    if not kd:
        return "Conceptual"
    s = str(kd).strip().lower()
    if s.startswith("fact"):
        return "Factual"
    if s.startswith("proc"):
        return "Procedural"
    if s.startswith("meta"):
        return "Metacognitive"
    return "Conceptual"


def get_allowed_kd_verbs(level: str, kd: str) -> list[str]:
    """Get list of allowed starting verbs for a given (Bloom level, Knowledge Dimension) pair."""
    lvl = normalize_level(level)
    k_dim = normalize_knowledge_dimension(kd)
    key = (lvl, k_dim)
    if key in KD_BLOOM_VERBS:
        return list(KD_BLOOM_VERBS[key])
    # Fallback to level bank if pair is unspecified
    return get_verbs(lvl)


def is_verb_allowed_for_kd(verb: str, level: str, kd: str) -> bool:
    """Check if a verb is allowed for the assigned Bloom's level and Knowledge Dimension."""
    if not verb:
        return False
    v = verb.strip().lower()
    allowed = get_allowed_kd_verbs(level, kd)
    return v in allowed


def nearest_valid_kd_verb(verb: str, level: str, kd: str) -> str:
    """Return the nearest valid verb from the same Bloom's level and KD."""
    allowed = get_allowed_kd_verbs(level, kd)
    if not allowed:
        return "Explain"
    v = (verb or "").strip().lower()
    if v in allowed:
        return v.capitalize()
    # Simple lexical prefix matching or fallback to first allowed verb
    for cand in allowed:
        if cand.startswith(v[:3]) or v.startswith(cand[:3]):
            return cand.capitalize()
    return allowed[0].capitalize()


def sample_kd_verbs(level: str, kd: str, k: int = 4, rng: Optional[random.Random] = None) -> list[str]:
    """Sample verbs constrained by both Bloom's level and Knowledge Dimension."""
    rng = rng or random
    allowed = get_allowed_kd_verbs(level, kd)
    if not allowed:
        return sample_verbs(level, k, rng)
    return rng.sample(allowed, min(k, len(allowed)))


def validate_question_kd(question: str, level: str, kd: str) -> tuple[bool, str]:
    """Validate that the question starts with a valid verb for its assigned (Bloom level, KD)."""
    ok, reason = validate_question(question, level)
    if not ok:
        return False, reason
    fw = first_word(question)
    if not is_verb_allowed_for_kd(fw, level, kd):
        norm_lvl = normalize_level(level)
        norm_kd = normalize_knowledge_dimension(kd)
        allowed = get_allowed_kd_verbs(norm_lvl, norm_kd)
        return False, f"verb '{fw}' is not allowed for ({norm_lvl}, {norm_kd}); expected one of {allowed}"
    return True, "ok"


def replace_question_verb(question: str, new_verb: str) -> str:
    """Replace the first word/verb of a question with new_verb while preserving punctuation and capitalization."""
    q = (question or "").strip()
    if not q:
        return f"{new_verb}?"
    # Find leading non-letter chars (e.g. quotes or numbering)
    prefix_m = re.match(r"^([\s\"'“”*_#\-•\d.)(]*)", q)
    prefix = prefix_m.group(1) if prefix_m else ""
    rest = q[len(prefix):]
    word_m = re.match(r"^[A-Za-z]+", rest)
    if word_m:
        old_word = word_m.group(0)
        remainder = rest[len(old_word):]
        return f"{prefix}{new_verb}{remainder}"
    return f"{prefix}{new_verb} {rest}"
