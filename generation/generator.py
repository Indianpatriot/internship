"""Step 3c / Issue 4 - question and answer generation with validation, retry, de-duplication, and multi-book assembly.

Optimized with:
- Parallel slot generation via ThreadPoolExecutor with strict order preservation
- Fast LLM result caching (keyed by prompt, model, temperature, seed)
- Bounded token caps (num_predict) tailored to questions vs answers
- Targeted Bloom's verb retries
- Reused chunk embeddings for grounded answer cosine similarity
"""
import concurrent.futures
import random
import re
import threading
from typing import Any, Optional, Union
import config

from blooms.revised_taxonomy import normalize_knowledge_dimension
from generation.diagnostics import (
    ReasonCode,
    classify_exception_to_reason_code,
    log_slot_attempt,
)
from blooms.verb_bank import (
    first_word,
    get_allowed_kd_verbs,
    get_verbs,
    is_verb_allowed_for_kd,
    nearest_valid_kd_verb,
    normalize_level,
    replace_question_verb,
    sample_kd_verbs,
    sample_verbs,
    validate_question,
    validate_question_kd,
)
from config import (
    ANSWER_GROUNDING_SENTENCE_SIMILARITY,
    ANSWER_MAX_UNGROUNDED_SENTENCE_RATIO,
    ANSWER_MIN_SIMILARITY,
    CO_SIMILARITY_THRESHOLD,
    GENERATION_WORKERS,
    MAX_CONTEXT_WORDS,
    MIN_CONTEXT_WORDS,
    QUESTION_SIMILARITY_DUPLICATE_THRESHOLD,
)
from generation.cache import get_cached_response, store_cached_response
from generation.prompts import build_answer_prompt, build_prompt
from retrieval.sources import is_content_chunk, select_chunks_for_unit

DANGLING_ENDINGS = {
    "and", "the", "with", "impl", "ho", "availab", "by", "to", "or", "of", "in", "on",
    "at", "for", "from", "as", "is", "are", "a", "an", "that", "this", "which", "into",
    "onto", "be", "been", "being", "have", "has", "had", "do", "does", "did", "but",
    "so", "if", "because", "while", "where", "when", "how", "what", "who",
}

TERMINAL_PUNCTUATION = (".", "!", "?", '."', ".'", '!"', "!'", '?"', "?'", "```", ")", "]", "}")

TEXTBOOK_REF_PATTERNS = [
    r"\b(this|the)\s+(book|text|chapter|section|author|authors|passage|diagram above|diagram below|figure\s*\d|table\s*\d)\b",
    r"\bchapter\s+\d+\b",
    r"\bfigure\s*\d+(?:\.\d+)?\b",
    r"\btable\s*\d+(?:\.\d+)?\b",
    r"\btarget\s+audience\b",
    r"\baim\s+of\s+chapter\b",
    r"\bas\s+described\s+in\b",
    r"\baccording\s+to\s+(?:the\s+author|the\s+text|the\s+book|chapter)\b",
    r"\bcovered\s+in\s+chapter\b",
    r"\bmentioned\s+in\s+chapter\b",
    r"\blook\s+at\s+(?:the\s+)?(?:diagram|figure|table)\b",
]
TEXTBOOK_REF_RE = re.compile("|".join(TEXTBOOK_REF_PATTERNS), re.I)

# Problem 4: Banned deictic and relative reference patterns
UNSEEN_MATERIAL_PATTERNS = [
    r"\bprovided\b",
    r"\babove\b",
    r"\bbelow\b",
    r"\bthe\s+example\b",
    r"\bthe\s+technique\b",
    r"\bthis\s+technique\b",
    r"\bthe\s+listing\b",
    r"\bListing\s+\d+\b",
    r"\bthe\s+describe\s+method\b",
    r"\bthe\s+context\b",
    r"\bin\s+the\s+passage\b",
]
UNSEEN_MATERIAL_RE = re.compile("|".join(UNSEEN_MATERIAL_PATTERNS), re.I)

CODE_REFERENCE_PATTERNS = [
    r"\bthe\s+following\s+code\b",
    r"\bthe\s+given\s+code\b",
    r"\bthe\s+code\s+snippet\b",
    r"\bthe\s+script\b",
]
CODE_REFERENCE_RE = re.compile("|".join(CODE_REFERENCE_PATTERNS), re.I)

BANNED_ANSWER_PATTERNS = [
    r"\baccording\s+to\s+the\s+provided\s+context\b",
    r"\bin\s+the\s+provided\s+example\b",
    r"\bas\s+shown\s+in\s+Listing\b",
    r"\bthe\s+provided\s+text\b",
    r"\bthe\s+provided\s+context\b",
    r"\bthe\s+text\s+does\s+not\b",
    r"\bnot\s+explicitly\s+mentioned\b",
    r"\bbased\s+on\s+general\s+knowledge\b",
    r"\bcannot\s+be\s+answered\b",
    r"\bthe\s+passage\s+doesn'?t\b",
    r"\bnot\s+mentioned\s+in\s+the\b",
    r"\bdoes\s+not\s+provide\b",
    r"\bnot\s+specified\s+in\s+the\b",
    r"\blacks\s+information\b",
    r"\b(?:the\s+)?(?:correct\s+)?(?:answer|option)\s+is\s+[a-d]\b",
    r"\boption\s+[a-d]\s+is\s+correct\b",
    r"^(?:the\s+)?answer:\s*[a-d]\b",
    r"^(?:option\s+)?[a-d][\.\)]?$",
]
BANNED_ANSWER_RE = re.compile("|".join(BANNED_ANSWER_PATTERNS), re.I)

BANNED_LEAK_PATTERNS = [
    r"\bintroductory\s+chapter\b",
    r"\bearlier\s+chapter\b",
    r"\bin\s+this\s+chapter\b",
    r"\bin\s+this\s+book\b",
    r"\bthe\s+author(?:s)?\b",
    r"/home/[a-zA-Z0-9_\.\-]+/",
    r"[A-Za-z]:\\(?:Users|Documents and Settings)\\[a-zA-Z0-9_\.\-\\]+",
    r"/(?:Users|home)/[a-zA-Z0-9_\.\-]+",
]
BANNED_LEAK_RE = re.compile("|".join(BANNED_LEAK_PATTERNS), re.I)


def has_unseen_material_reference(text: str, is_question: bool = True) -> tuple[bool, str]:
    """Check for dependencies on unseen material or textbook leaks (Problem 4 & Problem 6).

    Allows code/script references only if code is included inline in the question stem.
    """
    s = text or ""
    if BANNED_LEAK_RE.search(s):
        return True, "textbook leak or personal path reference"
    if is_question:
        # Check inline code
        has_inline_code = bool(re.search(r"`[^`]+`|```[\s\S]+?```", s))
        if CODE_REFERENCE_RE.search(s) and not has_inline_code:
            return True, "relative code reference without inline code in stem"
        if UNSEEN_MATERIAL_RE.search(s):
            return True, "unseen material reference"
    else:
        if BANNED_ANSWER_RE.search(s):
            return True, "banned phrase in model answer"
        if UNSEEN_MATERIAL_RE.search(s):
            return True, "unseen material reference in model answer"
    return False, ""


def is_circular_question(q: str) -> bool:
    """Problem 5.4: Reject stems where one concept is a case of the other or circular."""
    fw = first_word(q)
    if fw in ("integrate", "compare", "contrast", "differentiate", "distinguish"):
        m = re.search(
            r"\b(?:integrate|compare|contrast|differentiate|distinguish)\s+(?:the\s+role\s+of\s+|the\s+differences?\s+between\s+|between\s+)?(.+?)\s+(?:into|with(?:\s+that\s+of)?|and)\s+(.+?)(?:\?|,|\bin\b|\bterms\b|\busing\b|\bas\b|$)",
            q,
            re.I,
        )
        if m:
            left_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", m.group(1))]
            right_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", m.group(2))]
            stop = {"concept", "mechanism", "process", "approach", "system", "method", "using", "between", "role", "the", "for", "with", "into"}
            left = {w for w in left_words if w not in stop}
            right = {w for w in right_words if w not in stop}
            if left and right:
                if left == right or left.issubset(right) or right.issubset(left):
                    return True
                # Check for continuous deployment into continuous delivery
                if ("continuous" in left and "continuous" in right) and ("deployment" in left or "delivery" in left) and ("deployment" in right or "delivery" in right):
                    return True
    return False


def check_compare_entities_in_context(q: str, context: str) -> bool:
    """Problem 5.3 & Problem 3.1: For compare/contrast/distinguish questions, both named entities
    must be of the same category and both must appear in chunk context.
    """
    fw = first_word(q)
    if fw in ("compare", "contrast", "differentiate", "distinguish"):
        m = re.search(
            r"\b(?:compare|contrast|differentiate|distinguish)\s+(?:the\s+role\s+of\s+|the\s+differences?\s+between\s+|between\s+)?(.+?)\s+(?:with(?:\s+that\s+of)?|and)\s+(.+?)(?:\?|,|\bin\b|\bterms\b|\busing\b|\bas\b|$)",
            q,
            re.I,
        )
        if m:
            e1 = m.group(1).strip().lower()
            e2 = m.group(2).strip().lower()
            stop = {"the", "a", "an", "role", "of", "process", "concept", "approach", "task"}
            generic = {
                "network", "networks", "system", "systems", "component", "components",
                "method", "methods", "model", "models", "architecture", "architectures",
                "tool", "tools", "data", "technique", "techniques", "mechanism", "mechanisms",
                "pattern", "patterns", "design", "designs",
            }
            e1_words = [w for w in e1.split() if w not in stop]
            e2_words = [w for w in e2.split() if w not in stop]

            # Cross-category non-alternative checks (Problem 3.1)
            e1_str = " ".join(e1_words)
            e2_str = " ".join(e2_words)
            is_pipeline_1 = bool(re.search(r"\b(?:continuous\s+delivery|ci/cd|pipeline|delivery|integration)\b", e1_str))
            is_network_2 = bool(re.search(r"\b(?:networking|network|ip|dns|port|overlay)\b", e2_str))
            is_pipeline_2 = bool(re.search(r"\b(?:continuous\s+delivery|ci/cd|pipeline|delivery|integration)\b", e2_str))
            is_network_1 = bool(re.search(r"\b(?:networking|network|ip|dns|port|overlay)\b", e1_str))
            if (is_pipeline_1 and is_network_2) or (is_pipeline_2 and is_network_1):
                return False

            is_docker_task = bool(re.search(r"\b(?:docker\s+build|build\s+and\s+push|container\s+build)\b", e1_str))
            is_iac_task = bool(re.search(r"\b(?:terraform|installer|provisioning|packer)\b", e2_str))
            is_docker_task_2 = bool(re.search(r"\b(?:docker\s+build|build\s+and\s+push|container\s+build)\b", e2_str))
            is_iac_task_1 = bool(re.search(r"\b(?:terraform|installer|provisioning|packer)\b", e1_str))
            if (is_docker_task and is_iac_task) or (is_docker_task_2 and is_iac_task_1):
                return False

            e1_substantive = [w for w in e1_words if w not in generic]
            e2_substantive = [w for w in e2_words if w not in generic]
            e1_check = e1_substantive if e1_substantive else e1_words
            e2_check = e2_substantive if e2_substantive else e2_words
            ctx_lower = context.lower()
            has_e1 = all(w in ctx_lower for w in e1_check) if e1_check else True
            has_e2 = all(w in ctx_lower for w in e2_check) if e2_check else True
            return has_e1 and has_e2
    return True


UNDERSPECIFIED_STEM_PATTERNS = [
    r"\bwhich\s+of\s+the\s+(?:two\s+)?(?:tools|options|approaches|methods)\b",
    r"\bthe\s+two\s+(?:tools|options|approaches|methods)\b",
    r"\bthe\s+(?:tools|approaches|options)\s+discussed\b",
    r"\bthe\s+tools\s+that\s+will\s+be\s+used\b",
    r"\bthe\s+tools\s+to\s+be\s+used\b",
    r"\b(?:what\s+are|name|list|state|identify)\s+the\s+tools\b(?!\s+(?:in|of|for)\s+[A-Z])",
    r"\b(?:the|this)\s+(?:approach|technique|method|process)\s+(?:used\s+to|for|in)\b(?!\s+[A-Z])",
]
UNDERSPECIFIED_STEM_RE = re.compile("|".join(UNDERSPECIFIED_STEM_PATTERNS), re.I)


def is_underspecified_stem(q: str) -> bool:
    """Problem 6.2: Reject stems referring to 'the tools' / 'the approach' without naming the referent."""
    return bool(UNDERSPECIFIED_STEM_RE.search(q or ""))


def check_l1_marks_demand(q: str, level: str, marks: Optional[int]) -> tuple[bool, str]:
    """Problem 6.3: A slot with 8 or more marks at L1 must ask for a multi-part recall."""
    if level == "L1" and marks and marks >= 8:
        s = (q or "").lower()
        has_compound_recall = bool(re.search(r"\b(?:list|identify|name|state|recognize|recall)\b.*\band\s+(?:state|describe|explain)\s+(?:the\s+)?(?:purpose|role|function|operation)", s))
        has_large_count = bool(re.search(r"\b(?:four|five|six|seven|eight|multiple|several)\s+(?:[a-zA-Z-]+\s+)?(?:components|features|steps|types|categories|principles|properties|attributes|elements|factors|parts|phases|prerequisites|primitives)\b", s))
        if not (has_compound_recall or has_large_count):
            return False, "8+ mark L1 slot must ask for multi-part recall (e.g. four or more items, or list and state purpose/role), not a small single-item recall"
    return True, "ok"


def count_stem_demands(stem: str) -> int:
    """Problem 3.2: Count demands in stem by verbs/clauses and conjunctions."""
    s = (stem or "").strip()
    demands = 1
    demands += len(re.findall(
        r"(?:,\s*(?:compare|contrast|describe|explain|define|state|outline|illustrate|provide|give|evaluate|justify)|and\s+(?:state|explain|describe|outline|identify|list|justify|compare|evaluate|illustrate|provide|give)|also|considering|justify\s+your\s+choice|which\s+(?:option\s+)?would\s+you\s+choose)",
        s,
        re.I
    ))
    demands += len(re.findall(r";\s*[A-Za-z]", s))
    return demands


def simplify_stem_demands(stem: str, marks: Optional[int]) -> str:
    """Problem 3.2 & 3.3: Strip trailing 'Which option would you choose and why?' for <8 marks,
    and enforce demand limits: 4-5 marks <= 2, 6-8 marks <= 3, 10+ <= 4.
    """
    if not stem:
        return stem
    q = stem.strip()
    m_val = marks or 5

    # 1. Strip trailing choice questions for < 8 marks
    if m_val < 8:
        q = re.sub(
            r"[\s,\.]*\b(?:which\s+(?:option\s+)?would\s+you\s+(?:choose|prefer)(?:\s+and\s+why)?|which\s+(?:would\s+you|would\s+one)\s+(?:choose|prefer)(?:\s+and\s+why)?|justify\s+your\s+(?:choice|selection))\s*\?*$",
            "?",
            q,
            flags=re.I
        )

    # 2. Limit demands
    max_demands = 2 if m_val <= 5 else (3 if m_val <= 8 else 4)
    for _ in range(5):
        if count_stem_demands(q) <= max_demands:
            break
        trimmed = re.sub(
            r"[\s,]+(?:and\s+)?(?:explain|give|illustrate|provide|describe|outline|state|justify|compare|contrast)[^,?.!]+(?:\?|\.)?$",
            "?",
            q,
            flags=re.I
        )
        if trimmed != q and len(trimmed.split()) >= 4:
            q = trimmed
        else:
            break

    if not q.endswith("?"):
        q = q.rstrip(".!,") + "?"
    return q


TERM_STEM_RE = re.compile(
    r"\b(?:identify|recognize|name|recall)\s+(?:the\s+term|the\s+name|the\s+concept)\b|\bwhat\s+is\s+the\s+term\s+for\b",
    re.I
)


def validate_term_answer(question: str, answer: str) -> tuple[bool, str]:
    """Problem 2.1: 'Identify / Recognize / Name / Recall the term' answer must be under ~8 words
    and not echo or repeat question wording.
    """
    if not TERM_STEM_RE.search(question or ""):
        return True, "ok"

    words = [w for w in re.split(r"\s+", answer.strip()) if w]
    if len(words) > 8:
        return False, f"term answer is too long ({len(words)} words > 8; expected short term)"

    stop_words = {"what", "is", "the", "term", "for", "a", "an", "in", "of", "to", "that", "which", "and", "by", "or", "as"}
    q_tokens = {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", question) if w.lower() not in stop_words}
    a_tokens = {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", answer) if w.lower() not in stop_words}
    if a_tokens and (len(a_tokens & q_tokens) / len(a_tokens)) >= 0.70:
        return False, "term answer echoes or restates the question stem wording"
    return True, "ok"


DEPRECATED_CURRENCY_WATCHLIST = [
    "rkt",
    "ksonnet",
    "fabric8",
    "docker machine",
    "docker links",
    "dockershim",
    "replicationcontroller",
]


def check_currency_watchlist(text: str) -> list[str]:
    """Check text for deprecated or outdated technology topics from the currency watchlist."""
    s_lower = (text or "").lower()
    found = []
    for term in DEPRECATED_CURRENCY_WATCHLIST:
        if re.search(r"\b" + re.escape(term) + r"\b", s_lower):
            found.append(term)
    return found


DEICTIC_PATTERNS = [
    r"\bat the time of writing\b",
    r"\bthe author\b",
    r"\bin this chapter\b",
    r"\bin this section\b",
    r"\bthis chapter\b",
    r"\bthis section\b",
    r"\bthe following example\b",
    r"\bas seen above\b",
    r"\bas mentioned earlier\b",
    r"\bin the preceding section\b",
    r"\bthe provided context\b",
    r"\bthe provided passage\b",
]
DEICTIC_RE = re.compile("|".join(DEICTIC_PATTERNS), re.I)

PRESUPPOSITION_PATTERNS = [
    r"\bjustify\s+your\s+(?:choice|selection|decision|answer|opinion)\b",
    r"\bwhy\s+did\s+you\s+(?:choose|select|prefer)\b",
    r"\bwhich\s+(?:would|do)\s+you\s+(?:choose|prefer)\b",
    r"\bexplain\s+your\s+(?:choice|selection|decision|reason)\b",
    r"\bwhat\s+is\s+your\s+(?:choice|recommendation)\b",
]
PRESUPPOSITION_RE = re.compile("|".join(PRESUPPOSITION_PATTERNS), re.I)

PREMISE_IN_STEM_PATTERNS = [
    r"^(?:assuming|given\s+that|supposing)\s+[^,]+,\s*(?:why|how|what|explain)",
]
PREMISE_IN_STEM_RE = re.compile("|".join(PREMISE_IN_STEM_PATTERNS), re.I)


def lint_question_stem(stem: str, marks: Optional[int] = None) -> tuple[bool, str]:
    """Rule-based stem linter for academic paper standards (PART E)."""
    s = (stem or "").strip()
    if not s:
        return False, "empty stem"

    # 1. Banned deictic book-references
    m_deictic = DEICTIC_RE.search(s)
    if m_deictic:
        return False, f"stem contains book-deictic phrase: '{m_deictic.group(0)}'"

    # 2. Presuppositions ('Justify YOUR choice')
    m_presup = PRESUPPOSITION_RE.search(s)
    if m_presup:
        return False, f"stem contains personalized presupposition: '{m_presup.group(0)}'"

    # 3. Premise in stem
    m_premise = PREMISE_IN_STEM_RE.search(s)
    if m_premise:
        return False, f"stem contains unverified hypothetical premise: '{m_premise.group(0)}'"

    # 4. Stray quotes and typos
    if s.count('"') % 2 != 0:
        return False, "stem contains unbalanced double quotes"
    if s.startswith(('"', "'")) or s.endswith(('"', "'")):
        return False, "stem has stray leading or trailing quotation marks"
    if re.search(r"\?\?|\.\.|,,|!\?", s):
        return False, "stem contains duplicated punctuation marks (e.g. ?? or ..)"

    # 5. Two-part stems under 6 marks
    m_val = marks or 5
    if m_val < 6:
        if s.count("?") > 1:
            return False, f"two-part compound stem ({s.count('?')} questions) not permitted under 6 marks"
        if re.search(r"\b(?:explain|describe|define|state|what\s+is)\b.*\band\s+(?:evaluate|analyze|justify|compare|how\s+does)\b", s, re.I):
            return False, "two-part compound stem with multiple distinct cognitive tasks not permitted under 6 marks"

    return True, "ok"


def validate_known_technical_facts(question: str, answer: str) -> tuple[bool, str]:
    """Validate answers against known technical traps and hallucinations (PART E)."""
    q_lower = (question or "").lower()
    a_lower = (answer or "").lower()

    # Trap 1: docker export config loss
    if "docker export" in q_lower or "docker export" in a_lower:
        if any(w in a_lower for w in ["preserves config", "maintains env", "retains cmd", "keeps entrypoint", "saves metadata"]):
            return False, "docker export loses metadata (CMD, ENTRYPOINT, ENV, volumes) and only exports the container filesystem; claiming it preserves configuration is factually wrong"

    # Trap 2: docker build creates Dockerfile
    if "docker build" in q_lower or "docker build" in a_lower:
        if any(w in a_lower for w in ["creates a dockerfile", "generates the dockerfile", "writes the dockerfile", "creates the dockerfile"]):
            return False, "docker build reads from an existing Dockerfile to build an image; it does NOT create or generate a Dockerfile"

    # Trap 3: apt-get assumption on non-Debian (Alpine or RHEL)
    if "alpine" in q_lower or "alpine" in a_lower:
        if "apt-get" in a_lower or "apt install" in a_lower:
            return False, "Alpine Linux uses apk package manager (apk add), not apt-get or apt"

    # Trap 4: restartPolicy on ReplicationController
    if "replicationcontroller" in q_lower or "replicationcontroller" in a_lower:
        if any(w in a_lower for w in ["restartpolicy: onfailure", "restartpolicy: never", "restartpolicy onfailure", "restartpolicy never"]):
            return False, "ReplicationController only supports restartPolicy: Always; OnFailure and Never are invalid for ReplicationController pods"

    return True, "ok"


def extract_technical_tokens(answer: str) -> list[str]:
    """Extract commands, flags, tools, and code identifiers from answer text."""
    tokens = set()
    s = answer or ""

    # 1. Inline code tokens
    for code in re.findall(r"`([^`\n]+)`", s):
        tok = code.strip()
        if len(tok) >= 2 and not tok.isdigit():
            tokens.add(tok)

    # 2. CLI flags (--restart, -it, -d, --name, etc.)
    for flag in re.findall(r"(?:^|\s)(--[a-zA-Z0-9_-]{2,}|-[a-zA-Z0-9]{1,4})\b", s):
        tokens.add(flag.strip())

    # 3. CLI command pairs (docker run, kubectl get, apt-get, apk add, etc.)
    for cmd in re.findall(r"\b(docker\s+[a-z-]+|kubectl\s+[a-z-]+|apt-get|apk\s+add|systemctl|dockershim|rkt)\b", s, re.I):
        tokens.add(cmd.strip().lower())

    return sorted(list(tokens))


def check_hallucinated_tokens(answer: str, context: str) -> list[str]:
    """Check extracted commands, flags, and tools against chunk context; return ungrounded tokens."""
    tokens = extract_technical_tokens(answer)
    ctx_lower = (context or "").lower()
    ungrounded = []

    for tok in tokens:
        t_low = tok.lower()
        if t_low not in ctx_lower:
            bare = t_low.lstrip("-")
            if len(bare) >= 3 and bare in ctx_lower:
                continue
            ungrounded.append(tok)

    return ungrounded


def check_cross_slot_contradictions(questions: list[dict[str, Any]], embedder: Any = None) -> list[tuple[int, int, str]]:
    """Problem 2.3: Compare statements about the same concept across slots; flag contradictions for review."""
    contradictions = []
    items = []
    for idx, q in enumerate(questions):
        ans = q.get("answer", "")
        if ans and not ans.startswith("[NEEDS REVIEW"):
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ans) if len(s.split()) >= 4]
            items.append((idx, q, sents))

    NEGATION_WORDS = {"not", "never", "no", "cannot", "neither", "nor", "fails", "without", "lost", "destroy"}
    AFFIRMATION_WORDS = {"restarts", "retained", "survives", "preserves", "maintains", "handled", "recovers"}

    for a_i in range(len(items)):
        idx_a, q_a, sents_a = items[a_i]
        for b_i in range(a_i + 1, len(items)):
            idx_b, q_b, sents_b = items[b_i]
            for sa in sents_a:
                sa_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", sa.lower()))
                for sb in sents_b:
                    sb_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", sb.lower()))
                    overlap = sa_words & sb_words
                    if len(overlap) >= 3:
                        neg_a = bool(overlap & NEGATION_WORDS or (sa_words & NEGATION_WORDS))
                        neg_b = bool(overlap & NEGATION_WORDS or (sb_words & NEGATION_WORDS))
                        aff_a = bool(sa_words & AFFIRMATION_WORDS)
                        aff_b = bool(sb_words & AFFIRMATION_WORDS)
                        if (neg_a and aff_b) or (neg_b and aff_a) or (neg_a != neg_b and len(overlap) >= 4):
                            msg = f"potential cross-slot contradiction between {q_a.get('id', 'Slot')} and {q_b.get('id', 'Slot')}: '{sa[:50]}...' vs '{sb[:50]}...'"
                            contradictions.append((idx_a, idx_b, msg))
                            break
    return contradictions


CO_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "into", "over", "able", "student", "students", "understand", "explain",
    "apply", "analyze", "evaluate", "create", "concepts", "basics", "principles",
    "course", "outcome", "outcomes", "learn", "study", "general", "overview",
}


def verify_question_co_alignment(
    questions: list[dict[str, Any]],
    outcome_descs: dict[str, str],
    embedder: Any = None,
    threshold: float = CO_SIMILARITY_THRESHOLD,
    margin: float = 0.15,
) -> list[tuple[int, str]]:
    """Post-generation Course Outcome (CO) verification (PART F).

    Checks:
    1. Keyword anchors: Identifies content anchors per CO and flags strong cross-CO keywords with 0 target overlap.
    2. Vector embedding similarity & margin check when an embedder is provided.

    Returns:
        List of (question_index, mismatch_reason).
    """
    if not questions or not outcome_descs:
        return []

    mismatches: list[tuple[int, str]] = []

    # 1. Build keyword anchors per CO
    co_anchors: dict[str, set[str]] = {}
    for co_code, desc in outcome_descs.items():
        words = set(re.findall(r"\b[a-zA-Z]{4,}\b", (desc or "").lower()))
        co_anchors[co_code] = words - CO_STOPWORDS

    # 2. Vectorized embedding check (if embedder provided)
    if embedder is not None:
        try:
            import numpy as np
            candidate_qs = [
                (idx, q_rec, q_rec.get("co"), outcome_descs.get(q_rec.get("co")))
                for idx, q_rec in enumerate(questions)
                if q_rec.get("co") and outcome_descs.get(q_rec.get("co"))
                and not q_rec.get("needs_review") and q_rec.get("valid", True)
            ]
            if candidate_qs:
                unique_cos = list(outcome_descs.items())
                co_keys = [c[0] for c in unique_cos]
                co_texts = [c[1] for c in unique_cos]
                q_texts = [c[1].get("question", "") for c in candidate_qs]

                all_encoded = embedder.encode(co_texts + q_texts, normalize_embeddings=True)
                co_embs_map = {co_keys[k]: all_encoded[k] for k in range(len(co_keys))}
                q_embs = all_encoded[len(co_keys):]

                for k, (orig_idx, q_rec, co_code, _) in enumerate(candidate_qs):
                    co_vec = co_embs_map.get(co_code)
                    if co_vec is not None:
                        sim_val = float(np.dot(q_embs[k], co_vec))
                        # Check margin against other COs
                        other_sims = {c: float(np.dot(q_embs[k], co_embs_map[c])) for c in co_keys if c != co_code}
                        max_other = max(other_sims.values()) if other_sims else 0.0
                        best_other_co = max(other_sims, key=other_sims.get) if other_sims else None

                        if sim_val < threshold:
                            mismatches.append(
                                (orig_idx, f"CO mismatch - review (similarity {sim_val:.2f} < {threshold})")
                            )
                        elif max_other - sim_val >= margin and sim_val < 0.40:
                            mismatches.append(
                                (orig_idx, f"CO mismatch - review (closer to {best_other_co} by margin {max_other - sim_val:.2f})")
                            )
                return mismatches
        except Exception:
            pass

    # 3. Keyword anchor check (fallback or standalone when embedder is absent)
    for idx, q_rec in enumerate(questions):
        co_code = q_rec.get("co")
        if not co_code or co_code not in outcome_descs:
            continue
        if q_rec.get("needs_review") or not q_rec.get("valid", True):
            continue

        q_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", (q_rec.get("question", "") + " " + q_rec.get("answer", "")).lower())) - CO_STOPWORDS
        target_anchors = co_anchors.get(co_code, set())
        target_overlap = len(q_words & target_anchors)

        if target_overlap <= 1 and len(target_anchors) >= 2:
            # Check if another CO has strong keyword overlap
            other_overlaps = {
                c: len(q_words & anchors)
                for c, anchors in co_anchors.items()
                if c != co_code
            }
            max_other = max(other_overlaps.values()) if other_overlaps else 0
            best_other = max(other_overlaps, key=other_overlaps.get) if other_overlaps else None
            if max_other >= 3 and (max_other - target_overlap >= 2):
                mismatches.append(
                    (idx, f"CO mismatch - review (keyword overlap with {best_other} vs {co_code})")
                )

    return mismatches


def check_chapter_coverage_balance(
    questions: list[dict[str, Any]],
    unit_sources: Optional[dict[Any, Any]],
) -> tuple[list[str], dict[Any, dict[str, int]]]:
    """Check whether generated questions in each unit adequately balance across mapped chapters (PART G).

    Returns:
        (warnings, coverage_counts_by_unit)
    """
    if not unit_sources or not questions:
        return [], {}

    warnings = []
    coverage_by_unit: dict[Any, dict[str, int]] = {}

    for u, b_map in unit_sources.items():
        if not isinstance(b_map, dict):
            continue
        mapped_chapters = set()
        for b_id, chaps in b_map.items():
            if chaps:
                for ch in chaps:
                    mapped_chapters.add(str(ch).strip())

        if len(mapped_chapters) < 2:
            continue

        unit_qs = [q for q in questions if q.get("unit") == u or str(q.get("unit")) == str(u)]
        counts = {ch: 0 for ch in mapped_chapters}
        for q in unit_qs:
            q_chap = str(q.get("chapter", "")).strip()
            for ch in mapped_chapters:
                if ch.lower() in q_chap.lower() or q_chap.lower() in ch.lower():
                    counts[ch] += 1
                    break

        coverage_by_unit[u] = counts
        uncovered = [ch for ch, cnt in counts.items() if cnt == 0]
        max_cnt = max(counts.values()) if counts else 0
        if uncovered and max_cnt >= 2:
            warnings.append(
                f"Unit {u} chapter coverage imbalance: {', '.join(uncovered)} had 0 questions generated while other chapters received {max_cnt} question(s)."
            )

    return warnings, coverage_by_unit


DEMAND_KEYWORDS = {
    "trade-offs": ["trade-off", "tradeoff", "advantage", "disadvantage", "pros", "cons", "cost", "benefit"],
    "trade-off": ["trade-off", "tradeoff", "advantage", "disadvantage", "pros", "cons", "cost", "benefit"],
    "justify": ["because", "reason", "due to", "justif", "why", "since", "therefore"],
    "differences": ["whereas", "while", "unlike", "differ", "instead of", "contrast", "distinction", "however"],
    "advantages": ["advantage", "benefit", "strength", "superior", "gain"],
    "steps": ["step", "first", "next", "then", "finally", "1.", "2.", "procedure", "stage"],
    "consequences": ["impact", "lead to", "result in", "consequence", "risk", "damage", "affect"],
    "solution": ["solution", "mitigat", "resolve", "prevent", "address", "remedy", "fix"],
}


def check_demand_coverage(question: str, answer: str) -> tuple[bool, Optional[str]]:
    """Problem 5.2: Check that demanded question parts (trade-offs, justify, steps, etc.) appear in the model answer."""
    q_lower = question.lower()
    a_lower = answer.lower()
    for demand, markers in DEMAND_KEYWORDS.items():
        if demand in q_lower:
            has_marker = any(m in a_lower for m in markers)
            if not has_marker:
                return False, f"missing expected '{demand}' content in answer"
    return True, None


def is_truncated_text(text: str, is_question: bool = False) -> bool:
    """Return True if text appears truncated (cut off mid-sentence or mid-word)."""
    s = (text or "").strip()
    if not s:
        return True
    if is_question:
        if not s.endswith(("?", '?"', "?'")):
            return True
        last_word = re.sub(r"[^\w]", "", s.split()[-1].lower())
        if last_word in DANGLING_ENDINGS:
            return True
        return False

    ends_punct = any(s.endswith(p) for p in TERMINAL_PUNCTUATION)
    if not ends_punct:
        return True
    words = s.split()
    if words:
        last_word = re.sub(r"[^\w]", "", words[-1].lower())
        if last_word in DANGLING_ENDINGS:
            return True
    return False


def truncate_to_last_complete_sentence(text: str) -> str:
    """Fallback: truncate to the last sentence ending with terminal punctuation."""
    s = (text or "").strip()
    matches = list(re.finditer(r"[\.!\?]+(?:\s+|$)", s))
    if matches:
        last_end = matches[-1].end()
        return s[:last_end].strip()
    return s


def has_textbook_reference(text: str) -> bool:
    """Check if text refers to physical textbook rather than concept."""
    return bool(TEXTBOOK_REF_RE.search(text or ""))


def has_banned_answer_phrases(text: str) -> bool:
    """Check if model answer contains meta commentary or apology phrases."""
    return bool(BANNED_ANSWER_RE.search(text or ""))


def clean_answer_text(text: str) -> tuple[str, list[str]]:
    """Clean model answer: replace '― OR ―' and 'OR' with 'Alternative approach:'.

    Ensures ONE authoritative model answer or distinct alternative approaches.
    Never leaves '― OR ―' inside answer text.
    """
    raw = (text or "").strip()
    if not raw:
        return "", []

    lines = raw.splitlines()
    cur_buf: list[str] = []
    alternatives: list[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(?:[—―\-]+\s*)?OR(?:\s*[—―\-]+)?[:\.]?$", stripped, re.I):
            alt_text = "\n".join(cur_buf).strip()
            if alt_text:
                alternatives.append(alt_text)
            cur_buf = []
        elif re.match(r"^Alternative approach(?:\s*\d+)?[:\.]?", stripped, re.I):
            alt_text = "\n".join(cur_buf).strip()
            if alt_text:
                alternatives.append(alt_text)
            content_after = re.sub(r"^Alternative approach(?:\s*\d+)?[:\.]?\s*", "", stripped, flags=re.I).strip()
            cur_buf = [content_after] if content_after else []
        else:
            cur_buf.append(line)

    if cur_buf:
        alt_text = "\n".join(cur_buf).strip()
        if alt_text:
            alternatives.append(alt_text)

    if not alternatives:
        alternatives = [raw]

    unique_alts: list[str] = []
    for a in alternatives:
        # Strip any internal '― OR ―'
        a_clean = re.sub(r"^[—―-]+\s*OR\s*[—―-]+$", "", a, flags=re.I | re.M).strip()
        a_clean = re.sub(r"[—―-]+\s*OR\s*[—―-]+", "Alternative approach:", a_clean, flags=re.I).strip()
        if a_clean and a_clean not in unique_alts:
            unique_alts.append(a_clean)

    if len(unique_alts) > 1:
        full = unique_alts[0] + "\n\n" + "\n\n".join(f"Alternative approach:\n{a}" for a in unique_alts[1:])
    else:
        full = unique_alts[0] if unique_alts else raw

    full = re.sub(r"[—―]+\s*OR\s*[—―]+", "Alternative approach:", full, flags=re.I)
    return full.strip(), unique_alts


def clean_output(raw: str) -> str:
    """Strip fences, preambles, numbering and quotes from LLM output."""
    text = re.sub(r"^```\w*|```$", "", (raw or "").strip(), flags=re.M).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    q = lines[0]
    for l in lines:
        if not re.match(r"^(here('s| is)|sure|okay|certainly)\b.*:$", l, re.I):
            q = l
            break
    q = re.sub(r"^(question|q)\s*\d*\s*[:.)-]\s*", "", q, flags=re.I)
    q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", q)
    return q.strip(" \"'“”*_`")


def _jaccard(a: str, b: str) -> float:
    x, y = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(x & y) / len(x | y) if x | y else 0.0


def is_duplicate(q, seen, threshold=0.8):
    return any(_jaccard(q, s) >= threshold for s in seen)


def generate_question(
    chunk,
    level,
    client,
    rng=None,
    max_retries=3,
    marks=None,
    k_verbs=4,
    knowledge_dimension=None,
    used_topics=None,
    force_regenerate=False,
    exclude_deprecated=True,
):
    level = normalize_level(level)
    rng = rng or random.Random()
    kd = normalize_knowledge_dimension(knowledge_dimension) if knowledge_dimension else "Conceptual"
    context = " ".join(chunk["text"].split()[:MAX_CONTEXT_WORDS])
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", context))
    q, reason, attempts = "", "no output", 0
    ok = False

    # Proportional token cap
    num_predict = getattr(config, "QUESTION_NUM_PREDICT", 120)
    client_model = getattr(client, "model", "llama3.1:8b")

    for attempts in range(1, max_retries + 1):
        # Sample verbs constrained by both Bloom's level and Knowledge Dimension (Problem 6)
        verbs = sample_kd_verbs(level, kd, k_verbs, rng)
        if not has_numbers:
            verbs = [v for v in verbs if v.lower() not in ("compute", "calculate")]
            if not verbs:
                all_allowed = get_allowed_kd_verbs(level, kd)
                verbs = [v for v in all_allowed if v.lower() not in ("compute", "calculate")]
                verbs = rng.sample(verbs, min(k_verbs, len(verbs))) if verbs else sample_verbs(level, k_verbs, rng)

        prompt = build_prompt(context, level, verbs, marks=marks, knowledge_dimension=kd, used_topics=used_topics)
        temp = min(0.3 + 0.2 * (attempts - 1), 0.9)

        cached = get_cached_response(prompt, client_model, temperature=temp, force_regenerate=force_regenerate)
        if cached:
            raw = cached
        else:
            raw = client.generate(prompt, temperature=temp, num_predict=num_predict)

        q = clean_output(raw)
        ok, reason = validate_question(q, level)

        # Check unseen material / relative references (Problem 4)
        if ok:
            has_unseen, unseen_msg = has_unseen_material_reference(q, is_question=True)
            if has_unseen:
                ok = False
                reason = unseen_msg

        # Check textbook references
        if ok and has_textbook_reference(q):
            ok = False
            reason = "question contains textbook references"

        # Check truncation
        if ok and is_truncated_text(q, is_question=True):
            ok = False
            reason = "question is truncated"

        # Check circularity (Problem 5.4)
        if ok and is_circular_question(q):
            ok = False
            reason = "circular question"

        # Check compare entities in context (Problem 5.3)
        if ok and not check_compare_entities_in_context(q, context):
            ok = False
            reason = "compared entities not found in context"

        # Check numeric context for compute/calculate (Problem 5.5)
        if ok and not has_numbers and first_word(q) in ("compute", "calculate"):
            old_v = first_word(q)
            new_v = nearest_valid_kd_verb("demonstrate", level, kd)
            q = replace_question_verb(q, new_v)
            ok, reason = validate_question(q, level)

        # Check Knowledge Dimension verb alignment (Problem 5)
        if ok and knowledge_dimension:
            from blooms.verb_bank import correct_stem_verb_for_kd
            corr_q, was_corr, _ = correct_stem_verb_for_kd(q, level, kd)
            if was_corr:
                v_ok, _ = validate_question(corr_q, level)
                if v_ok:
                    q = corr_q
            else:
                ok_kd, kd_reason = validate_question_kd(q, level, kd)
                if not ok_kd:
                    if attempts < max_retries:
                        ok = False
                        reason = kd_reason
                    else:
                        old_verb = first_word(q)
                        new_verb = nearest_valid_kd_verb(old_verb, level, kd)
                        q = replace_question_verb(q, new_verb)
                        ok, reason = validate_question(q, level)

        # Check demand limits and trailing choices (Problem 3.2 & 3.3)
        if ok:
            q = simplify_stem_demands(q, marks)
            max_dem = 2 if (marks or 5) <= 5 else (3 if (marks or 5) <= 8 else 4)
            if count_stem_demands(q) > max_dem and attempts < max_retries:
                ok = False
                reason = f"overloaded stem ({count_stem_demands(q)} demands > max {max_dem})"

        # Check underspecified stem (Problem 6.2)
        if ok and is_underspecified_stem(q):
            ok = False
            reason = "underspecified stem: refers to 'the tools' / 'the approach' without naming referent"

        # Check 8+ marks L1 multi-part recall (Problem 6.3)
        if ok:
            ok_l1, l1_msg = check_l1_marks_demand(q, level, marks)
            if not ok_l1:
                ok = False
                reason = l1_msg

        # Check stem linter (PART E)
        if ok:
            ok_lint, lint_msg = lint_question_stem(q, marks=marks)
            if not ok_lint:
                ok = False
                reason = lint_msg

        # Check currency watchlist (PART E)
        if ok and exclude_deprecated:
            dep_terms = check_currency_watchlist(q)
            if dep_terms:
                ok = False
                reason = f"stem contains deprecated technology '{dep_terms[0]}'"

        # Outdated terms glossary replacement (Problem 6.4)
        if ok:
            from generation.glossary import apply_glossary_replacements
            q = apply_glossary_replacements(q)

        if ok:
            if not cached and not is_truncated_text(q, is_question=True):
                store_cached_response(prompt, client_model, raw, temperature=temp)
            break
        elif attempts < max_retries:
            targeted_verbs = sample_kd_verbs(level, kd, k_verbs, rng)
            if not has_numbers:
                targeted_verbs = [v for v in targeted_verbs if v.lower() not in ("compute", "calculate")]
            targeted_prompt = (
                f"{prompt}\n\nNote: The question must strictly start with one of: {', '.join(targeted_verbs)}. "
                f"Ensure the question is self-contained with no references to unseen material, 'provided', or 'the snippet'."
            )
            raw = client.generate(targeted_prompt, temperature=temp + 0.1, num_predict=num_predict)
            q = clean_output(raw)
            ok, reason = validate_question(q, level)
            if ok:
                has_unseen, unseen_msg = has_unseen_material_reference(q, is_question=True)
                if has_unseen:
                    ok = False
                    reason = unseen_msg
            if ok and has_textbook_reference(q):
                ok = False
                reason = "question contains textbook references"
            if ok and is_truncated_text(q, is_question=True):
                ok = False
                reason = "question is truncated"
            if ok and is_circular_question(q):
                ok = False
                reason = "circular question"
            if ok and not check_compare_entities_in_context(q, context):
                ok = False
                reason = "compared entities not found in context"
            if ok and not has_numbers and first_word(q) in ("compute", "calculate"):
                new_v = nearest_valid_kd_verb("demonstrate", level, kd)
                q = replace_question_verb(q, new_v)
                ok, reason = validate_question(q, level)
            if ok and knowledge_dimension:
                from blooms.verb_bank import correct_stem_verb_for_kd
                corr_q, was_corr, _ = correct_stem_verb_for_kd(q, level, kd)
                if was_corr:
                    v_ok, _ = validate_question(corr_q, level)
                    if v_ok:
                        q = corr_q
                else:
                    ok_kd, kd_reason = validate_question_kd(q, level, kd)
                    if not ok_kd:
                        old_verb = first_word(q)
                        new_verb = nearest_valid_kd_verb(old_verb, level, kd)
                        q = replace_question_verb(q, new_verb)
                        ok, reason = validate_question(q, level)
            if ok:
                q = simplify_stem_demands(q, marks)
                max_dem = 2 if (marks or 5) <= 5 else (3 if (marks or 5) <= 8 else 4)
                if count_stem_demands(q) > max_dem and attempts < max_retries:
                    ok = False
                    reason = f"overloaded stem ({count_stem_demands(q)} demands > max {max_dem})"
            if ok and is_underspecified_stem(q):
                ok = False
                reason = "underspecified stem: refers to 'the tools' / 'the approach' without naming referent"
            if ok:
                ok_l1, l1_msg = check_l1_marks_demand(q, level, marks)
                if not ok_l1:
                    ok = False
                    reason = l1_msg
            if ok:
                from generation.glossary import apply_glossary_replacements
                q = apply_glossary_replacements(q)
            attempts += 1
            if ok:
                if not is_truncated_text(q, is_question=True):
                    store_cached_response(prompt, client_model, raw, temperature=temp)
                break

    return {
        "question": q,
        "level": level,
        "verb": first_word(q),
        "marks": marks,
        "knowledge_dimension": kd or "Conceptual",
        "kd": kd or "Conceptual",
        "valid": ok,
        "reason": reason,
        "attempts": attempts,
        "chunk_id": chunk.get("chunk_id"),
        "book_id": chunk.get("book_id"),
        "chapter": chunk.get("chapter"),
        "section": chunk.get("section"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
    }


def extract_scheme_points_and_marks(answer: str) -> tuple[list[str], list[int]]:
    """Extract itemized points and associated mark splits from an answer.
    Recognizes:
      'Point 1 (2M): ...', '1. (2M) ...', 'Criteria (3M): ...', 'Verdict (2M): ...',
      'Architecture (4M): ...', '1. Point ... [2 Marks]'
    """
    lines = [line.strip() for line in (answer or "").splitlines() if line.strip()]
    points = []
    marks_list = []

    for line in lines:
        if line.startswith("― OR ―") or line == "OR" or line.startswith("Alternative approach"):
            continue
        m_match = re.search(r"\((\d+)\s*M(?:arks?)?\)|\[(\d+)\s*M(?:arks?)?\]|\b(\d+)\s*M(?:arks?)?:", line, re.I)
        is_point = bool(re.match(r"^(?:Point\s*\d+|Criteria|Verdict|Justification|Trade-offs?|Architecture|Plan(?:\s*Steps?)?|Rationale|\d+[\.\)])\b", line, re.I))
        if is_point or m_match:
            points.append(line)
            if m_match:
                val = int(m_match.group(1) or m_match.group(2) or m_match.group(3))
                marks_list.append(val)

    return points, marks_list


def validate_evaluation_scheme(
    answer: str,
    marks: int,
    level: Optional[str] = None,
) -> tuple[bool, str, list[str]]:
    """Validate itemized key points and marking scheme for the answer.

    Rules:
    1. For marks >= 5: reject single-sentence answers.
    2. Min points: ceil(marks / 2) for marks >= 5.
    3. sum(points) == marks when mark splits are provided.
    4. For L5/L6: check structured rubric (criteria/verdict/justification/trade-offs or architecture/rationale/trade-offs).
    """
    if not answer or not answer.strip():
        return False, "empty answer", []

    marks = marks if marks is not None and marks > 0 else 5

    # If multiple alternatives separated by OR or Alternative approach, validate each
    alt_blocks = [a.strip() for a in re.split(r"\n+OR\n+|\n+Alternative approach:?\n*", answer, flags=re.I) if a.strip()]
    if not alt_blocks:
        alt_blocks = [answer]

    for block in alt_blocks:
        if marks < 5:
            _, mark_splits = extract_scheme_points_and_marks(block)
            if mark_splits and sum(mark_splits) != marks:
                return False, f"itemized mark splits sum {sum(mark_splits)} != expected {marks}M", ["RUBRIC"]
            continue

        # Rule 1: Reject single-sentence answers for >= 5 marks
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", block) if s.strip() and len(s.split()) >= 3]
        if len(sents) <= 1:
            return False, f"single-sentence answer rejected for {marks}-mark question", ["RUBRIC"]

        min_points = (marks + 1) // 2  # ceil(marks / 2)
        norm_level = normalize_level(level) if level else None
        if norm_level in ("L5", "L6") and marks >= 8:
            min_points = min(min_points, 4)

        points, mark_splits = extract_scheme_points_and_marks(block)
        effective_point_count = len(points) if points else len(sents)
        if effective_point_count < min_points:
            return False, f"answer has {effective_point_count} points, below minimum {min_points} for {marks} marks", ["RUBRIC"]

        # Rule 3: sum(points) == marks
        if mark_splits:
            if sum(mark_splits) != marks:
                return False, f"itemized mark splits sum {sum(mark_splits)} != expected {marks}M", ["RUBRIC"]

        # Rule 4: Structured rubric for L5/L6
        if norm_level == "L5":
            ans_lower = block.lower()
            has_eval_aspects = any(kw in ans_lower for kw in ["criteria", "verdict", "justif", "trade-off", "tradeoff", "evaluation", "benchmark", "critique"])
            if not has_eval_aspects:
                return False, "L5 evaluation answer lacks structured evaluation rubric (criteria, verdict, justification, or trade-offs)", ["RUBRIC"]
        elif norm_level == "L6":
            ans_lower = block.lower()
            has_create_aspects = any(kw in ans_lower for kw in ["plan", "step", "architecture", "design", "rationale", "trade-off", "tradeoff", "workflow"])
            if not has_create_aspects:
                return False, "L6 design/creation answer lacks structured rubric (plan steps, architecture, rationale, or trade-offs)", ["RUBRIC"]

    return True, "ok", []


def check_or_twin_answer_depth_parity(questions: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    """Check that choice (OR) twin questions have balanced answer depth and points."""
    disparities = []
    by_unit_and_marks: dict[tuple[Any, int], list[tuple[int, dict[str, Any]]]] = {}

    for idx, q in enumerate(questions):
        u = q.get("unit")
        m = q.get("marks", 5)
        alt = q.get("alternative_group")
        if u is not None and alt and m >= 5:
            by_unit_and_marks.setdefault((u, m), []).append((idx, q))

    for (u, m), group in by_unit_and_marks.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                idx_a, q_a = group[i]
                idx_b, q_b = group[j]
                if q_a.get("alternative_group") != q_b.get("alternative_group"):
                    words_a = len(q_a.get("answer", "").split())
                    words_b = len(q_b.get("answer", "").split())
                    if words_a > 0 and words_b > 0:
                        ratio = min(words_a, words_b) / max(words_a, words_b)
                        if ratio < 0.5:
                            msg = f"RUBRIC: answer depth disparity between OR twins ({words_a}w vs {words_b}w)"
                            disparities.append((idx_a, idx_b, msg))
    return disparities


def generate_answer(
    question: str,
    chunk: dict[str, Any],
    marks: Optional[int],
    client: Any,
    rng: Optional[random.Random] = None,
    max_retries: int = 3,
    check_similarity: Optional[bool] = None,
    force_regenerate: bool = False,
    level: Optional[str] = None,
) -> dict[str, Any]:
    """Generate model answer grounded in chunk context.

    Validates non-empty, minimum length for marks, absence of banned meta phrases,
    absence of textbook references, non-truncation, demand coverage, and cosine similarity with chunk text.
    """
    rng = rng or random.Random()
    marks = marks if marks is not None and marks > 0 else 5
    context = " ".join(chunk.get("text", "").split()[:MAX_CONTEXT_WORDS])

    is_mock = getattr(client, "model", None) == "mock" or "Mock" in client.__class__.__name__
    should_check_similarity = (not is_mock) if check_similarity is None else bool(check_similarity)

    last_answer = ""
    last_alternatives: list[str] = []
    reason = "no output"
    ok = False
    ungrounded_sentences: list[str] = []
    demand_issue: Optional[str] = None

    if marks <= 5:
        num_predict = 180
    elif marks <= 8:
        num_predict = 320
    else:
        num_predict = 500

    client_model = getattr(client, "model", "llama3.1:8b")

    for attempt in range(1, max_retries + 1):
        temp = min(0.3 + 0.2 * (attempt - 1), 0.9)
        strict = (attempt > 1)
        prompt = build_answer_prompt(question, context, marks=marks, level=level, strict_grounding=strict)

        cached = get_cached_response(prompt, client_model, temperature=temp, force_regenerate=force_regenerate)
        if cached:
            raw = cached
        else:
            raw = client.generate(prompt, temperature=temp, num_predict=num_predict)

        # Clean output fences
        text = re.sub(r"^```\w*|```$", "", (raw or "").strip(), flags=re.M).strip()
        full_answer, alternatives = clean_answer_text(text)
        last_alternatives = alternatives
        last_answer = "\nOR\n".join(alternatives) if len(alternatives) > 1 else full_answer

        # Validation 1: non-empty
        if not last_answer:
            reason = "empty answer"
            continue

        # Validation 2: minimum length for marks (except for short term recall questions)
        is_term_q = bool(TERM_STEM_RE.search(question or ""))
        word_count = len(last_answer.split())
        min_words = min(20, max(5, int(marks * 2))) if not is_term_q else 1
        if word_count < min_words:
            reason = f"answer length {word_count} words below minimum {min_words} for {marks} marks"
            continue

        # Validation 2b: term answer validation (Problem 2.1)
        ok_term, term_msg = validate_term_answer(question, last_answer)
        if not ok_term:
            reason = term_msg
            continue

        # Validation 2d: known technical fact validation (PART E)
        ok_facts, fact_msg = validate_known_technical_facts(question, last_answer)
        if not ok_facts:
            reason = fact_msg
            continue

        # Outdated terms glossary replacement (Problem 6.4)
        from generation.glossary import apply_glossary_replacements
        last_answer = apply_glossary_replacements(last_answer)
        last_alternatives = [apply_glossary_replacements(a) for a in last_alternatives]

        # Validation 3: banned meta-commentary phrases & unseen references (Problem 4 & 5)
        has_unseen, unseen_msg = has_unseen_material_reference(last_answer, is_question=False)
        if has_unseen:
            reason = unseen_msg
            continue
        if has_banned_answer_phrases(last_answer):
            reason = "answer contains banned phrases (meta commentary / apology)"
            continue

        # Validation 4: textbook references
        if has_textbook_reference(last_answer):
            reason = "answer contains textbook references"
            continue

        # Validation 5: truncation check
        if is_truncated_text(last_answer, is_question=False):
            if attempt < max_retries:
                reason = "answer is truncated"
                continue
            else:
                fallback = truncate_to_last_complete_sentence(last_answer)
                if len(fallback.split()) >= min_words:
                    last_answer = fallback
                    last_alternatives = [fallback]
                else:
                    reason = "answer is truncated"
                    continue

        # Validation 6: demand coverage check (Problem 5.2)
        cov_ok, cov_msg = check_demand_coverage(question, last_answer)
        if not cov_ok and attempt < max_retries:
            reason = cov_msg or "missing demanded question content"
            demand_issue = cov_msg
            continue

        # Validation 7: grounded cosine similarity check
        ungrounded_sentences = []
        if should_check_similarity:
            from retrieval.co_mapper import _m
            embedder = _m()
            if embedder is not None:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", last_answer) if s.strip() and len(s.split()) >= 4]
                try:
                    import numpy as np
                    all_texts = [context, last_answer] + sentences
                    all_embs = embedder.encode(all_texts, normalize_embeddings=True)
                    chunk_emb = all_embs[0]
                    ans_emb = all_embs[1]
                    overall_sim = float(np.dot(chunk_emb, ans_emb))
                    if overall_sim < ANSWER_MIN_SIMILARITY:
                        reason = f"similarity {overall_sim:.2f} below threshold {ANSWER_MIN_SIMILARITY}"
                        continue

                    if sentences:
                        sent_embs = all_embs[2:]
                        cos_sims = np.dot(sent_embs, chunk_emb)
                        for s_txt, cos_sim in zip(sentences, cos_sims):
                            if float(cos_sim) < ANSWER_GROUNDING_SENTENCE_SIMILARITY:
                                ungrounded_sentences.append(s_txt)
                        ungrounded_ratio = len(ungrounded_sentences) / len(sentences)
                        if ungrounded_ratio > ANSWER_MAX_UNGROUNDED_SENTENCE_RATIO and attempt < max_retries:
                            reason = f"ungrounded sentences ({len(ungrounded_sentences)}/{len(sentences)})"
                            continue
                except Exception:
                    pass

        # Validation 8: scheme of evaluation validation (PART D)
        ok_scheme, scheme_msg, _ = validate_evaluation_scheme(last_answer, marks, level=level)
        if not ok_scheme:
            reason = scheme_msg
            continue

        ok = True
        reason = "ok"
        if not cached and not is_truncated_text(last_answer, is_question=False):
            store_cached_response(prompt, client_model, raw, temperature=temp)
        break

    res_dict: dict[str, Any] = {
        "answer": last_answer,
        "answer_alternatives": last_alternatives,
        "valid": ok,
        "reason": reason,
    }
    if ungrounded_sentences:
        res_dict["ungrounded_sentences"] = ungrounded_sentences
    if demand_issue and not ok:
        res_dict["demand_issue"] = demand_issue
    return res_dict


def generate_paper(chunks, level_counts, client, seed=None, marks=None, on_progress=None):
    """level_counts: {'L1': 3, 'L4': 2}. Returns {'questions': [...], 'rejected': [...]}."""
    rng = random.Random(seed)
    usable = [c for c in chunks if c.get("word_count", len(c["text"].split())) >= MIN_CONTEXT_WORDS]
    questions, rejected, seen = [], [], []
    for level, n in level_counts.items():
        level = normalize_level(level)
        order = usable[:]
        rng.shuffle(order)
        produced = 0
        for chunk in order * 2:                      # at most two passes over the chunks
            if produced >= n:
                break
            rec = generate_question(chunk, level, client, rng, marks=marks)
            if not rec["valid"] or is_duplicate(rec["question"], seen):
                rejected.append(rec)
                continue
            seen.append(rec["question"])
            rec["id"] = f"Q{len(questions) + 1:03d}"
            questions.append(rec)
            produced += 1
            if on_progress:
                on_progress(rec)
    return {"questions": questions, "rejected": rejected}


def check_question_duplicates(
    questions: list[dict[str, Any]],
    embedding_model: Any,
    threshold: float = QUESTION_SIMILARITY_DUPLICATE_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Compute pairwise cosine similarities across all question texts."""
    valid_qs = [q for q in questions if q.get("valid", True) and not q.get("question", "").startswith("[NEEDS REVIEW")]
    if len(valid_qs) < 2 or embedding_model is None:
        return []
    try:
        texts = [q.get("question", "") for q in valid_qs]
        embs = embedding_model.encode(texts, normalize_embeddings=True)
        import numpy as np
        sim_matrix = np.dot(embs, embs.T)

        dups = []
        for i in range(len(valid_qs)):
            for j in range(i + 1, len(valid_qs)):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    # Map back to original indices in questions
                    orig_i = questions.index(valid_qs[i])
                    orig_j = questions.index(valid_qs[j])
                    dups.append((orig_i, orig_j, sim))
        return dups
    except Exception:
        return []


def check_question_co_similarity(
    question_text: str,
    co_desc: str,
    embedding_model: Any,
    threshold: float = CO_SIMILARITY_THRESHOLD,
) -> tuple[bool, float]:
    """Compute cosine similarity between generated question and assigned CO description."""
    if not co_desc or embedding_model is None:
        return True, 1.0
    try:
        import numpy as np
        embs = embedding_model.encode([question_text, co_desc], normalize_embeddings=True)
        sim = float(np.dot(embs[0], embs[1]))
        return (sim >= threshold), sim
    except Exception:
        return True, 1.0


def check_paper_identifier_consistency(questions: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """Extract backticked or snake_case identifiers across paper and flag conflicting spellings."""
    from difflib import SequenceMatcher
    seen_identifiers: dict[str, int] = {}
    inconsistencies = []

    for q_idx, q in enumerate(questions):
        text = (q.get("question", "") + " " + q.get("answer", ""))
        tokens = set(re.findall(r"`([a-zA-Z0-9_\-]+)`", text))
        tokens.update(re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text))

        for tok in tokens:
            if len(tok) < 5 or tok.startswith("http"):
                continue
            tok_lower = tok.lower()
            for prev_tok, prev_idx in seen_identifiers.items():
                if prev_idx != q_idx and prev_tok != tok_lower:
                    sm = SequenceMatcher(None, prev_tok, tok_lower)
                    if sm.ratio() >= 0.88 and abs(len(prev_tok) - len(tok_lower)) <= 2:
                        inconsistencies.append((q_idx, f"Inconsistent identifier spelling: '{tok}' vs earlier '{prev_tok}'"))
                        break
            seen_identifiers[tok_lower] = q_idx

    return inconsistencies


def verify_paper_completeness(
    slots: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Check that every blueprint slot has question/answer and OR alternatives mirror each other."""
    # Robust argument order: detect if caller passed (questions, slots) instead of (slots, questions)
    if slots and any("question" in item for item in slots) and not any("question" in item for item in questions):
        slots, questions = questions, slots

    issues = []
    if len(questions) != len(slots):
        issues.append(f"Slot count mismatch: paper has {len(questions)} questions, expected {len(slots)} from blueprint")

    # Check for placeholder / failed slots
    for i, q in enumerate(questions):
        q_text = q.get("question", "")
        a_text = q.get("answer", "")
        q_id = q.get("id") or f"Slot {i+1}"
        if not q_text or q_text.startswith("[NEEDS REVIEW") or q.get("needs_review") or not q.get("valid", True):
            reason = q.get("review_reason") or q.get("reason") or "generation incomplete"
            hint = q.get("hint")
            hint_str = f" (💡 {hint})" if hint else ""
            issues.append(f"{q_id}: {reason}{hint_str}")
        elif not a_text or a_text.startswith("[NEEDS REVIEW"):
            issues.append(f"{q_id}: missing model answer")

    # Check OR symmetry within units
    by_unit_alt: dict[Any, dict[str, list[dict[str, Any]]]] = {}
    for q in questions:
        u = q.get("unit")
        alt = q.get("alternative_group")
        if u is not None and alt:
            by_unit_alt.setdefault(u, {})
            q_no = q.get("question_no")
            by_unit_alt[u].setdefault(str(q_no), []).append(q)

    for u, q_groups in by_unit_alt.items():
        if len(q_groups) >= 2:
            group_keys = list(q_groups.keys())
            g1_qs = q_groups[group_keys[0]]
            g2_qs = q_groups[group_keys[1]]
            if len(g1_qs) != len(g2_qs):
                issues.append(
                    f"Unit {u}: Alternative options do not mirror each other (Option {group_keys[0]} has {len(g1_qs)} parts, Option {group_keys[1]} has {len(g2_qs)} parts)"
                )
            else:
                g1_marks = [q.get("marks") for q in g1_qs]
                g2_marks = [q.get("marks") for q in g2_qs]
                if g1_marks != g2_marks:
                    issues.append(
                        f"Unit {u}: Sub-question marks do not match between options ({g1_marks} vs {g2_marks})"
                    )

    return len(issues) == 0, issues


def normalize_paper_question_numbers(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure question numbers always start at Q1 for the first unit and run continuously."""
    if not questions:
        return []
    units_order = []
    for q in questions:
        u = q.get("unit")
        if u not in units_order and u is not None:
            units_order.append(u)

    if not units_order:
        q_map: dict[Any, int] = {}
        curr_q = 1
        for q in questions:
            orig_qno = q.get("question_no")
            if orig_qno not in q_map:
                q_map[orig_qno] = curr_q
                curr_q += 1
            new_qno = q_map[orig_qno]
            q["question_no"] = new_qno
            sub = q.get("sub_index")
            if sub:
                letter = chr(96 + int(sub))
                q["id"] = f"{new_qno}({letter})"
            else:
                q["id"] = f"{new_qno}"
        return questions

    curr_q_no = 1
    for u in units_order:
        unit_qs = [q for q in questions if q.get("unit") == u]
        orig_q_nos = []
        for q in unit_qs:
            oq = q.get("question_no")
            if oq not in orig_q_nos:
                orig_q_nos.append(oq)

        for oq in orig_q_nos:
            matching_qs = [q for q in unit_qs if q.get("question_no") == oq]
            has_subs = len(matching_qs) > 1 or any(q.get("sub_index") is not None for q in matching_qs)
            for s_idx, q in enumerate(matching_qs, 1):
                q["question_no"] = curr_q_no
                sub = q.get("sub_index") or (s_idx if has_subs else None)
                if sub:
                    letter = chr(96 + int(sub))
                    q["id"] = f"{curr_q_no}({letter})"
                else:
                    q["id"] = f"{curr_q_no}"
            curr_q_no += 1

    return questions


def diagnose_slot_failure(
    slot: dict[str, Any],
    candidate_chunks: list[dict[str, Any]],
    slot_rejected: list[dict[str, Any]],
    unit_sources: Optional[dict[Any, Any]] = None,
) -> tuple[str, str, str]:
    """Diagnose why a question slot could not be filled and provide actionable hint and reason code."""
    unit = slot.get("unit")
    co = slot.get("co", "CO")
    bt = slot.get("level") or slot.get("bloom", "L2")
    kd = slot.get("knowledge_dimension") or slot.get("kd", "Conceptual")

    # 1. No chunks at all in candidate pool or selected chapters
    if not candidate_chunks:
        return (
            f"no text chunks found for Unit {unit} in selected chapters",
            "Select more chapters in the Textbook Mapping section or expand chapter filter.",
            ReasonCode.NO_CHUNKS.value,
        )

    # 2. Chunks exist, but zero relevance for the required CO
    co_scores = [c.get("co_scores", {}).get(co, 0.0) for c in candidate_chunks]
    max_score = max(co_scores) if co_scores else 0.0
    if max_score <= 0.05:
        return (
            f"no content chunks found matching {co} in the selected chapters for Unit {unit}",
            f"Select chapters covering {co} or adjust Course Outcome weightages.",
            ReasonCode.NO_CHUNKS.value,
        )

    # 3. LLM generation attempts were made and rejected
    if slot_rejected:
        reasons = [str(r.get("reason", "")) for r in slot_rejected if r.get("reason")]
        codes = [str(r.get("reason_code", "")) for r in slot_rejected if r.get("reason_code")]

        # Timeout
        if any("timeout" in r.lower() or c == ReasonCode.OLLAMA_TIMEOUT.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "Ollama request timed out after multiple attempts",
                "Lower worker count to 1-2 in settings or increase timeout.",
                ReasonCode.OLLAMA_TIMEOUT.value,
            )
        # Connection
        if any("connection" in r.lower() or c == ReasonCode.OLLAMA_CONN.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "Cannot connect to Ollama service",
                "Ensure Ollama is running ('ollama serve') on configured host.",
                ReasonCode.OLLAMA_CONN.value,
            )
        # Model not loaded
        if any("not found" in r.lower() or "pull" in r.lower() or c == ReasonCode.MODEL_NOT_LOADED.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "Model is not loaded or available in Ollama",
                "Run 'ollama pull <model>' to download the model.",
                ReasonCode.MODEL_NOT_LOADED.value,
            )
        # Grounding check failure
        if any("grounding" in r.lower() or c == ReasonCode.GROUNDING_FAIL.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "all attempts failed factual grounding check against textbook text",
                f"Select chapters with more detailed explanations for Unit {unit} or adjust target CO.",
                ReasonCode.GROUNDING_FAIL.value,
            )
        # Bloom's / KD verb validation failure
        if any("verb" in r.lower() or "bloom" in r.lower() or "kd" in r.lower() or c == ReasonCode.BLOOM_VERB_FAIL.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                f"generated questions failed {bt} / {kd} verb validation",
                f"Ensure source chapters contain material suitable for {bt} ({kd}) questions.",
                ReasonCode.BLOOM_VERB_FAIL.value,
            )
        # Duplicate question failure
        if any("duplicate" in r.lower() or "similarity" in r.lower() or c == ReasonCode.DUPLICATE.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "generated questions duplicated other questions in the paper",
                f"Map additional chapters to Unit {unit} to provide greater topical diversity.",
                ReasonCode.DUPLICATE.value,
            )
        # Empty response
        if any("empty" in r.lower() or c == ReasonCode.EMPTY_RESPONSE.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "LLM returned an empty response",
                "Check model temperature and prompt parameters.",
                ReasonCode.EMPTY_RESPONSE.value,
            )
        # JSON parse
        if any("json" in r.lower() or c == ReasonCode.JSON_PARSE.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "model failed to output structured JSON",
                "Retry regeneration with fresh seed.",
                ReasonCode.JSON_PARSE.value,
            )
        # Stem lint
        if any("lint" in r.lower() or "premise" in r.lower() or "deictic" in r.lower() or "banned" in r.lower() or c == ReasonCode.STEM_LINT.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "generated question failed stem quality linting",
                "Select a different source chunk or revise prompt.",
                ReasonCode.STEM_LINT.value,
            )
        # Answer rubric
        if any("rubric" in r.lower() or c == ReasonCode.ANSWER_RUBRIC.value for r, c in zip(reasons, codes + [""] * len(reasons))):
            return (
                "model answer failed rubric criteria",
                "Ensure points sum to slot marks and required criteria are present.",
                ReasonCode.ANSWER_RUBRIC.value,
            )
        # Exception
        if any("exception" in r.lower() for r in reasons):
            last_ex = [r for r in reasons if "exception" in r.lower()][-1]
            return (
                f"generation error: {last_ex}",
                "Ensure Ollama service is responsive or switch to Mock Client.",
                ReasonCode.UNKNOWN_EXCEPTION.value,
            )
        last_reason = reasons[-1] if reasons else "generation retries exhausted"
        last_code = codes[-1] if codes else ReasonCode.UNKNOWN_EXCEPTION.value
        return (
            f"generation failed validation: {last_reason}",
            f"Review Unit {unit} source chapters and outcome targets.",
            last_code,
        )

    return (
        f"no suitable content chunks found matching slot constraints for Unit {unit}",
        "Select more chapters in the Textbook Mapping section.",
        ReasonCode.NO_CHUNKS.value,
    )


def generate_paper_by_slots(
    chunks: list[dict[str, Any]],
    slots: list[dict[str, Any]],
    client: Any,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
    unit_sources: Optional[dict[Any, Any]] = None,
    workers: Optional[int] = None,
    force_regenerate: bool = False,
    keep_placeholders: bool = False,
    outcome_descs: Optional[dict[str, str]] = None,
    pool_size: int = 10,
    existing_questions: Optional[list[str]] = None,
    excluded_chunk_ids: Optional[set[str]] = None,
    exclude_deprecated: bool = True,
) -> dict[str, Any]:
    """Generate questions and model answers matching blueprint slots.

    Executes concurrently using a ThreadPoolExecutor while preserving exact original slot order.

    Args:
        chunks: List of text chunk dicts.
        slots: Blueprint slot dictionaries.
        client: LLM client (OllamaClient or MockClient).
        seed: Random seed for repeatability.
        on_progress: Optional callback on question accepted.
        unit_sources: Optional mapping {unit: {book_id: [chapters] or None}}.
        workers: Optional max workers count (defaults to config.GENERATION_WORKERS).
        force_regenerate: If True, bypasses generation cache.
        keep_placeholders: If True, keeps failed slots as [NEEDS REVIEW] placeholders in questions.
        outcome_descs: Optional mapping of CO code -> CO description for similarity validation.
    """
    usable = [
        c for c in chunks
        if c.get("word_count", len(c.get("text", "").split())) >= MIN_CONTEXT_WORDS
        and is_content_chunk(c)
    ]
    if exclude_deprecated and usable:
        clean_usable = [c for c in usable if not check_currency_watchlist(c.get("text", ""))]
        if len(clean_usable) >= len(slots):
            usable = clean_usable
    if not slots:
        return {"questions": [], "rejected": [], "unfilled": []}
    if not usable:
        if keep_placeholders:
            placeholder_questions = []
            unfilled_slots = []
            for idx, slot in enumerate(slots):
                slot_kd = normalize_knowledge_dimension(slot.get("knowledge_dimension") or slot.get("kd"))
                slot_id = slot.get("id") or slot.get("label") or f"Q{idx + 1:03d}"
                reason_code = ReasonCode.NO_CHUNKS.value
                specific_reason = f"no usable text chunks found for Unit {slot.get('unit')}"
                hint = "Select more chapters in the Textbook Mapping section or expand chapter filter."
                placeholder_q = {
                    "question": f"[NEEDS REVIEW: {specific_reason}]",
                    "answer": f"[NEEDS REVIEW: {hint}]",
                    "valid": False,
                    "needs_review": True,
                    "review_reason": f"needs review [{reason_code}]: {specific_reason}",
                    "reason_code": reason_code,
                    "reason": specific_reason,
                    "hint": hint,
                    "review_flags": [f"generation_failed: {reason_code}"],
                    "marks": slot.get("marks", 0),
                    "unit": slot.get("unit"),
                    "co": slot.get("co", "CO1"),
                    "bloom": slot.get("level") or slot.get("bloom", "L2"),
                    "level": slot.get("level") or slot.get("bloom", "L2"),
                    "bt_level": slot.get("bt_level") or slot.get("level") or "L2",
                    "knowledge_dimension": slot_kd,
                    "kd": slot_kd,
                    "id": slot_id,
                    "question_no": slot.get("question_no"),
                    "sub_index": slot.get("sub_index"),
                    "alternative_group": slot.get("alternative_group", ""),
                    "book_id": "",
                    "book_title": "",
                    "display_label": "",
                    "chapter": "",
                    "page": "",
                }
                placeholder_questions.append(placeholder_q)
                slot_copy = dict(slot)
                slot_copy["reason_code"] = reason_code
                slot_copy["reason"] = specific_reason
                slot_copy["hint"] = hint
                unfilled_slots.append(slot_copy)
            return {"questions": placeholder_questions, "rejected": [], "unfilled": unfilled_slots}
        return {"questions": [], "rejected": [], "unfilled": list(slots)}

    if workers is not None:
        num_workers = workers
    elif getattr(client, "model", "") == "mock":
        num_workers = max(1, min(GENERATION_WORKERS, len(slots)))
    else:
        # Default Ollama workers 1-2 to prevent VRAM exhaustion / timeouts on 4 GB GPU
        num_workers = min(2, GENERATION_WORKERS)
        if getattr(client, "consecutive_timeouts", 0) >= 2:
            num_workers = 1
        num_workers = max(1, min(num_workers, len(slots)))

    seen_lock = threading.Lock()
    group_lock = threading.Lock()
    progress_lock = threading.Lock()

    used_chunks_per_group: dict[str, set[str]] = {}
    used_chunks_per_unit: dict[Any, set[str]] = {}
    used_pages_per_unit_group: dict[tuple[Any, str], set[int]] = {}
    used_source_pages: set[int] = set()
    bp_warnings: dict[Any, str] = {}
    seen_questions_in_unit: dict[Any, list[str]] = {}
    seen_questions: list[str] = list(existing_questions or [])
    completed_questions_by_slot: dict[int, str] = {}

    def _process_single_slot(slot_idx: int, slot: dict[str, Any]) -> dict[str, Any]:
        slot_rng = random.Random(seed + slot_idx * 1000 if seed is not None else None)
        slot_unit = slot.get("unit")
        alt_group = slot.get("alternative_group")
        slot_kd = slot.get("knowledge_dimension") or slot.get("kd") or "Conceptual"
        slot_id = slot.get("id") or slot.get("label") or f"Q{slot_idx + 1:03d}"

        candidate_chunks = select_chunks_for_unit(usable, unit_sources, slot_unit)
        if not candidate_chunks:
            candidate_chunks = usable
        if excluded_chunk_ids:
            unexcluded = [c for c in candidate_chunks if c.get("chunk_id") not in excluded_chunk_ids]
            if unexcluded:
                candidate_chunks = unexcluded
        if not candidate_chunks:
            specific_reason, hint, reason_code = diagnose_slot_failure(slot, [], [], unit_sources)
            placeholder_q = {
                "question": f"[NEEDS REVIEW: {specific_reason}]",
                "answer": f"[NEEDS REVIEW: {hint}]",
                "valid": False,
                "needs_review": True,
                "review_reason": f"needs review [{reason_code}]: {specific_reason}",
                "reason_code": reason_code,
                "reason": specific_reason,
                "hint": hint,
                "review_flags": [f"generation_failed: {reason_code}"],
                "marks": slot.get("marks", 0),
                "unit": slot.get("unit"),
                "co": slot.get("co", "CO1"),
                "bloom": slot.get("level") or slot.get("bloom", "L2"),
                "level": slot.get("level") or slot.get("bloom", "L2"),
                "bt_level": slot.get("bt_level") or slot.get("level") or "L2",
                "knowledge_dimension": slot_kd,
                "kd": slot_kd,
                "id": slot_id,
                "question_no": slot.get("question_no"),
                "sub_index": slot.get("sub_index"),
                "alternative_group": alt_group,
                "book_id": "",
                "book_title": "",
                "display_label": "",
                "chapter": "",
                "page": "",
            }
            slot_copy = dict(slot)
            slot_copy["reason_code"] = reason_code
            slot_copy["reason"] = specific_reason
            slot_copy["hint"] = hint
            log_slot_attempt(slot_id, 0, "failed", reason_code=reason_code)
            return {"status": "placeholder" if keep_placeholders else "unfilled", "question": placeholder_q, "slot": slot_copy, "rejected": [], "slot_idx": slot_idx}

        # Deterministic slot chunk partitioning (Problem 2.2)
        # Disjoint chunk start for OR alternatives (Option A vs Option B) and sub-questions
        sub_i = slot.get("sub_index", 1) or 1
        q_no = slot.get("question_no", 1) or 1
        slot_offset = ((q_no - 1) * 3 + (sub_i - 1)) if (q_no and sub_i) else slot_idx

        slot_co = slot.get("co")
        has_co_scoring = any(bool(c.get("co_scores")) for c in candidate_chunks)
        if slot_co and has_co_scoring:
            matching_chunks = [c for c in candidate_chunks if c.get("co_scores", {}).get(slot_co, 0.0) > 0.05]
            if not matching_chunks:
                specific_reason, hint, reason_code = diagnose_slot_failure(slot, candidate_chunks, [], unit_sources)
                placeholder_q = {
                    "question": f"[NEEDS REVIEW: {specific_reason}]",
                    "answer": f"[NEEDS REVIEW: {hint}]",
                    "valid": False,
                    "needs_review": True,
                    "review_reason": f"needs review [{reason_code}]: {specific_reason}",
                    "reason_code": reason_code,
                    "reason": specific_reason,
                    "hint": hint,
                    "review_flags": [f"generation_failed: {reason_code}"],
                    "marks": slot.get("marks", 0),
                    "unit": slot.get("unit"),
                    "co": slot.get("co", "CO1"),
                    "bloom": slot.get("level") or slot.get("bloom", "L2"),
                    "level": slot.get("level") or slot.get("bloom", "L2"),
                    "bt_level": slot.get("bt_level") or slot.get("level") or "L2",
                    "knowledge_dimension": slot_kd,
                    "kd": slot_kd,
                    "id": slot_id,
                    "question_no": slot.get("question_no"),
                    "sub_index": slot.get("sub_index"),
                    "alternative_group": alt_group,
                    "book_id": "",
                    "book_title": "",
                    "display_label": "",
                    "chapter": "",
                    "page": "",
                }
                slot_copy = dict(slot)
                slot_copy["reason_code"] = reason_code
                slot_copy["reason"] = specific_reason
                slot_copy["hint"] = hint
                log_slot_attempt(slot_id, 0, "failed", reason_code=reason_code)
                return {"status": "placeholder" if keep_placeholders else "unfilled", "question": placeholder_q, "slot": slot_copy, "rejected": [], "slot_idx": slot_idx}
            pool = sorted(
                matching_chunks,
                key=lambda c: c.get("co_scores", {}).get(slot_co, float("-inf")),
                reverse=True,
            )[:pool_size]
        elif slot_co:
            pool = sorted(
                candidate_chunks,
                key=lambda c: c.get("co_scores", {}).get(slot_co, float("-inf")),
                reverse=True,
            )[:pool_size]
        else:
            pool = candidate_chunks[:pool_size]

        if pool:
            # Shift pool deterministically so alternatives and sub-questions use disjoint chunks
            shift = slot_offset % len(pool)
            pool = pool[shift:] + pool[:shift]

        # Prioritize disjoint chapters, sections, and pages between OR counterpart alternatives (Problem 1 & PART G)
        # Deterministically partition chapters, sections, and pages between Option A and Option B
        unit_slots = [s for s in slots if s.get("unit") == slot_unit]
        is_choice_unit = len({s.get("question_no") for s in unit_slots}) >= 2
        unit_all_pages = sorted(list({c.get("page_start") for c in candidate_chunks if c.get("page_start") is not None}))
        unit_all_chaps = sorted(list({c.get("chapter") for c in candidate_chunks if c.get("chapter")}))
        unit_all_secs = sorted(list({c.get("section") for c in candidate_chunks if c.get("section")}))

        if is_choice_unit:
            unit_q_nums = sorted(list({s.get("question_no", 1) for s in unit_slots}))
            is_option_b = (len(unit_q_nums) >= 2 and q_no == unit_q_nums[1]) or (alt_group == "B")

            # 1. Partition across distinct chapters if multiple chapters are mapped
            if len(unit_all_chaps) >= 2:
                assigned_chaps = set(unit_all_chaps[1::2] if is_option_b else unit_all_chaps[0::2])
                chap_pool = [c for c in pool if c.get("chapter") in assigned_chaps]
                if chap_pool:
                    pool = chap_pool

            # 2. Partition across distinct sections if single chapter with multiple sections
            elif len(unit_all_secs) >= 2:
                assigned_secs = set(unit_all_secs[1::2] if is_option_b else unit_all_secs[0::2])
                sec_pool = [c for c in pool if c.get("section") in assigned_secs]
                if sec_pool:
                    pool = sec_pool

            # 3. Deterministically partition pages
            if unit_all_pages:
                if len(unit_all_pages) < 2:
                    with group_lock:
                        if slot_unit not in bp_warnings:
                            bp_warnings[slot_unit] = f"Unit {slot_unit} mapped chapters have too little distinct content ({len(unit_all_pages)} page(s)) for disjoint OR alternatives."
                else:
                    assigned_pages = set(unit_all_pages[1::2] if is_option_b else unit_all_pages[0::2])
                    disjoint_pool = [c for c in pool if c.get("page_start") in assigned_pages]
                    if disjoint_pool:
                        pool = disjoint_pool
                    else:
                        disjoint_cands = [c for c in candidate_chunks if c.get("page_start") in assigned_pages]
                        if disjoint_cands:
                            pool = disjoint_cands[:pool_size]

        # Prioritize unused pages across the entire paper (PART E)
        with group_lock:
            unused_page_pool = [c for c in pool if c.get("page_start") not in used_source_pages]
            if unused_page_pool:
                pool = unused_page_pool

        slot_rng.shuffle(pool)
        slot_kd = slot.get("knowledge_dimension") or slot.get("kd") or "Conceptual"
        slot_rejected = []

        # Bounded retries: max 2 retries per failing slot (up to 3 attempts total)
        max_slot_tries = min(max(len(pool), 1) * 2, 3)
        for attempt_i in range(max_slot_tries):
            chunk = pool[attempt_i % len(pool)]
            if not is_content_chunk(chunk):
                continue
            try:
                rec = generate_question(
                    chunk,
                    slot["level"],
                    client,
                    slot_rng,
                    marks=slot.get("marks"),
                    knowledge_dimension=slot_kd,
                    used_topics=None,
                    force_regenerate=force_regenerate,
                    exclude_deprecated=exclude_deprecated,
                )
            except Exception as e:
                code_i, _, _ = classify_exception_to_reason_code(e)
                c_val = code_i.value if hasattr(code_i, "value") else str(code_i)
                rec = {"valid": False, "reason": f"generation exception: {e}", "reason_code": c_val}

            with seen_lock:
                is_dup = is_duplicate(rec.get("question", ""), seen_questions) if rec.get("valid") else False

            if not rec.get("valid") or is_dup:
                slot_rejected.append(rec)
                continue

            try:
                ans_rec = generate_answer(
                    rec["question"],
                    chunk,
                    slot.get("marks"),
                    client,
                    rng=slot_rng,
                    force_regenerate=force_regenerate,
                    level=slot.get("level"),
                )
            except Exception as e:
                code_i, _, _ = classify_exception_to_reason_code(e)
                c_val = code_i.value if hasattr(code_i, "value") else str(code_i)
                ans_rec = {"valid": False, "reason": f"answer exception: {e}", "reason_code": c_val}

            if not ans_rec.get("valid"):
                rec["valid"] = False
                rec["reason"] = f"answer generation failed: {ans_rec.get('reason')}"
                slot_rejected.append(rec)
                continue

            rec["answer"] = ans_rec["answer"]
            rec["answer_alternatives"] = ans_rec["answer_alternatives"]
            rec["co"] = slot.get("co", "CO1")
            rec["id"] = slot.get("id") or slot.get("label") or f"Q{slot_idx + 1:03d}"
            rec["unit"] = slot.get("unit")
            rec["question_no"] = slot.get("question_no")
            rec["sub_index"] = slot.get("sub_index")
            rec["marks"] = slot.get("marks")
            rec["alternative_group"] = alt_group
            rec["book_id"] = chunk.get("book_id")
            rec["book_title"] = chunk.get("display_label") or chunk.get("book_title")
            rec["display_label"] = chunk.get("display_label")
            rec["chapter"] = chunk.get("chapter")
            rec["page"] = chunk.get("page_start")
            rec["context_text"] = chunk.get("text", "")
            rec["knowledge_dimension"] = slot_kd
            rec["kd"] = slot_kd
            rec["review_flags"] = []
            rec["needs_review"] = False

            if ans_rec.get("ungrounded_sentences"):
                rec["ungrounded_sentences"] = ans_rec["ungrounded_sentences"]
                rec["needs_review"] = True
                rec["review_flags"].append(f"ungrounded sentences in model answer ({len(ans_rec['ungrounded_sentences'])})")

            cid = chunk.get("chunk_id")
            if cid:
                with group_lock:
                    if alt_group:
                        used_chunks_per_group.setdefault(alt_group, set()).add(cid)
                    if slot_unit is not None:
                        used_chunks_per_unit.setdefault(slot_unit, set()).add(cid)
                    if alt_group and chunk.get("page_start"):
                        used_pages_per_unit_group.setdefault((slot_unit, alt_group), set()).add(chunk["page_start"])
                    if chunk.get("page_start"):
                        used_source_pages.add(chunk["page_start"])

            with seen_lock:
                seen_questions.append(rec["question"])
                if slot_unit is not None:
                    seen_questions_in_unit.setdefault(slot_unit, []).append(rec["question"])

            return {"status": "ok", "question": rec, "rejected": slot_rejected, "slot_idx": slot_idx}

        # Try a fresh chunk set outside candidate pool once more before falling back
        outside_pool = [
            c for c in candidate_chunks
            if c not in pool and is_content_chunk(c) and (not is_choice_unit or not unit_all_pages or len(unit_all_pages) < 2 or c.get("page_start") in assigned_pages)
        ]
        if outside_pool:
            fresh_chunk = slot_rng.choice(outside_pool)
            try:
                rec = generate_question(
                    fresh_chunk,
                    slot["level"],
                    client,
                    slot_rng,
                    marks=slot.get("marks"),
                    knowledge_dimension=slot_kd,
                    used_topics=None,
                    force_regenerate=force_regenerate,
                )
                if rec.get("valid"):
                    ans_rec = generate_answer(
                        rec["question"],
                        fresh_chunk,
                        slot.get("marks"),
                        client,
                        rng=slot_rng,
                        force_regenerate=force_regenerate,
                        level=slot.get("level"),
                    )
                    if ans_rec.get("valid"):
                        rec["answer"] = ans_rec["answer"]
                        rec["answer_alternatives"] = ans_rec["answer_alternatives"]
                        rec["co"] = slot.get("co", "CO1")
                        rec["id"] = slot.get("id") or slot.get("label") or f"Q{slot_idx + 1:03d}"
                        rec["unit"] = slot.get("unit")
                        rec["question_no"] = slot.get("question_no")
                        rec["sub_index"] = slot.get("sub_index")
                        rec["marks"] = slot.get("marks")
                        rec["alternative_group"] = alt_group
                        rec["book_id"] = fresh_chunk.get("book_id")
                        rec["book_title"] = fresh_chunk.get("display_label") or fresh_chunk.get("book_title")
                        rec["display_label"] = fresh_chunk.get("display_label")
                        rec["chapter"] = fresh_chunk.get("chapter")
                        rec["page"] = fresh_chunk.get("page_start")
                        rec["context_text"] = fresh_chunk.get("text", "")
                        rec["knowledge_dimension"] = slot_kd
                        rec["kd"] = slot_kd
                        rec["review_flags"] = []
                        rec["needs_review"] = False
                        with group_lock:
                            if alt_group and fresh_chunk.get("page_start"):
                                used_pages_per_unit_group.setdefault((slot_unit, alt_group), set()).add(fresh_chunk["page_start"])
                            if fresh_chunk.get("page_start"):
                                used_source_pages.add(fresh_chunk["page_start"])
                        return {"status": "ok", "question": rec, "rejected": slot_rejected, "slot_idx": slot_idx}
            except Exception as ex:
                code_i, _, _ = classify_exception_to_reason_code(ex)
                c_val = code_i.value if hasattr(code_i, "value") else str(code_i)
                slot_rejected.append({"valid": False, "reason": f"fresh chunk exception: {ex}", "reason_code": c_val})

        # All retries failed: build placeholder (Problem 1)
        specific_reason, hint, reason_code = diagnose_slot_failure(slot, candidate_chunks, slot_rejected, unit_sources)
        placeholder_q = {
            "question": f"[NEEDS REVIEW: {specific_reason}]",
            "answer": f"[NEEDS REVIEW: Model answer generation failed - {specific_reason}]",
            "valid": False,
            "needs_review": True,
            "review_reason": f"needs review [{reason_code}]: {specific_reason}",
            "reason_code": reason_code,
            "reason": specific_reason,
            "hint": hint,
            "review_flags": [f"generation_failed: {reason_code}"],
            "marks": slot.get("marks", 0),
            "unit": slot.get("unit"),
            "co": slot.get("co", "CO1"),
            "bloom": slot.get("level") or slot.get("bloom", "L2"),
            "level": slot.get("level") or slot.get("bloom", "L2"),
            "bt_level": slot.get("bt_level") or slot.get("level") or "L2",
            "knowledge_dimension": slot_kd,
            "kd": slot_kd,
            "id": slot.get("id") or slot.get("label") or f"Q{slot_idx + 1:03d}",
            "question_no": slot.get("question_no"),
            "sub_index": slot.get("sub_index"),
            "alternative_group": alt_group,
            "book_id": "",
            "book_title": "",
            "display_label": "",
            "chapter": "",
            "page": "",
        }
        slot_copy = dict(slot)
        slot_copy["reason_code"] = reason_code
        slot_copy["reason"] = specific_reason
        slot_copy["hint"] = hint
        log_slot_attempt(slot_id, 0, "failed", reason_code=reason_code)
        return {"status": "placeholder", "question": placeholder_q, "slot": slot_copy, "rejected": slot_rejected, "slot_idx": slot_idx}

    results_by_index: list[Optional[dict[str, Any]]] = [None] * len(slots)

    if num_workers <= 1 or len(slots) <= 1:
        for idx, slot in enumerate(slots):
            res = _process_single_slot(idx, slot)
            results_by_index[idx] = res
            if on_progress:
                try:
                    on_progress(res.get("question") or slot)
                except Exception:
                    pass
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(_process_single_slot, idx, slot): idx
                for idx, slot in enumerate(slots)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    results_by_index[idx] = res
                    if on_progress:
                        try:
                            on_progress(res.get("question") or slots[idx])
                        except Exception:
                            pass
                except Exception as ex:
                    reason_code, specific_reason, hint = classify_exception_to_reason_code(ex)
                    slot_i = slots[idx]
                    slot_id = slot_i.get("id") or slot_i.get("label") or f"Q{idx + 1:03d}"
                    log_slot_attempt(slot_id, 0, "failed", reason_code=reason_code, error=str(ex))
                    slot_kd = normalize_knowledge_dimension(
                        slot_i.get("knowledge_dimension") or slot_i.get("kd")
                    )
                    placeholder_q = {
                        "question": f"[NEEDS REVIEW: {specific_reason}]",
                        "answer": f"[NEEDS REVIEW: {hint}]",
                        "valid": False,
                        "needs_review": True,
                        "review_reason": f"needs review [{reason_code}]: {specific_reason}",
                        "reason_code": reason_code,
                        "reason": specific_reason,
                        "hint": hint,
                        "review_flags": [f"generation_failed: {reason_code}"],
                        "marks": slot_i.get("marks", 0),
                        "unit": slot_i.get("unit"),
                        "co": slot_i.get("co", "CO1"),
                        "bloom": slot_i.get("level") or slot_i.get("bloom", "L2"),
                        "level": slot_i.get("level") or slot_i.get("bloom", "L2"),
                        "bt_level": slot_i.get("bt_level") or slot_i.get("level") or "L2",
                        "knowledge_dimension": slot_kd,
                        "kd": slot_kd,
                        "id": slot_id,
                        "question_no": slot_i.get("question_no"),
                        "sub_index": slot_i.get("sub_index"),
                        "alternative_group": slot_i.get("alternative_group", ""),
                        "book_id": "",
                        "book_title": "",
                        "display_label": "",
                        "chapter": "",
                        "page": "",
                    }
                    slot_copy = dict(slot_i)
                    slot_copy["reason_code"] = reason_code
                    slot_copy["reason"] = specific_reason
                    slot_copy["hint"] = hint
                    results_by_index[idx] = {
                        "status": "placeholder" if keep_placeholders else "unfilled",
                        "question": placeholder_q,
                        "slot": slot_copy,
                        "rejected": [{"valid": False, "reason": str(ex), "reason_code": reason_code}],
                        "slot_idx": idx,
                    }
                    if on_progress:
                        try:
                            on_progress(placeholder_q if keep_placeholders else slots[idx])
                        except Exception:
                            pass

    questions = []
    rejected = []
    unfilled = []
    for item in results_by_index:
        if not item:
            continue
        if item.get("rejected"):
            rejected.extend(item["rejected"])
        if item["status"] == "ok":
            questions.append(item["question"])
        elif item["status"] == "placeholder":
            if keep_placeholders:
                questions.append(item["question"])
            unfilled.append(item.get("slot", item["question"]))
        else:
            unfilled.append(item["slot"])

    # Post-generation embedding checks
    if questions:
        from retrieval.co_mapper import _m
        embedder = _m()

        # 1. Duplicate check (Problem 1 & Problem 2.3)
        if len(questions) >= 2 and embedder is not None:
            dups = check_question_duplicates(questions, embedder)
            for orig_i, orig_j, sim in dups:
                q_i = questions[orig_i]
                q_j = questions[orig_j]
                is_same_unit = (q_i.get("unit") is not None and q_i.get("unit") == q_j.get("unit"))
                is_or_counterpart = is_same_unit and (q_i.get("alternative_group") != q_j.get("alternative_group"))
                flag_name = "OR counterpart similarity" if is_or_counterpart else "duplicate question"
                q_j["needs_review"] = True
                q_j.setdefault("review_flags", []).append(
                    f"{flag_name}: cosine similarity {sim:.2f} >= {QUESTION_SIMILARITY_DUPLICATE_THRESHOLD} with {q_i.get('id', 'earlier slot')}"
                )

        # 2. Batched grounding check after first pass (Problem 2.2)
        candidate_grounding = [
            q for q in questions
            if q.get("valid", True) and not q.get("question", "").startswith("[NEEDS REVIEW")
            and q.get("answer") and not q.get("answer", "").startswith("[NEEDS REVIEW")
        ]
        if candidate_grounding and embedder is not None:
            try:
                import numpy as np
                chunk_map = {c.get("chunk_id"): c.get("text", "") for c in usable if c.get("chunk_id")}
                q_sentence_items = []
                for q in candidate_grounding:
                    ans = q.get("answer", "")
                    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ans) if s.strip() and len(s.split()) >= 4]
                    if sents:
                        ctx = q.get("context_text") or (chunk_map.get(q.get("chunk_id")) if q.get("chunk_id") else "")
                        if not ctx and q.get("page"):
                            matching_c = [c for c in usable if c.get("page_start") == q.get("page")]
                            if matching_c:
                                ctx = matching_c[0].get("text", "")
                        if ctx:
                            q_sentence_items.append((q, ctx, sents))

                if q_sentence_items:
                    all_ctx_texts = [item[1] for item in q_sentence_items]
                    all_sent_texts = [s for item in q_sentence_items for s in item[2]]
                    encoded_all = embedder.encode(all_ctx_texts + all_sent_texts, normalize_embeddings=True)
                    ctx_embs = encoded_all[:len(all_ctx_texts)]
                    sent_embs = encoded_all[len(all_ctx_texts):]

                    sent_idx = 0
                    for c_idx, (q, _, sents) in enumerate(q_sentence_items):
                        c_vec = ctx_embs[c_idx]
                        unsupported = []
                        for s in sents:
                            s_vec = sent_embs[sent_idx]
                            sim_val = float(np.dot(s_vec, c_vec))
                            if sim_val < ANSWER_GROUNDING_SENTENCE_SIMILARITY:
                                unsupported.append(s)
                            sent_idx += 1
                        if unsupported:
                            q["ungrounded_sentences"] = unsupported
                            q["needs_review"] = True
                            q.setdefault("review_flags", []).append(
                                f"ungrounded sentences in model answer ({len(unsupported)})"
                            )
            except Exception as ex:
                code_i, _, _ = classify_exception_to_reason_code(ex)
                c_val = code_i.value if hasattr(code_i, "value") else str(code_i)
                slot_rejected.append({"valid": False, "reason": f"fresh chunk exception: {ex}", "reason_code": c_val})

        # 3. Cross-slot consistency contradiction check (Problem 2.3)
        contradictions = check_cross_slot_contradictions(questions, embedder)
        for orig_i, orig_j, c_msg in contradictions:
            if orig_j < len(questions):
                questions[orig_j]["needs_review"] = True
                questions[orig_j].setdefault("review_flags", []).append(c_msg)
            if orig_i < len(questions):
                questions[orig_i]["needs_review"] = True
                questions[orig_i].setdefault("review_flags", []).append(c_msg)

        # 4. CO similarity and keyword anchor check (PART F)
        if outcome_descs:
            co_mismatches = verify_question_co_alignment(questions, outcome_descs, embedder)
            for q_idx, m_msg in co_mismatches:
                if q_idx < len(questions):
                    questions[q_idx]["needs_review"] = True
                    questions[q_idx].setdefault("review_flags", []).append(m_msg)
                    questions[q_idx].setdefault("qa_flags", []).append("CO_MISMATCH")

        # 5. Consistency across paper (Problem 5.6)
        inconsistencies = check_paper_identifier_consistency(questions)
        for q_idx, inc_msg in inconsistencies:
            if q_idx < len(questions):
                questions[q_idx].setdefault("review_flags", []).append(inc_msg)

        # 6. OR twin answer depth parity check (PART D)
        depth_disparities = check_or_twin_answer_depth_parity(questions)
        for orig_i, orig_j, d_msg in depth_disparities:
            if orig_i < len(questions):
                questions[orig_i]["needs_review"] = True
                questions[orig_i].setdefault("review_flags", []).append(d_msg)
            if orig_j < len(questions):
                questions[orig_j]["needs_review"] = True
                questions[orig_j].setdefault("review_flags", []).append(d_msg)

        # 7. Hallucination guard: command/flag/tool token extraction check (PART E)
        chunk_map = {c.get("chunk_id"): c.get("text", "") for c in usable if c.get("chunk_id")}
        for q in questions:
            ctx_text = q.get("context_text") or (chunk_map.get(q.get("chunk_id")) if q.get("chunk_id") else "")
            if not ctx_text and q.get("page"):
                matching_c = [c for c in usable if c.get("page_start") == q.get("page")]
                if matching_c:
                    ctx_text = matching_c[0].get("text", "")
            if ctx_text and q.get("answer"):
                ungrounded = check_hallucinated_tokens(q["answer"], ctx_text)
                if ungrounded:
                    q["ungrounded_tokens"] = ungrounded
                    q["needs_review"] = True
                    q.setdefault("review_flags", []).append(f"UNGROUNDED_TOKEN: {', '.join(ungrounded[:3])}")

        # 8. Currency watchlist check (PART E)
        for q in questions:
            combined = (q.get("question", "") + " " + q.get("answer", "")).lower()
            outdated = check_currency_watchlist(combined)
            if outdated:
                q["outdated_topics"] = outdated
                q["needs_review"] = True
                q.setdefault("review_flags", []).append(f"OUTDATED: {', '.join(outdated)}")

        # 9. Paper-wide source page dedupe check (PART E)
        seen_paper_pages: dict[int, int] = {}
        for idx, q in enumerate(questions):
            p = q.get("page")
            if p is not None:
                if p in seen_paper_pages:
                    orig_idx = seen_paper_pages[p]
                    q_orig = questions[orig_idx]
                    is_or = (q.get("unit") == q_orig.get("unit") and q.get("alternative_group") != q_orig.get("alternative_group"))
                    flag = "DUPLICATE: OR twins share source page" if is_or else "DUPLICATE: shared source page"
                    q["needs_review"] = True
                    q.setdefault("review_flags", []).append(f"{flag} {p} with {q_orig.get('id', 'earlier question')}")
                else:
                    seen_paper_pages[p] = idx

    # Normalize question numbers only if paper starts above Q1 (Problem 1.4)
    if questions and any(q.get("question_no") is not None for q in questions):
        q_nos = [q["question_no"] for q in questions if q.get("question_no") is not None]
        if q_nos and min(q_nos) > 1:
            questions = normalize_paper_question_numbers(questions)

    # 10. Chapter coverage balance check across mapped units (PART G)
    chap_warnings, coverage_by_unit = check_chapter_coverage_balance(questions, unit_sources)
    if chap_warnings and bp_warnings is not None:
        for cw in chap_warnings:
            bp_warnings[f"cov_balance_{len(bp_warnings)}"] = cw

    res_dict = {
        "questions": questions,
        "rejected": rejected,
        "unfilled": unfilled,
        "chapter_coverage": coverage_by_unit,
    }
    if bp_warnings:
        res_dict["warnings"] = list(bp_warnings.values())
    return res_dict


def regenerate_missing_slots(
    ctx: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
    result: Optional[dict[str, Any]] = None,
    seed_bump: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Regenerate ONLY failed/incomplete slots, leaving all accepted questions untouched.

    Args:
        ctx: Generation context dict containing scored_chunks, slots/blueprint, outcomes,
             unit_sources, client/client_config, seed, etc.
        result: Current generation result dict containing questions, unfilled, rejected.
        seed_bump: Multiplier for seed bump to ensure fresh random generation.
        **kwargs: Backward compatibility arguments (questions, slots, chunks, client, etc.).

    Returns:
        Updated result dict with questions, unfilled, rejected, and covered_chapters.
    """
    # 1. Handle legacy signature: (questions, slots, chunks, client, seed, unit_sources, outcomes)
    if isinstance(ctx, list) or "questions" in kwargs:
        q_list = ctx if isinstance(ctx, list) else kwargs["questions"]
        s_list = result if isinstance(result, list) else kwargs.get("slots", [])
        c_list = kwargs.get("chunks", [])
        cl = kwargs.get("client")
        sd = kwargs.get("seed", 42)
        us = kwargs.get("unit_sources")
        oc = kwargs.get("outcomes")
        ctx = {
            "scored_chunks": c_list,
            "slots": s_list,
            "blueprint": s_list,
            "unit_sources": us,
            "outcomes": oc,
            "client": cl,
            "seed": sd,
        }
        result = {
            "questions": q_list,
            "unfilled": [],
            "rejected": [],
        }

    # 2. Validate ctx and result
    if not ctx or not isinstance(ctx, dict):
        raise ValueError("Please click Generate first before regenerating. Missing or empty generation context.")

    if not result:
        result = ctx.get("result") or ctx.get("gen_result")
    if not result or not isinstance(result, dict):
        raise ValueError("Please click Generate first before regenerating. Missing or empty generation result.")

    original_slots = list(ctx.get("slots") or ctx.get("blueprint") or [])
    original_questions = list(result.get("questions") or [])
    unfilled_records = list(result.get("unfilled") or [])

    if not original_slots and original_questions:
        original_slots = [dict(q) for q in original_questions]

    total_expected = max(len(original_slots), len(original_questions))
    if total_expected == 0:
        res_copy = dict(result)
        res_copy["message"] = "All slots are already complete"
        return res_copy

    # 3. Identify incomplete slot indices
    incomplete_indices: list[int] = []
    for i in range(total_expected):
        q = original_questions[i] if i < len(original_questions) else None
        s = original_slots[i] if i < len(original_slots) else None
        if q is None:
            incomplete_indices.append(i)
            continue
        q_text = str(q.get("question", "")).strip()
        is_inc = (
            not q.get("valid", True)
            or not q_text
            or q_text.startswith("[NEEDS REVIEW")
            or (q.get("needs_review") and ("generation_failed" in str(q.get("review_flags", [])) or "generation retries exhausted" in str(q.get("review_reason", "")).lower()))
        )
        if is_inc:
            incomplete_indices.append(i)
            continue
        # Also check if s is explicitly in unfilled
        if s and unfilled_records:
            s_id = s.get("id") or s.get("label")
            for u in unfilled_records:
                u_id = u.get("id") or u.get("label")
                if s_id and u_id and s_id == u_id:
                    incomplete_indices.append(i)
                    break
                elif (
                    s.get("unit") is not None
                    and s.get("unit") == u.get("unit")
                    and s.get("question_no") is not None
                    and s.get("question_no") == u.get("question_no")
                    and s.get("sub_index") == u.get("sub_index")
                ):
                    incomplete_indices.append(i)
                    break

    # If all slots are complete, return immediately
    if not incomplete_indices:
        res_copy = dict(result)
        res_copy["message"] = "All slots are already complete"
        return res_copy

    # 4. Extract context resources
    chunks = ctx.get("scored_chunks") or ctx.get("chunks", [])
    outcomes = ctx.get("outcomes")
    unit_sources = ctx.get("unit_sources")

    client = ctx.get("client")
    if client is None:
        cfg = ctx.get("client_config", {})
        if cfg.get("use_mock", True):
            from generation.mock_client import MockClient
            client = MockClient()
        else:
            from generation.ollama_client import OllamaClient
            client = OllamaClient(model=cfg.get("model_name", "llama3.1:8b"))

    # 5. Extract accepted questions, used chunk IDs, and existing questions
    accepted_questions = [
        q for idx, q in enumerate(original_questions)
        if idx not in incomplete_indices and q and q.get("valid", True)
    ]
    used_chunk_ids = {q.get("chunk_id") for q in accepted_questions if q.get("chunk_id")}
    existing_question_texts = [
        q.get("question") for q in accepted_questions
        if q.get("question") and not str(q.get("question", "")).startswith("[NEEDS REVIEW")
    ]

    # 6. Prepare slots to regenerate
    slots_to_regen = []
    for idx in incomplete_indices:
        if idx < len(original_slots):
            slots_to_regen.append(dict(original_slots[idx]))
        else:
            slots_to_regen.append(dict(original_questions[idx]))

    # 7. Seed bump
    base_seed = ctx.get("seed", 42) or 42
    retry_count = ctx.get("retry_count", 0)
    bumped_seed = base_seed + (seed_bump * 1009) + (retry_count * 1337) + 17

    # 8. Generate paper by slots for incomplete slots only
    new_results = generate_paper_by_slots(
        chunks=chunks,
        slots=slots_to_regen,
        client=client,
        seed=bumped_seed,
        unit_sources=unit_sources,
        force_regenerate=True,
        keep_placeholders=True,
        outcome_descs=outcomes if isinstance(outcomes, dict) else None,
        pool_size=20,
        existing_questions=existing_question_texts,
        excluded_chunk_ids=used_chunk_ids,
        exclude_deprecated=ctx.get("exclude_deprecated", True),
    )

    new_questions = new_results.get("questions", [])

    # 9. Merge new questions into original positions, preserving blueprint metadata
    merged_questions: list[Any] = list(original_questions)
    if len(merged_questions) < total_expected:
        merged_questions.extend([None] * (total_expected - len(merged_questions)))

    for k, orig_idx in enumerate(incomplete_indices):
        if k < len(new_questions):
            new_q = dict(new_questions[k])
            orig_slot = original_slots[orig_idx] if orig_idx < len(original_slots) else (original_questions[orig_idx] or {})
            for field in [
                "id", "question_no", "sub_index", "marks", "co",
                "level", "bloom", "bt_level", "knowledge_dimension",
                "kd", "unit", "alternative_group"
            ]:
                if field in orig_slot and orig_slot[field] is not None:
                    new_q[field] = orig_slot[field]
            merged_questions[orig_idx] = new_q

    # 10. Compute updated unfilled and rejected
    updated_unfilled = []
    for idx, q in enumerate(merged_questions):
        if not q or not q.get("valid", True) or str(q.get("question", "")).startswith("[NEEDS REVIEW"):
            orig_slot = original_slots[idx] if idx < len(original_slots) else (q or {})
            u_rec = dict(orig_slot)
            if q:
                u_rec["reason"] = q.get("reason") or q.get("review_reason") or "Generation incomplete"
                u_rec["hint"] = q.get("hint") or "Select additional chapters or adjust CO target weights"
            updated_unfilled.append(u_rec)

    updated_rejected = list(result.get("rejected", [])) + list(new_results.get("rejected", []))
    merged_warnings = list(dict.fromkeys(result.get("warnings", []) + new_results.get("warnings", [])))

    return {
        "questions": merged_questions,
        "unfilled": updated_unfilled,
        "rejected": updated_rejected,
        "covered_chapters": result.get("covered_chapters", []),
        "warnings": merged_warnings,
    }


def generate_full_paper(
    chunks: list[dict[str, Any]],
    blueprint: Any,
    outcomes: dict[str, str],
    unit_sources: Optional[dict[Any, Any]],
    client: Any,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
    workers: Optional[int] = None,
    force_regenerate: bool = False,
    keep_placeholders: bool = True,
) -> dict[str, Any]:
    """Generate a complete exam paper with questions and model answers from blueprint and outcomes."""
    from retrieval.co_mapper import score_chunks_against_outcomes

    # 1. Score chunks against outcomes
    scored_chunks = score_chunks_against_outcomes(chunks, outcomes)

    # 2. Extract slots and report from blueprint
    if isinstance(blueprint, dict):
        slots = blueprint.get("slots", [])
        blueprint_report = blueprint.get("report", {})
    else:
        slots = list(blueprint)
        blueprint_report = {}

    # 3. Generate paper by slots
    result = generate_paper_by_slots(
        chunks=scored_chunks,
        slots=slots,
        client=client,
        seed=seed,
        on_progress=on_progress,
        unit_sources=unit_sources,
        workers=workers,
        force_regenerate=force_regenerate,
        keep_placeholders=keep_placeholders,
        outcome_descs=outcomes if isinstance(outcomes, dict) else None,
    )

    return {
        "questions": result["questions"],
        "rejected": result["rejected"],
        "unfilled": result["unfilled"],
        "blueprint_report": blueprint_report,
    }
