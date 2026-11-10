"""Revised Bloom's Taxonomy (Anderson & Krathwohl, 2001).

Implements the two-dimensional taxonomy framework:
1. The Cognitive Process Dimension:
   - L1: Remember (Retrieve relevant knowledge from long-term memory)
   - L2: Understand (Construct meaning from instructional messages)
   - L3: Apply (Carry out or use a procedure in a given situation)
   - L4: Analyze (Break material into constituent parts and determine relationships)
   - L5: Evaluate (Make judgements based on criteria and standards)
   - L6: Create (Put elements together to form a coherent whole / new pattern)

2. The Knowledge Dimension:
   - Factual: Basic elements students must know to be acquainted with a discipline or solve problems.
   - Conceptual: Interrelationships among basic elements within a larger structure.
   - Procedural: How to do something, inquiry methods, algorithms, and techniques.
   - Metacognitive: Knowledge of cognition in general and awareness of one's own cognition.

Contains the complete 4x6 matrix data with exemplar tasks from the reference framework.
"""
import re
from typing import Any, Optional

# Knowledge Dimensions
KNOWLEDGE_DIMENSIONS = ["Factual", "Conceptual", "Procedural", "Metacognitive"]

ALLOWED_KDS_BY_LEVEL: dict[str, list[str]] = {
    "L1": ["Factual", "Conceptual"],
    "L2": ["Conceptual", "Factual"],
    "L3": ["Procedural", "Conceptual"],
    "L4": ["Conceptual", "Procedural", "Metacognitive"],
    "L5": ["Conceptual", "Procedural", "Metacognitive"],
    "L6": ["Conceptual", "Procedural", "Metacognitive"],
}

KNOWLEDGE_DIMENSION_DESCRIPTIONS = {
    "Factual": (
        "The basic elements students must know to be acquainted with a discipline "
        "or solve problems in it (terminology, specific details, symbols, core elements)."
    ),
    "Conceptual": (
        "The interrelationships among the basic elements within a larger structure "
        "that enable them to function together (classifications, categories, principles, "
        "theories, models, system architectures)."
    ),
    "Procedural": (
        "How to do something, methods of inquiry, and criteria for using skills, "
        "algorithms, techniques, and step-by-step methods."
    ),
    "Metacognitive": (
        "Knowledge of cognition in general as well as awareness and knowledge "
        "of one's own cognition (strategic knowledge, self-assessment, reflection)."
    ),
}

# Cognitive Process Dimension (L1-L6)
COGNITIVE_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6"]

COGNITIVE_LEVEL_NAMES = {
    "L1": "Remember",
    "L2": "Understand",
    "L3": "Apply",
    "L4": "Analyze",
    "L5": "Evaluate",
    "L6": "Create",
}

COGNITIVE_LEVEL_DESCRIPTIONS = {
    "L1": "Retrieve relevant knowledge from long-term memory.",
    "L2": "Construct meaning from instructional messages, including oral, written, and graphic communication.",
    "L3": "Carry out or use a procedure in a given situation.",
    "L4": "Break material into constituent parts and determine how parts relate to one another and to an overall structure or purpose.",
    "L5": "Make judgements based on criteria and standards.",
    "L6": "Put elements together to form a coherent whole; reorganize into a new pattern or structure.",
}

# 4x6 Revised Taxonomy Matrix Exemplars (from the reference diagram)
TAXONOMY_MATRIX_2D: dict[tuple[str, str], dict[str, str]] = {
    # --- Row 1: Factual Knowledge ---
    ("Factual", "L1"): {
        "verb": "List",
        "task": "List primary and secondary colors.",
        "focus": "Recall terms, definitions, symbols, or specific details.",
    },
    ("Factual", "L2"): {
        "verb": "Summarize",
        "task": "Summarize features of a new product.",
        "focus": "Summarize or paraphrase factual data and properties.",
    },
    ("Factual", "L3"): {
        "verb": "Respond",
        "task": "Respond to frequently asked questions.",
        "focus": "Supply correct factual information in response to specific queries.",
    },
    ("Factual", "L4"): {
        "verb": "Select",
        "task": "Select the most complete list of activities.",
        "focus": "Distinguish and filter relevant factual items from irrelevant ones.",
    },
    ("Factual", "L5"): {
        "verb": "Check",
        "task": "Check for consistency among sources.",
        "focus": "Evaluate the factual accuracy and consistency of statements.",
    },
    ("Factual", "L6"): {
        "verb": "Generate",
        "task": "Generate a log of daily activities.",
        "focus": "Compile and assemble factual records or data summaries.",
    },

    # --- Row 2: Conceptual Knowledge ---
    ("Conceptual", "L1"): {
        "verb": "Recognize",
        "task": "Recognize symptoms of exhaustion.",
        "focus": "Identify categories, conceptual schemas, or design patterns.",
    },
    ("Conceptual", "L2"): {
        "verb": "Classify",
        "task": "Classify adhesives by toxicity.",
        "focus": "Categorize concepts, structures, or architectural models.",
    },
    ("Conceptual", "L3"): {
        "verb": "Provide",
        "task": "Provide advice to novices.",
        "focus": "Apply theories, models, or principles to practical scenarios.",
    },
    ("Conceptual", "L4"): {
        "verb": "Differentiate",
        "task": "Differentiate high and low culture.",
        "focus": "Analyze structural relationships, differences, and trade-offs.",
    },
    ("Conceptual", "L5"): {
        "verb": "Determine",
        "task": "Determine relevance of results.",
        "focus": "Assess conceptual soundness, validity, and significance.",
    },
    ("Conceptual", "L6"): {
        "verb": "Assemble",
        "task": "Assemble a team of experts.",
        "focus": "Synthesize multiple concepts into a unified structural model.",
    },

    # --- Row 3: Procedural Knowledge ---
    ("Procedural", "L1"): {
        "verb": "Recall",
        "task": "Recall how to perform CPR.",
        "focus": "Recall the steps of an algorithm, technique, or standard procedure.",
    },
    ("Procedural", "L2"): {
        "verb": "Clarify",
        "task": "Clarify assembly instructions.",
        "focus": "Explain the operational mechanics or execution steps of a process.",
    },
    ("Procedural", "L3"): {
        "verb": "Carry out",
        "task": "Carry out pH tests of water samples.",
        "focus": "Execute or implement an algorithm, protocol, or computation.",
    },
    ("Procedural", "L4"): {
        "verb": "Integrate",
        "task": "Integrate compliance with regulations.",
        "focus": "Analyze procedures to identify bottlenecks, dependencies, or flaws.",
    },
    ("Procedural", "L5"): {
        "verb": "Judge",
        "task": "Judge efficiency of sampling techniques.",
        "focus": "Critique and evaluate procedural efficiency, accuracy, or safety.",
    },
    ("Procedural", "L6"): {
        "verb": "Design",
        "task": "Design an efficient project workflow.",
        "focus": "Design an original algorithm, pipeline, workflow, or protocol.",
    },

    # --- Row 4: Metacognitive Knowledge ---
    ("Metacognitive", "L1"): {
        "verb": "Identify",
        "task": "Identify strategies for retaining information.",
        "focus": "Identify cognitive strategies, learning heuristics, or self-monitoring methods.",
    },
    ("Metacognitive", "L2"): {
        "verb": "Predict",
        "task": "Predict one's response to culture shock.",
        "focus": "Anticipate cognitive performance, biases, or comprehension difficulties.",
    },
    ("Metacognitive", "L3"): {
        "verb": "Use",
        "task": "Use techniques that match one's strengths.",
        "focus": "Select and apply appropriate problem-solving strategies given personal strengths.",
    },
    ("Metacognitive", "L4"): {
        "verb": "Deconstruct",
        "task": "Deconstruct one's biases.",
        "focus": "Critically analyze one's own assumptions, reasoning patterns, or cognitive biases.",
    },
    ("Metacognitive", "L5"): {
        "verb": "Reflect",
        "task": "Reflect on one's progress.",
        "focus": "Evaluate the effectiveness of one's reasoning, methodology, or learning outcomes.",
    },
    ("Metacognitive", "L6"): {
        "verb": "Create",
        "task": "Create an innovative learning portfolio.",
        "focus": "Formulate a personalized learning, meta-learning, or reflection framework.",
    },
}


def normalize_knowledge_dimension(kd: Optional[str]) -> str:
    """Normalize knowledge dimension input (e.g. 'factual', 'FACTUAL', 'Factual' -> 'Factual')."""
    if not kd:
        return "Conceptual"
    s = str(kd).strip().capitalize()
    for valid in KNOWLEDGE_DIMENSIONS:
        if s.lower() == valid.lower():
            return valid
    return "Conceptual"


def get_taxonomy_cell(cognitive_level: str, knowledge_dim: str) -> dict[str, str]:
    """Retrieve the matrix cell metadata for (knowledge_dimension, cognitive_level)."""
    from blooms.verb_bank import normalize_level
    lvl = normalize_level(cognitive_level)
    kd = normalize_knowledge_dimension(knowledge_dim)
    return TAXONOMY_MATRIX_2D.get(
        (kd, lvl),
        {
            "verb": "Analyze",
            "task": f"Examine {kd} aspects at {lvl} cognitive level.",
            "focus": f"Apply {lvl} processing to {kd} knowledge.",
        },
    )


def compute_2d_taxonomy_matrix(slots_or_questions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Generate a 4x6 count table for all combinations of Knowledge Dimension and Cognitive Level.

    Returns:
        dict: {
            "Factual": {"L1": count, "L2": count, ..., "L6": count},
            "Conceptual": {...},
            "Procedural": {...},
            "Metacognitive": {...}
        }
    """
    from blooms.verb_bank import normalize_level

    matrix = {
        kd: {lvl: 0 for lvl in COGNITIVE_LEVELS}
        for kd in KNOWLEDGE_DIMENSIONS
    }

    for item in slots_or_questions:
        raw_lvl = item.get("level") or item.get("bt") or "L1"
        try:
            lvl = normalize_level(raw_lvl)
        except ValueError:
            lvl = "L1"

        raw_kd = item.get("knowledge_dimension") or item.get("kd") or "Conceptual"
        kd = normalize_knowledge_dimension(raw_kd)

        if kd in matrix and lvl in matrix[kd]:
            matrix[kd][lvl] += 1

    return matrix


def infer_knowledge_dimension(
    question: str,
    level: Optional[str] = None,
    answer: Optional[str] = None,
) -> str:
    """Infer the Knowledge Dimension (Factual, Conceptual, Procedural, Metacognitive)
    of an exam question using dynamic 2D BT x KD analysis of verbs, question patterns,
    stem keywords, and answer structure.
    """
    q = (question or "").strip()
    if not q:
        return "Conceptual"

    q_low = q.lower()
    from blooms.verb_bank import first_word, normalize_level
    fw = first_word(q).lower()
    norm_lvl = normalize_level(level) if level else None
    ans_low = (answer or "").lower()

    # 1. Metacognitive signals (reflection, strategic justification, evaluating trade-offs, defending choices)
    if (
        re.search(r"\bjustify\s+(?:the|your)?\s*(?:choice|recommendation|decision|selection|strategy)\b", q_low)
        or re.search(r"\bchoose\s+between\b.*\band\s+defend\b", q_low)
        or re.search(r"\b(?:assess\s+and\s+justify|evaluate\s+a\s+strategy|strategic\s+trade-offs?)\b", q_low)
        or fw in ("justify", "recommend", "defend", "reflect")
        or ("verdict" in ans_low and "justification" in ans_low and "trade-offs" in ans_low)
    ):
        return "Metacognitive"

    # 2. Procedural signals (steps, algorithms, code modification, calculations, commands, workflows)
    if (
        re.search(r"\b(?:outline|describe|state|explain)\s+the\s+steps\b", q_low)
        or re.search(r"\b(?:step-by-step|procedure\s+to|procedure\s+for|workflow\s+to)\b", q_low)
        or re.search(r"\bmodify\s+(?:the\s+)?code\b", q_low)
        or fw in ("modify", "calculate", "compute", "solve", "implement", "execute", "demonstrate")
        or re.search(r"\b(?:docker\s+run|kubectl|cli\s+command|apk\s+add|apt-get)\b", q_low)
        or ("step 1" in ans_low and "step 2" in ans_low)
    ):
        return "Procedural"

    # 3. Conceptual signals (explanations, comparisons, distinctions, classifications, models, theories)
    if (
        re.search(r"\b(?:distinguish|contrast|compare|differentiate|classify|paraphrase)\b", q_low)
        or fw in ("explain", "distinguish", "contrast", "paraphrase", "classify", "compare", "differentiate")
        or re.search(r"\bhow\s+.*\b(?:functions|operates|interacts|behaves)\b", q_low)
        or re.search(r"\b(?:difference\s+between|relationship\s+between|architectural\s+model)\b", q_low)
    ):
        return "Conceptual"

    # 4. Factual signals (direct recall of terms, definitions, symbols, lists)
    if (
        fw in ("name", "define", "list", "state", "identify", "recall", "label", "enumerate", "mention")
        or re.search(r"\b(?:what\s+is\s+the\s+term|state\s+the\s+definition|list\s+the\s+properties)\b", q_low)
    ):
        return "Factual"

    # 5. Level-based fallback if stem is ambiguous
    if norm_lvl == "L1":
        return "Factual"
    elif norm_lvl == "L3":
        return "Procedural"
    elif norm_lvl in ("L5", "L6"):
        return "Conceptual"

    return "Conceptual"

