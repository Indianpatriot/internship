# Handoff Documentation: Blueprint Engine, Answers & Multi-Book Retrieval

**Author**: Nitin  
**Teammates**: Alice (Ingestion & Syllabus Scheme Parsing), Bob (UI & Document Export)

---

## 1. Overview & Ownership

| Component | Owner | Status | Purpose |
| :--- | :--- | :--- | :--- |
| `contracts.py` | Nitin | **Complete** | Core data structures (`Slot`, `QuestionRecord`, `Weights`), Bloom's weights & validator |
| `blueprint.py` (`build_blueprint`) | Nitin | **Complete** | Deficit-based greedy allocation engine for 50-mark and 100-mark presets |
| `generation/generator.py` (`generate_answer`, `generate_full_paper`) | Nitin | **Complete** | Grounded model answer generation with `OR` alternative splitting, and full paper assembly |
| `retrieval/sources.py` (`select_chunks_for_unit`) | Nitin | **Complete** | Multi-textbook chunk filtering by unit and chapter |
| `stubs.py` (`parse_scheme`) | Alice | **Stub** (replace) | Syllabus PDF parser to extract units, outcomes, and textbook mappings |
| `stubs.py` (`summarize`) | Bob / Alice | **Stub** (replace) | Cumulative totals and target difference summary for UI display |

> **Note**: `app.py` and `export_utils.py` are strictly owned by Bob. Bob can import the plain functions documented below directly.

---

## 2. Core Data Formats (`contracts.py`)

### `Slot`
Defines a question or sub-question slot in a blueprint:
```python
class Slot(TypedDict, total=False):
    unit: int                       # 1-based unit identifier (e.g., 1..5)
    question_no: int                # 1-based question number (e.g., 1..10)
    sub_index: int | str            # Sub-question index (e.g., 1, 2 or 'a', 'b')
    co: str                         # Target Course Outcome code (e.g., "CO1")
    level: str                      # Bloom's cognitive level ("L1" - "L6")
    marks: int                      # Exact integer marks for this slot
    alternative_group: Optional[str]# Group ID for choice questions (e.g., "U1"); None for compulsory
    id: str                         # Display ID (e.g., "1(a)", "4")
    part: str                       # Exam section (e.g., "Unit 1" or "Part A")
    knowledge_dimension: str        # Revised Bloom's knowledge dimension (e.g., "Conceptual")
    kd: str                         # Short alias for knowledge_dimension
```

### `QuestionRecord`
Contains the generated question, verified model answer, and textbook provenance:
```python
class QuestionRecord(TypedDict, total=False):
    id: str                         # Display ID (e.g., "1(a)")
    unit: Optional[int]             # Unit number
    question_no: Optional[int]      # Main question number
    sub_index: Optional[int | str]  # Sub-question index
    question: str                   # Generated question text ending with '?'
    answer: str                     # Primary model answer (or OR-joined alternatives)
    answer_alternatives: list[str]  # List of acceptable answer alternatives
    level: str                      # Bloom's level (L1-L6)
    co: str                         # Course Outcome (e.g., "CO1")
    marks: int                      # Integer marks
    book_id: Optional[str]          # Source textbook stem / identifier
    chapter: Optional[str | int]    # Source chapter
    page: Optional[int]             # Source start page (page_start)
    valid: bool                     # Whether generation passed validator & similarity check
    reason: str                     # Acceptance reason ("ok") or failure cause
    attempts: int                   # Number of LLM sampling attempts
```

### `Weights`
Defines target weight fractions for Bloom's cognitive bands and Course Outcomes:
```python
class Weights(TypedDict, total=False):
    bt: dict[str, float]  # E.g. {"L1-L2": 0.40, "L3-L4": 0.40, "L5-L6": 0.20}
    co: dict[str, float]  # E.g. {"CO1": 0.20, "CO2": 0.20, ...}
```
- Teacher's allowed BT ranges: `TEACHER_BT_RANGES = {"L1-L2": (0.30, 0.40), "L3-L4": (0.30, 0.40), "L5-L6": (0.10, 0.20)}`.
- Default BT weights: `DEFAULT_BT_WEIGHTS = {"L1-L2": 0.40, "L3-L4": 0.40, "L5-L6": 0.20}`.
- `validate_weights(weights) -> tuple[bool, list[str]]`: requires fractions to sum to 1.0 (tolerance 1e-6); warns if outside `TEACHER_BT_RANGES`.

---

## 3. Implemented Functions & Signatures

### `build_blueprint`
```python
def build_blueprint(
    preset: Optional[Any] = None,
    total_marks: Optional[int] = None,
    units: Optional[Any] = None,
    questions_per_unit: Optional[int] = None,
    sub_question_counts: Optional[Any] = None,
    sub_question_marks: Optional[Any] = None,
    bt_weights: Optional[dict[str, float]] = None,
    co_weights: Optional[dict[str, float]] = None,
    unit_co_map: Optional[dict[Any, list[str]]] = None,
    seed: Optional[int] = None,
) -> dict[str, Any]:
```
**Returns**: `{"slots": list[Slot], "report": dict, "warnings": list[str]}`

**Example Call**:
```python
from blueprint import build_blueprint

blueprint = build_blueprint(
    preset=100,             # 5 units x 2 questions = 10 questions, each 20 marks
    sub_question_counts=2,  # 2 sub-questions per question (10 + 10 marks)
    seed=42,
)
print("Total slots:", len(blueprint["slots"]))
print("Max deviation:", blueprint["report"]["max_deviation"])
```

---

### `generate_full_paper`
```python
def generate_full_paper(
    chunks: list[dict[str, Any]],
    blueprint: Any,
    outcomes: dict[str, str],
    unit_sources: Optional[dict[Any, Any]],
    client: Any,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
) -> dict[str, Any]:
```
**Returns**: `{"questions": list[QuestionRecord], "rejected": list[dict], "unfilled": list[Slot], "blueprint_report": dict}`

**Example Call**:
```python
from generation.generator import generate_full_paper
from generation.mock_client import MockClient

client = MockClient()
outcomes = {"CO1": "Understand algorithmic complexity", "CO2": "Apply graph algorithms"}
unit_sources = {
    1: {"textbook_a": ["Chapter 1", "Chapter 2"]},
    2: {"textbook_b": None},  # All chapters
}

paper = generate_full_paper(
    chunks=chunks,
    blueprint=blueprint,
    outcomes=outcomes,
    unit_sources=unit_sources,
    client=client,
    seed=42,
)
print("Generated questions:", len(paper["questions"]))
```

---

### `generate_answer`
```python
def generate_answer(
    question: str,
    chunk: dict[str, Any],
    marks: Optional[int],
    client: Any,
    rng: Optional[random.Random] = None,
    max_retries: int = 3,
    check_similarity: Optional[bool] = None,
) -> dict[str, Any]:
```
**Returns**: `{"answer": str, "answer_alternatives": list[str], "valid": bool, "reason": str}`

**Example Call**:
```python
from generation.generator import generate_answer
from generation.mock_client import MockClient

ans = generate_answer(
    question="Explain the operational characteristics of binary search trees?",
    chunk=sample_chunk,
    marks=5,
    client=MockClient(),
)
print("Answer text:", ans["answer"])
print("Alternative count:", len(ans["answer_alternatives"]))
```

---

### `select_chunks_for_unit`
```python
def select_chunks_for_unit(
    chunks: list[dict[str, Any]],
    unit_sources: Optional[dict[Any, Any]],
    unit: Optional[Any],
) -> list[dict[str, Any]]:
```
**Returns**: Filtered `list[dict[str, Any]]` matching the mapped textbooks and chapters for the specified unit.

**Example Call**:
```python
from retrieval.sources import select_chunks_for_unit

unit_sources = {
    1: {"data_structures_pdf": ["Chapter 1", "Chapter 2"]},
    2: {"db_systems_pdf": None},
}

unit_1_chunks = select_chunks_for_unit(chunks, unit_sources, unit=1)
```

---

## 4. Teammate Stubs (`stubs.py`)

### `parse_scheme` (OWNER: Alice)
```python
def parse_scheme(pdf_path: str) -> dict[str, Any]:
    """OWNER: Alice
    STUB - replace
    Extracts units, course outcomes, and textbook mappings from syllabus PDF.
    """
```
**Example Call**:
```python
from stubs import parse_scheme

scheme_data = parse_scheme("syllabus.pdf")
units = scheme_data["units"]
outcomes = scheme_data["outcomes"]
unit_sources = scheme_data["unit_sources"]
```

### `summarize` (OWNER: Bob)
```python
def summarize(
    questions: list[dict[str, Any]],
    targets: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """OWNER: Bob
    STUB - replace
    Summarizes question totals across COs, BT levels, and target differences.
    """
```
**Example Call**:
```python
from stubs import summarize

summary = summarize(paper["questions"])
print("Total marks:", summary["total_marks"])
print("Marks by CO:", summary["by_co"])
```
