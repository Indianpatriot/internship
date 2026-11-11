# Handoff Documentation: Blueprint Engine, Answers & Multi-Book Retrieval

**Authors**: Nitin (Contracts, Blueprint, Answers, Multi-Book Retrieval), Alice (Multi-Book Ingestion, Syllabus Parsing, Cumulative Totals)  
**Teammates**: Alice, Bob (UI & Document Export), Nitin

---

## 1. Overview & Ownership

| Component | Owner | Status | Purpose |
| :--- | :--- | :--- | :--- |
| `contracts.py` | Nitin | **Complete** | Core data structures (`Slot`, `QuestionRecord`, `Weights`), Bloom's weights & validator |
| `blueprint.py` (`build_blueprint`) | Nitin | **Complete** | Deficit-based greedy allocation engine for 50-mark and 100-mark presets |
| `generation/generator.py` (`generate_answer`, `generate_full_paper`) | Nitin | **Complete** | Grounded model answer generation with `OR` alternative splitting, and full paper assembly |
| `retrieval/sources.py` (`select_chunks_for_unit`) | Nitin | **Complete** | Multi-textbook chunk filtering by unit and chapter |
| `ingestion/library.py` (`add_book`, `load_library`, `list_chapters`, `suggest_unit_sources`, `validate_unit_sources`) | Alice | **Complete** | Multi-textbook ingestion, disk caching, chapter indexing, and source mapping |
| `scheme.py` (`parse_scheme`, `parse_scheme_text`, `map_topics_to_chapters`) | Alice | **Complete** | Syllabus PDF/text parsing (units, outcomes, hours, textbooks) and topic semantic mapping |
| `totals.py` (`summarize`, `totals_table_rows`) | Alice | **Complete** | Cumulative totals, choice question weighting, target deviations, and export tables |
| `stubs.py` | Alice / Nitin | **Implemented** | Forwards to `scheme.parse_scheme` and `totals.summarize` |
| `app.py` & `export_utils.py` | Bob | **Bob's Domain** | Streamlit UI and Word/PDF export integration |

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
    unit: int                       # 1-based unit identifier (e.g., 1..5)
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
    marking_scheme: Optional[MarkingScheme] # Itemized examiner marking scheme
```

### `MarkingScheme` & `SchemeCriterion`
Defines an itemized, examiner-checkable criteria breakdown attached to each question:
```python
class SchemeCriterion(TypedDict, total=False):
    no: int                         # Sequential criterion number (1, 2, ...)
    criterion: str                  # Short label of what is expected
    expected: str                   # What examiner should look for (specific items, keywords, steps)
    marks: Union[int, float]        # Marks allocated to this criterion (> 0)
    kind: str                       # One of "definition" | "explanation" | "points" | "diagram" | "example" | "steps" | "code" | "comparison" | "analysis" | "justification" | "conclusion"
    reason: str                     # Pedagogical justification explaining why this part earns these marks
    partial_credit: Optional[str]   # How to split marks if only partially answered
    alternatives: Optional[list[str]] # List of alternative acceptable answers / approaches

class MarkingScheme(TypedDict, total=False):
    total_marks: Union[int, float]  # Question total marks (must match question's marks exactly)
    criteria: list[SchemeCriterion] # Examiner checkable criteria summing exactly to total_marks
    notes: Optional[str]            # Guidance / instructions for the examiner
    is_fallback: Optional[bool]     # True if generated via rule-based fallback mechanism
```

- `validate_marking_scheme(scheme: dict[str, Any], expected_total: Union[int, float]) -> tuple[bool, list[str]]`:
  Requires criteria marks to sum exactly to `total_marks`, and `total_marks` to equal `expected_total`. Rejects non-positive marks, missing reasons, or missing expected text. Allows half-marks (`0.5`) only when `config.ALLOW_HALF_MARKS` is `True`.


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

## 3. Nitin's Functions & Signatures

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

### `generate_scheme` (`generation/scheme_generator.py`)
```python
def generate_scheme(
    question_record: dict[str, Any],
    chunk: Optional[dict[str, Any]] = None,
    client: Any = None,
    rng: Optional[random.Random] = None,
    max_retries: int = 3,
    force_regenerate: bool = False,
) -> MarkingScheme:
```
**Returns**: `MarkingScheme` dict containing `total_marks`, `criteria`, `notes`, and `is_fallback`.
- Uses `build_scheme_prompt` to query the model for an itemized, examiner-checkable criteria breakdown.
- Automatically repairs small mark discrepancies by adjusting candidate criteria.
- Backed by `build_fallback_scheme(...)` guaranteeing valid, non-empty criteria with pedagogical reasons that sum exactly to the slot's marks.

### `build_scheme_prompt` (`generation/prompts.py`)
```python
def build_scheme_prompt(
    question: str,
    level: str,
    marks: int,
    answer: str,
    context: Optional[str] = "",
) -> str:
```
**Returns**: Formatted prompt instructing Ollama/LLM to produce strict JSON-only marking scheme criteria breakdowns based on Bloom's cognitive level, marks, and model answer.


### `select_chunks_for_unit`
```python
def select_chunks_for_unit(
    chunks: list[dict[str, Any]],
    unit_sources: Optional[dict[Any, Any]],
    unit: Optional[Any],
) -> list[dict[str, Any]]:
```
**Returns**: Filtered `list[dict[str, Any]]` matching the mapped textbooks and chapters for the specified unit.

---

## 4. Alice's Functions: Ingestion Library, Scheme Parsing & Totals

### A. Multi-Book Ingestion Library (`ingestion/library.py`)

#### `load_library(paths: list[str | Path], cache_dir: Optional[Path | str] = None) -> dict[str, Any]`
Ingests multiple textbook PDFs with SHA256 disk caching under `.cache/ingestion/` and automatic duplicate-stem disambiguation (e.g. `book`, `book_2`).
- **Returns**: `{"books": {book_id: {"title": str, "path": str, "chapters": list[str], "n_chunks": int, "page_range": tuple[int, int]}}, "chunks": list[dict]}`
- **Example**:
  ```python
  from ingestion.library import load_library
  lib = load_library(["uploads/book1.pdf", "uploads/book2.pdf"])
  print("Loaded books:", list(lib["books"].keys()))
  ```

#### `list_chapters(library: dict[str, Any], book_id: str) -> list[dict[str, Any]]`
Provides chapters ordered by page appearance with chunk counts for Bob's chapter selector dropdown.
- **Returns**: `[{"chapter": str, "chunks": int, "chunk_count": int, "page_start": int, "page_end": int}, ...]`
- **Example**:
  ```python
  from ingestion.library import list_chapters
  chaps = list_chapters(lib, "book1")
  # [ {"chapter": "Chapter 1: Intro", "chunks": 8, "page_start": 1, "page_end": 14}, ... ]
  ```

#### `suggest_unit_sources(library: dict[str, Any], units: Any) -> dict[Any, dict[str, Any]]`
Creates an initial best-effort textbook and chapter assignment across units in page order.
- **Returns**: `{unit: {book_id: [chapters]}}`
- **Example**:
  ```python
  from ingestion.library import suggest_unit_sources
  sources = suggest_unit_sources(lib, units=[1, 2, 3, 4, 5])
  ```

#### `validate_unit_sources(library: dict[str, Any], unit_sources: Optional[dict[Any, Any]]) -> list[str]`
Checks for unknown `book_id`s, chapters not present in the book, and units with zero usable chunks.
- **Returns**: `list[str]` of human-readable problem messages (empty list if completely valid).
- **Example**:
  ```python
  from ingestion.library import validate_unit_sources
  problems = validate_unit_sources(lib, user_selected_sources)
  if problems:
      print("Configuration errors:", problems)
  ```

---

### B. Syllabus Scheme Parsing (`scheme.py`)

#### `parse_scheme(pdf_path: str, library: Optional[dict[str, Any]] = None) -> dict[str, Any]`
#### `parse_scheme_text(text: str, library: Optional[dict[str, Any]] = None) -> dict[str, Any]`
Extracts units (Arabic/Roman numerals), lecture hours, topic lists, Course Outcomes (CO1..COn), textbooks, and fuzzy-matched unit sources using `difflib`.
- **Returns**:
  ```python
  {
      "units": [{"unit": 1, "title": "Divide & Conquer", "topics": ["Merge sort", ...], "hours": 8}],
      "outcomes": {"CO1": "Understand algorithmic principles..."},
      "unit_sources": {1: {"book1": None}},
      "textbooks": ["Cormen et al., Introduction to Algorithms..."],
      "confidence": 0.85,  # float in [0.0, 1.0]
      "warnings": []       # list[str]
  }
  ```
- **Example**:
  ```python
  from scheme import parse_scheme, parse_scheme_text
  scheme = parse_scheme("syllabus.pdf", library=lib)
  # or from user text input:
  scheme = parse_scheme_text(pasted_text, library=lib)
  ```

#### `map_topics_to_chapters(scheme_units: list[dict[str, Any]], library: dict[str, Any]) -> dict[Any, list[dict[str, Any]]]`
Computes semantic embeddings between each unit's topics and library chapters using SentenceTransformer.
- **Returns**: `{unit_num: [{"book_id": str, "chapter": str, "score": float}, ...]}` sorted descending by relevance.
- **Example**:
  ```python
  from scheme import map_topics_to_chapters
  matches = map_topics_to_chapters(scheme["units"], lib)
  best_for_unit_1 = matches[1][0]
  print(f"Recommended: {best_for_unit_1['chapter']} (score {best_for_unit_1['score']})")
  ```

#### Scheme confirm / manual fallback (`scheme.py`)
- `parse_scheme(...)` / `parse_scheme_text(...)` now also return `needs_manual: bool`. With a `library`, `unit_sources` is pre-filled as `{unit: {book_id: [chapter]}}` (semantic match, keyword-overlap fallback if the embedding model is missing, page-order split as last resort) so the UI only asks the user to confirm.
- `needs_manual` is True when no units/outcomes were found, confidence < 0.5, or the PDF is unreadable/scanned. The UI should then show `manual_scheme_template(n_units, n_outcomes)` as editable fields.
- `merge_manual_entry(scheme, units=, outcomes=, unit_sources=)` overlays what the user typed/confirmed and returns the final scheme (`confirmed=True`).
- `prefill_unit_sources(scheme, library, top_k=1)` is the underlying helper.

---

### C. Cumulative Totals & Export Tables (`totals.py`)

#### `summarize(questions: list[dict[str, Any]], targets: Optional[dict[str, Any]] = None, attempted_only: bool = True) -> dict[str, Any]`
Calculates exact marks, percentages, and question counts by Course Outcome, Bloom's level (L1-L6), and Bloom's band (L1-L2, L3-L4, L5-L6).
- When `attempted_only=True`, alternative choice questions in the same `alternative_group` count once toward attempted marks while reporting full `offered_marks`.
- **Returns**:
  ```python
  {
      "by_co": {code: {"marks": int, "pct": float, "questions": int}},
      "by_bt": {"L1": {"marks": int, "pct": float, "questions": int}, ..., "L6": ...},
      "by_bt_band": {"L1-L2": {"marks": int, "pct": float, "questions": int}, ...},
      "total_marks": int,
      "offered_marks": int,
      "targets_diff": {"bt_band": {...}, "co": {...}}
  }
  ```
- **Example**:
  ```python
  from totals import summarize
  stats = summarize(paper["questions"], targets=blueprint["weights"])
  print(f"Total Attempted: {stats['total_marks']} / Offered: {stats['offered_marks']}")
  ```

#### `totals_table_rows(summary: dict[str, Any]) -> dict[str, list[list[Any]]]`
Transforms summary analytics into structured list-of-lists tables with totals ready for Bob's Word/PDF export without formatting logic.
- **Returns**: `{"co_table": list[list[Any]], "bt_table": list[list[Any]]}`
- **Example**:
  ```python
  from totals import totals_table_rows
  tables = totals_table_rows(stats)
  co_rows = tables["co_table"]  # [ ["Course Outcome", "Marks", "Percentage", "Questions Count"], ... ]
  bt_rows = tables["bt_table"]  # [ ["Bloom's Taxonomy Band / Level", "Marks", ...], ... ]
  ```

---

## 5. UI Integration Guide for Bob

Bob should call the following plain functions directly in `app.py` and `export_utils.py`:

| UI Step / Component | Function to Call | Import Location |
| :--- | :--- | :--- |
| **1. Textbook Upload** | `load_library(paths)` | `from ingestion.library import load_library` |
| **2. Chapter Selector UI** | `list_chapters(library, book_id)` | `from ingestion.library import list_chapters` |
| **3. Syllabus Upload / Paste** | `parse_scheme(pdf)` or `parse_scheme_text(text)` | `from scheme import parse_scheme, parse_scheme_text` |
| **4. Default Source Mapping** | `suggest_unit_sources(library, units)` | `from ingestion.library import suggest_unit_sources` |
| **5. Source Mapping Check** | `validate_unit_sources(library, unit_sources)` | `from ingestion.library import validate_unit_sources` |
| **6. Semantic Chapter Auto-Map**| `map_topics_to_chapters(scheme_units, library)` | `from scheme import map_topics_to_chapters` |
| **7. Paper Generation** | `build_blueprint(...)`<br>`generate_full_paper(...)` | `from blueprint import build_blueprint`<br>`from generation.generator import generate_full_paper` |
| **8. Paper Analytics & Dashboard**| `summarize(questions, targets=...)` | `from totals import summarize` |
| **9. Word / PDF Table Export** | `totals_table_rows(summary)` | `from totals import totals_table_rows` |
| **10. Marking Scheme Generation** | `generate_scheme(question_record, chunk, client)` | `from generation.scheme_generator import generate_scheme` |
| **11. Word / PDF Export with Schemes** | `create_docx_exam_paper(..., include_marking_scheme=True, include_reasons=True)`<br>`create_pdf_exam_paper(..., include_marking_scheme=True, include_reasons=True)` | `from export_utils import create_docx_exam_paper, create_pdf_exam_paper` |

---

## 6. Examiner Marking Scheme Breakdown & Export Integration

### Marking Scheme Structure
Each question in `QuestionRecord` contains an itemized `marking_scheme`:
- **Criteria**: Each criterion specifies `no`, `criterion`, `expected` (what examiner looks for), `marks`, `kind` (`definition`, `explanation`, `points`, `diagram`, `example`, `steps`, `code`, `comparison`, `analysis`, `justification`, `conclusion`), `reason` (pedagogical justification explaining why this part earns these marks), `partial_credit`, and `alternatives` (acceptable alternative solutions).
- **Sum Rule**: Sum of criteria marks equals the question's total marks exactly.
- **Granularity**: Whole marks by default; half-marks (`0.5`) supported when `config.ALLOW_HALF_MARKS` is enabled.
- **Rule-based Fallback**: If LLM output fails validation after retries, `build_fallback_scheme` generates structured criteria from the model answer sentences and BT level templates (`is_fallback=True`).

### UI & Document Export Integration
- **Streamlit Results UI (`app.py`)**:
  - Displays a collapsible expander `📋 Marking scheme: Question {id}` showing a structured table: `No | Criterion | What the examiner looks for | Marks | Reason`, along with a bold **Total Marks** row.
  - Fallback schemes display subtle notice: `* (auto-generated structure; review before use)`.
  - Keeps `💡 Model answer: Question {id}` as a separate collapsed expander.
  - Export controls provide toggles: `"Include marking scheme in export"` (default on) and `"Include reasons"` (default on).
- **Word & PDF Documents (`export_utils.py`)**:
  - In faculty copies, adds an itemized marking scheme table per question under `SCHEME OF EVALUATION & MODEL ANSWERS`.
  - Displays alternative solutions with `― OR ―`.
  - In student copies (`profile="student"`), schemes and solutions are strictly omitted.

