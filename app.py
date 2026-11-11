"""Streamlit UI for Outcome-Aware Question Paper Generation.

Zero-cost, local RAG question paper generation mapped to Course Outcomes (CO/LO)
and Revised Bloom's Taxonomy (Cognitive Process Dimension L1-L6 & Knowledge Dimension).
"""
import os
import re
import hashlib
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

import config
from paper_preset import (
    PaperPreset,
    QuestionPlan,
    UnitPlan,
    make_preset,
    format_paper,
    PRESETS as PAPER_PRESETS,
)
from blueprint import (
    build_blueprint,
    get_50_mark_blueprint,
    get_25_mark_blueprint,
    calculate_blueprint_stats,
    validate_outcomes,
    validate_weightage_totals,
)
from blooms.revised_taxonomy import (
    KNOWLEDGE_DIMENSIONS,
    COGNITIVE_LEVELS,
    COGNITIVE_LEVEL_NAMES,
    normalize_knowledge_dimension,
)
from contracts import DEFAULT_BT_WEIGHTS, TEACHER_BT_RANGES
from chapter_filter import (
    filter_chunks,
    get_chapter_metadata,
    get_filter_summary,
)
from export_utils import clean_book_name, create_docx_exam_paper, create_pdf_exam_paper
from generation.mock_client import MockClient
from generation.ollama_client import OllamaClient
from generation.generator import generate_paper_by_slots
from ingestion.pipeline import ingest
from ingestion.library import (
    load_library,
    suggest_unit_sources,
    list_chapters,
    validate_unit_sources,
)
from retrieval.co_mapper import score_chunks_against_outcomes, _m
from retrieval.llm_mapper import auto_map_units_with_llm
from scheme import parse_scheme, parse_scheme_text
from totals import summarize, totals_table_rows

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Outcome-Aware Exam Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        background: #f8fafc;
        padding: 0.45rem 0.85rem;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        display: inline-block;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-success {
        background: #ecfdf5;
        border-color: #6ee7b7;
        color: #065f46;
    }
    .badge-warning {
        background: #fffbeb;
        border-color: #fde68a;
        color: #92400e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Caching Functions (st.cache_resource / st.cache_data)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (BAAI/bge-small-en-v1.5)...")
def get_cached_embedding_model():
    """Cache the lazy-loaded SentenceTransformer model resource."""
    return _m()


@st.cache_data(show_spinner="Parsing & chunking textbook PDF...")
def get_cached_chunks(file_bytes: bytes, filename: str):
    """Cache the ingestion pipeline result keyed on PDF content."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        chunks = ingest(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return chunks


@st.cache_data(show_spinner="Scoring chunks against outcomes...")
def get_cached_scored_chunks(_chunks, outcomes_tuple: tuple[tuple[str, str], ...]):
    """Cache outcome scoring keyed on outcome descriptions."""
    outcomes_dict = dict(outcomes_tuple)
    chunks_copy = [dict(c) for c in _chunks]
    return score_chunks_against_outcomes(chunks_copy, outcomes_dict)


@st.cache_data(show_spinner="Preparing Word document...")
def get_cached_docx_paper(
    questions_data,
    course_name,
    subject_code,
    outcomes_tuple,
    institution_name,
    covered_chaps_tuple,
    include_answers,
    include_totals,
    totals_summary_data,
    profile="faculty",
    exam_type="SEE",
    semester="",
    include_qa_flags=None,
    include_marking_scheme=True,
    include_reasons=True,
):
    outcomes_map = dict(outcomes_tuple) if outcomes_tuple else {}
    return create_docx_exam_paper(
        questions=questions_data,
        course_name=course_name,
        subject_code=subject_code,
        outcomes=outcomes_map,
        institution_name=institution_name,
        covered_chapters=list(covered_chaps_tuple) if covered_chaps_tuple else None,
        include_source=True,
        include_knowledge_dim=True,
        include_answers=include_answers,
        include_totals=include_totals,
        include_marking_scheme=include_marking_scheme,
        include_reasons=include_reasons,
        totals_summary=totals_summary_data,
        profile=profile,
        exam_type=exam_type,
        semester=semester,
        include_qa_flags=include_qa_flags,
    )


@st.cache_data(show_spinner="Preparing PDF document...")
def get_cached_pdf_paper(
    questions_data,
    course_name,
    subject_code,
    outcomes_tuple,
    institution_name,
    covered_chaps_tuple,
    include_answers,
    include_totals,
    totals_summary_data,
    profile="faculty",
    exam_type="SEE",
    semester="",
    include_qa_flags=None,
    include_marking_scheme=True,
    include_reasons=True,
):
    outcomes_map = dict(outcomes_tuple) if outcomes_tuple else {}
    return create_pdf_exam_paper(
        questions=questions_data,
        course_name=course_name,
        subject_code=subject_code,
        outcomes=outcomes_map,
        institution_name=institution_name,
        covered_chapters=list(covered_chaps_tuple) if covered_chaps_tuple else None,
        include_source=True,
        include_knowledge_dim=True,
        include_answers=include_answers,
        include_totals=include_totals,
        include_marking_scheme=include_marking_scheme,
        include_reasons=include_reasons,
        totals_summary=totals_summary_data,
        profile=profile,
        exam_type=exam_type,
        semester=semester,
        include_qa_flags=include_qa_flags,
    )


def _get_unit_descriptors(active_units, scheme, slots, outcomes):
    """Compile syllabus or blueprint metadata for each active unit for LLM mapping."""
    scheme_units_map = {
        u.get("unit"): u
        for u in (scheme.get("units", []) if scheme and isinstance(scheme, dict) else [])
        if u.get("unit") is not None
    }
    unit_objs = []
    for u in active_units:
        if u in scheme_units_map:
            unit_objs.append(scheme_units_map[u])
        else:
            u_cos = {s.get("co") for s in slots if s.get("unit") == u}
            co_descs = [o["desc"] for o in outcomes if o.get("code") in u_cos and o.get("desc")]
            unit_objs.append({
                "unit": u,
                "title": f"Unit {u}",
                "topics": co_descs,
            })
    return unit_objs



# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "outcomes" not in st.session_state:
    st.session_state.outcomes = [
        {"code": "CO1", "desc": "Understand foundational concepts, terminology, and principles."},
        {"code": "CO2", "desc": "Analyze system architectures, evaluate performance metrics, and solve problems."},
        {"code": "CO3", "desc": "Apply design principles and evaluate trade-offs in real-world scenarios."},
        {"code": "CO4", "desc": "Design and integrate modular components and validate system correctness."},
    ]

available_cos = [o["code"].strip() for o in st.session_state.outcomes if o["code"].strip()]
if not available_cos:
    available_cos = ["CO1"]

if "slots" not in st.session_state:
    st.session_state.slots = get_50_mark_blueprint(available_cos)
if "slots_version" not in st.session_state:
    st.session_state.slots_version = 0
if "outcomes_version" not in st.session_state:
    st.session_state.outcomes_version = 0

if "results" not in st.session_state:
    st.session_state.results = None
if "gen_ctx" not in st.session_state:
    st.session_state.gen_ctx = None
if "gen_result" not in st.session_state:
    st.session_state.gen_result = None
if "regen_count" not in st.session_state:
    st.session_state.regen_count = 0

if "institution_name" not in st.session_state:
    st.session_state.institution_name = "EXAMINATION QUESTION PAPER"

if "subject_code" not in st.session_state:
    st.session_state.subject_code = "CS-302"

if "scheme" not in st.session_state:
    st.session_state.scheme = None

if "target_weights" not in st.session_state:
    st.session_state.target_weights = {
        "bt": dict(DEFAULT_BT_WEIGHTS),
        "co": {co: 1.0 / len(available_cos) for co in available_cos},
    }


# -----------------------------------------------------------------------------
# Sidebar: LLM Configuration & Exam Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Generator Settings")
    st.caption("Configure generation mode and exam header details.")

    # Probe Ollama status to determine smart default
    probe_client = OllamaClient(model=config.OLLAMA_MODEL)
    ollama_online = probe_client.is_up()

    use_mock = st.toggle(
        "Use mock client",
        value=not ollama_online,
        help="Toggle between offline MockClient and local Ollama server.",
    )

    model_name = st.text_input(
        "Model Name",
        value=config.OLLAMA_MODEL,
        help="Name of the model running on your local Ollama server.",
    )

    if use_mock:
        st.warning("⚠️ **Mock Client Active**: Offline mode enabled. Questions generate instantaneously without Ollama.")
    else:
        active_client = OllamaClient(model=model_name)
        if active_client.is_up():
            st.success(f"🟢 **Ollama Connected**: `{model_name}` ({config.OLLAMA_HOST})")
        else:
            st.error(f"🔴 Ollama server not reachable at `{config.OLLAMA_HOST}`. Start it with `ollama serve`.")

    exclude_deprecated = st.checkbox(
        "Exclude deprecated topics",
        value=True,
        help="Exclude outdated or deprecated tools/APIs (e.g. rkt, Docker links, dockershim, ReplicationController).",
    )

    st.divider()
    st.subheader("📝 Exam Paper Details")
    st.session_state.institution_name = st.text_input(
        "Exam Title / Institute",
        value=st.session_state.institution_name,
        help="Appears as the top header on exported DOCX and PDF papers.",
    )
    st.session_state.subject_code = st.text_input(
        "Subject Code",
        value=st.session_state.subject_code,
        help="Course or subject code (e.g., CS-302).",
    )

    st.divider()
    st.markdown(
        """
        **Revised Bloom's Taxonomy 2D Framework:**
        - **Cognitive Process Dimension:**
          L1 (Remember) → L6 (Create)
        - **Knowledge Dimension:**
          Factual, Conceptual, Procedural, Metacognitive
        - **Model:** `BAAI/bge-small-en-v1.5`
        - **Zero API Cost**
        """
    )


# -----------------------------------------------------------------------------
# Main Header & Scheme Upload Banner (Substep 1.3)
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Outcome-Aware Question Paper Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a textbook PDF, select chapters, define Course Outcomes (CO/LO), '
    'and generate a balanced 50/100-mark exam paper mapped to <b>Revised Bloom\'s Taxonomy</b> (Cognitive & Knowledge dimensions).</div>',
    unsafe_allow_html=True,
)

# Substep 1.3: Banner: "Upload your scheme for a more precise paper."
st.info("📋 **Upload your scheme for a more precise paper.**")
with st.expander("📋 Upload your scheme for a more precise paper.", expanded=False):
    st.caption("Upload your syllabus scheme (PDF or text) to automatically extract Course Outcomes (COs), unit modules, hours, and textbook mappings.")
    col_sch_f, col_sch_t = st.columns([1, 1])
    with col_sch_f:
        scheme_file = st.file_uploader(
            "Upload Syllabus / Scheme (PDF or TXT)",
            type=["pdf", "txt"],
            key="scheme_uploader",
            help="Syllabus PDF or text containing Units, Hours, Course Outcomes, and Prescribed Textbooks.",
        )
    with col_sch_t:
        scheme_paste = st.text_area(
            "Or paste syllabus text directly:",
            placeholder="Unit 1: Introduction (8 hours)...\nCO1: Understand core principles...\nTextbooks: ...",
            key="scheme_text_area",
            height=110,
        )

    if st.button("Parse & Apply Scheme", key="btn_parse_scheme"):
        parsed = None
        if scheme_file:
            if scheme_file.name.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(scheme_file.getvalue())
                    tmp_p = tmp.name
                try:
                    parsed = parse_scheme(tmp_p)
                finally:
                    if os.path.exists(tmp_p):
                        try:
                            os.remove(tmp_p)
                        except OSError:
                            pass
            else:
                parsed = parse_scheme_text(scheme_file.getvalue().decode("utf-8", errors="ignore"))
        elif scheme_paste.strip():
            parsed = parse_scheme_text(scheme_paste)

        if parsed:
            st.session_state.scheme = parsed
            if parsed.get("outcomes"):
                st.session_state.outcomes = [
                    {"code": k, "desc": v} for k, v in parsed["outcomes"].items()
                ]
                st.session_state.outcomes_version = st.session_state.get("outcomes_version", 0) + 1
                for k in list(st.session_state.keys()):
                    if k.startswith("outcome_code_") or k.startswith("outcome_desc_") or k.startswith("del_outcome_"):
                        del st.session_state[k]
                n_parsed = len(parsed["outcomes"])
                if n_parsed < config.MIN_COS:
                    st.session_state.scheme_warning = (
                        f"Extracted {n_parsed} Course Outcomes from scheme, but a paper requires at least "
                        f"{config.MIN_COS}. Please add more outcomes below."
                    )
                elif n_parsed > config.MAX_COS:
                    st.session_state.scheme_warning = (
                        f"Extracted {n_parsed} Course Outcomes from scheme, but a paper allows at most "
                        f"{config.MAX_COS}. Please remove outcomes below."
                    )
                else:
                    st.session_state.scheme_warning = None
                    st.success(f"✓ Extracted {n_parsed} Course Outcomes and {len(parsed.get('units', []))} Units from scheme!")
                st.rerun()
            else:
                st.warning("Scheme parsed, but no Course Outcomes (CO1..COn) were detected.")

# -----------------------------------------------------------------------------
# Stage 1: Course Outcomes (Dynamic CRUD)
# -----------------------------------------------------------------------------
st.subheader("1. Course / Learning Outcomes")
st.caption("Define the outcomes specific to your syllabus. Click '+ Add outcome' to append more.")

if st.session_state.get("scheme_warning"):
    st.warning(st.session_state.scheme_warning)

ov = st.session_state.get("outcomes_version", 0)
total_outcomes = len(st.session_state.outcomes)
can_delete = total_outcomes > 1
can_add = total_outcomes < config.MAX_COS

outcomes_to_remove = []
for i, oc in enumerate(st.session_state.outcomes):
    col_code, col_desc, col_del = st.columns([1.2, 5, 0.8])
    with col_code:
        new_code = st.text_input(
            "Code",
            value=oc["code"],
            key=f"outcome_code_{ov}_{i}",
            label_visibility="collapsed" if i > 0 else "visible",
            placeholder="e.g. CO1",
        )
        oc["code"] = new_code
    with col_desc:
        new_desc = st.text_input(
            "Outcome Description",
            value=oc["desc"],
            key=f"outcome_desc_{ov}_{i}",
            label_visibility="collapsed" if i > 0 else "visible",
            placeholder="Describe what students should be able to do...",
        )
        oc["desc"] = new_desc
    with col_del:
        if i == 0:
            st.caption("Action")
        if st.button("🗑️", key=f"del_outcome_{ov}_{i}", help="Delete outcome", disabled=not can_delete):
            outcomes_to_remove.append(i)

if outcomes_to_remove:
    for idx in sorted(outcomes_to_remove, reverse=True):
        st.session_state.outcomes.pop(idx)
    st.session_state.outcomes_version = ov + 1
    for k in list(st.session_state.keys()):
        if k.startswith("outcome_code_") or k.startswith("outcome_desc_") or k.startswith("del_outcome_"):
            del st.session_state[k]
    st.rerun()

col_btn, col_cap = st.columns([2, 5])
with col_btn:
    if st.button("➕ Add outcome", key="btn_add_outcome", disabled=not can_add):
        existing_nums = [
            int(m.group(1))
            for o in st.session_state.outcomes
            if (m := re.match(r"^CO(\d+)$", o["code"].strip(), re.I))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else len(st.session_state.outcomes) + 1
        st.session_state.outcomes.append({"code": f"CO{next_num}", "desc": ""})
        st.session_state.outcomes_version = ov + 1
        for k in list(st.session_state.keys()):
            if k.startswith("outcome_code_") or k.startswith("outcome_desc_") or k.startswith("del_outcome_"):
                del st.session_state[k]
        st.rerun()
    if not can_add:
        st.caption(f"Maximum {config.MAX_COS} outcomes")

with col_cap:
    if total_outcomes < config.MIN_COS:
        st.caption(f"Minimum {config.MIN_COS} outcomes required for paper generation")


st.divider()

# -----------------------------------------------------------------------------
# Stage 2: Question Blueprint Slots (Revised Bloom's Taxonomy 2D)
# -----------------------------------------------------------------------------
st.subheader("2. Question Paper Blueprint (Slots)")
st.caption(
    "Configure blueprint preset, sub-questions per question, and target weightages. "
    "Map each question across the <b>Cognitive Process Dimension (L1–L6)</b> and <b>Knowledge Dimension (Factual, Conceptual, Procedural, Metacognitive)</b>."
)

# Ignore outcomes with an empty description (they do not count toward the total)
valid_cos = []
seen_cos = set()
for o in st.session_state.outcomes:
    c = o.get("code", "").strip()
    d = o.get("desc", "").strip()
    if c and d and c not in seen_cos:
        seen_cos.add(c)
        valid_cos.append(c)

outcomes_valid, outcome_err_msg = validate_outcomes(len(valid_cos))
if outcomes_valid and "scheme_warning" in st.session_state:
    st.session_state.scheme_warning = None

available_cos = valid_cos if valid_cos else [o["code"].strip() for o in st.session_state.outcomes if o["code"].strip()]
if not available_cos:
    available_cos = ["CO1"]

def rebuild_co_weights(current_cos: list[str]) -> dict[str, float]:
    """Rebuild CO weights dict from current outcome codes, dropping stale keys."""
    # Drop stale session_state keys for deleted or renamed outcomes
    stale_keys = [
        k for k in list(st.session_state.keys())
        if (k.startswith("w_co_") and k[len("w_co_"):] not in current_cos)
        or (k.startswith("slider_co_") and k[len("slider_co_"):] not in current_cos)
    ]
    for k in stale_keys:
        del st.session_state[k]

    equal_pct = round(100.0 / max(len(current_cos), 1), 1)
    weights = {}
    for c in current_cos:
        val = st.session_state.get(f"w_co_{c}", equal_pct)
        st.session_state[f"w_co_{c}"] = val
        weights[c] = float(val)

    return weights

# Substep 1.1: UI Controls (Preset dropdown 50/100, Sub-questions input, Weightage sliders)
col_preset, col_subq = st.columns([1.5, 1.5])
with col_preset:
    preset_options = [
        "50 Marks (Standard)",
        "100 Marks (Choice)",
        "25 Marks (Internal Assessment)",
        "Custom Blueprint",
    ]
    selected_preset = st.selectbox(
        "Exam Preset",
        options=preset_options,
        index=0 if "selected_preset_idx" not in st.session_state else st.session_state.selected_preset_idx,
        help="Select 50-mark standard exam, 100-mark exam with internal choice, or custom structure.",
    )

current_preset = PAPER_PRESETS.get(
    selected_preset,
    make_preset(selected_preset, marks_per_unit=20, has_choice=("100" in selected_preset)),
)
preset_has_choice = bool(getattr(current_preset, "has_choice", False))

target_unit_marks = (
    current_preset.units[0].options[0].marks
    if current_preset.units and current_preset.units[0].options
    else (20 if preset_has_choice else (10 if "50" in selected_preset else (5 if "25" in selected_preset else 20)))
)
num_preset_units = len(current_preset.units) if current_preset.units else 5

# Reset sub-question marks defaults if target_unit_marks changes
prev_target_marks = st.session_state.get("_prev_target_unit_marks")
if prev_target_marks != target_unit_marks:
    st.session_state._prev_target_unit_marks = target_unit_marks
    for u in range(1, num_preset_units + 1):
        if f"unit_{u}_marks_input" in st.session_state:
            del st.session_state[f"unit_{u}_marks_input"]

# Switching between presets updates the UI immediately with no leftover choice elements
prev_active_preset = st.session_state.get("_active_preset_name")
if prev_active_preset is not None and prev_active_preset != selected_preset:
    st.session_state._active_preset_name = selected_preset
    st.session_state.slots_version = st.session_state.get("slots_version", 0) + 1
    for k in list(st.session_state.keys()):
        if k.startswith("slot_") or (k.startswith("unit_") and k.endswith("_marks_input")):
            del st.session_state[k]

    if not preset_has_choice:
        if "25" in selected_preset:
            st.session_state.slots = get_25_mark_blueprint(available_cos)
            st.session_state.selected_preset_idx = 2
        else:
            st.session_state.slots = get_50_mark_blueprint(available_cos)
            st.session_state.selected_preset_idx = 0
    else:
        st.session_state.selected_preset_idx = 1 if "100" in selected_preset else 3
        p_obj = PAPER_PRESETS.get(
            selected_preset,
            make_preset(selected_preset, units=num_preset_units, marks_per_unit=target_unit_marks, has_choice=True),
        )
        bp = build_blueprint(preset=p_obj, outcomes=available_cos, seed=42)
        st.session_state.slots = bp["slots"]
    st.session_state.bp_warnings = []
    st.rerun()
elif prev_active_preset is None:
    st.session_state._active_preset_name = selected_preset

with col_subq:
    sub_questions_input = st.number_input(
        "Sub-questions per question",
        min_value=0,
        max_value=4,
        value=int(st.session_state.get("sub_questions_count", 2)),
        step=1,
        help="Default number of subparts (0 for full question, 2 for (a) & (b), etc.) per question slot.",
    )
    st.session_state.sub_questions_count = sub_questions_input

# Step 5: Unit Question & Sub-question Structure Configuration
unit_sub_marks: dict[int, list[int]] = {}
unit_structure_valid = True
unit_validation_errors: list[str] = []

with st.expander("📐 Configure Questions & Sub-questions Per Unit", expanded=preset_has_choice):
    st.caption(
        f"For each unit, configure the **number of sub-questions** (0 for none) and the **marks of each**, "
        f"with a live check that marks add up to the unit total ({target_unit_marks} marks)."
    )

    h_u, h_cnt, h_m, h_chk = st.columns([1.2, 1.6, 2.4, 2.2])
    h_u.caption("**Unit**")
    h_cnt.caption("**Sub-questions (0 = None)**")
    h_m.caption(f"**Marks per Sub-question (Sum = {target_unit_marks})**")
    h_chk.caption("**Live Marks Check**")

    for u in range(1, num_preset_units + 1):
        u_c1, u_c2, u_c3, u_c4 = st.columns([1.2, 1.6, 2.4, 2.2])
        with u_c1:
            st.markdown(f"**Unit {u}**")
            st.caption(f"{target_unit_marks}m" if not preset_has_choice else f"{target_unit_marks}m / option")

        with u_c2:
            sq_cnt = st.number_input(
                f"U{u} Sub-questions",
                min_value=0,
                max_value=4,
                value=int(st.session_state.get(f"unit_{u}_sq_count", sub_questions_input)),
                step=1,
                key=f"unit_{u}_sq_count",
                label_visibility="collapsed",
            )

        with u_c3:
            if sq_cnt == 0:
                st.text_input(
                    f"U{u} Marks",
                    value=f"Full {target_unit_marks}m question",
                    disabled=True,
                    key=f"unit_{u}_marks_display",
                    label_visibility="collapsed",
                )
                unit_sub_marks[u] = []
                m_valid = True
                m_status = f"✅ {target_unit_marks} / {target_unit_marks} marks"
            else:
                base_m = target_unit_marks // sq_cnt
                rem_m = target_unit_marks % sq_cnt
                calc_default_str = ", ".join(str(base_m + 1 if i < rem_m else base_m) for i in range(sq_cnt))
                default_val = st.session_state.get(f"unit_{u}_marks_input", calc_default_str)
                try:
                    curr_parsed = [int(x.strip()) for x in default_val.split(",") if x.strip()]
                    if len(curr_parsed) != sq_cnt:
                        default_val = calc_default_str
                except Exception:
                    default_val = calc_default_str

                marks_str = st.text_input(
                    f"U{u} Marks",
                    value=default_val,
                    key=f"unit_{u}_marks_input",
                    label_visibility="collapsed",
                    placeholder="e.g. 10, 10",
                )
                try:
                    parsed_m = [int(x.strip()) for x in marks_str.split(",") if x.strip()]
                except ValueError:
                    parsed_m = []

                if len(parsed_m) != sq_cnt:
                    m_valid = False
                    m_status = f"❌ Enter {sq_cnt} values"
                    unit_validation_errors.append(f"Unit {u}: Expected {sq_cnt} values, got {len(parsed_m)}.")
                elif sum(parsed_m) != target_unit_marks:
                    m_valid = False
                    m_status = f"❌ Sum: {sum(parsed_m)}m (needs {target_unit_marks}m)"
                    unit_validation_errors.append(f"Unit {u}: Sub-question marks sum to {sum(parsed_m)}, expected {target_unit_marks}.")
                else:
                    m_valid = True
                    m_status = f"✅ {sum(parsed_m)} / {target_unit_marks} marks"
                    unit_sub_marks[u] = parsed_m

                if not m_valid:
                    unit_structure_valid = False

        with u_c4:
            st.markdown(f"**{m_status}**")

with st.expander("⚖️ Bloom's Taxonomy & Course Outcome Weightage Sliders", expanded=False):
    col_bt_w, col_co_w = st.columns(2)
    with col_bt_w:
        st.markdown("**Bloom's Taxonomy (BT) Cognitive Band Weightage (%)**")
        st.caption("Standard target ranges: L1–L2 (30–40%), L3–L4 (30–40%), L5–L6 (10–20%)")
        w_l12 = st.slider("L1–L2 (Remember / Understand) %", min_value=0.0, max_value=100.0, value=float(st.session_state.get("w_l12", 40.0)), step=1.0)
        w_l34 = st.slider("L3–L4 (Apply / Analyze) %", min_value=0.0, max_value=100.0, value=float(st.session_state.get("w_l34", 40.0)), step=1.0)
        w_l56 = st.slider("L5–L6 (Evaluate / Create) %", min_value=0.0, max_value=100.0, value=float(st.session_state.get("w_l56", 20.0)), step=1.0)
        st.session_state.w_l12 = w_l12
        st.session_state.w_l34 = w_l34
        st.session_state.w_l56 = w_l56

        tot_bt_w = round(w_l12 + w_l34 + w_l56, 3)
        st.info(f"📊 Running Total (Bloom's): **{tot_bt_w:g}%** / 100%")

    with col_co_w:
        st.markdown("**Course Outcomes (CO) Weightage (%)**")
        st.caption("Allocate percentage emphasis across syllabus Course Outcomes.")
        co_weights_input = {}
        equal_pct = round(100.0 / max(len(available_cos), 1), 1)
        for c in available_cos:
            saved_val = st.session_state.get(f"w_co_{c}", equal_pct)
            co_weights_input[c] = st.slider(f"{c} Weightage %", min_value=0.0, max_value=100.0, value=float(saved_val), step=1.0, key=f"slider_co_{c}")
            st.session_state[f"w_co_{c}"] = co_weights_input[c]

        tot_co_w = round(sum(co_weights_input.values()), 3)
        st.info(f"📊 Running Total (COs): **{tot_co_w:g}%** / 100%")

# Show any blueprint warnings or weightage apply errors
if st.session_state.get("weightage_validation_errors"):
    for err in st.session_state.weightage_validation_errors:
        st.error(err)

if st.session_state.get("bp_warnings"):
    for w in st.session_state.bp_warnings:
        st.warning(w)

# Preset Action Buttons
col_p1, col_p2, col_p3 = st.columns([2.5, 2.5, 2])
with col_p1:
    can_apply = outcomes_valid and unit_structure_valid
    apply_help = outcome_err_msg if not outcomes_valid else (unit_validation_errors[0] if not unit_structure_valid else None)
    btn_apply = st.button(
        "⚙️ Apply Preset & Weightages",
        use_container_width=True,
        disabled=not can_apply,
        help=apply_help,
    )
    if not can_apply:
        st.caption(f"⚠️ {apply_help}")
    if btn_apply:
        bt_input_map = {
            "L1-L2": float(w_l12),
            "L3-L4": float(w_l34),
            "L5-L6": float(w_l56),
        }
        co_input_map = {c: float(co_weights_input.get(c, 0.0)) for c in valid_cos}

        valid_weights, weight_errors = validate_weightage_totals(
            co_weights=co_input_map,
            bt_weights=bt_input_map,
            existing_cos=valid_cos,
            tolerance=0.01,
        )
        if not valid_weights:
            st.session_state.weightage_validation_errors = weight_errors
            for err in weight_errors:
                st.error(err)
        else:
            if "weightage_validation_errors" in st.session_state:
                del st.session_state.weightage_validation_errors
            eff_bt_weights = {k: v / 100.0 for k, v in bt_input_map.items()}
            eff_co_weights = {k: v / 100.0 for k, v in co_input_map.items()}
            st.session_state.target_weights = {"bt": eff_bt_weights, "co": eff_co_weights}

            try:
                if "25" in selected_preset:
                    st.session_state.slots = get_25_mark_blueprint(valid_cos)
                    st.session_state.selected_preset_idx = 2
                    st.session_state.bp_warnings = []
                else:
                    preset_obj = make_preset(
                        name=selected_preset,
                        units=num_preset_units,
                        marks_per_unit=target_unit_marks,
                        options=2 if preset_has_choice else 1,
                        sub_marks=unit_sub_marks,
                        has_choice=preset_has_choice,
                    )
                    bp = build_blueprint(
                        preset=preset_obj,
                        bt_weights=eff_bt_weights,
                        co_weights=eff_co_weights,
                        outcomes=valid_cos,
                        seed=42,
                    )
                    st.session_state.slots = bp["slots"]
                    st.session_state.selected_preset_idx = 1 if preset_has_choice else (0 if "50" in selected_preset else 3)
                    st.session_state.bp_warnings = bp.get("warnings", [])

                # Clear stale slot widget keys and bump version
                st.session_state.slots_version = st.session_state.get("slots_version", 0) + 1
                for k in list(st.session_state.keys()):
                    if k.startswith("slot_"):
                        del st.session_state[k]
                st.rerun()
            except ValueError as e:
                st.error(str(e))


with col_p2:
    if st.button("🎓 Standard 50-Mark Blueprint", use_container_width=True, disabled=not outcomes_valid):
        st.session_state.slots = get_50_mark_blueprint(valid_cos)
        st.session_state.selected_preset_idx = 0
        st.session_state.bp_warnings = []
        st.session_state.slots_version = st.session_state.get("slots_version", 0) + 1
        for k in list(st.session_state.keys()):
            if k.startswith("slot_"):
                del st.session_state[k]
        st.rerun()

with col_p3:
    if st.button("🔄 Reset to Blank", use_container_width=True):
        st.session_state.slots = [
            {
                "unit": 1,
                "question_no": 1,
                "id": "1",
                "co": available_cos[0],
                "level": "L1",
                "knowledge_dimension": "Factual",
                "marks": 5,
            }
        ]
        st.session_state.selected_preset_idx = 3
        st.session_state.slots_version = st.session_state.get("slots_version", 0) + 1
        for k in list(st.session_state.keys()):
            if k.startswith("slot_"):
                del st.session_state[k]
        st.rerun()

# Calculate Blueprint Statistics
stats = calculate_blueprint_stats(st.session_state.slots)
total_marks = stats["total_marks"]
attempted_m = stats.get("attempted_marks", total_marks)
offered_m = stats.get("offered_marks", total_marks)
is_choice_paper = stats.get("is_choice", False) and preset_has_choice

if is_choice_paper:
    mark_target_label = f"Attempted: {attempted_m}m / Offered: {offered_m}m"
    badge_class = "badge-success" if attempted_m in (50, 100) else "badge-warning"
else:
    mark_target_label = f"Total: {total_marks} Marks"
    badge_class = "badge-success" if total_marks in (25, 50, 100) else "badge-warning"

unit_offered = stats.get("unit_marks", {})
unit_attempted = stats.get("unit_attempted_marks", {})

unit_badges = "".join(
    f'<span class="metric-badge"><b>Unit {u}:</b> {unit_attempted.get(u, m // 2)}m / {m}m</span> '
    if is_choice_paper else
    f'<span class="metric-badge"><b>Unit {u}:</b> {m}m</span> '
    for u, m in sorted(unit_offered.items())
)

st.markdown(
    f"""
    <div style="margin-top: 0.6rem; margin-bottom: 0.8rem;">
        <span class="metric-badge {badge_class}"><b>Exam Structure:</b> {mark_target_label}</span>
        <span class="metric-badge"><b>Total Slots:</b> {len(st.session_state.slots)}</span>
        {unit_badges}
    </div>
    """,
    unsafe_allow_html=True,
)

# Bloom's 2D Spread & Analytics
with st.expander("📊 View Revised Bloom's Taxonomy 2D Matrix & Spread", expanded=False):
    col_bt_dist, col_kd_dist = st.columns(2)
    with col_bt_dist:
        st.markdown("**1. Cognitive Process Dimension (L1 to L6):**")
        bt_spread_str = " | ".join(
            f"**{lvl} ({COGNITIVE_LEVEL_NAMES.get(lvl, lvl)})**: {marks}m ({stats['bt_percent'].get(lvl, 0)}%)"
            for lvl, marks in sorted(stats["bt_marks"].items())
        )
        st.info(bt_spread_str or "No slots defined")
    with col_kd_dist:
        st.markdown("**2. Knowledge Dimension (4 Types):**")
        kd_spread_str = " | ".join(
            f"**{kd}**: {marks}m ({stats['kd_percent'].get(kd, 0)}%)"
            for kd, marks in stats["kd_marks"].items()
        )
        st.success(kd_spread_str or "No slots defined")

    st.markdown("##### 🏛️ Revised Bloom's 2D Taxonomy Matrix (Question Count per Cell)")
    matrix_data = stats.get("matrix_2d", {})
    matrix_df = pd.DataFrame(
        matrix_data
    ).T[COGNITIVE_LEVELS]
    matrix_df.columns = [f"{lvl} ({COGNITIVE_LEVEL_NAMES[lvl]})" for lvl in COGNITIVE_LEVELS]
    st.dataframe(matrix_df, use_container_width=True)

    st.caption("Matches Anderson & Krathwohl's (2001) 4x6 Revised Taxonomy Table.")

bt_options = ["L1", "L2", "L3", "L4", "L5", "L6"]
bt_labels = {
    "L1": "L1 - Remember",
    "L2": "L2 - Understand",
    "L3": "L3 - Apply",
    "L4": "L4 - Analyze",
    "L5": "L5 - Evaluate",
    "L6": "L6 - Create",
}
kd_options = KNOWLEDGE_DIMENSIONS

# Slots Table Header
hdr_col1, hdr_col2, hdr_col3, hdr_col4, hdr_col5, hdr_col6, hdr_col7 = st.columns([1.0, 1.0, 1.4, 2.0, 1.8, 1.0, 0.6])
with hdr_col1:
    st.caption("Unit")
with hdr_col2:
    st.caption("Q.No")
with hdr_col3:
    st.caption("Outcome (CO)")
with hdr_col4:
    st.caption("Cognitive Level (BT)")
with hdr_col5:
    st.caption("Knowledge Dim (KD)")
with hdr_col6:
    st.caption("Marks")
with hdr_col7:
    st.caption("Action")

v = st.session_state.get("slots_version", 0)
slots_to_remove = []
for j, slot in enumerate(st.session_state.slots):
    col_unit, col_id, col_co, col_bt, col_kd, col_marks, col_del_slot = st.columns([1.0, 1.0, 1.4, 2.0, 1.8, 1.0, 0.6])

    with col_unit:
        slot["unit"] = st.number_input(
            "Unit",
            min_value=1,
            max_value=10,
            value=int(slot.get("unit", 1)),
            step=1,
            key=f"slot_unit_{v}_{j}",
            label_visibility="collapsed",
        )

    with col_id:
        slot["id"] = st.text_input(
            "Q.No",
            value=str(slot.get("id", f"{j+1}")),
            key=f"slot_id_{v}_{j}",
            label_visibility="collapsed",
            placeholder="e.g. 1(a)",
        )

    with col_co:
        co_val = slot.get("co", available_cos[0])
        co_index = available_cos.index(co_val) if co_val in available_cos else 0
        slot["co"] = st.selectbox(
            "Outcome",
            options=available_cos,
            index=co_index,
            key=f"slot_co_{v}_{j}",
            label_visibility="collapsed",
        )

    with col_bt:
        current_lvl = slot.get("level", "L1")
        lvl_index = bt_options.index(current_lvl) if current_lvl in bt_options else 0
        slot["level"] = st.selectbox(
            "Cognitive Level",
            options=bt_options,
            format_func=lambda x: bt_labels.get(x, x),
            index=lvl_index,
            key=f"slot_bt_{v}_{j}",
            label_visibility="collapsed",
        )

    with col_kd:
        curr_kd = normalize_knowledge_dimension(slot.get("knowledge_dimension") or slot.get("kd"))
        kd_index = kd_options.index(curr_kd) if curr_kd in kd_options else 1
        slot["knowledge_dimension"] = st.selectbox(
            "Knowledge Dimension",
            options=kd_options,
            index=kd_index,
            key=f"slot_kd_{v}_{j}",
            label_visibility="collapsed",
        )

    with col_marks:
        slot["marks"] = int(
            st.number_input(
                "Marks",
                min_value=1,
                max_value=50,
                value=int(slot.get("marks", 2)),
                step=1,
                key=f"slot_marks_{v}_{j}",
                label_visibility="collapsed",
            )
        )

    with col_del_slot:
        if st.button("🗑️", key=f"del_slot_{v}_{j}", help="Delete question slot"):
            slots_to_remove.append(j)

if slots_to_remove:
    for idx in sorted(slots_to_remove, reverse=True):
        st.session_state.slots.pop(idx)
    st.rerun()

col_add_q, _ = st.columns([2, 5])
with col_add_q:
    if st.button("➕ Add question slot", key="btn_add_slot"):
        st.session_state.slots.append({
            "unit": 1,
            "id": f"{len(st.session_state.slots) + 1}(a)",
            "co": available_cos[0],
            "level": "L3",
            "knowledge_dimension": "Procedural",
            "marks": 5,
        })
        st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Stage 3: File Upload & Chapter Selection (Task Steps 1, 2, 3, 4)
# -----------------------------------------------------------------------------
st.subheader("3. Document Ingestion & Chapter Selection")

uploaded_pdfs = st.file_uploader(
    "Upload Textbook PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    key="textbook_uploader",
    help="Upload one or more reference textbooks or course materials in PDF format.",
)

filtered_chunks = []
selected_chapters = []
selected_sections = None
page_range = None
can_generate = False
library = {"books": {}, "chunks": []}
books = {}
unmapped_units = []

if uploaded_pdfs:
    upload_dir = Path(".cache/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    titles = []
    for idx, up_file in enumerate(uploaded_pdfs, 1):
        file_bytes = up_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:12]
        orig_stem = Path(up_file.name).stem
        clean_stem = re.sub(r"^[0-9a-f]{8,64}[_-]+", "", orig_stem, flags=re.I).strip("_- ")
        clean_stem = clean_stem.replace("-", " ").replace("_", " ").strip()
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", up_file.name)
        save_path = upload_dir / f"{file_hash}_{safe_name}"
        if not save_path.exists() or save_path.stat().st_size != len(file_bytes):
            with open(save_path, "wb") as f:
                f.write(file_bytes)
        saved_paths.append(str(save_path))
        titles.append(f"{idx} {clean_stem}")

    library = load_library(saved_paths, titles=titles)
    raw_chunks = library.get("chunks", [])
    books = library.get("books", {})

    from ingestion.library import display_name
    if "course_title" not in st.session_state or getattr(st.session_state, "_course_title_is_auto", True):
        if len(uploaded_pdfs) == 1:
            single_binfo = list(books.values())[0] if books else None
            title_cand = single_binfo.get("title", titles[0]) if single_binfo else titles[0]
            st.session_state.course_title = display_name(title_cand)
        else:
            book_titles = [display_name(binfo, order=i + 1) for i, binfo in enumerate(books.values())]
            st.session_state.course_title = " & ".join([b for b in book_titles if b])
        st.session_state._course_title_is_auto = True

    st.markdown(f"##### 📚 Uploaded Textbooks ({len(books)})")
    b_cols = st.columns(min(max(len(books), 1), 4))
    for i, (bid, binfo) in enumerate(books.items()):
        with b_cols[i % len(b_cols)]:
            p_min, p_max = binfo.get("page_range", (1, 1))
            n_ch = len(binfo.get("chapters", []))
            st.metric(
                label=display_name(binfo, order=i + 1),
                value=f"{binfo.get('n_chunks', 0)} chunks",
                delta=f"{n_ch} chaps (pp. {p_min}–{p_max})",
                delta_color="off",
            )

    chapter_metadata = get_chapter_metadata(raw_chunks)

    if chapter_metadata:
        all_chapter_names = [m["chapter"] for m in chapter_metadata]
        chap_display_map = {m["chapter"]: m["display_label"] for m in chapter_metadata}
        chap_sections_map = {m["chapter"]: m["sections"] for m in chapter_metadata}

        if "selected_chapters" not in st.session_state:
            st.session_state.selected_chapters = list(all_chapter_names)

        valid_selected = [c for c in st.session_state.selected_chapters if c in all_chapter_names]
        if not valid_selected and all_chapter_names and "chapter_cleared" not in st.session_state:
            valid_selected = list(all_chapter_names)
            st.session_state.selected_chapters = valid_selected

        st.markdown("##### 📖 Select Chapters for Question Paper")
        st.caption("Every question will be generated strictly from your selected chapters. Other chapters will be ignored.")

        btn_c1, btn_c2, _ = st.columns([1.5, 1.5, 5])
        with btn_c1:
            if st.button("✓ Select all", use_container_width=True):
                st.session_state.selected_chapters = list(all_chapter_names)
                if "chapter_cleared" in st.session_state:
                    del st.session_state.chapter_cleared
                st.rerun()
        with btn_c2:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.selected_chapters = []
                st.session_state.chapter_cleared = True
                st.rerun()

        selected_chapters = st.multiselect(
            "Choose Chapters",
            options=all_chapter_names,
            default=st.session_state.selected_chapters,
            format_func=lambda x: chap_display_map.get(x, x),
            key="chapter_multiselect",
        )
        st.session_state.selected_chapters = selected_chapters

        if selected_chapters:
            all_available_secs = []
            for ch in selected_chapters:
                all_available_secs.extend(chap_sections_map.get(ch, []))
            all_available_secs = sorted(list(set(all_available_secs)))

            if all_available_secs:
                with st.expander("🔍 Filter by Specific Sections (Optional)", expanded=False):
                    selected_sections = st.multiselect(
                        "Include sections:",
                        options=all_available_secs,
                        default=all_available_secs,
                        help="Uncheck any sections you wish to exclude from the paper.",
                    )

        filtered_chunks = filter_chunks(
            raw_chunks,
            selected_chapters=selected_chapters,
            selected_sections=selected_sections,
        )

        if selected_chapters:
            st.info(f"ℹ️ {get_filter_summary(filtered_chunks, selected_chapters)}")
            can_generate = len(filtered_chunks) > 0
        else:
            st.error("⚠️ **Select at least one chapter** to enable question paper generation.")
            can_generate = False

        # Unit-to-Textbook Mapping
        active_units = sorted(list({s["unit"] for s in st.session_state.slots if "unit" in s}))
        if not active_units:
            active_units = [1, 2, 3, 4, 5]

        if "unit_sources" not in st.session_state or not st.session_state.unit_sources:
            if st.session_state.scheme and st.session_state.scheme.get("unit_sources"):
                st.session_state.unit_sources = dict(st.session_state.scheme["unit_sources"])
            elif len(books) > 1:
                st.session_state.unit_sources = suggest_unit_sources(library, active_units)

        mapping_meta = (st.session_state.scheme.get("unit_mapping_meta") or {}) if st.session_state.scheme else {}
        manually_mapped = st.session_state.setdefault("manually_mapped_units", set())
        auto_suggested_units = st.session_state.setdefault("auto_suggested_units", set())

        with st.expander("🗺️ Unit-to-Textbook Mapping (Optional)", expanded=False):
            st.caption(
                "**Optional:** Map specific textbook chapters to syllabus units to focus question generation. "
                "Unmapped units automatically search across all available textbook content."
            )

            # Show book-level fallback notices (e.g. pseudo-sections)
            for bid, binfo in books.items():
                if binfo.get("is_pseudo_sections"):
                    st.warning(f"⚠️ Could not detect chapters for '{binfo.get('title', bid)}'; partitioned by topic into estimated sections Pages X-Y. Please verify.")

            current_sources = st.session_state.get("unit_sources") or {}

            btn_af_col1, btn_af_col2 = st.columns([2.5, 4.5])
            with btn_af_col1:
                if st.button("✨ Auto-fill mapping", key="btn_autofill_mapping", help="Automatically propose textbook chapters for unmapped units using the model"):
                    to_autofill = [
                        u for u in active_units
                        if u not in manually_mapped
                        and (current_sources.get(u) is None or not any(bool(v) for v in (current_sources[u] or {}).values() if v is not None))
                    ]
                    if not to_autofill:
                        st.info("All units are already mapped or manually assigned!")
                    else:
                        client_to_use = MockClient() if use_mock else OllamaClient(model=model_name)
                        unit_descs = [
                            obj for obj in _get_unit_descriptors(active_units, st.session_state.scheme, st.session_state.slots, st.session_state.outcomes)
                            if obj.get("unit") in to_autofill
                        ]
                        auto_res = auto_map_units_with_llm(
                            units=unit_descs,
                            library=library,
                            client=client_to_use,
                            min_confidence=0.50,
                            manually_mapped_units=manually_mapped,
                        )
                        mapped_count = 0
                        for u_num, info in auto_res.items():
                            if info.get("status") == "auto_suggested" and info.get("chapters"):
                                current_sources[u_num] = info["chapters"]
                                auto_suggested_units.add(u_num)
                                mapping_meta[u_num] = info
                                for bid in books:
                                    st.session_state[f"u_{u_num}_b_{bid}_chaps"] = info["chapters"].get(bid, [])
                                mapped_count += 1
                            else:
                                mapping_meta[u_num] = info
                        st.session_state.unit_sources = current_sources
                        if st.session_state.scheme:
                            st.session_state.scheme["unit_sources"] = current_sources
                            st.session_state.scheme["unit_mapping_meta"] = mapping_meta
                        if mapped_count > 0:
                            st.success(f"✓ Auto-mapped {mapped_count} unit(s). Review chapters below and edit as needed.")
                        else:
                            st.warning("Could not confidently map remaining units. Please select chapters manually.")
                        st.rerun()

            updated_sources = {}
            for u in active_units:
                u_sources = current_sources.get(u) or {}
                meta = mapping_meta.get(u, {})

                if u in auto_suggested_units:
                    col_u_hdr, col_u_conf = st.columns([5, 2])
                    with col_u_hdr:
                        conf = meta.get("confidence", 0.90)
                        st.markdown(f"**Unit {u} Sources:** :blue-background[🏷️ Auto-suggested] *(confidence: {conf:.2f})*")
                    with col_u_conf:
                        if st.button("✓ Confirm", key=f"confirm_u_{u}", help="Accept auto-suggested chapters as confirmed"):
                            auto_suggested_units.discard(u)
                            manually_mapped.add(u)
                            st.rerun()
                elif u in manually_mapped:
                    st.markdown(f"**Unit {u} Sources:** :green-background[✍️ Manually mapped]")
                else:
                    st.markdown(f"**Unit {u} Sources:**")

                if meta and u not in auto_suggested_units and u not in manually_mapped:
                    status = meta.get("status")
                    conf = meta.get("confidence", 0.0)
                    method = meta.get("method", "semantic+lexical")
                    if status == "suggested":
                        st.caption(f"💡 Suggested mapping (confidence: {conf:.2f}). You can adjust or leave as is.")
                    elif status == "unmapped":
                        st.caption(f"ℹ️ Unit {u} is unmapped (optional). It will draw from all textbook content.")
                    elif status == "auto":
                        st.caption(f"✅ Auto-mapped via {method} (confidence: {conf:.2f})")

                u_b_cols = st.columns(min(max(len(books), 1), 3))
                unit_book_map = {}
                for b_idx, (bid, binfo) in enumerate(books.items()):
                    with u_b_cols[b_idx % len(u_b_cols)]:
                        b_chaps = binfo.get("chapters", [])
                        def_chaps = u_sources.get(bid) if isinstance(u_sources, dict) else None
                        valid_def = [c for c in def_chaps if c in b_chaps] if def_chaps else []
                        from ingestion.library import display_name
                        book_lbl = display_name(binfo, order=b_idx + 1)
                        sel_for_b = st.multiselect(
                            f"{book_lbl}",
                            options=b_chaps,
                            default=valid_def,
                            key=f"u_{u}_b_{bid}_chaps",
                            help=f"Select chapters in {book_lbl} for Unit {u}. Leave empty to exclude this book from Unit {u}.",
                        )
                        if set(sel_for_b) != set(valid_def):
                            manually_mapped.add(u)
                            auto_suggested_units.discard(u)
                        if sel_for_b:
                            unit_book_map[bid] = sel_for_b
                if unit_book_map:
                    updated_sources[u] = unit_book_map
                else:
                    if meta.get("status") == "unmapped" and u not in manually_mapped:
                        updated_sources[u] = None
                    elif len(books) > 1:
                        updated_sources[u] = None
                    else:
                        updated_sources[u] = {bid: None for bid in books}

            st.session_state.unit_sources = updated_sources
            source_warnings = validate_unit_sources(library, updated_sources)
            for sw in source_warnings:
                st.warning(f"⚠️ {sw}")

            # Problem 3.3: Warning when mapped chapters have low similarity to Course Outcomes
            if outcomes_valid:
                from config import CO_MAPPING_WARNING_THRESHOLD
                for u in active_units:
                    if updated_sources.get(u) and any(bool(v) for v in (updated_sources[u] or {}).values() if v is not None):
                        u_chunks = select_chunks_for_unit(raw_chunks, updated_sources, u)
                        if u_chunks:
                            co_sims = []
                            for c in u_chunks:
                                scores = c.get("co_scores", {})
                                if scores:
                                    co_sims.extend(scores.values())
                            if co_sims and max(co_sims) < CO_MAPPING_WARNING_THRESHOLD:
                                st.warning(f"⚠️ Mapped chapters for Unit {u} do not match the Course Outcomes well.")

        # Soft hint for unmapped units (strictly optional, does not block generation)
        unmapped_units = [
            u for u in active_units
            if (updated_sources.get(u) is None or not any(bool(v) for v in (updated_sources[u] or {}).values() if v is not None))
            and u not in manually_mapped
        ]
        if unmapped_units and len(books) > 1:
            st.info(
                f"💡 **Tip:** Units {', '.join(map(str, unmapped_units))} aren't mapped to textbook chapters. "
                f"Mapping them can improve how closely questions match your textbook. This is optional."
            )

    else:
        st.warning("⚠️ No chapter headings detected in document TOC. Falling back to page range selection.")
        all_pages = [c.get("page_start", 1) for c in raw_chunks] + [c.get("page_end", 1) for c in raw_chunks]
        min_p = min(all_pages) if all_pages else 1
        max_p = max(all_pages) if all_pages else min_p

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            from_page = st.number_input("From Page", min_value=1, max_value=max_p, value=min_p, step=1)
        with p_col2:
            to_page = st.number_input("To Page", min_value=int(from_page), max_value=max_p, value=max_p, step=1)

        page_range = (int(from_page), int(to_page))
        filtered_chunks = filter_chunks(raw_chunks, page_range=page_range)
        st.info(f"ℹ️ Selected pages {from_page}–{to_page} ({len(filtered_chunks)} chunks).")
        can_generate = len(filtered_chunks) > 0

    usable_chunks = [
        c for c in filtered_chunks
        if c.get("word_count", len(c.get("text", "").split())) >= config.MIN_CONTEXT_WORDS
    ]
    needed_questions = len(st.session_state.slots)
    if filtered_chunks and len(usable_chunks) < needed_questions:
        st.warning(
            f"⚠️ **Selection is small**: Selected filter has only {len(usable_chunks)} usable text chunk(s) "
            f"(with ≥{config.MIN_CONTEXT_WORDS} words), but the blueprint requires {needed_questions} questions. "
            f"Some questions might be duplicated or left unfilled."
        )

col_gen, col_gen_force, col_gen_msg = st.columns([2.5, 2.0, 2.5])
with col_gen:
    generate_disabled = (not can_generate) or (not outcomes_valid)
    gen_help_text = outcome_err_msg if not outcomes_valid else ("Select at least one chapter to enable." if not can_generate else "Generate exam questions")
    generate_clicked = st.button(
        "🚀 Generate Question Paper",
        type="primary",
        use_container_width=True,
        disabled=generate_disabled,
        help=gen_help_text,
    )
with col_gen_force:
    force_regenerate = st.checkbox(
        "⚡ Force Regenerate",
        value=False,
        help="Bypass generation cache and regenerate fresh questions from the LLM.",
    )
with col_gen_msg:
    if not outcomes_valid:
        st.caption(f"⚠️ {outcome_err_msg}")
    elif unmapped_units and len(books) > 1:
        st.caption("ℹ️ Unmapped units search across all textbook content.")

if generate_clicked:
    if not uploaded_pdfs:
        st.error("Please upload at least one textbook PDF before generating.")
    elif not outcomes_valid:
        st.error(outcome_err_msg)
    elif not st.session_state.slots:
        st.error("Please add at least one question slot in the blueprint.")
    elif not filtered_chunks:
        st.error("No text chunks match your chapter/section selection.")
    else:
        if not use_mock:
            probe_client = OllamaClient(model=model_name)
            if not probe_client.is_up():
                st.error("⚠️ **Ollama service is not reachable.** Please ensure Ollama is running (`ollama serve`) or switch to 'Mock Client' in the sidebar.")
                st.stop()

        with st.status("Executing Generation Pipeline...", expanded=True) as status_box:
            status_box.write(f"📄 Filtered to {len(filtered_chunks)} chunk(s) from selected chapter(s)...")

            # 2. Embedding & Scoring ONLY on filtered chunks
            status_box.write("🧠 Scoring filtered chunks against course outcomes...")
            outcomes_dict = {
                o["code"].strip(): o["desc"].strip()
                for o in st.session_state.outcomes
                if o["code"].strip()
            }
            outcomes_tuple = tuple(sorted(outcomes_dict.items()))
            scored_chunks = get_cached_scored_chunks(filtered_chunks, outcomes_tuple)
            status_box.write(f"✓ Computed semantic embeddings for {len(outcomes_dict)} outcome(s) across selected chapters.")

            # 3. Client initialization
            if use_mock:
                client = MockClient()
            else:
                client = OllamaClient(model=model_name)

            # 4. Generate by Slots (aligned with Revised Bloom's Taxonomy)
            status_box.write("✍️ Generating questions matching Revised Bloom's Taxonomy & blueprint slots...")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            total_slots = len(st.session_state.slots)
            progress_state = {"count": 0}

            def on_progress_callback(rec):
                progress_state["count"] += 1
                cnt = progress_state["count"]
                progress_bar.progress(min(cnt / max(total_slots, 1), 1.0))
                s_id = rec.get("slot_id") or rec.get("id") or f"Slot {cnt}"
                status_text.markdown(f"⏳ **Generated {cnt} of {total_slots}** question slots (completed `{s_id}`)...")

            unit_sources = st.session_state.get("unit_sources") or (st.session_state.scheme.get("unit_sources") if st.session_state.scheme else None)
            results = generate_paper_by_slots(
                chunks=scored_chunks,
                slots=st.session_state.slots,
                client=client,
                seed=42,
                on_progress=on_progress_callback,
                unit_sources=unit_sources,
                workers=config.GENERATION_WORKERS,
                force_regenerate=force_regenerate,
                keep_placeholders=True,
                outcome_descs=outcomes_dict,
                exclude_deprecated=exclude_deprecated,
            )
            progress_bar.progress(1.0)
            status_text.empty()

            # Map chapters specifically for paper's active units only (Problem 7.4)
            mapped_chaps_for_units = []
            if unit_sources:
                for u in active_units:
                    b_map = unit_sources.get(u) or unit_sources.get(str(u)) or {}
                    for b_id, chaps in b_map.items():
                        if chaps:
                            b_info = books.get(b_id, {})
                            b_title = b_info.get("title", b_id)
                            for ch in chaps:
                                mapped_chaps_for_units.append(f"{b_title} - {ch}")
            if not mapped_chaps_for_units:
                mapped_chaps_for_units = [q.get("chapter") for q in results.get("questions", []) if q.get("chapter")]

            results["covered_chapters"] = mapped_chaps_for_units
            st.session_state.results = results
            st.session_state["gen_result"] = results
            st.session_state["gen_ctx"] = {
                "scored_chunks": scored_chunks,
                "outcomes": outcomes_dict,
                "slots": list(st.session_state.slots),
                "blueprint": list(st.session_state.slots),
                "unit_sources": unit_sources,
                "weights": st.session_state.get("target_weights"),
                "client": client,
                "client_config": {"use_mock": use_mock, "model_name": model_name},
                "seed": 42,
                "retry_count": 0,
                "has_choice": preset_has_choice,
                "selected_chapters": list(selected_chapters) if selected_chapters else [],
                "active_units": list(active_units) if active_units else [],
                "exclude_deprecated": exclude_deprecated,
            }
            st.session_state["regen_count"] = 0
            status_box.update(label="Question paper generated successfully!", state="complete", expanded=False)

# -----------------------------------------------------------------------------
# Stage 4: Results Display & Export (Step 1.2, 2.1, 2.2)
# -----------------------------------------------------------------------------
if st.session_state.results:
    res = st.session_state.get("gen_result") or st.session_state.results
    questions = res.get("questions", [])
    unfilled = res.get("unfilled", [])
    raw_chaps = res.get("covered_chapters") or (st.session_state.get("gen_ctx", {}) or {}).get("selected_chapters") or selected_chapters
    covered_chaps = [clean_book_name(ch) for ch in raw_chaps] if raw_chaps else []
    paper_has_choice = (st.session_state.get("gen_ctx", {}) or {}).get("has_choice", preset_has_choice)

    st.subheader("📋 Generated Question Paper")

    if covered_chaps:
        st.markdown(f"**Generated from Chapters:** `{', '.join(covered_chaps)}`")

    # Problem 1.2: Completeness check and "Regenerate missing slots" button
    from generation.generator import regenerate_missing_slots, verify_paper_completeness
    is_complete, completeness_issues = verify_paper_completeness(st.session_state.slots, questions)
    has_unfilled_slots = bool(unfilled) or (not is_complete)
    if has_unfilled_slots:
        st.error(f"⚠️ **Paper Completeness Issues Detected ({len(completeness_issues)}):**")
        for iss in completeness_issues:
            st.markdown(f"- {iss}")
        col_reg1, col_reg2 = st.columns([2.5, 4.5])
        with col_reg1:
            if st.button("🔄 Regenerate missing slots", key="btn_regen_missing", use_container_width=True):
                if not st.session_state.get("gen_ctx") or not (st.session_state.get("gen_result") or st.session_state.results):
                    st.warning("⚠️ Generation context missing. Please click '🚀 Generate Question Paper' first.")
                else:
                    with st.spinner("🔄 Regenerating incomplete slots with expanded chunk pool..."):
                        regen_count = st.session_state.get("regen_count", 0) + 1
                        st.session_state["regen_count"] = regen_count
                        try:
                            current_gen_result = st.session_state.get("gen_result") or st.session_state.results
                            regen_res = regenerate_missing_slots(
                                ctx=st.session_state["gen_ctx"],
                                result=current_gen_result,
                                seed_bump=regen_count,
                            )
                            st.session_state["gen_result"] = regen_res
                            st.session_state.results = regen_res
                            if "gen_ctx" in st.session_state and isinstance(st.session_state["gen_ctx"], dict):
                                st.session_state["gen_ctx"]["retry_count"] = regen_count
                            st.rerun()
                        except Exception as ex:
                            st.error(f"Error regenerating missing slots: {ex}")

    # Problem 5.7: Faculty Review Panel for flagged slots
    flagged_slots = [q for q in questions if q.get("needs_review") or q.get("review_flags")]
    if flagged_slots:
        with st.expander(f"🚩 Review Flags ({len(flagged_slots)} items flagged)", expanded=True):
            st.caption("Faculty Review Panel: Automated quality validators flagged the following slots for review.")
            for fq in flagged_slots:
                f_id = fq.get("id", "Slot")
                f_flags = fq.get("review_flags", [fq.get("review_reason", "Review needed")])
                st.markdown(f"**{f_id}** ({fq.get('level')}, {fq.get('co')}): `{', '.join(f_flags)}`")
                if fq.get("ungrounded_sentences"):
                    st.caption("Flagged ungrounded statements:")
                    for us in fq["ungrounded_sentences"]:
                        st.markdown(f"> *{us}*")

            report_lines = ["QUESTION PAPER REVIEW REPORT", "=" * 30]
            for fq in flagged_slots:
                report_lines.append(f"Question: {fq.get('id')} ({fq.get('level')}, {fq.get('co')})")
                report_lines.append(f"Stem: {fq.get('question')}")
                report_lines.append(f"Flags: {', '.join(fq.get('review_flags', []))}")
                if fq.get("ungrounded_sentences"):
                    report_lines.append("Ungrounded sentences:")
                    for us in fq["ungrounded_sentences"]:
                        report_lines.append(f"  - {us}")
                report_lines.append("-" * 30)
            st.download_button(
                "📥 Download Review Report",
                data="\n".join(report_lines),
                file_name="paper_review_report.txt",
                mime="text/plain",
            )

    if unfilled:
        st.warning(f"⚠️ **{len(unfilled)} slot(s) could not be filled:**", icon="⚠️")
        for u in unfilled:
            u_id = u.get("id") or f"Slot {u.get('question_no', '')}"
            u_unit = f"Unit {u.get('unit')} " if u.get('unit') is not None else ""
            u_co = u.get("co", "")
            u_reason = u.get("reason") or u.get("review_reason") or "No suitable content or validation failed"
            u_hint = u.get("hint")
            hint_str = f" Try: *{u_hint}*" if u_hint else ""
            st.markdown(f"- **{u_unit}{u_id}** ({u_co}, {u.get('level', '')}, {u.get('marks', '')}m): {u_reason}.{hint_str}")

    if questions:
        table_data = [
            {
                "Unit": f"Unit {q.get('unit', 1)}",
                "S.No": q.get("id", ""),
                "Question": q.get("question", ""),
                "Marks": q.get("marks", ""),
                "BT (Cognitive)": q.get("level", ""),
                "Knowledge Dim": q.get("knowledge_dimension", "Conceptual"),
                "Outcome": q.get("co", ""),
                "Source Chapter": q.get("chapter", "N/A"),
                "Source Page": f"p. {q.get('page_start', '')}" if q.get("page_start") else "N/A",
            }
            for q in questions
        ]
        df = pd.DataFrame(table_data)

        tab_label = "📄 Formatted Question Paper (Unit & Choice Layout)" if paper_has_choice else "📄 Formatted Question Paper (Unit Layout)"
        tab_fmt, tab_tbl = st.tabs([tab_label, "📊 Detailed Table View"])
        with tab_fmt:
            paper_hdr = "##### 📝 Exam Paper Layout (Internal Choice OR Format)" if paper_has_choice else "##### 📝 Exam Paper Layout"
            st.markdown(paper_hdr)
            st.code(format_paper(questions, has_choice=paper_has_choice), language="markdown")

            st.write("---")
            st.markdown("##### 📋 Question Details, Marking Schemes & Model Answers")
            for q in questions:
                q_id = q.get("id", "")
                q_txt = q.get("question", "")
                q_marks = q.get("marks", "")
                q_lvl = q.get("level", "")
                q_co = q.get("co", "")
                st.markdown(f"**Question {q_id}** [{q_marks} Marks] *({q_lvl}, {q_co})*: {q_txt}")

                scheme = q.get("marking_scheme")
                is_fb = scheme.get("is_fallback") if isinstance(scheme, dict) else False
                badge = " *(auto-generated structure; review before use)*" if is_fb else ""

                with st.expander(f"📋 Marking scheme: Question {q_id}{badge}", expanded=False):
                    if scheme and isinstance(scheme, dict) and scheme.get("criteria"):
                        rows = []
                        for c in scheme["criteria"]:
                            exp_text = str(c.get("expected", ""))
                            alts = c.get("alternatives")
                            if alts and isinstance(alts, list) and len(alts) > 0:
                                exp_text += "\n\n*OR (Acceptable Alternatives):*\n" + "\n".join(f"- {a}" for a in alts)
                            rows.append({
                                "No": c.get("no", ""),
                                "Criterion": c.get("criterion", ""),
                                "What the examiner looks for": exp_text,
                                "Marks": c.get("marks", ""),
                                "Reason": c.get("reason", ""),
                            })
                        tot = scheme.get("total_marks", sum(r["Marks"] for r in rows if isinstance(r["Marks"], (int, float))))
                        rows.append({
                            "No": "",
                            "Criterion": "**Total**",
                            "What the examiner looks for": "",
                            "Marks": f"**{tot}**",
                            "Reason": "",
                        })
                        df_scheme = pd.DataFrame(rows)
                        st.dataframe(df_scheme, hide_index=True, use_container_width=True)
                        if scheme.get("notes"):
                            st.caption(f"**Examiner Notes:** {scheme['notes']}")
                        if is_fb:
                            st.caption("⚠️ * This marking scheme was auto-generated from rule-based templates. Please review before use.")
                    else:
                        st.info("Marking scheme not available.")

                with st.expander(f"💡 Model answer: Question {q_id}", expanded=False):
                    ans = q.get("answer", "Model answer pending evaluation.")
                    st.markdown(ans)
        with tab_tbl:
            st.dataframe(
                df,
                column_config={
                    "Unit": st.column_config.TextColumn("Unit", width="small"),
                    "S.No": st.column_config.TextColumn("S.No", width="small"),
                    "Question": st.column_config.TextColumn("Question Text", width="large"),
                    "Marks": st.column_config.NumberColumn("Marks", width="small"),
                    "BT (Cognitive)": st.column_config.TextColumn("BT Level", width="small"),
                    "Knowledge Dim": st.column_config.TextColumn("Knowledge (KD)", width="medium"),
                    "Outcome": st.column_config.TextColumn("Course Outcome", width="small"),
                    "Source Chapter": st.column_config.TextColumn("Source Chapter", width="medium"),
                    "Source Page": st.column_config.TextColumn("Page", width="small"),
                },
                hide_index=True,
                use_container_width=True,
            )

        # Substep 1.2 & Step 2: Export Controls & Cumulative Totals Checkbox
        st.write("#### 📥 Export Formatted Question Paper & Evaluation Scheme")
        course_title = getattr(st.session_state, "course_title", "Course Examination")
        outcomes_map = {
            o["code"].strip(): o["desc"].strip()
            for o in st.session_state.outcomes
            if o["code"].strip()
        }

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            # Substep 1.2: Checkbox: "Add cumulative CO/BT totals"
            add_totals = st.checkbox(
                "Add cumulative CO/BT totals",
                value=True,
                help="Append cumulative Course Outcome (CO) and Bloom's Taxonomy (BT) distribution tables to DOCX, PDF, and UI summary.",
            )
            include_marking_scheme = st.checkbox(
                "Include marking scheme in export",
                value=True,
                help="Include examiner marking scheme criteria breakdown tables in DOCX and PDF papers.",
            )
        with col_opt2:
            ans_help = (
                "Include verified model answers (with OR alternative format) in DOCX and PDF papers."
                if paper_has_choice else
                "Include verified model answers in DOCX and PDF papers."
            )
            include_answers = st.checkbox(
                "Include Model Answers in export",
                value=True,
                help=ans_help,
            )
            include_reasons = st.checkbox(
                "Include reasons",
                value=True,
                help="Include pedagogical justification / examiner reasoning column in the marking scheme table.",
            )

        # Calculate totals summary if requested
        totals_summary = None
        if add_totals:
            totals_summary = summarize(
                questions,
                targets=st.session_state.get("target_weights"),
                attempted_only=True,
            )
            with st.expander("📊 View Cumulative CO / BT Distribution Totals", expanded=False):
                if paper_has_choice:
                    st.markdown(
                        f"**Total Attempted Marks:** `{totals_summary.get('total_marks', 0)}` | "
                        f"**Total Offered Marks:** `{totals_summary.get('offered_marks', 0)}`"
                    )
                else:
                    st.markdown(
                        f"**Total Marks:** `{totals_summary.get('total_marks', 0)}`"
                    )
                tbl_rows = totals_table_rows(totals_summary)
                co_tbl = tbl_rows.get("co_table", [])
                bt_tbl = tbl_rows.get("bt_table", [])
                kd_tbl = tbl_rows.get("kd_table", [])

                c_co, c_bt, c_kd = st.columns(3)
                with c_co:
                    st.markdown("**Course Outcomes (CO) Distribution:**")
                    if co_tbl:
                        df_co = pd.DataFrame(co_tbl[1:], columns=co_tbl[0])
                        st.dataframe(df_co, hide_index=True, use_container_width=True)
                with c_bt:
                    st.markdown("**Bloom's Taxonomy (BT) Distribution:**")
                    if bt_tbl:
                        df_bt = pd.DataFrame(bt_tbl[1:], columns=bt_tbl[0])
                        st.dataframe(df_bt, hide_index=True, use_container_width=True)
                with c_kd:
                    st.markdown("**Knowledge Dimension (KD) Distribution:**")
                    if kd_tbl:
                        df_kd = pd.DataFrame(kd_tbl[1:], columns=kd_tbl[0])
                        st.dataframe(df_kd, hide_index=True, use_container_width=True)

        outcomes_tuple = tuple(sorted(outcomes_map.items()))
        chaps_tuple = tuple(covered_chaps) if covered_chaps else None

        # Student Copy (Clean, Exam-Ready, no sources, no solutions)
        student_docx = get_cached_docx_paper(
            questions_data=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes_tuple=outcomes_tuple,
            institution_name=st.session_state.institution_name,
            covered_chaps_tuple=chaps_tuple,
            include_answers=False,
            include_totals=False,
            totals_summary_data=totals_summary,
            profile="student",
        )
        student_pdf = get_cached_pdf_paper(
            questions_data=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes_tuple=outcomes_tuple,
            institution_name=st.session_state.institution_name,
            covered_chaps_tuple=chaps_tuple,
            include_answers=False,
            include_totals=False,
            totals_summary_data=totals_summary,
            profile="student",
        )

        # Faculty Copy (Includes solutions, sources, QA flags, KD/CO/BT totals)
        faculty_docx = get_cached_docx_paper(
            questions_data=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes_tuple=outcomes_tuple,
            institution_name=st.session_state.institution_name,
            covered_chaps_tuple=chaps_tuple,
            include_answers=include_answers,
            include_totals=True,
            include_marking_scheme=include_marking_scheme,
            include_reasons=include_reasons,
            totals_summary_data=totals_summary,
            profile="faculty",
            include_qa_flags=True,
        )
        faculty_pdf = get_cached_pdf_paper(
            questions_data=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes_tuple=outcomes_tuple,
            institution_name=st.session_state.institution_name,
            covered_chaps_tuple=chaps_tuple,
            include_answers=include_answers,
            include_totals=True,
            include_marking_scheme=include_marking_scheme,
            include_reasons=include_reasons,
            totals_summary_data=totals_summary,
            profile="faculty",
            include_qa_flags=True,
        )
        csv_data = df.to_csv(index=False).encode("utf-8")

        has_placeholders = any(
            q.get("needs_review") or str(q.get("question", "")).startswith("[NEEDS REVIEW")
            for q in questions
        )
        can_download = True
        if has_placeholders:
            st.warning("⚠️ **Warning**: This paper contains incomplete placeholder slots marked for review. Please review or regenerate them before exporting.")
            can_download = st.checkbox(
                "I acknowledge some slots are placeholders and wish to export anyway",
                value=False,
                key="confirm_export_incomplete",
            )
            if not can_download:
                st.info("ℹ️ Check the box above to enable downloading this paper.")

        tab_student, tab_faculty = st.tabs([
            "🎓 Student Copy (Exam-Ready)",
            "👨‍🏫 Faculty Copy (With Solutions & QA Flags)",
        ])

        with tab_student:
            st.info("📌 **Student Copy:** Formal university exam layout with USN roll-number boxes, course code, time & marks instructions. Syllabus dump, source citations, answers, and internal analytics are omitted.")
            col_s_docx, col_s_pdf = st.columns(2)
            with col_s_docx:
                st.download_button(
                    label="📄 Download Student Paper (.docx)",
                    data=student_docx,
                    file_name=f"{course_title.replace(' ', '_')}_Student_Exam_Paper.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary",
                    disabled=not can_download,
                    use_container_width=True,
                    key="dl_student_docx",
                )
            with col_s_pdf:
                st.download_button(
                    label="📑 Download Student Paper (.pdf)",
                    data=student_pdf,
                    file_name=f"{course_title.replace(' ', '_')}_Student_Exam_Paper.pdf",
                    mime="application/pdf",
                    type="primary",
                    disabled=not can_download,
                    use_container_width=True,
                    key="dl_student_pdf",
                )

        with tab_faculty:
            st.info("📌 **Faculty Copy:** Comprehensive master copy featuring mapped textbook sources, QA validation flags, itemized marking schemes & model solutions, and cumulative CO/BT/KD distributions.")
            col_f_docx, col_f_pdf, col_f_csv = st.columns([2.5, 2.5, 2])
            with col_f_docx:
                st.download_button(
                    label="📄 Download Faculty Master (.docx)",
                    data=faculty_docx,
                    file_name=f"{course_title.replace(' ', '_')}_Faculty_Master_Paper.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary",
                    disabled=not can_download,
                    use_container_width=True,
                    key="dl_faculty_docx",
                )
            with col_f_pdf:
                st.download_button(
                    label="📑 Download Faculty Master (.pdf)",
                    data=faculty_pdf,
                    file_name=f"{course_title.replace(' ', '_')}_Faculty_Master_Paper.pdf",
                    mime="application/pdf",
                    type="primary",
                    disabled=not can_download,
                    use_container_width=True,
                    key="dl_faculty_pdf",
                )
            with col_f_csv:
                st.download_button(
                    label="📊 Download Evaluation Data (.csv)",
                    data=csv_data,
                    file_name=f"{course_title.replace(' ', '_')}_Exam_Paper.csv",
                    mime="text/csv",
                    disabled=not can_download,
                    use_container_width=True,
                    key="dl_faculty_csv",
                )

        with st.expander("🔍 View Technical Question Metadata & Evaluation Details"):
            st.json(questions)
    else:
        st.error("No questions could be generated. Please check your uploaded PDF, chapter selection, and outcomes.")
