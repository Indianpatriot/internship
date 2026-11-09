"""Streamlit UI for Outcome-Aware Question Paper Generation.

Zero-cost, local RAG question paper generation mapped to Course Outcomes (CO/LO)
and Revised Bloom's Taxonomy (Cognitive Process Dimension L1-L6 & Knowledge Dimension).
"""
import os
import re
import tempfile
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
from export_utils import create_docx_exam_paper, create_pdf_exam_paper
from generation.mock_client import MockClient
from generation.ollama_client import OllamaClient
from generation.generator import generate_paper_by_slots
from ingestion.pipeline import ingest
from retrieval.co_mapper import score_chunks_against_outcomes, _m
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

if "results" not in st.session_state:
    st.session_state.results = None

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
# Section 1: Course Outcomes (Dynamic CRUD)
# -----------------------------------------------------------------------------
st.subheader("1. Course / Learning Outcomes")
st.caption("Define the outcomes specific to your syllabus. Click '+ Add outcome' to append more.")

if st.session_state.get("scheme_warning"):
    st.warning(st.session_state.scheme_warning)

total_outcomes = len(st.session_state.outcomes)
can_delete = total_outcomes > config.MIN_COS
can_add = total_outcomes < config.MAX_COS

outcomes_to_remove = []
for i, oc in enumerate(st.session_state.outcomes):
    col_code, col_desc, col_del = st.columns([1.2, 5, 0.8])
    with col_code:
        new_code = st.text_input(
            "Code",
            value=oc["code"],
            key=f"outcome_code_{i}",
            label_visibility="collapsed" if i > 0 else "visible",
            placeholder="e.g. CO1",
        )
        oc["code"] = new_code
    with col_desc:
        new_desc = st.text_input(
            "Outcome Description",
            value=oc["desc"],
            key=f"outcome_desc_{i}",
            label_visibility="collapsed" if i > 0 else "visible",
            placeholder="Describe what students should be able to do...",
        )
        oc["desc"] = new_desc
    with col_del:
        if i == 0:
            st.caption("Action")
        if st.button("🗑️", key=f"del_outcome_{i}", help="Delete outcome", disabled=not can_delete):
            outcomes_to_remove.append(i)

if outcomes_to_remove:
    for idx in sorted(outcomes_to_remove, reverse=True):
        st.session_state.outcomes.pop(idx)
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
        st.rerun()
    if not can_add:
        st.caption(f"Maximum {config.MAX_COS} outcomes")

with col_cap:
    if not can_delete:
        st.caption(f"Minimum {config.MIN_COS} outcomes")


st.divider()

# -----------------------------------------------------------------------------
# Section 2: Question Blueprint Slots (Revised Bloom's Taxonomy 2D)
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

    equal_pct = max(5, int(100 / max(len(current_cos), 1)))
    weights = {}
    for c in current_cos:
        val = st.session_state.get(f"w_co_{c}", equal_pct)
        st.session_state[f"w_co_{c}"] = val
        weights[c] = val

    tot = max(sum(weights.values()), 1)
    return {c: v / tot for c, v in weights.items()}

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

is_100_choice = "100" in selected_preset
target_unit_marks = 20 if is_100_choice else (10 if "50" in selected_preset else 20)
num_preset_units = 5

# Reset sub-question marks defaults if target_unit_marks changes
prev_target_marks = st.session_state.get("_prev_target_unit_marks")
if prev_target_marks != target_unit_marks:
    st.session_state._prev_target_unit_marks = target_unit_marks
    for u in range(1, num_preset_units + 1):
        if f"unit_{u}_marks_input" in st.session_state:
            del st.session_state[f"unit_{u}_marks_input"]

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

with st.expander("📐 Configure Questions & Sub-questions Per Unit", expanded=is_100_choice):
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
            st.caption(f"{target_unit_marks}m / option")

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
        w_l12 = st.slider("L1–L2 (Remember / Understand) %", min_value=10, max_value=60, value=int(st.session_state.get("w_l12", 40)), step=5)
        w_l34 = st.slider("L3–L4 (Apply / Analyze) %", min_value=10, max_value=60, value=int(st.session_state.get("w_l34", 40)), step=5)
        w_l56 = st.slider("L5–L6 (Evaluate / Create) %", min_value=5, max_value=40, value=int(st.session_state.get("w_l56", 20)), step=5)
        st.session_state.w_l12 = w_l12
        st.session_state.w_l34 = w_l34
        st.session_state.w_l56 = w_l56

        tot_bt_w = w_l12 + w_l34 + w_l56
        if tot_bt_w != 100:
            st.caption(f"Current BT total: {tot_bt_w}% (will be normalized proportionally)")

    with col_co_w:
        st.markdown("**Course Outcomes (CO) Weightage (%)**")
        st.caption("Allocate percentage emphasis across syllabus Course Outcomes.")
        co_weights_input = {}
        equal_pct = max(5, int(100 / max(len(available_cos), 1)))
        for c in available_cos:
            saved_val = st.session_state.get(f"w_co_{c}", equal_pct)
            co_weights_input[c] = st.slider(f"{c} Weightage %", min_value=5, max_value=100, value=int(saved_val), step=5, key=f"slider_co_{c}")
            st.session_state[f"w_co_{c}"] = co_weights_input[c]

        tot_co_w = sum(co_weights_input.values())
        if tot_co_w != 100:
            st.caption(f"Current CO total: {tot_co_w}% (will be normalized proportionally)")

# Show any blueprint warnings
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
        tot_bt_val = max(w_l12 + w_l34 + w_l56, 1)
        eff_bt_weights = {
            "L1-L2": w_l12 / tot_bt_val,
            "L3-L4": w_l34 / tot_bt_val,
            "L5-L6": w_l56 / tot_bt_val,
        }
        eff_co_weights = rebuild_co_weights(valid_cos)
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
                    options=2 if (is_100_choice or "Custom" in selected_preset) else 1,
                    sub_marks=unit_sub_marks,
                )
                bp = build_blueprint(
                    preset=preset_obj,
                    bt_weights=eff_bt_weights,
                    co_weights=eff_co_weights,
                    outcomes=valid_cos,
                    seed=42,
                )
                st.session_state.slots = bp["slots"]
                st.session_state.selected_preset_idx = 1 if is_100_choice else (0 if "50" in selected_preset else 3)
                st.session_state.bp_warnings = bp.get("warnings", [])
            st.rerun()
        except ValueError as e:
            st.error(str(e))


with col_p2:
    if st.button("🎓 Standard 50-Mark Blueprint", use_container_width=True, disabled=not outcomes_valid):
        st.session_state.slots = get_50_mark_blueprint(valid_cos)
        st.session_state.selected_preset_idx = 0
        st.session_state.bp_warnings = []
        st.rerun()

with col_p3:
    if st.button("🔄 Reset to Blank", use_container_width=True):
        st.session_state.slots = [
            {
                "part": "Part A",
                "id": "1",
                "co": available_cos[0],
                "level": "L1",
                "knowledge_dimension": "Factual",
                "marks": 2,
            }
        ]
        st.session_state.selected_preset_idx = 3
        st.rerun()

# Calculate Blueprint Statistics
stats = calculate_blueprint_stats(st.session_state.slots)
total_marks = stats["total_marks"]
total_questions = stats["total_questions"]
part_marks = stats["part_marks"]

badge_class = "badge-success" if total_marks == 50 else "badge-warning"
mark_target_label = "✅ Target 50 Marks Met" if total_marks == 50 else f"Target: 50 Marks (Current: {total_marks})"

st.markdown(
    f"""
    <div style="margin-top: 0.6rem; margin-bottom: 0.8rem;">
        <span class="metric-badge {badge_class}"><b>Total Marks:</b> {total_marks} ({mark_target_label})</span>
        <span class="metric-badge"><b>Total Slots:</b> {total_questions}</span>
        <span class="metric-badge"><b>Part A:</b> {part_marks.get('Part A', 0)} Marks</span>
        <span class="metric-badge"><b>Part B:</b> {part_marks.get('Part B', 0)} Marks</span>
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
hdr_col1, hdr_col2, hdr_col3, hdr_col4, hdr_col5, hdr_col6, hdr_col7 = st.columns([1.3, 1.0, 1.4, 2.0, 1.8, 1.0, 0.6])
with hdr_col1:
    st.caption("Section / Part")
with hdr_col2:
    st.caption("Q.No / Subpart")
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

slots_to_remove = []
for j, slot in enumerate(st.session_state.slots):
    col_part, col_id, col_co, col_bt, col_kd, col_marks, col_del_slot = st.columns([1.3, 1.0, 1.4, 2.0, 1.8, 1.0, 0.6])

    with col_part:
        part_options = ["Part A", "Part B", "Part C"]
        curr_part = slot.get("part", "Part A")
        part_idx = part_options.index(curr_part) if curr_part in part_options else 0
        slot["part"] = st.selectbox(
            "Section",
            options=part_options,
            index=part_idx,
            key=f"slot_part_{j}",
            label_visibility="collapsed",
        )

    with col_id:
        slot["id"] = st.text_input(
            "Q.No",
            value=str(slot.get("id", f"{j+1}")),
            key=f"slot_id_{j}",
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
            key=f"slot_co_{j}",
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
            key=f"slot_bt_{j}",
            label_visibility="collapsed",
        )

    with col_kd:
        curr_kd = normalize_knowledge_dimension(slot.get("knowledge_dimension") or slot.get("kd"))
        kd_index = kd_options.index(curr_kd) if curr_kd in kd_options else 1
        slot["knowledge_dimension"] = st.selectbox(
            "Knowledge Dimension",
            options=kd_options,
            index=kd_index,
            key=f"slot_kd_{j}",
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
                key=f"slot_marks_{j}",
                label_visibility="collapsed",
            )
        )

    with col_del_slot:
        if st.button("🗑️", key=f"del_slot_{j}", help="Delete question slot"):
            slots_to_remove.append(j)

if slots_to_remove:
    for idx in sorted(slots_to_remove, reverse=True):
        st.session_state.slots.pop(idx)
    st.rerun()

col_add_q, _ = st.columns([2, 5])
with col_add_q:
    if st.button("➕ Add question slot", key="btn_add_slot"):
        st.session_state.slots.append({
            "part": "Part B",
            "id": f"{len(st.session_state.slots) + 1}(a)",
            "co": available_cos[0],
            "level": "L3",
            "knowledge_dimension": "Procedural",
            "marks": 5,
        })
        st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Section 3: File Upload & Chapter Selection (Task Steps 1, 2, 3, 4)
# -----------------------------------------------------------------------------
st.subheader("3. Document Ingestion & Chapter Selection")

uploaded_pdf = st.file_uploader(
    "Upload Textbook PDF",
    type=["pdf"],
    help="Upload the reference textbook or course material in PDF format.",
)

filtered_chunks = []
selected_chapters = []
selected_sections = None
page_range = None
can_generate = False

if uploaded_pdf:
    file_bytes = uploaded_pdf.getvalue()
    raw_chunks = get_cached_chunks(file_bytes, uploaded_pdf.name)
    st.session_state.course_title = os.path.splitext(uploaded_pdf.name)[0].replace("-", " ").replace("_", " ").title()

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

col_gen, col_gen_msg = st.columns([2.5, 4.5])
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
with col_gen_msg:
    if not outcomes_valid:
        st.caption(f"⚠️ {outcome_err_msg}")

if generate_clicked:
    if not uploaded_pdf:
        st.error("Please upload a textbook PDF before generating.")
    elif not outcomes_valid:
        st.error(outcome_err_msg)
    elif not st.session_state.slots:
        st.error("Please add at least one question slot in the blueprint.")
    elif not filtered_chunks:
        st.error("No text chunks match your chapter/section selection.")
    else:
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
            total_slots = len(st.session_state.slots)
            progress_state = {"count": 0}

            def on_progress_callback(rec):
                progress_state["count"] += 1
                progress_bar.progress(min(progress_state["count"] / max(total_slots, 1), 1.0))

            unit_sources = st.session_state.scheme.get("unit_sources") if st.session_state.scheme else None
            results = generate_paper_by_slots(
                chunks=scored_chunks,
                slots=st.session_state.slots,
                client=client,
                seed=42,
                on_progress=on_progress_callback,
                unit_sources=unit_sources,
            )
            progress_bar.progress(1.0)
            results["covered_chapters"] = selected_chapters
            st.session_state.results = results
            status_box.update(label="Question paper generated successfully!", state="complete", expanded=False)

# -----------------------------------------------------------------------------
# Section 4: Results Display & Export (Step 1.2, 2.1, 2.2)
# -----------------------------------------------------------------------------
if st.session_state.results:
    res = st.session_state.results
    questions = res.get("questions", [])
    unfilled = res.get("unfilled", [])
    covered_chaps = res.get("covered_chapters", selected_chapters)

    st.subheader("📋 Generated Question Paper")

    if covered_chaps:
        st.markdown(f"**Generated from Chapters:** `{', '.join(covered_chaps)}`")

    if unfilled:
        unfilled_details = ", ".join(
            f"{u.get('co')} ({u.get('level')}, {u.get('knowledge_dimension', 'Conceptual')}, {u.get('marks')}m)" for u in unfilled
        )
        st.warning(
            f"⚠️ Could not find suitable text or valid questions for {len(unfilled)} slot(s): "
            f"**{unfilled_details}**.<br/>"
            f"💡 *Note: These Course Outcomes (COs) might belong to textbook chapters that were excluded in your chapter selection.*",
            icon="⚠️",
        )

    if questions:
        table_data = [
            {
                "Section": q.get("part", "General"),
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

        tab_fmt, tab_tbl = st.tabs(["📄 Formatted Question Paper (Unit & Choice Layout)", "📊 Detailed Table View"])
        with tab_fmt:
            st.markdown("##### 📝 Exam Paper Layout (Internal Choice OR Format)")
            st.code(format_paper(questions), language="markdown")
        with tab_tbl:
            st.dataframe(
                df,
                column_config={
                    "Section": st.column_config.TextColumn("Section", width="small"),
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
        with col_opt2:
            include_answers = st.checkbox(
                "Include Model Answers & Scheme of Evaluation",
                value=True,
                help="Include verified model answers (with OR alternative format) in DOCX and PDF papers.",
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
                st.markdown(
                    f"**Total Attempted Marks:** `{totals_summary.get('total_marks', 0)}` | "
                    f"**Total Offered Marks:** `{totals_summary.get('offered_marks', 0)}`"
                )
                tbl_rows = totals_table_rows(totals_summary)
                co_tbl = tbl_rows.get("co_table", [])
                bt_tbl = tbl_rows.get("bt_table", [])

                c_co, c_bt = st.columns(2)
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

        docx_data = create_docx_exam_paper(
            questions=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes=outcomes_map,
            institution_name=st.session_state.institution_name,
            covered_chapters=covered_chaps,
            include_source=True,
            include_knowledge_dim=True,
            include_answers=include_answers,
            include_totals=add_totals,
            totals_summary=totals_summary,
        )
        pdf_data = create_pdf_exam_paper(
            questions=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes=outcomes_map,
            institution_name=st.session_state.institution_name,
            covered_chapters=covered_chaps,
            include_source=True,
            include_knowledge_dim=True,
            include_answers=include_answers,
            include_totals=add_totals,
            totals_summary=totals_summary,
        )
        csv_data = df.to_csv(index=False).encode("utf-8")

        col_docx, col_pdf, col_csv = st.columns([2.5, 2.5, 2])
        with col_docx:
            st.download_button(
                label="📄 Download as Word (.docx)",
                data=docx_data,
                file_name=f"{course_title.replace(' ', '_')}_Exam_Paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
            )
        with col_pdf:
            st.download_button(
                label="📑 Download as PDF (.pdf)",
                data=pdf_data,
                file_name=f"{course_title.replace(' ', '_')}_Exam_Paper.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        with col_csv:
            st.download_button(
                label="📊 Download as CSV (.csv)",
                data=csv_data,
                file_name=f"{course_title.replace(' ', '_')}_Exam_Paper.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("🔍 View Technical Question Metadata & Evaluation Details"):
            st.json(questions)
    else:
        st.error("No questions could be generated. Please check your uploaded PDF, chapter selection, and outcomes.")
