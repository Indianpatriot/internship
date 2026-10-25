"""Streamlit UI for Outcome-Aware Question Paper Generation.

Zero-cost, local RAG question paper generation mapped to Course Outcomes (CO/LO)
and Bloom's Taxonomy (BT) levels.
"""
import os
import re
import tempfile
import pandas as pd
import streamlit as st

import config
from export_utils import create_docx_exam_paper, create_pdf_exam_paper
from generation.mock_client import MockClient
from generation.ollama_client import OllamaClient
from generation.generator import generate_paper_by_slots
from ingestion.pipeline import ingest
from retrieval.co_mapper import score_chunks_against_outcomes, _m

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Outcome-Aware Exam Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
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
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        background: #f1f5f9;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
        display: inline-block;
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
    ]

if "slots" not in st.session_state:
    st.session_state.slots = [
        {"co": "CO1", "level": "L1", "marks": 5},
        {"co": "CO1", "level": "L2", "marks": 5},
        {"co": "CO2", "level": "L3", "marks": 10},
        {"co": "CO2", "level": "L4", "marks": 10},
    ]

if "results" not in st.session_state:
    st.session_state.results = None


# -----------------------------------------------------------------------------
# Sidebar: LLM Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Generator Settings")
    st.markdown("Configure the LLM generation client and model parameters.")

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
        st.warning("⚠️ **Mock Client Active**: Questions are generated heuristically offline without an LLM. Toggle OFF to use Ollama for AI generation.")
    else:
        active_client = OllamaClient(model=model_name)
        if active_client.is_up():
            st.success(f"🟢 **Ollama Connected**: `{model_name}` ({config.OLLAMA_HOST})")
        else:
            st.error(f"🔴 Ollama server not reachable at `{config.OLLAMA_HOST}`. Start it with `ollama serve`.")

    st.divider()
    st.markdown(
        """
        **Pipeline Information:**
        - **Model:** `BAAI/bge-small-en-v1.5`
        - **Embedding Mode:** Normalized Cosine Similarity
        - **Bloom's Levels:** L1 to L5
        - **Constraint:** Zero external API cost
        """
    )


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Outcome-Aware Question Paper Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a textbook PDF, define custom Course/Learning Outcomes (CO/LO), '
    'and generate a balanced exam paper mapped to Bloom\'s Taxonomy.</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Section 1: Course Outcomes (Dynamic CRUD)
# -----------------------------------------------------------------------------
st.subheader("1. Course / Learning Outcomes")
st.caption("Define the outcomes specific to your syllabus. Click '+ Add outcome' to append more.")

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
        if st.button("🗑️", key=f"del_outcome_{i}", help="Delete outcome"):
            outcomes_to_remove.append(i)

if outcomes_to_remove:
    for idx in sorted(outcomes_to_remove, reverse=True):
        st.session_state.outcomes.pop(idx)
    st.rerun()

col_btn, _ = st.columns([2, 5])
with col_btn:
    if st.button("➕ Add outcome", key="btn_add_outcome"):
        # Auto-detect next CO code number
        existing_nums = [
            int(m.group(1))
            for o in st.session_state.outcomes
            if (m := re.match(r"^CO(\d+)$", o["code"].strip(), re.I))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else len(st.session_state.outcomes) + 1
        st.session_state.outcomes.append({"code": f"CO{next_num}", "desc": ""})
        st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Section 2: Question Blueprint Slots
# -----------------------------------------------------------------------------
st.subheader("2. Question Paper Blueprint (Slots)")
st.caption("Specify questions with target Bloom's Taxonomy level, Course Outcome, and marks.")

# Extract current outcome codes
available_cos = [o["code"].strip() for o in st.session_state.outcomes if o["code"].strip()]
if not available_cos:
    available_cos = ["CO1"]

total_marks = sum(int(s.get("marks", 0)) for s in st.session_state.slots)
total_questions = len(st.session_state.slots)

m_col1, m_col2 = st.columns([1, 4])
with m_col1:
    st.markdown(f'<div class="metric-badge">Total Marks: {total_marks}</div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="metric-badge">Questions: {total_questions}</div>', unsafe_allow_html=True)

st.write("")

bt_options = ["L1", "L2", "L3", "L4", "L5"]
bt_labels = {
    "L1": "L1 - Remember",
    "L2": "L2 - Understand",
    "L3": "L3 - Apply",
    "L4": "L4 - Analyze",
    "L5": "L5 - Evaluate",
}

slots_to_remove = []
for j, slot in enumerate(st.session_state.slots):
    col_co, col_bt, col_marks, col_del_slot = st.columns([2, 2.5, 1.5, 0.8])

    # Choose default index for outcome
    co_val = slot.get("co", available_cos[0])
    co_index = available_cos.index(co_val) if co_val in available_cos else 0

    with col_co:
        sel_co = st.selectbox(
            "Course Outcome",
            options=available_cos,
            index=co_index,
            key=f"slot_co_{j}",
            label_visibility="collapsed" if j > 0 else "visible",
        )
        slot["co"] = sel_co

    with col_bt:
        current_lvl = slot.get("level", "L1")
        lvl_index = bt_options.index(current_lvl) if current_lvl in bt_options else 0
        sel_bt = st.selectbox(
            "Bloom's Level",
            options=bt_options,
            format_func=lambda x: bt_labels.get(x, x),
            index=lvl_index,
            key=f"slot_bt_{j}",
            label_visibility="collapsed" if j > 0 else "visible",
        )
        slot["level"] = sel_bt

    with col_marks:
        sel_marks = st.number_input(
            "Marks",
            min_value=1,
            max_value=100,
            value=int(slot.get("marks", 5)),
            step=1,
            key=f"slot_marks_{j}",
            label_visibility="collapsed" if j > 0 else "visible",
        )
        slot["marks"] = int(sel_marks)

    with col_del_slot:
        if j == 0:
            st.caption("Action")
        if st.button("🗑️", key=f"del_slot_{j}", help="Delete question slot"):
            slots_to_remove.append(j)

if slots_to_remove:
    for idx in sorted(slots_to_remove, reverse=True):
        st.session_state.slots.pop(idx)
    st.rerun()

col_add_q, _ = st.columns([2, 5])
with col_add_q:
    if st.button("➕ Add question", key="btn_add_slot"):
        st.session_state.slots.append({
            "co": available_cos[0],
            "level": "L1",
            "marks": 5,
        })
        st.rerun()

st.divider()

# -----------------------------------------------------------------------------
# Section 3: File Upload & Generation Execution
# -----------------------------------------------------------------------------
st.subheader("3. Document Ingestion & Generation")

uploaded_pdf = st.file_uploader(
    "Upload Textbook PDF",
    type=["pdf"],
    help="Upload the reference textbook or course material in PDF format.",
)

col_gen, _ = st.columns([2, 5])
with col_gen:
    generate_clicked = st.button("🚀 Generate Question Paper", type="primary", use_container_width=True)

if generate_clicked:
    # Form Validation
    if not uploaded_pdf:
        st.error("Please upload a textbook PDF before generating.")
    elif not st.session_state.outcomes or not any(o["desc"].strip() for o in st.session_state.outcomes):
        st.error("Please provide at least one Course Outcome with a description.")
    elif not st.session_state.slots:
        st.error("Please add at least one question slot in the blueprint.")
    else:
        with st.status("Executing Generation Pipeline...", expanded=True) as status_box:
            # 1. Ingest PDF
            status_box.write("📄 Ingesting and chunking PDF document...")
            file_bytes = uploaded_pdf.getvalue()
            chunks = get_cached_chunks(file_bytes, uploaded_pdf.name)
            st.session_state.course_title = os.path.splitext(uploaded_pdf.name)[0].replace("-", " ").replace("_", " ").title()
            status_box.write(f"✓ Ingested {len(chunks)} chunks from `{uploaded_pdf.name}`.")

            # 2. Embedding & Scoring
            status_box.write("🧠 Scoring chunks against course outcomes...")
            outcomes_dict = {
                o["code"].strip(): o["desc"].strip()
                for o in st.session_state.outcomes
                if o["code"].strip()
            }
            # Convert to tuple of tuples for cache hashing
            outcomes_tuple = tuple(sorted(outcomes_dict.items()))
            scored_chunks = get_cached_scored_chunks(chunks, outcomes_tuple)
            status_box.write(f"✓ Computed semantic embeddings for {len(outcomes_dict)} outcome(s).")

            # 3. Client initialization
            if use_mock:
                client = MockClient()
            else:
                client = OllamaClient(model=model_name)

            # 4. Generate by Slots
            status_box.write("✍️ Generating questions matching blueprint slots...")
            progress_bar = st.progress(0.0)
            total_slots = len(st.session_state.slots)
            progress_state = {"count": 0}

            def on_progress_callback(rec):
                progress_state["count"] += 1
                progress_bar.progress(min(progress_state["count"] / max(total_slots, 1), 1.0))

            results = generate_paper_by_slots(
                chunks=scored_chunks,
                slots=st.session_state.slots,
                client=client,
                seed=42,
                on_progress=on_progress_callback,
            )
            progress_bar.progress(1.0)
            st.session_state.results = results
            status_box.update(label="Question paper generated successfully!", state="complete", expanded=False)

# -----------------------------------------------------------------------------
# Section 4: Results Display & Download
# -----------------------------------------------------------------------------
if st.session_state.results:
    res = st.session_state.results
    questions = res.get("questions", [])
    unfilled = res.get("unfilled", [])

    st.subheader("📋 Generated Question Paper")

    # Unfilled slots warning
    if unfilled:
        unfilled_details = ", ".join(
            f"{u.get('co')} ({u.get('level')}, {u.get('marks')} marks)" for u in unfilled
        )
        st.warning(
            f"⚠️ Could not find suitable text or valid questions for {len(unfilled)} slot(s): "
            f"**{unfilled_details}**."
        )

    if questions:
        table_data = [
            {
                "id": q["id"],
                "question": q["question"],
                "BT level": q["level"],
                "CO": q["co"],
                "marks": q["marks"],
            }
            for q in questions
        ]
        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            column_config={
                "id": st.column_config.TextColumn("ID", width="small"),
                "question": st.column_config.TextColumn("Question", width="large"),
                "BT level": st.column_config.TextColumn("BT Level", width="small"),
                "CO": st.column_config.TextColumn("Outcome", width="small"),
                "marks": st.column_config.NumberColumn("Marks", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Export Buttons (DOCX and PDF)
        st.write("#### 📥 Export Question Paper")
        course_title = getattr(st.session_state, "course_title", "Course Examination")
        outcomes_map = {
            o["code"].strip(): o["desc"].strip()
            for o in st.session_state.outcomes
            if o["code"].strip()
        }

        docx_data = create_docx_exam_paper(
            questions=questions,
            course_name=course_title,
            subject_code="CO-RAG",
            outcomes=outcomes_map,
        )
        pdf_data = create_pdf_exam_paper(
            questions=questions,
            course_name=course_title,
            subject_code="CO-RAG",
            outcomes=outcomes_map,
        )

        col_docx, col_pdf, _ = st.columns([2, 2, 3])
        with col_docx:
            st.download_button(
                label="📄 Download as DOCX",
                data=docx_data,
                file_name="question_paper.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
            )
        with col_pdf:
            st.download_button(
                label="📑 Download as PDF",
                data=pdf_data,
                file_name="question_paper.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

        with st.expander("🔍 View Technical Question Metadata"):
            st.json(questions)
    else:
        st.error("No questions could be generated. Please check your uploaded PDF and outcomes.")
