"""Streamlit UI for Outcome-Aware Question Paper Generation.

Zero-cost, local RAG question paper generation mapped to Course Outcomes (CO/LO)
and Bloom's Taxonomy (BT) levels with Chapter Selection and Paper Assembly.
"""
import os
import re
import tempfile
import pandas as pd
import streamlit as st

import config
from blueprint import (
    get_50_mark_blueprint,
    get_25_mark_blueprint,
    calculate_blueprint_stats,
)
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
        **Pipeline Information:**
        - **Model:** `BAAI/bge-small-en-v1.5`
        - **Embedding Mode:** Normalized Cosine Similarity
        - **Bloom's Levels:** L1 to L5
        - **Features:** 50-Mark Blueprint, Chapter Filtering & Verification
        """
    )


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Outcome-Aware Question Paper Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a textbook PDF, select specific chapters, define Course Outcomes (CO/LO), '
    'and generate a balanced 50-mark exam paper mapped to Bloom\'s Taxonomy.</div>',
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
# Section 2: Question Blueprint Slots (Vikas Step 1)
# -----------------------------------------------------------------------------
st.subheader("2. Question Paper Blueprint (Slots)")
st.caption(
    "Define the 50-mark structure: Part A (Short Questions) and Part B (Long questions with a/b subparts). "
    "Select a preset below or customize individual question slots."
)

available_cos = [o["code"].strip() for o in st.session_state.outcomes if o["code"].strip()]
if not available_cos:
    available_cos = ["CO1"]

col_p1, col_p2, col_p3 = st.columns([2.5, 2.5, 2])
with col_p1:
    if st.button("🎓 Load 50-Mark Blueprint (Standard)", use_container_width=True):
        st.session_state.slots = get_50_mark_blueprint(available_cos)
        st.rerun()
with col_p2:
    if st.button("📝 Load 25-Mark Blueprint (Internal)", use_container_width=True):
        st.session_state.slots = get_25_mark_blueprint(available_cos)
        st.rerun()
with col_p3:
    if st.button("🔄 Reset to Blank", use_container_width=True):
        st.session_state.slots = [
            {"part": "Part A", "id": "1", "co": available_cos[0], "level": "L1", "marks": 2}
        ]
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

# Bloom's & CO Spread Badges
with st.expander("📊 View Bloom's Taxonomy & Course Outcome Spread", expanded=False):
    col_bt_dist, col_co_dist = st.columns(2)
    with col_bt_dist:
        st.markdown("**Bloom's Level Distribution:**")
        bt_spread_str = " | ".join(
            f"**{lvl}**: {marks}m ({stats['bt_percent'].get(lvl, 0)}%)"
            for lvl, marks in sorted(stats["bt_marks"].items())
        )
        st.info(bt_spread_str or "No slots defined")
    with col_co_dist:
        st.markdown("**Course Outcome Coverage:**")
        co_spread_str = " | ".join(
            f"**{co}**: {marks}m ({stats['co_counts'].get(co, 0)} Qs)"
            for co, marks in sorted(stats["co_marks"].items())
        )
        st.success(co_spread_str or "No slots defined")

bt_options = ["L1", "L2", "L3", "L4", "L5"]
bt_labels = {
    "L1": "L1 - Remember",
    "L2": "L2 - Understand",
    "L3": "L3 - Apply",
    "L4": "L4 - Analyze",
    "L5": "L5 - Evaluate",
}

# Slots Table Header
hdr_col1, hdr_col2, hdr_col3, hdr_col4, hdr_col5, hdr_col6 = st.columns([1.5, 1.2, 1.8, 2.2, 1.2, 0.7])
with hdr_col1:
    st.caption("Section / Part")
with hdr_col2:
    st.caption("Q.No / Subpart")
with hdr_col3:
    st.caption("Outcome (CO)")
with hdr_col4:
    st.caption("Bloom Level")
with hdr_col5:
    st.caption("Marks")
with hdr_col6:
    st.caption("Action")

slots_to_remove = []
for j, slot in enumerate(st.session_state.slots):
    col_part, col_id, col_co, col_bt, col_marks, col_del_slot = st.columns([1.5, 1.2, 1.8, 2.2, 1.2, 0.7])

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
            "Bloom's Level",
            options=bt_options,
            format_func=lambda x: bt_labels.get(x, x),
            index=lvl_index,
            key=f"slot_bt_{j}",
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
    # Substep 1.1: Run ingestion immediately upon upload and cache
    file_bytes = uploaded_pdf.getvalue()
    raw_chunks = get_cached_chunks(file_bytes, uploaded_pdf.name)
    st.session_state.course_title = os.path.splitext(uploaded_pdf.name)[0].replace("-", " ").replace("_", " ").title()

    # Substep 1.2: Build chapter metadata in book order (by page_start)
    chapter_metadata = get_chapter_metadata(raw_chunks)

    if chapter_metadata:
        all_chapter_names = [m["chapter"] for m in chapter_metadata]
        chap_display_map = {m["chapter"]: m["display_label"] for m in chapter_metadata}
        chap_sections_map = {m["chapter"]: m["sections"] for m in chapter_metadata}

        # Initialize session state for selected chapters
        if "selected_chapters" not in st.session_state:
            st.session_state.selected_chapters = list(all_chapter_names)

        # Validate against current PDF chapters
        valid_selected = [c for c in st.session_state.selected_chapters if c in all_chapter_names]
        if not valid_selected and all_chapter_names and "chapter_cleared" not in st.session_state:
            valid_selected = list(all_chapter_names)
            st.session_state.selected_chapters = valid_selected

        st.markdown("##### 📖 Select Chapters for Question Paper")
        st.caption("Every question will be generated strictly from your selected chapters. Other chapters will be ignored.")

        # Substep 2.2: Select all and Clear buttons
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

        # Substep 2.1: st.multiselect with all chapters selected by default
        selected_chapters = st.multiselect(
            "Choose Chapters",
            options=all_chapter_names,
            default=st.session_state.selected_chapters,
            format_func=lambda x: chap_display_map.get(x, x),
            key="chapter_multiselect",
        )
        st.session_state.selected_chapters = selected_chapters

        # Substep 2.3: Optional section narrowing
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

        # Substep 3.1 & 3.2: Filter chunks BEFORE outcome scoring
        filtered_chunks = filter_chunks(
            raw_chunks,
            selected_chapters=selected_chapters,
            selected_sections=selected_sections,
        )

        # Substep 2.4: Live summary
        if selected_chapters:
            st.info(f"ℹ️ {get_filter_summary(filtered_chunks, selected_chapters)}")
            can_generate = len(filtered_chunks) > 0
        else:
            # Step 4 Edge case: No chapter selected
            st.error("⚠️ **Select at least one chapter** to enable question paper generation.")
            can_generate = False

    else:
        # Step 4 Edge case: Chapter detection found nothing
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

    # Step 4 Edge case: Selection is too small
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

col_gen, _ = st.columns([2.5, 4.5])
with col_gen:
    generate_clicked = st.button(
        "🚀 Generate Question Paper",
        type="primary",
        use_container_width=True,
        disabled=not can_generate,
        help="Select at least one chapter to enable." if not can_generate else "Generate exam questions",
    )

if generate_clicked:
    if not uploaded_pdf:
        st.error("Please upload a textbook PDF before generating.")
    elif not st.session_state.outcomes or not any(o["desc"].strip() for o in st.session_state.outcomes):
        st.error("Please provide at least one Course Outcome with a description.")
    elif not st.session_state.slots:
        st.error("Please add at least one question slot in the blueprint.")
    elif not filtered_chunks:
        st.error("No text chunks match your chapter/section selection.")
    else:
        with st.status("Executing Generation Pipeline...", expanded=True) as status_box:
            # Substep 3.2 & 3.3: Ingest -> Filter by Chapter -> Score Outcomes -> Generate
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
            results["covered_chapters"] = selected_chapters
            st.session_state.results = results
            status_box.update(label="Question paper generated successfully!", state="complete", expanded=False)

# -----------------------------------------------------------------------------
# Section 4: Results Display & Export (Step 5)
# -----------------------------------------------------------------------------
if st.session_state.results:
    res = st.session_state.results
    questions = res.get("questions", [])
    unfilled = res.get("unfilled", [])
    covered_chaps = res.get("covered_chapters", selected_chapters)

    st.subheader("📋 Generated Question Paper")

    # Substep 5.1: Results header with selected chapters
    if covered_chaps:
        st.markdown(f"**Generated from Chapters:** `{', '.join(covered_chaps)}`")

    # Step 4: Unfilled slots with CO coverage note
    if unfilled:
        unfilled_details = ", ".join(
            f"{u.get('co')} ({u.get('level')}, {u.get('marks')} marks)" for u in unfilled
        )
        st.warning(
            f"⚠️ Could not find suitable text or valid questions for {len(unfilled)} slot(s): "
            f"**{unfilled_details}**.<br/>"
            f"💡 *Note: These Course Outcomes (COs) might belong to textbook chapters that were excluded in your chapter selection.*",
            icon="⚠️",
        )

    if questions:
        # Substep 5.2: Add source chapter and page as columns for verification
        table_data = [
            {
                "Section": q.get("part", "General"),
                "S.No": q.get("id", ""),
                "Question": q.get("question", ""),
                "Marks": q.get("marks", ""),
                "BT Level": q.get("level", ""),
                "Outcome": q.get("co", ""),
                "Source Chapter": q.get("chapter", "N/A"),
                "Source Page": f"p. {q.get('page_start', '')}" if q.get("page_start") else "N/A",
            }
            for q in questions
        ]
        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            column_config={
                "Section": st.column_config.TextColumn("Section", width="small"),
                "S.No": st.column_config.TextColumn("S.No", width="small"),
                "Question": st.column_config.TextColumn("Question Text", width="large"),
                "Marks": st.column_config.NumberColumn("Marks", width="small"),
                "BT Level": st.column_config.TextColumn("BT Level", width="small"),
                "Outcome": st.column_config.TextColumn("Course Outcome", width="small"),
                "Source Chapter": st.column_config.TextColumn("Source Chapter", width="medium"),
                "Source Page": st.column_config.TextColumn("Page", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )

        # Export Section (DOCX, PDF, CSV)
        st.write("#### 📥 Export Formatted Question Paper")
        course_title = getattr(st.session_state, "course_title", "Course Examination")
        outcomes_map = {
            o["code"].strip(): o["desc"].strip()
            for o in st.session_state.outcomes
            if o["code"].strip()
        }

        docx_data = create_docx_exam_paper(
            questions=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes=outcomes_map,
            institution_name=st.session_state.institution_name,
            covered_chapters=covered_chaps,
            include_source=True,
        )
        pdf_data = create_pdf_exam_paper(
            questions=questions,
            course_name=course_title,
            subject_code=st.session_state.subject_code,
            outcomes=outcomes_map,
            institution_name=st.session_state.institution_name,
            covered_chapters=covered_chaps,
            include_source=True,
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
