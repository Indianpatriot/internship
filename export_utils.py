"""Export utilities for generating professional DOCX and PDF question papers.

Implements Paper Assembly & Export (Step 2 of UI and Export requirements):
- Substep 2.1: Show answers in DOCX/PDF, including the OR format (choice questions and answer alternatives).
- Substep 2.2: Add the optional totals table at the end (Course Outcomes and Bloom's Taxonomy cumulative totals).
- Table layout: (S.No, Question Text, Marks, BT (Cognitive), Knowledge Dimension (KD), CO, [Source])
- Unit grouping (Unit 1, Unit 2, ... with choice questions)
- Academic header formatting (University/Course, Duration, Max Marks, Covered Chapters)
"""
import io
import os
import re
import xml.sax.saxutils as saxutils
from typing import Any, Optional, Union

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)

from ingestion.heading_detect import is_non_content_title


HEX_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8,64}[_\s\-]+", re.I)


def clean_book_name(name: str) -> str:
    """Strip any leading hex hash (8-64 chars), extensions, and clean up book titles."""
    if not name or str(name).strip().lower() in ("none", "null", "undefined"):
        return ""
    s = str(name).strip()
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^[•*–—\-]\s*", "", s).strip()
    s = re.sub(r"^[0-9a-fA-F]{8,64}\s*[:_\-\s]+", "", s).strip()
    s = re.sub(r"\b[0-9a-fA-F]{8,64}\b\s*[:_\-\s]*", "", s).strip()
    if "_" in s and " " not in s:
        s = s.replace("_", " ")
    if re.match(r"^chapter:\s*pages\s+\d+", s, re.IGNORECASE):
        s = re.sub(r"^chapter:\s*", "Section: ", s, flags=re.IGNORECASE)
    elif re.match(r"^pages\s+\d+", s, re.IGNORECASE):
        s = f"Section: {s}"
    return re.sub(r"\s+", " ", s).strip()


def clean_course_title(title: str) -> str:
    """Clean course title or book concatenation for headers (e.g. 1_Docker_in_Practice_&_2_Kubernetes_in_Action)."""
    if not title:
        return "Course Assessment"
    s = str(title).strip()
    s = re.sub(r"^[•*–—\-]\s*", "", s).strip()
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"_Exam_Paper$", "", s, flags=re.IGNORECASE).strip()
    # Strip numeric prefixes in concatenated book titles like 1_Docker... & 2_Kubernetes...
    s = re.sub(r"^\(?\d+\)?[_\.\-\s]+", "", s).strip()
    s = re.sub(r"([&_\s]+)\(?\d+\)?[_\.\-\s]+", r"\1", s).strip()
    if "_" in s and " " not in s:
        s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def format_covered_chapters_header(
    covered_chapters: Optional[list[str]] = None,
    questions: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Format 'Syllabus / Covered Chapters' for academic paper export headers.

    Lists ONLY chapters mapped to the paper's units, grouped by book, using clean titles.
    Never includes Parts, Sections, Roadmap, About the code, About Packt, or appendix items.
    """
    from ingestion.heading_detect import is_non_content_title

    book_to_chaps: dict[str, list[str]] = {}

    def _add_entry(b_title: str, ch_name: str):
        b_clean = clean_book_name(b_title)
        ch_clean = clean_book_name(ch_name)
        if not ch_clean or is_non_content_title(ch_clean):
            return
        if re.match(r"^(?:section|part)\s+\d+", ch_clean, re.I) and not re.search(r"\bchapter\b", ch_clean, re.I):
            return
        if b_clean and ch_clean.lower().startswith(b_clean.lower()):
            ch_clean = ch_clean[len(b_clean):].lstrip(" -:•")
        if not ch_clean:
            return
        book_key = b_clean or "Textbook"
        if book_key not in book_to_chaps:
            book_to_chaps[book_key] = []
        if ch_clean not in book_to_chaps[book_key]:
            book_to_chaps[book_key].append(ch_clean)

    # 1. From questions
    if questions:
        for q in questions:
            b_title = q.get("display_label") or q.get("book_title") or ""
            if not b_title and q.get("book_id"):
                from ingestion.library import display_name
                b_title = display_name(q["book_id"])
            ch = q.get("chapter") or ""
            if ch:
                _add_entry(b_title, ch)

    # 2. From covered_chapters if empty or supplemental
    if not book_to_chaps and covered_chapters:
        for item in covered_chapters:
            s = clean_book_name(item)
            if not s or is_non_content_title(s):
                continue
            if re.match(r"^(?:section|part)\s+\d+", s, re.I) and not re.search(r"\bchapter\b", s, re.I):
                continue
            if " - " in s:
                parts = s.split(" - ", 1)
                _add_entry(parts[0], parts[1])
            elif ": " in s and not s.lower().startswith("chapter"):
                parts = s.split(": ", 1)
                _add_entry(parts[0], parts[1])
            else:
                _add_entry("", s)

    if not book_to_chaps:
        return ""

    groups = []
    for b_title, chaps in book_to_chaps.items():
        ch_str = ", ".join(chaps)
        if b_title and b_title != "Textbook":
            groups.append(f"{b_title}: {ch_str}")
        else:
            groups.append(ch_str)
    return "; ".join(groups)


def _get_paper_header_values(
    questions: list[dict[str, Any]],
    totals_summary: Optional[dict[str, Any]] = None,
) -> tuple[int, str]:
    """Compute (attempted_marks, question_count_label)."""
    from totals import _filter_attempted_questions

    if totals_summary and "total_marks" in totals_summary:
        total_marks = int(totals_summary["total_marks"])
    else:
        attempted_qs = _filter_attempted_questions(questions)
        total_marks = sum(int(q.get("marks", 0)) for q in attempted_qs)

    units = sorted(list({q.get("unit") for q in questions if q.get("unit") is not None}))
    has_choices = any(q.get("alternative_group") for q in questions)
    if units:
        n_units = len(units)
        if has_choices:
            q_count_text = f"Total Units: {n_units} (Choice in each)"
        else:
            q_count_text = f"Total Units: {n_units}"
    else:
        main_q_count = len({q.get("question_no") for q in questions if q.get("question_no") is not None})
        if main_q_count > 0:
            q_count_text = f"Total Questions: {main_q_count}"
        else:
            q_count_text = f"Total Questions: {len(questions)}"

    return total_marks, q_count_text


def _split_answer_alternatives(q: dict[str, Any]) -> list[str]:
    """Extract distinct answer alternatives from QuestionRecord, cleaning away '― OR ―'."""
    raw_alts = q.get("answer_alternatives")
    if raw_alts and isinstance(raw_alts, list):
        alts = [str(a).strip() for a in raw_alts if str(a).strip()]
    else:
        raw_ans = q.get("answer", "")
        if not raw_ans:
            return []
        lines = raw_ans.splitlines()
        alts = []
        curr = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^(?:[—―\-]+\s*)?OR(?:\s*[—―\-]+)?[:\.]?$", stripped, re.I):
                if curr:
                    alts.append("\n".join(curr).strip())
                    curr = []
            elif re.match(r"^Alternative approach(?:\s*\d+)?[:\.]?", stripped, re.I):
                if curr:
                    alts.append("\n".join(curr).strip())
                    curr = []
                content_after = re.sub(r"^Alternative approach(?:\s*\d+)?[:\.]?\s*", "", stripped, flags=re.I).strip()
                if content_after:
                    curr.append(content_after)
            else:
                curr.append(line)
        if curr:
            alts.append("\n".join(curr).strip())

    cleaned_alts = []
    for a in alts:
        cleaned_a = re.sub(r"^[—―-]+\s*OR\s*[—―-]+$", "", a, flags=re.I | re.M).strip()
        cleaned_a = re.sub(r"^[—―-]+\s*OR\s*[—―-]+", "Alternative approach:", cleaned_a, flags=re.I).strip()
        if cleaned_a and cleaned_a not in cleaned_alts:
            cleaned_alts.append(cleaned_a)

    return cleaned_alts if cleaned_alts else ([q.get("answer", "").strip()] if q.get("answer") else [])


def safe_xml_text(text: Any) -> str:
    """Safely escape text for ReportLab Paragraphs, converting inline `code` to Courier font."""
    if text is None:
        return ""
    s = str(text)
    escaped = saxutils.escape(s)
    escaped = re.sub(r"`([^`]+)`", lambda m: f'<font name="Courier">{m.group(1)}</font>', escaped)
    escaped = escaped.replace("\n", "<br/>")
    return escaped


def format_pdf_cell_content(
    text: Any,
    default_style: ParagraphStyle,
    code_style: Optional[ParagraphStyle] = None,
) -> Any:
    """Format table cell or document block for ReportLab.
    If text contains multi-line markdown code blocks (```...```),
    splits into Paragraphs and XPreformatted flowables, preserving exact newlines and indentation.
    Bracketed tokens (<container-id>) and inline code are safely preserved.
    """
    if text is None:
        return Paragraph("", default_style)
    s = str(text)
    if "```" not in s:
        return Paragraph(safe_xml_text(s), default_style)

    parts = re.split(r"```(?:\w+)?\n?(.*?)```", s, flags=re.DOTALL)
    flowables = []
    c_style = code_style or ParagraphStyle(
        "PDFCodeBlock",
        parent=default_style,
        fontName="Courier",
        fontSize=max(7.5, default_style.fontSize - 1),
        leading=max(9.5, default_style.leading - 1),
        textColor=colors.HexColor("#0F172A"),
    )

    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 1:
            # Code block: preserve newlines
            flowables.append(XPreformatted(saxutils.escape(part.strip("\r\n")), c_style))
        else:
            cleaned = part.strip("\r\n")
            if cleaned:
                flowables.append(Paragraph(safe_xml_text(cleaned), default_style))

    if not flowables:
        return Paragraph("", default_style)
    if len(flowables) == 1:
        return flowables[0]
    return flowables


def _add_inline_runs_docx(p: Any, text: str, default_font_size: Pt = Pt(9.5), is_bold: bool = False) -> None:
    tokens = re.split(r"(`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("`") and tok.endswith("`") and len(tok) >= 2:
            r = p.add_run(tok[1:-1])
            r.font.name = "Courier New"
            r.font.size = default_font_size
            r.bold = is_bold
        else:
            r = p.add_run(tok)
            r.font.size = default_font_size
            r.bold = is_bold


def add_docx_formatted_content(
    target: Any,
    text: Any,
    default_font_size: Pt = Pt(9.5),
    is_bold: bool = False,
) -> None:
    """Format text into a docx Paragraph or Table Cell:
    - Inline `code` is rendered as monospace runs (Courier New).
    - Multi-line ```code``` blocks are rendered in Courier New with explicit line breaks.
    - Preserves <container-id> and other bracketed tokens literally.
    """
    s = str(text or "")
    if "```" in s:
        parts = re.split(r"```(?:\w+)?\n?(.*?)```", s, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if not part:
                continue
            if i % 2 == 1:
                # Multi-line code block
                if isinstance(target, docx.text.paragraph.Paragraph):
                    p = target.insert_paragraph_before()
                elif hasattr(target, "add_paragraph"):
                    p = target.add_paragraph()
                else:
                    p = target
                p.paragraph_format.left_indent = Inches(0.2)
                lines = part.strip("\r\n").split("\n")
                for l_idx, line in enumerate(lines):
                    r = p.add_run(line)
                    r.font.name = "Courier New"
                    r.font.size = Pt(8.5)
                    if l_idx < len(lines) - 1:
                        p.add_run("\n")
            else:
                cleaned = part.strip("\r\n")
                if cleaned:
                    p = target.add_paragraph() if hasattr(target, "add_paragraph") else target
                    _add_inline_runs_docx(p, cleaned, default_font_size, is_bold)
    else:
        p = target.add_paragraph() if hasattr(target, "add_paragraph") else target
        _add_inline_runs_docx(p, s, default_font_size, is_bold)


def extract_qa_flags_cell(q: dict[str, Any]) -> str:
    """Extract and normalize QA flags for faculty review column."""
    raw_flags = list(q.get("qa_flags") or q.get("review_flags") or [])
    if q.get("needs_review") and not raw_flags:
        raw_flags = [q.get("review_reason") or "NEEDS_REVIEW"]
    if not raw_flags:
        return "-"
    norm_flags = []
    for f in raw_flags:
        fl = str(f).lower()
        if "duplicate" in fl:
            norm_flags.append("DUPLICATE")
        elif "outdated" in fl or "deprecated" in fl:
            norm_flags.append("OUTDATED")
        elif "ungrounded" in fl:
            norm_flags.append("UNGROUNDED_TOKEN")
        elif "co_mismatch" in fl or "co mismatch" in fl:
            norm_flags.append("CO_MISMATCH")
        elif "lint" in fl or "stem" in fl:
            norm_flags.append("STEM_LINT")
        elif "rubric" in fl:
            norm_flags.append("RUBRIC")
        elif "generation_failed" in fl:
            code = q.get("reason_code")
            norm_flags.append(f"FAILED:{code}" if code else "FAILED")
        else:
            cleaned = str(f).strip().upper()[:16]
            if cleaned:
                norm_flags.append(cleaned)
    seen = set()
    uniq = [x for x in norm_flags if not (x in seen or seen.add(x))]
    return ", ".join(uniq) if uniq else "-"


def create_docx_exam_paper(
    questions: list[dict],
    course_name: str = "Course Assessment",
    subject_code: str = "EXAM-101",
    outcomes: dict[str, str] = None,
    institution_name: str = "EXAMINATION QUESTION PAPER",
    covered_chapters: list[str] = None,
    include_source: bool = True,
    include_knowledge_dim: bool = True,
    include_answers: bool = True,
    include_totals: bool = True,
    include_marking_scheme: bool = True,
    include_reasons: bool = True,
    totals_summary: Optional[dict[str, Any]] = None,
    profile: str = "faculty",
    exam_type: str = "SEE",
    semester: str = "",
    date_str: str = "",
    include_qa_flags: Optional[bool] = None,
    include_figures: bool = True,
) -> bytes:
    """Generate a formatted Microsoft Word (.docx) exam question paper.

    Produces table layout with Revised Bloom's Taxonomy, OR choice formatting,
    optional itemized marking schemes, optional model answers, and optional cumulative totals tables.
    """
    doc = docx.Document()

    is_student = (str(profile).strip().lower() == "student")
    if is_student:
        include_source = False
        include_answers = False
        include_marking_scheme = False
        include_totals = False
        include_knowledge_dim = False
        include_qa_flags = False
    elif include_qa_flags is None:
        include_qa_flags = True if str(profile).strip().lower() == "faculty" else False if str(profile).strip().lower() == "faculty" else False

    # Standard Margins (0.75 in)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    clean_course = clean_course_title(course_name)
    clean_covered = [clean_book_name(c) for c in covered_chapters] if covered_chapters else None

    if is_student:
        usn_p = doc.add_paragraph()
        usn_r = usn_p.add_run("USN:   [   ][   ][   ][   ][   ][   ][   ][   ][   ][   ]\n")
        usn_r.font.name = "Courier New"
        usn_r.font.size = Pt(9.5)
        usn_r.bold = True

    # 1. Header Information
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(f"{institution_name.upper()}\n")
    title_run.bold = True
    title_run.font.size = Pt(15)
    title_run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    exam_banner = "SEMESTER END EXAMINATION (SEE)" if str(exam_type).upper() == "SEE" else "CONTINUOUS INTERNAL EVALUATION (CIE)"
    eb_run = title_p.add_run(f"{exam_banner}\n")
    eb_run.bold = True
    eb_run.font.size = Pt(10.5)
    eb_run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    sub_run = title_p.add_run(f"Subject: {clean_course} ({subject_code})")
    sub_run.bold = True
    sub_run.font.size = Pt(11)

    if not is_student:
        header_covered_str = format_covered_chapters_header(covered_chapters, questions=questions)
        if header_covered_str:
            ch_p = doc.add_paragraph()
            ch_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            ch_run = ch_p.add_run(f"Syllabus / Covered Chapters: {header_covered_str}")
            ch_run.font.size = Pt(9.5)
            ch_run.font.italic = True
            ch_run.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    total_marks, q_count_text = _get_paper_header_values(questions, totals_summary)
    time_str = "1.5 Hours" if total_marks <= 30 else ("2 Hours" if total_marks <= 50 else "3 Hours")

    # Meta Table: Date / Time / Total Marks
    meta_table = doc.add_table(rows=1, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    col_widths = [Inches(2.3), Inches(2.3), Inches(2.3)]
    for cell, w in zip(meta_table.rows[0].cells, col_widths):
        cell.width = w

    meta_cells = meta_table.rows[0].cells
    meta_cells[0].text = f"Time Allowed: {time_str}"
    sem_str = f"Semester: {semester}" if semester else q_count_text
    meta_cells[1].text = sem_str
    meta_cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_cells[2].text = f"Max. Marks: {total_marks}"
    meta_cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

    for cell in meta_cells:
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)

    doc.add_paragraph("―" * 58).alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 2. Instructions to Candidates
    inst_p = doc.add_paragraph()
    inst_r1 = inst_p.add_run("Instructions to Candidates:\n")
    inst_r1.bold = True
    inst_r1.font.size = Pt(10)
    has_choices = any(q.get("alternative_group") for q in questions)
    inst_choice_text = (
        "Answer ONE full question from each unit (Internal Choice marked with OR)."
        if has_choices
        else "Answer all questions."
    )
    inst_r2 = inst_p.add_run(
        f"1. {inst_choice_text}\n"
        "2. Figures indicate allocated Marks, Cognitive Bloom's Level (BT), Knowledge Dimension (KD), and Course Outcome (CO).\n"
        "3. Read each question carefully before attempting."
    )
    inst_r2.font.size = Pt(9.5)

    doc.add_paragraph("")

    # 3. Main Question Table
    has_source = include_source and any(q.get("chapter") or q.get("page_start") or q.get("page") for q in questions)
    has_kd = include_knowledge_dim and any(q.get("knowledge_dimension") or q.get("kd") for q in questions)
    has_qa = bool(include_qa_flags)

    if is_student:
        headers = ["Q.No", "Question Text", "Marks", "BT", "CO"]
        t_widths = [Inches(0.75), Inches(4.1), Inches(0.7), Inches(0.7), Inches(0.75)]
    else:
        headers = ["Q.No", "Question Text", "Marks", "BT"]
        fixed_widths = [0.6, 0.55, 0.55]  # Q.No, Marks, BT
        if has_kd:
            headers.append("KD")
            fixed_widths.append(0.75)
        headers.append("CO")
        fixed_widths.append(0.55)
        if has_source:
            headers.append("Source")
            fixed_widths.append(0.85)
        if has_qa:
            headers.append("QA Flags")
            fixed_widths.append(0.75)

        q_w = max(2.5, 7.0 - sum(fixed_widths))
        t_widths = [Inches(fixed_widths[0]), Inches(q_w)] + [Inches(w) for w in fixed_widths[1:]]

    num_cols = len(headers)
    q_table = doc.add_table(rows=1, cols=num_cols)
    q_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    q_table.autofit = False

    for cell, w in zip(q_table.rows[0].cells, t_widths):
        cell.width = w

    hdr_cells = q_table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        p = hdr_cells[i].paragraphs[0]
        if i != 1:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        # Navy blue header background
        shading = parse_xml(r'<w:shd {} w:fill="1E3A8A"/>'.format(nsdecls("w")))
        hdr_cells[i]._tc.get_or_add_tcPr().append(shading)

    current_unit = None
    prev_q: Optional[dict[str, Any]] = None

    for q in questions:
        q_unit = q.get("unit")
        if q_unit is not None and q_unit != current_unit:
            current_unit = q_unit
            unit_row = q_table.add_row()
            merged_cell = unit_row.cells[0]
            for c in unit_row.cells[1:]:
                merged_cell.merge(c)

            merged_cell.text = f"UNIT {current_unit}"
            p = merged_cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
                r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

            shading = parse_xml(r'<w:shd {} w:fill="F1F5F9"/>'.format(nsdecls("w")))
            merged_cell._tc.get_or_add_tcPr().append(shading)

        # Substep 2.1: Check for OR format between choice questions
        if prev_q:
            prev_grp = prev_q.get("alternative_group")
            curr_grp = q.get("alternative_group")
            if (
                prev_grp
                and curr_grp
                and prev_grp == curr_grp
                and prev_q.get("question_no") != q.get("question_no")
            ):
                or_row = q_table.add_row()
                or_cell = or_row.cells[0]
                for c in or_row.cells[1:]:
                    or_cell.merge(c)
                or_cell.text = "— OR —"
                p = or_cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(10)
                    r.font.color.rgb = RGBColor(0xDC, 0x26, 0x26)
                shading = parse_xml(r'<w:shd {} w:fill="FEF2F2"/>'.format(nsdecls("w")))
                or_cell._tc.get_or_add_tcPr().append(shading)

        row_cells = q_table.add_row().cells
        for cell, w in zip(row_cells, t_widths):
            cell.width = w

        col_idx = 0
        row_cells[col_idx].text = str(q.get("id", ""))
        row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        col_idx += 1

        row_cells[col_idx].text = ""
        add_docx_formatted_content(row_cells[col_idx], q.get("question", ""), default_font_size=Pt(9.5))
        col_idx += 1

        row_cells[col_idx].text = f"[{q.get('marks', '')}]"
        row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        col_idx += 1

        row_cells[col_idx].text = str(q.get("level", ""))
        row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        col_idx += 1

        if has_kd:
            kd_val = q.get("knowledge_dimension") or q.get("kd", "Conceptual")
            row_cells[col_idx].text = str(kd_val)
            row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            col_idx += 1

        row_cells[col_idx].text = str(q.get("co", ""))
        row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        col_idx += 1

        if has_source:
            ch = clean_book_name(q.get("chapter", ""))
            if is_non_content_title(ch):
                ch = ""
            b_title = clean_book_name(q.get("display_label") or q.get("book_title") or "")
            if not b_title and q.get("book_id"):
                from ingestion.library import display_name
                b_title = display_name(q["book_id"])
            p_s = q.get("page_start") or q.get("page", "")
            p_str = f"p.{p_s}" if p_s else ""
            if b_title and ch.lower().startswith(b_title.lower()):
                ch = ch[len(b_title):].lstrip(" -:•")
            if b_title and ch:
                src_label = f"{b_title} - {ch}"
            elif ch:
                src_label = ch
            elif b_title:
                src_label = b_title
            else:
                src_label = ""
            src_str = f"{src_label} ({p_str})" if src_label and p_str else (src_label or p_str)
            row_cells[col_idx].text = src_str
            row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            col_idx += 1

        if has_qa:
            row_cells[col_idx].text = extract_qa_flags_cell(q)
            row_cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in row_cells[col_idx].paragraphs[0].runs:
                r.font.size = Pt(8.0)
            col_idx += 1

        for cell in row_cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.0)

        prev_q = q

    # 4. Reference Block: Course Outcomes & Revised Bloom's Taxonomy
    if outcomes:
        doc.add_paragraph("")
        doc.add_paragraph("―" * 58).alignment = WD_ALIGN_PARAGRAPH.CENTER
        co_head = doc.add_paragraph()
        co_head.add_run("Course Outcomes (CO / LO) Reference:\n").bold = True
        for code, desc in outcomes.items():
            doc.add_paragraph(f"• {code}: {desc}", style="List Bullet")

    bt_note = doc.add_paragraph()
    bt_run = bt_note.add_run(
        "\nRevised Bloom's Taxonomy 2D Framework (Anderson & Krathwohl, 2001):\n"
        "• Cognitive Process Dimension: L1-Remember | L2-Understand | L3-Apply | L4-Analyze | L5-Evaluate | L6-Create\n"
        "• Knowledge Dimension: Factual | Conceptual | Procedural | Metacognitive"
    )
    bt_run.italic = True
    bt_run.font.size = Pt(8.5)
    bt_run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # 5. Substep 2.1: Model Answers & Scheme of Evaluation (including OR format)
    has_any_answers = (include_answers or include_marking_scheme) and any(
        (q.get("answer") or q.get("marking_scheme")) for q in questions
    )
    if has_any_answers:
        doc.add_page_break()
        ans_heading = doc.add_paragraph()
        ans_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ans_run = ans_heading.add_run("SCHEME OF EVALUATION & MODEL ANSWERS\n")
        ans_run.bold = True
        ans_run.font.size = Pt(14)
        ans_run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

        sub_ans = ans_heading.add_run(f"Subject: {clean_course} ({subject_code})")
        sub_ans.bold = True
        sub_ans.font.size = Pt(10.5)

        doc.add_paragraph("―" * 58).alignment = WD_ALIGN_PARAGRAPH.CENTER

        prev_ans_q: Optional[dict[str, Any]] = None
        for q in questions:
            # Check for choice group separator between distinct questions
            if prev_ans_q:
                prev_grp = prev_ans_q.get("alternative_group")
                curr_grp = q.get("alternative_group")
                if (
                    prev_grp
                    and curr_grp
                    and prev_grp == curr_grp
                    and prev_ans_q.get("question_no") != q.get("question_no")
                ):
                    or_p = doc.add_paragraph()
                    or_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    or_r = or_p.add_run("― OR ―")
                    or_r.bold = True
                    or_r.font.size = Pt(11)
                    or_r.font.color.rgb = RGBColor(0xDC, 0x26, 0x26)

            q_id = q.get("id", "")
            q_txt = q.get("question", "")
            q_marks = q.get("marks", "")
            q_lvl = q.get("level", "")
            q_co = q.get("co", "")

            qp = doc.add_paragraph()
            q_run = qp.add_run(f"Question {q_id}: {q_txt} [{q_marks} Marks]\n")
            q_run.bold = True
            q_run.font.size = Pt(10)
            meta_r = qp.add_run(f"Target: {q_lvl} | Course Outcome: {q_co}")
            meta_r.italic = True
            meta_r.font.size = Pt(8.5)
            meta_r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

            # Examiner Marking Scheme Table
            scheme = q.get("marking_scheme")
            if include_marking_scheme and scheme and isinstance(scheme, dict) and scheme.get("criteria"):
                sp = doc.add_paragraph()
                s_label = sp.add_run("Marking Scheme & Evaluation Breakdown:\n")
                s_label.bold = True
                s_label.font.size = Pt(9.5)
                s_label.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

                crit_list = scheme.get("criteria", [])
                cols = 5 if include_reasons else 4
                s_table = doc.add_table(rows=len(crit_list) + 2, cols=cols)
                s_table.alignment = WD_TABLE_ALIGNMENT.CENTER
                s_table.autofit = False

                if include_reasons:
                    widths = [Inches(0.5), Inches(1.8), Inches(2.2), Inches(0.8), Inches(1.9)]
                    headers = ["No.", "Criterion", "What Examiner Looks For", "Marks", "Reason"]
                else:
                    widths = [Inches(0.6), Inches(2.2), Inches(3.4), Inches(1.0)]
                    headers = ["No.", "Criterion", "What Examiner Looks For", "Marks"]

                # Header row
                hdr_cells = s_table.rows[0].cells
                for ci, (cell, w, h_text) in enumerate(zip(hdr_cells, widths, headers)):
                    cell.width = w
                    cell.text = h_text
                    p = cell.paragraphs[0]
                    if ci in (0, 3):
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.bold = True
                        r.font.size = Pt(8.5)
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    shd = parse_xml(r'<w:shd {} w:fill="1E3A8A"/>'.format(nsdecls("w")))
                    cell._tc.get_or_add_tcPr().append(shd)

                tot_marks = 0.0
                for ri, crit in enumerate(crit_list, 1):
                    row_cells = s_table.rows[ri].cells
                    c_marks = crit.get("marks", 0)
                    try:
                        tot_marks += float(c_marks)
                    except (ValueError, TypeError):
                        pass

                    exp_text = str(crit.get("expected", ""))
                    alts = crit.get("alternatives")
                    if alts and isinstance(alts, list) and len(alts) > 0:
                        exp_text += "\n― OR ―\n" + "\n".join(f"• {a}" for a in alts)

                    vals = [
                        str(crit.get("no", ri)),
                        str(crit.get("criterion", "")),
                        exp_text,
                        str(c_marks),
                    ]
                    if include_reasons:
                        vals.append(str(crit.get("reason", "")))

                    for ci, (cell, w, val) in enumerate(zip(row_cells, widths, vals)):
                        cell.width = w
                        cell.text = val
                        p = cell.paragraphs[0]
                        if ci in (0, 3):
                            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for r in p.runs:
                            r.font.size = Pt(8.5)

                # Total Row
                tot_cells = s_table.rows[-1].cells
                display_tot = scheme.get(
                    "total_marks",
                    int(tot_marks) if tot_marks.is_integer() else tot_marks,
                )
                tot_vals = ["", "Total Marks", "", str(display_tot)]
                if include_reasons:
                    tot_vals.append("")

                for ci, (cell, w, val) in enumerate(zip(tot_cells, widths, tot_vals)):
                    cell.width = w
                    cell.text = val
                    p = cell.paragraphs[0]
                    if ci in (0, 3):
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.bold = True
                        r.font.size = Pt(8.5)
                    shd = parse_xml(r'<w:shd {} w:fill="F1F5F9"/>'.format(nsdecls("w")))
                    cell._tc.get_or_add_tcPr().append(shd)

                if scheme.get("notes"):
                    p_note = doc.add_paragraph()
                    r_note = p_note.add_run(f"Examiner Guidance: {scheme['notes']}")
                    r_note.italic = True
                    r_note.font.size = Pt(8.0)
                    r_note.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

                # Textbook Diagram / Figure rendering
                if include_figures:
                    all_figs = []
                    for c in crit_list:
                        if c.get("figures"):
                            all_figs.extend(c["figures"])
                    if scheme.get("reference_figures"):
                        all_figs.extend(scheme["reference_figures"])

                    has_diagram_criterion = any(c.get("kind") == "diagram" for c in crit_list)

                    if all_figs:
                        for fig_item in all_figs:
                            img_p = fig_item.get("image_path")
                            if img_p and os.path.exists(img_p):
                                try:
                                    doc.add_paragraph("")
                                    doc.add_picture(img_p, width=Inches(min(5.5, 5.0)))
                                    cap_p = doc.add_paragraph()
                                    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                    c_run = cap_p.add_run(f"Figure: {fig_item.get('caption', 'Textbook Figure')}\n")
                                    c_run.bold = True
                                    c_run.font.size = Pt(8.5)
                                    src_run = cap_p.add_run(f"Source: {fig_item.get('book_title', 'Textbook')}, Page {fig_item.get('page', '')}")
                                    src_run.italic = True
                                    src_run.font.size = Pt(8.0)
                                    src_run.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
                                except Exception:
                                    pass
                    elif has_diagram_criterion:
                        diag_crits = [c for c in crit_list if c.get("kind") == "diagram"]
                        exp_labels = diag_crits[0].get("expected", "key architectural components")
                        no_fig_p = doc.add_paragraph()
                        no_fig_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        nf_run = no_fig_p.add_run(
                            f"Diagram Guidance: No figure available in the textbook for this topic. "
                            f"Expected: hand-drawn diagram showing: {exp_labels}"
                        )
                        nf_run.italic = True
                        nf_run.font.size = Pt(8.5)
                        nf_run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

                if scheme.get("is_fallback"):
                    p_fb = doc.add_paragraph()
                    r_fb = p_fb.add_run("* Auto-generated structure; review before use.")
                    r_fb.italic = True
                    r_fb.font.size = Pt(8.0)
                    r_fb.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)

                doc.add_paragraph("")

            # Model Answer Section
            if include_answers:
                alternatives = _split_answer_alternatives(q)
                if len(alternatives) > 1:
                    # Render answer alternatives with clear labeling (no internal OR)
                    for a_idx, alt in enumerate(alternatives, 1):
                        ap = doc.add_paragraph()
                        label = "Model Answer:\n" if a_idx == 1 else f"Alternative approach {a_idx - 1}:\n"
                        a_label = ap.add_run(label)
                        a_label.bold = True
                        a_label.font.size = Pt(9.5)
                        a_label.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                        add_docx_formatted_content(doc, alt, default_font_size=Pt(9.0))
                elif alternatives:
                    ap = doc.add_paragraph()
                    a_label = ap.add_run("Model Answer / Key Points:\n")
                    a_label.bold = True
                    a_label.font.size = Pt(9.5)
                    a_label.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                    add_docx_formatted_content(doc, alternatives[0], default_font_size=Pt(9.0))
                else:
                    doc.add_paragraph("Model answer pending evaluation.").italic = True

                doc.add_paragraph("")

            prev_ans_q = q


    # 6. Substep 2.2: Add optional totals table at the end
    if include_totals:
        from totals import summarize, totals_table_rows

        summary = totals_summary or summarize(questions, attempted_only=True)
        table_rows_dict = totals_table_rows(summary)
        co_rows = table_rows_dict.get("co_table", [])
        bt_rows = table_rows_dict.get("bt_table", [])

        doc.add_page_break()
        tot_heading = doc.add_paragraph()
        tot_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tot_run = tot_heading.add_run("CUMULATIVE COURSE OUTCOME (CO) & BLOOM'S TAXONOMY (BT) TOTALS\n")
        tot_run.bold = True
        tot_run.font.size = Pt(13)
        tot_run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

        sub_tot = tot_heading.add_run(
            f"Attempted Marks: {summary.get('total_marks', 0)} | "
            f"Offered Marks: {summary.get('offered_marks', 0)}"
        )
        sub_tot.bold = True
        sub_tot.font.size = Pt(9.5)
        sub_tot.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

        doc.add_paragraph("―" * 58).alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Table 1: Course Outcomes Totals Table
        if co_rows:
            p_co = doc.add_paragraph()
            r_co = p_co.add_run("1. Course Outcomes (CO) Distribution:")
            r_co.bold = True
            r_co.font.size = Pt(10)

            t_co = doc.add_table(rows=len(co_rows), cols=len(co_rows[0]))
            t_co.alignment = WD_TABLE_ALIGNMENT.CENTER
            t_co.autofit = False

            co_col_widths = [Inches(2.5), Inches(1.3), Inches(1.4), Inches(1.5)]
            for r_idx, row_data in enumerate(co_rows):
                row_cells = t_co.rows[r_idx].cells
                is_hdr = (r_idx == 0)
                is_total = (r_idx == len(co_rows) - 1)

                for c_idx, val in enumerate(row_data):
                    cell = row_cells[c_idx]
                    cell.width = co_col_widths[c_idx]
                    cell.text = str(val)
                    p = cell.paragraphs[0]
                    if c_idx > 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    for r in p.runs:
                        r.font.size = Pt(9.0)
                        if is_hdr:
                            r.bold = True
                            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        elif is_total:
                            r.bold = True

                    if is_hdr:
                        shd = parse_xml(r'<w:shd {} w:fill="1E3A8A"/>'.format(nsdecls("w")))
                        cell._tc.get_or_add_tcPr().append(shd)
                    elif is_total:
                        shd = parse_xml(r'<w:shd {} w:fill="F1F5F9"/>'.format(nsdecls("w")))
                        cell._tc.get_or_add_tcPr().append(shd)

            doc.add_paragraph("")

        # Table 2: Bloom's Taxonomy Totals Table
        if bt_rows:
            p_bt = doc.add_paragraph()
            r_bt = p_bt.add_run("2. Bloom's Taxonomy (BT) Cognitive Distribution:")
            r_bt.bold = True
            r_bt.font.size = Pt(10)

            t_bt = doc.add_table(rows=len(bt_rows), cols=len(bt_rows[0]))
            t_bt.alignment = WD_TABLE_ALIGNMENT.CENTER
            t_bt.autofit = False

            bt_col_widths = [Inches(2.5), Inches(1.3), Inches(1.4), Inches(1.5)]
            for r_idx, row_data in enumerate(bt_rows):
                row_cells = t_bt.rows[r_idx].cells
                is_hdr = (r_idx == 0)
                is_total = (r_idx == len(bt_rows) - 1)

                for c_idx, val in enumerate(row_data):
                    cell = row_cells[c_idx]
                    cell.width = bt_col_widths[c_idx]
                    cell.text = str(val)
                    p = cell.paragraphs[0]
                    if c_idx > 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    for r in p.runs:
                        r.font.size = Pt(9.0)
                        if is_hdr:
                            r.bold = True
                            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        elif is_total:
                            r.bold = True

                    if is_hdr:
                        shd = parse_xml(r'<w:shd {} w:fill="1E3A8A"/>'.format(nsdecls("w")))
                        cell._tc.get_or_add_tcPr().append(shd)
                    elif is_total:
                        shd = parse_xml(r'<w:shd {} w:fill="F1F5F9"/>'.format(nsdecls("w")))
                        cell._tc.get_or_add_tcPr().append(shd)

            doc.add_paragraph("")

        # Table 3: Knowledge Dimension (KD) Totals Table
        kd_rows = table_rows_dict.get("kd_table", [])
        if kd_rows:
            p_kd = doc.add_paragraph()
            r_kd = p_kd.add_run("3. Knowledge Dimension (KD) Distribution:")
            r_kd.bold = True
            r_kd.font.size = Pt(10)

            t_kd = doc.add_table(rows=len(kd_rows), cols=len(kd_rows[0]))
            t_kd.alignment = WD_TABLE_ALIGNMENT.CENTER
            t_kd.autofit = False

            kd_col_widths = [Inches(2.5), Inches(1.3), Inches(1.4), Inches(1.5)]
            for r_idx, row_data in enumerate(kd_rows):
                row_cells = t_kd.rows[r_idx].cells
                is_hdr = (r_idx == 0)
                is_total = (r_idx == len(kd_rows) - 1)

                for c_idx, val in enumerate(row_data):
                    cell = row_cells[c_idx]
                    cell.width = kd_col_widths[c_idx]
                    cell.text = str(val)
                    p = cell.paragraphs[0]
                    if c_idx > 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    for r in p.runs:
                        r.font.size = Pt(9.0)
                        if is_hdr:
                            r.bold = True
                            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        elif is_total:
                            r.bold = True

                    if is_hdr:
                        shd = parse_xml(r'<w:shd {} w:fill="1E3A8A"/>'.format(nsdecls("w")))
                        cell._tc.get_or_add_tcPr().append(shd)
                    elif is_total:
                        shd = parse_xml(r'<w:shd {} w:fill="F1F5F9"/>'.format(nsdecls("w")))
                        cell._tc.get_or_add_tcPr().append(shd)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def create_pdf_exam_paper(
    questions: list[dict],
    course_name: str = "Course Assessment",
    subject_code: str = "EXAM-101",
    outcomes: dict[str, str] = None,
    institution_name: str = "EXAMINATION QUESTION PAPER",
    covered_chapters: list[str] = None,
    include_source: bool = True,
    include_knowledge_dim: bool = True,
    include_answers: bool = True,
    include_totals: bool = True,
    include_marking_scheme: bool = True,
    include_reasons: bool = True,
    totals_summary: Optional[dict[str, Any]] = None,
    profile: str = "faculty",
    exam_type: str = "SEE",
    semester: str = "",
    date_str: str = "",
    include_qa_flags: Optional[bool] = None,
    include_figures: bool = True,
) -> bytes:
    """Generate a formatted PDF exam question paper using ReportLab.

    Produces table layout with Revised Bloom's Taxonomy, OR choice formatting,
    optional itemized marking schemes, optional model answers, and optional cumulative totals tables.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    is_student = (str(profile).strip().lower() == "student")
    if is_student:
        include_source = False
        include_answers = False
        include_marking_scheme = False
        include_totals = False
        include_knowledge_dim = False
        include_qa_flags = False
    elif include_qa_flags is None:
        include_qa_flags = True if str(profile).strip().lower() == "faculty" else False

    title_style = ParagraphStyle(
        "ExamTitle",
        parent=styles["Heading1"],
        fontSize=15,
        leading=19,
        alignment=1,
        textColor=colors.HexColor("#1E3A8A"),
        fontName="Helvetica-Bold",
    )
    sub_style = ParagraphStyle(
        "ExamSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#334155"),
        fontName="Helvetica-Bold",
    )
    ch_style = ParagraphStyle(
        "ExamChapters",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#475569"),
        fontName="Helvetica-Oblique",
    )
    q_cell_style = ParagraphStyle(
        "QuestionCell",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        fontName="Helvetica",
    )
    q_bold_cell = ParagraphStyle(
        "QuestionCellBold",
        parent=q_cell_style,
        alignment=1,
        fontName="Helvetica-Bold",
    )
    hdr_style = ParagraphStyle(
        "HeaderCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    unit_style = ParagraphStyle(
        "UnitCell",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
        fontName="Helvetica-Bold",
    )
    part_style = unit_style
    or_style = ParagraphStyle(
        "OrChoiceCell",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        alignment=1,
        textColor=colors.HexColor("#DC2626"),
        fontName="Helvetica-Bold",
    )
    ans_text_style = ParagraphStyle(
        "AnswerText",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#1E293B"),
        fontName="Helvetica",
    )
    ans_or_sep_style = ParagraphStyle(
        "AnswerOrSep",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#2563EB"),
        fontName="Helvetica-Bold",
    )
    qa_cell_style = ParagraphStyle(
        "QACellStyle",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=9.5,
        alignment=1,
        textColor=colors.HexColor("#DC2626"),
        fontName="Helvetica-Bold",
    )

    clean_course = clean_course_title(course_name)
    clean_covered = [clean_book_name(c) for c in covered_chapters] if covered_chapters else None

    elements = []
    if is_student:
        usn_data = [["USN:"] + ["" for _ in range(10)]]
        usn_table = Table(usn_data, colWidths=[35] + [16 for _ in range(10)], rowHeights=[16])
        usn_table.setStyle(
            TableStyle([
                ("GRID", (1, 0), (-1, -1), 1, colors.HexColor("#0F172A")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (0, 0), 9),
            ])
        )
        elements.append(usn_table)
        elements.append(Spacer(1, 6))

    elements.append(Paragraph(safe_xml_text(institution_name.upper()), title_style))
    exam_banner = "SEMESTER END EXAMINATION (SEE)" if str(exam_type).upper() == "SEE" else "CONTINUOUS INTERNAL EVALUATION (CIE)"
    elements.append(Paragraph(exam_banner, sub_style))
    elements.append(Paragraph(f"Subject: {safe_xml_text(clean_course)} ({safe_xml_text(subject_code)})", sub_style))

    if not is_student:
        header_covered_str = format_covered_chapters_header(covered_chapters, questions=questions)
        if header_covered_str:
            elements.append(Paragraph(f"Syllabus / Covered Chapters: {safe_xml_text(header_covered_str)}", ch_style))
    elements.append(Spacer(1, 8))

    total_marks, q_count_text = _get_paper_header_values(questions, totals_summary)
    time_str = "1.5 Hours" if total_marks <= 30 else ("2 Hours" if total_marks <= 50 else "3 Hours")

    # Meta banner
    sem_str = f"Semester: {semester}" if semester else q_count_text
    meta_data = [
        [
            Paragraph(f"<b>Time Allowed:</b> {time_str}", styles["Normal"]),
            Paragraph(f"<b>{sem_str}</b>", styles["Normal"]),
            Paragraph(f"<b>Max. Marks:</b> {total_marks}", styles["Normal"]),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[170, 180, 170])
    meta_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ])
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 8))

    # Instructions
    elements.append(Paragraph("<b>Instructions to Candidates:</b>", styles["Normal"]))
    has_choices = any(q.get("alternative_group") for q in questions)
    inst_choice_text = (
        "Answer ONE full question from each unit (Internal Choice marked with OR)."
        if has_choices
        else "Answer all questions."
    )
    elements.append(
        Paragraph(
            f"1. {inst_choice_text}<br/>"
            "2. Figures indicate allocated Marks, Cognitive Bloom's Level (BT), Knowledge Dimension (KD), and Course Outcome (CO).<br/>"
            "3. Read each question carefully before attempting.",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 10))

    has_source = include_source and any(q.get("chapter") or q.get("page_start") or q.get("page") for q in questions)
    has_kd = include_knowledge_dim and any(q.get("knowledge_dimension") or q.get("kd") for q in questions)
    has_qa = bool(include_qa_flags)

    if is_student:
        table_headers = [
            Paragraph("Q.No", hdr_style),
            Paragraph("Question Text", hdr_style),
            Paragraph("Marks", hdr_style),
            Paragraph("BT", hdr_style),
            Paragraph("CO", hdr_style),
        ]
        col_widths = [42, 345, 45, 45, 45]
    else:
        table_headers = [
            Paragraph("Q.No", hdr_style),
            Paragraph("Question Text", hdr_style),
            Paragraph("Marks", hdr_style),
            Paragraph("BT", hdr_style),
        ]
        fixed_widths = [34, 36, 36]  # Q.No, Marks, BT
        if has_kd:
            table_headers.append(Paragraph("KD", hdr_style))
            fixed_widths.append(58)
        table_headers.append(Paragraph("CO", hdr_style))
        fixed_widths.append(36)
        if has_source:
            table_headers.append(Paragraph("Source", hdr_style))
            fixed_widths.append(72)
        if has_qa:
            table_headers.append(Paragraph("QA Flags", hdr_style))
            fixed_widths.append(66)

        q_w = max(180, 522 - sum(fixed_widths))
        col_widths = [fixed_widths[0], q_w] + fixed_widths[1:]

    table_data = [table_headers]
    current_unit = None
    unit_rows_indices = []
    or_rows_indices = []
    prev_q: Optional[dict[str, Any]] = None

    for q in questions:
        q_unit = q.get("unit")
        if q_unit is not None and q_unit != current_unit:
            current_unit = q_unit
            row_idx = len(table_data)
            unit_rows_indices.append(row_idx)
            empty_cells = [""] * (len(col_widths) - 1)
            table_data.append([
                Paragraph(f"UNIT {current_unit}", unit_style),
                *empty_cells,
            ])

        # Substep 2.1: Check for OR choice format between choice questions
        if prev_q:
            prev_grp = prev_q.get("alternative_group")
            curr_grp = q.get("alternative_group")
            if (
                prev_grp
                and curr_grp
                and prev_grp == curr_grp
                and prev_q.get("question_no") != q.get("question_no")
            ):
                or_idx = len(table_data)
                or_rows_indices.append(or_idx)
                empty_cells = [""] * (len(col_widths) - 1)
                table_data.append([
                    Paragraph("— OR —", or_style),
                    *empty_cells,
                ])

        row = [
            Paragraph(safe_xml_text(str(q.get("id", ""))), q_bold_cell),
            format_pdf_cell_content(q.get("question", ""), q_cell_style),
            Paragraph(safe_xml_text(f"[{q.get('marks', '')}]"), q_bold_cell),
            Paragraph(safe_xml_text(str(q.get("level", ""))), q_bold_cell),
        ]

        if has_kd:
            kd_val = q.get("knowledge_dimension") or q.get("kd", "Conceptual")
            row.append(Paragraph(safe_xml_text(str(kd_val)), q_bold_cell))

        row.append(Paragraph(safe_xml_text(str(q.get("co", ""))), q_bold_cell))

        if has_source:
            ch = clean_book_name(q.get("chapter", ""))
            if is_non_content_title(ch):
                ch = ""
            b_title = clean_book_name(q.get("display_label") or q.get("book_title") or "")
            if not b_title and q.get("book_id"):
                from ingestion.library import display_name
                b_title = display_name(q["book_id"])
            p_s = q.get("page_start") or q.get("page", "")
            p_str = f"p.{p_s}" if p_s else ""
            if b_title and ch.lower().startswith(b_title.lower()):
                ch = ch[len(b_title):].lstrip(" -:•")
            if b_title and ch:
                src_label = f"{b_title} - {ch}"
            elif ch:
                src_label = ch
            elif b_title:
                src_label = b_title
            else:
                src_label = ""
            src_str = f"{src_label} ({p_str})" if src_label and p_str else (src_label or p_str)
            row.append(Paragraph(safe_xml_text(src_str), q_cell_style))

        if has_qa:
            row.append(Paragraph(safe_xml_text(extract_qa_flags_cell(q)), qa_cell_style))

        table_data.append(row)
        prev_q = q

    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]

    for u_idx in unit_rows_indices:
        t_style.append(("SPAN", (0, u_idx), (-1, u_idx)))
        t_style.append(("BACKGROUND", (0, u_idx), (-1, u_idx), colors.HexColor("#F1F5F9")))

    for or_idx in or_rows_indices:
        t_style.append(("SPAN", (0, or_idx), (-1, or_idx)))
        t_style.append(("BACKGROUND", (0, or_idx), (-1, or_idx), colors.HexColor("#FEF2F2")))
        t_style.append(("ALIGN", (0, or_idx), (-1, or_idx), "CENTER"))

    q_table = Table(table_data, colWidths=col_widths, repeatRows=1, splitByRow=1)
    q_table.setStyle(TableStyle(t_style))
    elements.append(q_table)

    if outcomes:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Course Outcomes (CO / LO) Reference:</b>", styles["Normal"]))
        for code, desc in outcomes.items():
            elements.append(Paragraph(f"• <b>{safe_xml_text(code)}</b>: {safe_xml_text(desc)}", styles["Normal"]))

    elements.append(Spacer(1, 8))
    elements.append(
        Paragraph(
            "<i>Revised Bloom's Taxonomy (Anderson & Krathwohl, 2001): Cognitive (L1-Remember to L6-Create) | "
            "Knowledge Dimensions (Factual, Conceptual, Procedural, Metacognitive)</i>",
            styles["Normal"],
        )
    )

    # 5. Substep 2.1: Model Answers in PDF
    has_any_answers = (include_answers or include_marking_scheme) and any(
        (q.get("answer") or q.get("marking_scheme")) for q in questions
    )
    if has_any_answers:
        elements.append(PageBreak())
        elements.append(Paragraph("SCHEME OF EVALUATION & MODEL ANSWERS", title_style))
        elements.append(Paragraph(f"Subject: {clean_course} ({subject_code})", sub_style))
        elements.append(Spacer(1, 10))

        prev_ans_q: Optional[dict[str, Any]] = None
        for q in questions:
            if prev_ans_q:
                prev_grp = prev_ans_q.get("alternative_group")
                curr_grp = q.get("alternative_group")
                if (
                    prev_grp
                    and curr_grp
                    and prev_grp == curr_grp
                    and prev_ans_q.get("question_no") != q.get("question_no")
                ):
                    elements.append(Paragraph("<b>— OR —</b>", or_style))
                    elements.append(Spacer(1, 4))

            q_id = q.get("id", "")
            q_txt = q.get("question", "")
            q_marks = q.get("marks", "")
            q_lvl = q.get("level", "")
            q_co = q.get("co", "")

            q_banner = f"<b>Question {safe_xml_text(q_id)}:</b> {safe_xml_text(q_txt)} [<b>{safe_xml_text(q_marks)} Marks</b>] <i>({safe_xml_text(q_lvl)}, {safe_xml_text(q_co)})</i>"
            elements.append(Paragraph(q_banner, styles["Normal"]))
            elements.append(Spacer(1, 2))

            # Marking Scheme Table in PDF
            scheme = q.get("marking_scheme")
            if include_marking_scheme and scheme and isinstance(scheme, dict) and scheme.get("criteria"):
                elements.append(Paragraph("<b>Marking Scheme & Evaluation Breakdown:</b>", ans_text_style))
                elements.append(Spacer(1, 3))

                crit_list = scheme.get("criteria", [])
                sch_flow_data = []

                if include_reasons:
                    hdr_row = ["No.", "Criterion", "What Examiner Looks For", "Marks", "Reason"]
                    sch_col_widths = [30, 110, 175, 45, 160]
                else:
                    hdr_row = ["No.", "Criterion", "What Examiner Looks For", "Marks"]
                    sch_col_widths = [35, 145, 270, 70]

                sch_flow_data.append([
                    Paragraph(f"<b>{safe_xml_text(h)}</b>", hdr_style) for h in hdr_row
                ])

                tot_marks = 0.0
                for ri, crit in enumerate(crit_list, 1):
                    c_marks = crit.get("marks", 0)
                    try:
                        tot_marks += float(c_marks)
                    except (ValueError, TypeError):
                        pass

                    exp_text = str(crit.get("expected", ""))
                    alts = crit.get("alternatives")
                    if alts and isinstance(alts, list) and len(alts) > 0:
                        exp_text = (
                            safe_xml_text(exp_text)
                            + "<br/><b>— OR —</b><br/>"
                            + "<br/>".join(f"• {safe_xml_text(a)}" for a in alts)
                        )
                    else:
                        exp_text = safe_xml_text(exp_text)

                    row_vals = [
                        safe_xml_text(str(crit.get("no", ri))),
                        safe_xml_text(str(crit.get("criterion", ""))),
                        exp_text,
                        safe_xml_text(str(c_marks)),
                    ]
                    if include_reasons:
                        row_vals.append(safe_xml_text(str(crit.get("reason", ""))))

                    sch_flow_data.append([
                        Paragraph(v, q_bold_cell if ci in (0, 3) else q_cell_style)
                        for ci, v in enumerate(row_vals)
                    ])

                display_tot = scheme.get(
                    "total_marks",
                    int(tot_marks) if tot_marks.is_integer() else tot_marks,
                )
                tot_row = ["", "<b>Total Marks</b>", "", f"<b>{safe_xml_text(str(display_tot))}</b>"]
                if include_reasons:
                    tot_row.append("")

                sch_flow_data.append([
                    Paragraph(v, q_bold_cell) for v in tot_row
                ])

                t_sch = Table(sch_flow_data, colWidths=sch_col_widths, repeatRows=1, splitByRow=1)
                t_sch.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                    ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("ALIGN", (3, 0), (3, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
                elements.append(t_sch)

                if scheme.get("notes"):
                    elements.append(Spacer(1, 2))
                    elements.append(Paragraph(f"<i>Examiner Guidance: {safe_xml_text(scheme['notes'])}</i>", styles["Normal"]))

                if scheme.get("is_fallback"):
                    elements.append(Spacer(1, 2))
                    elements.append(Paragraph("<i>* Auto-generated structure; review before use.</i>", styles["Normal"]))

                # Textbook Diagram / Figure rendering
                if include_figures:
                    all_figs = []
                    for c in crit_list:
                        if c.get("figures"):
                            all_figs.extend(c["figures"])
                    if scheme.get("reference_figures"):
                        all_figs.extend(scheme["reference_figures"])

                    has_diagram_criterion = any(c.get("kind") == "diagram" for c in crit_list)

                    if all_figs:
                        for fig_item in all_figs:
                            img_p = fig_item.get("image_path")
                            if img_p and os.path.exists(img_p):
                                try:
                                    img_flow = [
                                        RLImage(img_p, width=min(450, 400), height=min(260, 230), kind="proportional"),
                                        Paragraph(
                                            f"<b>Figure:</b> {safe_xml_text(fig_item.get('caption', 'Diagram'))} | "
                                            f"<i>Source: {safe_xml_text(fig_item.get('book_title', 'Textbook'))}, Page {fig_item.get('page', '')}</i>",
                                            styles["Normal"]
                                        ),
                                    ]
                                    elements.append(Spacer(1, 4))
                                    elements.append(KeepTogether(img_flow))
                                except Exception:
                                    pass
                    elif has_diagram_criterion:
                        diag_crits = [c for c in crit_list if c.get("kind") == "diagram"]
                        exp_labels = diag_crits[0].get("expected", "key architectural components")
                        elements.append(Spacer(1, 3))
                        elements.append(Paragraph(
                            f"<i>Diagram Guidance: No figure available in the textbook for this topic. "
                            f"Expected: hand-drawn diagram showing: {safe_xml_text(exp_labels)}</i>",
                            styles["Normal"]
                        ))

                elements.append(Spacer(1, 6))

            # Model Answer Section
            if include_answers:
                alternatives = _split_answer_alternatives(q)
                if len(alternatives) > 1:
                    for a_idx, alt in enumerate(alternatives, 1):
                        label = "Model Answer:" if a_idx == 1 else f"Alternative approach {a_idx - 1}:"
                        elements.append(Paragraph(f"<b>{safe_xml_text(label)}</b>", ans_text_style))
                        flow = format_pdf_cell_content(alt, ans_text_style)
                        if isinstance(flow, list):
                            elements.extend(flow)
                        else:
                            elements.append(flow)
                elif alternatives:
                    elements.append(Paragraph("<b>Model Answer / Key Points:</b>", ans_text_style))
                    flow = format_pdf_cell_content(alternatives[0], ans_text_style)
                    if isinstance(flow, list):
                        elements.extend(flow)
                    else:
                        elements.append(flow)

            elements.append(Spacer(1, 8))
            prev_ans_q = q


    # 6. Substep 2.2: Cumulative Totals Tables in PDF
    if include_totals:
        from totals import summarize, totals_table_rows

        summary = totals_summary or summarize(questions, attempted_only=True)
        table_rows_dict = totals_table_rows(summary)
        co_rows = table_rows_dict.get("co_table", [])
        bt_rows = table_rows_dict.get("bt_table", [])

        elements.append(PageBreak())
        elements.append(Paragraph("CUMULATIVE CO & BT TOTALS", title_style))
        elements.append(
            Paragraph(
                f"Attempted Marks: {summary.get('total_marks', 0)} | "
                f"Offered Marks: {summary.get('offered_marks', 0)}",
                sub_style,
            )
        )
        elements.append(Spacer(1, 10))

        # Render CO table
        if co_rows:
            elements.append(Paragraph("<b>1. Course Outcomes (CO) Distribution:</b>", styles["Normal"]))
            elements.append(Spacer(1, 4))
            co_flow_data = [
                [Paragraph(str(c), hdr_style if r_i == 0 else (q_bold_cell if r_i == len(co_rows) - 1 else q_cell_style))
                 for c in row]
                for r_i, row in enumerate(co_rows)
            ]
            t_co = Table(co_flow_data, colWidths=[180, 100, 110, 110], repeatRows=1, splitByRow=1)
            t_co.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(t_co)
            elements.append(Spacer(1, 10))

        # Render BT table
        if bt_rows:
            elements.append(Paragraph("<b>2. Bloom's Taxonomy (BT) Cognitive Distribution:</b>", styles["Normal"]))
            elements.append(Spacer(1, 4))
            bt_flow_data = [
                [Paragraph(str(c), hdr_style if r_i == 0 else (q_bold_cell if r_i == len(bt_rows) - 1 else q_cell_style))
                 for c in row]
                for r_i, row in enumerate(bt_rows)
            ]
            t_bt = Table(bt_flow_data, colWidths=[180, 100, 110, 110], repeatRows=1, splitByRow=1)
            t_bt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(t_bt)
            elements.append(Spacer(1, 10))

        # Render KD table
        kd_rows = table_rows_dict.get("kd_table", [])
        if kd_rows:
            elements.append(Paragraph("<b>3. Knowledge Dimension (KD) Distribution:</b>", styles["Normal"]))
            elements.append(Spacer(1, 4))
            kd_flow_data = [
                [Paragraph(str(c), hdr_style if r_i == 0 else (q_bold_cell if r_i == len(kd_rows) - 1 else q_cell_style))
                 for c in row]
                for r_i, row in enumerate(kd_rows)
            ]
            t_kd = Table(kd_flow_data, colWidths=[180, 100, 110, 110], repeatRows=1, splitByRow=1)
            t_kd.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            elements.append(t_kd)

    doc.build(elements)
    return buf.getvalue()


def create_student_pdf_paper(questions: list[dict], **kwargs) -> bytes:
    """Convenience helper: Generate student exam PDF paper without sources or solutions."""
    kwargs["profile"] = "student"
    return create_pdf_exam_paper(questions, **kwargs)


def create_faculty_pdf_paper(questions: list[dict], **kwargs) -> bytes:
    """Convenience helper: Generate faculty exam PDF paper with solutions, sources, QA flags, and totals."""
    kwargs["profile"] = "faculty"
    return create_pdf_exam_paper(questions, **kwargs)


def create_student_docx_paper(questions: list[dict], **kwargs) -> bytes:
    """Convenience helper: Generate student exam Word DOCX paper without sources or solutions."""
    kwargs["profile"] = "student"
    return create_docx_exam_paper(questions, **kwargs)


def create_faculty_docx_paper(questions: list[dict], **kwargs) -> bytes:
    """Convenience helper: Generate faculty exam Word DOCX paper with solutions, sources, QA flags, and totals."""
    kwargs["profile"] = "faculty"
    return create_docx_exam_paper(questions, **kwargs)
