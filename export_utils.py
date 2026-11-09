"""Export utilities for generating professional DOCX and PDF question papers.

Implements Paper Assembly & Export (Step 2 of UI and Export requirements):
- Substep 2.1: Show answers in DOCX/PDF, including the OR format (choice questions and answer alternatives).
- Substep 2.2: Add the optional totals table at the end (Course Outcomes and Bloom's Taxonomy cumulative totals).
- Table layout: (S.No, Question Text, Marks, BT (Cognitive), Knowledge Dimension (KD), CO, [Source])
- Unit grouping (Unit 1, Unit 2, ... with choice questions)
- Academic header formatting (University/Course, Duration, Max Marks, Covered Chapters)
"""
import io
import re
from typing import Any, Optional

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
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


HEX_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8,64}[_\s\-]+", re.I)


def clean_book_name(name: str) -> str:
    """Strip any leading hex hash (8-64 chars) and clean up book titles."""
    if not name:
        return ""
    s = str(name).strip()
    s = re.sub(r"^[•*–—\-]\s*", "", s).strip()
    s = re.sub(r"^[0-9a-fA-F]{8,64}\s*[:_\-\s]+", "", s).strip()
    s = re.sub(r"\b[0-9a-fA-F]{8,64}\b\s*[:_\-\s]*", "", s).strip()
    if "_" in s and " " not in s:
        s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


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
    totals_summary: Optional[dict[str, Any]] = None,
) -> bytes:
    """Generate a formatted Microsoft Word (.docx) exam question paper.

    Produces table layout with Revised Bloom's Taxonomy, OR choice formatting,
    optional model answers, and optional cumulative totals tables at the end.
    """
    doc = docx.Document()

    # Standard Margins (0.75 in)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    clean_course = clean_book_name(course_name)
    clean_covered = [clean_book_name(c) for c in covered_chapters] if covered_chapters else None

    # 1. Header Information
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(f"{institution_name.upper()}\n")
    title_run.bold = True
    title_run.font.size = Pt(15)
    title_run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    sub_run = title_p.add_run(f"Subject: {clean_course} ({subject_code})")
    sub_run.bold = True
    sub_run.font.size = Pt(11)

    if clean_covered:
        ch_p = doc.add_paragraph()
        ch_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        ch_run = ch_p.add_run(f"Syllabus / Covered Chapters: {', '.join(clean_covered)}")
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
    meta_cells[1].text = q_count_text
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

    if has_kd and has_source:
        headers = ["S.No", "Question Text", "Marks", "BT", "KD", "CO", "Source"]
        t_widths = [Inches(0.6), Inches(3.2), Inches(0.55), Inches(0.55), Inches(0.75), Inches(0.55), Inches(0.75)]
    elif has_kd and not has_source:
        headers = ["S.No", "Question Text", "Marks", "BT", "KD", "CO"]
        t_widths = [Inches(0.65), Inches(3.7), Inches(0.6), Inches(0.6), Inches(0.8), Inches(0.6)]
    elif not has_kd and has_source:
        headers = ["S.No", "Question Text", "Marks", "BT", "CO", "Source"]
        t_widths = [Inches(0.65), Inches(3.6), Inches(0.6), Inches(0.6), Inches(0.6), Inches(0.9)]
    else:
        headers = ["S.No", "Question Text", "Marks", "BT", "CO"]
        t_widths = [Inches(0.75), Inches(4.1), Inches(0.7), Inches(0.7), Inches(0.7)]

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

        row_cells[col_idx].text = str(q.get("question", ""))
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
            b_title = clean_book_name(q.get("book_title", ""))
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
    has_any_answers = include_answers and any(q.get("answer") for q in questions)
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
                    ap.add_run(alt).font.size = Pt(9.0)
            elif alternatives:
                ap = doc.add_paragraph()
                a_label = ap.add_run("Model Answer / Key Points:\n")
                a_label.bold = True
                a_label.font.size = Pt(9.5)
                a_label.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                ap.add_run(alternatives[0]).font.size = Pt(9.0)
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
    totals_summary: Optional[dict[str, Any]] = None,
) -> bytes:
    """Generate a formatted PDF exam question paper using ReportLab.

    Produces table layout with Revised Bloom's Taxonomy, OR choice formatting,
    optional model answers, and optional cumulative totals tables at the end.
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

    clean_course = clean_book_name(course_name)
    clean_covered = [clean_book_name(c) for c in covered_chapters] if covered_chapters else None

    elements = []
    elements.append(Paragraph(institution_name.upper(), title_style))
    elements.append(Paragraph(f"Subject: {clean_course} ({subject_code})", sub_style))
    if clean_covered:
        elements.append(Paragraph(f"Syllabus / Covered Chapters: {', '.join(clean_covered)}", ch_style))
    elements.append(Spacer(1, 8))

    total_marks, q_count_text = _get_paper_header_values(questions, totals_summary)
    time_str = "1.5 Hours" if total_marks <= 30 else ("2 Hours" if total_marks <= 50 else "3 Hours")

    # Meta banner
    meta_data = [
        [
            Paragraph(f"<b>Time Allowed:</b> {time_str}", styles["Normal"]),
            Paragraph(f"<b>{q_count_text}</b>", styles["Normal"]),
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

    if has_kd and has_source:
        table_headers = [
            Paragraph("S.No", hdr_style),
            Paragraph("Question Text", hdr_style),
            Paragraph("Marks", hdr_style),
            Paragraph("BT", hdr_style),
            Paragraph("KD", hdr_style),
            Paragraph("CO", hdr_style),
            Paragraph("Source", hdr_style),
        ]
        col_widths = [35, 215, 38, 42, 65, 42, 85]
    elif has_kd and not has_source:
        table_headers = [
            Paragraph("S.No", hdr_style),
            Paragraph("Question Text", hdr_style),
            Paragraph("Marks", hdr_style),
            Paragraph("BT", hdr_style),
            Paragraph("KD", hdr_style),
            Paragraph("CO", hdr_style),
        ]
        col_widths = [40, 275, 42, 45, 75, 45]
    elif not has_kd and has_source:
        table_headers = [
            Paragraph("S.No", hdr_style),
            Paragraph("Question Text", hdr_style),
            Paragraph("Marks", hdr_style),
            Paragraph("BT", hdr_style),
            Paragraph("CO", hdr_style),
            Paragraph("Source", hdr_style),
        ]
        col_widths = [40, 260, 45, 45, 45, 85]
    else:
        table_headers = [
            Paragraph("S.No", hdr_style),
            Paragraph("Question Text", hdr_style),
            Paragraph("Marks", hdr_style),
            Paragraph("BT", hdr_style),
            Paragraph("CO", hdr_style),
        ]
        col_widths = [45, 335, 45, 45, 50]

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
            Paragraph(str(q.get("id", "")), q_bold_cell),
            Paragraph(str(q.get("question", "")), q_cell_style),
            Paragraph(f"[{q.get('marks', '')}]", q_bold_cell),
            Paragraph(str(q.get("level", "")), q_bold_cell),
        ]

        if has_kd:
            kd_val = q.get("knowledge_dimension") or q.get("kd", "Conceptual")
            row.append(Paragraph(str(kd_val), q_bold_cell))

        row.append(Paragraph(str(q.get("co", "")), q_bold_cell))

        if has_source:
            ch = clean_book_name(q.get("chapter", ""))
            b_title = clean_book_name(q.get("book_title", ""))
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
            row.append(Paragraph(src_str, q_cell_style))

        table_data.append(row)
        prev_q = q

    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]

    for u_idx in unit_rows_indices:
        t_style.append(("SPAN", (0, u_idx), (-1, u_idx)))
        t_style.append(("BACKGROUND", (0, u_idx), (-1, u_idx), colors.HexColor("#F1F5F9")))

    for or_idx in or_rows_indices:
        t_style.append(("SPAN", (0, or_idx), (-1, or_idx)))
        t_style.append(("BACKGROUND", (0, or_idx), (-1, or_idx), colors.HexColor("#FEF2F2")))
        t_style.append(("ALIGN", (0, or_idx), (-1, or_idx), "CENTER"))

    q_table = Table(table_data, colWidths=col_widths)
    q_table.setStyle(TableStyle(t_style))
    elements.append(q_table)

    if outcomes:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Course Outcomes (CO / LO) Reference:</b>", styles["Normal"]))
        for code, desc in outcomes.items():
            elements.append(Paragraph(f"• <b>{code}</b>: {desc}", styles["Normal"]))

    elements.append(Spacer(1, 8))
    elements.append(
        Paragraph(
            "<i>Revised Bloom's Taxonomy (Anderson & Krathwohl, 2001): Cognitive (L1-Remember to L6-Create) | "
            "Knowledge Dimensions (Factual, Conceptual, Procedural, Metacognitive)</i>",
            styles["Normal"],
        )
    )

    # 5. Substep 2.1: Model Answers in PDF
    has_any_answers = include_answers and any(q.get("answer") for q in questions)
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

            q_banner = f"<b>Question {q_id}:</b> {q_txt} [<b>{q_marks} Marks</b>] <i>({q_lvl}, {q_co})</i>"
            elements.append(Paragraph(q_banner, styles["Normal"]))
            elements.append(Spacer(1, 2))

            alternatives = _split_answer_alternatives(q)
            if len(alternatives) > 1:
                for a_idx, alt in enumerate(alternatives, 1):
                    label = "Model Answer:" if a_idx == 1 else f"Alternative approach {a_idx - 1}:"
                    elements.append(Paragraph(f"<b>{label}</b>", ans_text_style))
                    elements.append(Paragraph(alt.replace("\n", "<br/>"), ans_text_style))
            elif alternatives:
                elements.append(Paragraph("<b>Model Answer / Key Points:</b>", ans_text_style))
                elements.append(Paragraph(alternatives[0].replace("\n", "<br/>"), ans_text_style))

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
            t_co = Table(co_flow_data, colWidths=[180, 100, 110, 110])
            t_co.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
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
            t_bt = Table(bt_flow_data, colWidths=[180, 100, 110, 110])
            t_bt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_bt)

    doc.build(elements)
    return buf.getvalue()
