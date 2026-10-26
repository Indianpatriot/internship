"""Export utilities for generating professional DOCX and PDF question papers.

Implements Step 2 of Paper Assembly:
- Exact table layout: (S.No, Question, Marks, BT, CO)
- Grouping by sections/parts (Part A: 10 Marks, Part B: 40 Marks)
- Academic header formatting (University/Course, Duration, Max Marks)
- Bloom's Taxonomy & Course Outcome reference tables
"""
import io
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_docx_exam_paper(
    questions: list[dict],
    course_name: str = "Course Assessment",
    subject_code: str = "EXAM-101",
    outcomes: dict[str, str] = None,
    institution_name: str = "EXAMINATION QUESTION PAPER",
) -> bytes:
    """Generate a formatted Microsoft Word (.docx) exam question paper.

    Produces table layout: (S.No, Question, Marks, BT, CO) with section grouping.
    """
    doc = docx.Document()

    # Standard Margins (0.75 in)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # 1. Header Section
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run(f"{institution_name.upper()}\n")
    title_run.bold = True
    title_run.font.size = Pt(15)
    title_run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    sub_run = title_p.add_run(f"Subject: {course_name} ({subject_code})")
    sub_run.bold = True
    sub_run.font.size = Pt(11)

    total_marks = sum(int(q.get("marks", 0)) for q in questions)
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
    meta_cells[1].text = f"Total Questions: {len(questions)}"
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
    inst_r2 = inst_p.add_run(
        "1. Answer all questions.\n"
        "2. Figures to the right indicate allocated Marks, Bloom's Taxonomy Level (BT), and Course Outcome (CO).\n"
        "3. Read each question carefully before attempting."
    )
    inst_r2.font.size = Pt(9.5)

    doc.add_paragraph("")

    # 3. Main Question Table: (S.No, Question, Marks, BT, CO)
    q_table = doc.add_table(rows=1, cols=5)
    q_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    q_table.autofit = False

    # Widths: S.No (0.75"), Question (4.1"), Marks (0.7"), BT (0.7"), CO (0.7") => Total = 6.95"
    t_widths = [Inches(0.75), Inches(4.1), Inches(0.7), Inches(0.7), Inches(0.7)]
    for cell, w in zip(q_table.rows[0].cells, t_widths):
        cell.width = w

    hdr_cells = q_table.rows[0].cells
    headers = ["S.No", "Question Text", "Marks", "BT", "CO"]
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        p = hdr_cells[i].paragraphs[0]
        if i != 1:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold = True
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        # Navy blue header background
        shading = parse_xml(r'<w:shd {} w:fill="1E3A8A"/>'.format(nsdecls("w")))
        hdr_cells[i]._tc.get_or_add_tcPr().append(shading)

    # Detect if questions have parts
    has_parts = any("part" in q for q in questions)
    current_part = None

    for q in questions:
        q_part = q.get("part")
        if has_parts and q_part and q_part != current_part:
            current_part = q_part
            part_qs = [x for x in questions if x.get("part") == current_part]
            part_m = sum(int(x.get("marks", 0)) for x in part_qs)

            # Insert Part separator row
            part_row = q_table.add_row()
            merged_cell = part_row.cells[0]
            for c in part_row.cells[1:]:
                merged_cell.merge(c)

            merged_cell.text = f"{current_part.upper()} — ({part_m} Marks)"
            p = merged_cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

            shading = parse_xml(r'<w:shd {} w:fill="F1F5F9"/>'.format(nsdecls("w")))
            merged_cell._tc.get_or_add_tcPr().append(shading)

        # Standard question row: S.No, Question, Marks, BT, CO
        row_cells = q_table.add_row().cells
        for cell, w in zip(row_cells, t_widths):
            cell.width = w

        row_cells[0].text = str(q.get("id", ""))
        row_cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        row_cells[1].text = str(q.get("question", ""))

        row_cells[2].text = f"[{q.get('marks', '')}]"
        row_cells[2].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        row_cells[3].text = str(q.get("level", ""))
        row_cells[3].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        row_cells[4].text = str(q.get("co", ""))
        row_cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        for cell in row_cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)

    # 4. Course Outcomes and Bloom's Reference
    if outcomes:
        doc.add_paragraph("")
        doc.add_paragraph("―" * 58).alignment = WD_ALIGN_PARAGRAPH.CENTER
        co_head = doc.add_paragraph()
        co_head.add_run("Course Outcomes (CO / LO) Reference:\n").bold = True
        for code, desc in outcomes.items():
            doc.add_paragraph(f"• {code}: {desc}", style="List Bullet")

    bt_note = doc.add_paragraph()
    bt_run = bt_note.add_run(
        "\nBloom's Taxonomy Levels: L1-Remember | L2-Understand | L3-Apply | L4-Analyze | L5-Evaluate | L6-Create"
    )
    bt_run.italic = True
    bt_run.font.size = Pt(8.5)
    bt_run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def create_pdf_exam_paper(
    questions: list[dict],
    course_name: str = "Course Assessment",
    subject_code: str = "EXAM-101",
    outcomes: dict[str, str] = None,
    institution_name: str = "EXAMINATION QUESTION PAPER",
) -> bytes:
    """Generate a formatted PDF exam question paper using ReportLab."""
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
    q_cell_style = ParagraphStyle(
        "QuestionCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
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
        fontSize=9.5,
        leading=12,
        alignment=1,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    part_style = ParagraphStyle(
        "PartCell",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1E3A8A"),
        fontName="Helvetica-Bold",
    )

    elements = []
    elements.append(Paragraph(institution_name.upper(), title_style))
    elements.append(Paragraph(f"Subject: {course_name} ({subject_code})", sub_style))
    elements.append(Spacer(1, 8))

    total_marks = sum(int(q.get("marks", 0)) for q in questions)
    time_str = "1.5 Hours" if total_marks <= 30 else ("2 Hours" if total_marks <= 50 else "3 Hours")

    # Meta banner
    meta_data = [
        [
            Paragraph(f"<b>Time Allowed:</b> {time_str}", styles["Normal"]),
            Paragraph(f"<b>Total Questions:</b> {len(questions)}", styles["Normal"]),
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
    elements.append(
        Paragraph(
            "1. Answer all questions.<br/>"
            "2. Figures to the right indicate allocated Marks, Bloom's Taxonomy Level (BT), and Course Outcome (CO).<br/>"
            "3. Read each question carefully before attempting.",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 10))

    # Questions Table: S.No, Question, Marks, BT, CO
    table_data = [[
        Paragraph("S.No", hdr_style),
        Paragraph("Question Text", hdr_style),
        Paragraph("Marks", hdr_style),
        Paragraph("BT", hdr_style),
        Paragraph("CO", hdr_style),
    ]]

    has_parts = any("part" in q for q in questions)
    current_part = None
    part_rows_indices = []

    for q in questions:
        q_part = q.get("part")
        if has_parts and q_part and q_part != current_part:
            current_part = q_part
            part_qs = [x for x in questions if x.get("part") == current_part]
            part_m = sum(int(x.get("marks", 0)) for x in part_qs)

            row_idx = len(table_data)
            part_rows_indices.append(row_idx)
            table_data.append([
                Paragraph(f"{current_part.upper()} — ({part_m} Marks)", part_style),
                "", "", "", "",
            ])

        table_data.append([
            Paragraph(str(q.get("id", "")), q_bold_cell),
            Paragraph(str(q.get("question", "")), q_cell_style),
            Paragraph(f"[{q.get('marks', '')}]", q_bold_cell),
            Paragraph(str(q.get("level", "")), q_bold_cell),
            Paragraph(str(q.get("co", "")), q_bold_cell),
        ])

    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]

    for p_idx in part_rows_indices:
        t_style.append(("SPAN", (0, p_idx), (-1, p_idx)))
        t_style.append(("BACKGROUND", (0, p_idx), (-1, p_idx), colors.HexColor("#F1F5F9")))

    q_table = Table(table_data, colWidths=[45, 335, 45, 45, 50])
    q_table.setStyle(TableStyle(t_style))
    elements.append(q_table)

    if outcomes:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<b>Course Outcomes (CO / LO) Reference:</b>", styles["Normal"]))
        for code, desc in outcomes.items():
            elements.append(Paragraph(f"• <b>{code}</b>: {desc}", styles["Normal"]))

    doc.build(elements)
    return buf.getvalue()
