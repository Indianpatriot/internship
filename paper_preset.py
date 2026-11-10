"""Paper preset module defining exam architectures, question plans, and unit structures.

Implements PaperPreset, UnitPlan, and QuestionPlan for configurable
multi-unit exams with internal choice alternatives (OR) and sub-question layouts.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QuestionPlan:
    marks: int                                         # total marks of this question
    sub_marks: list[int] = field(default_factory=list)  # [] = no sub-questions

    def validate(self):
        if self.sub_marks and sum(self.sub_marks) != self.marks:
            raise ValueError(f"Sub-question marks {self.sub_marks} must add up to {self.marks}")


@dataclass
class UnitPlan:
    unit: int
    options: list[QuestionPlan]                        # alternatives joined by OR; student answers ONE

    def validate(self):
        if len(self.options) < 2:
            raise ValueError(f"Unit {self.unit}: needs at least 2 options")
        if len({q.marks for q in self.options}) != 1:
            raise ValueError(f"Unit {self.unit}: all options must carry the same marks")
        for q in self.options:
            q.validate()


@dataclass
class PaperPreset:
    name: str
    units: list[UnitPlan]
    has_choice: bool = True

    def validate(self):
        for u in self.units:
            u.validate()

    @property
    def attempt_marks(self) -> int:      # what a student answers (e.g. 100)
        return sum(u.options[0].marks for u in self.units)

    @property
    def generated_marks(self) -> int:    # what we must generate (e.g. 200)
        return sum(q.marks for u in self.units for q in u.options)


def make_preset(
    name: str,
    units: int = 5,
    marks_per_unit: int = 20,
    options: int = 2,
    sub_marks: Optional[dict[int, list[int]]] = None,
    has_choice: Optional[bool] = None,
) -> PaperPreset:
    """Create a PaperPreset with uniform or customized sub-question marks per unit.

    sub_marks: optional {unit_no: [m1, m2, ...]} to split each question of that unit.
               Empty list [] or omitted unit means a single question with no sub-parts.
    """
    sub_marks = sub_marks or {}
    if has_choice is None:
        has_choice = "100" in name or "Choice" in name or (options > 1 and "50" not in name and "25" not in name)
    plan = [
        UnitPlan(
            unit=n,
            options=[QuestionPlan(marks_per_unit, list(sub_marks.get(n, []))) for _ in range(options)],
        )
        for n in range(1, units + 1)
    ]
    p = PaperPreset(name, plan, has_choice=has_choice)
    p.validate()
    return p


PRESETS: dict[str, PaperPreset] = {
    "100 marks": make_preset("100 marks", units=5, marks_per_unit=20, has_choice=True),
    "100 Marks (Choice)": make_preset("100 Marks (Choice)", units=5, marks_per_unit=20, has_choice=True),
    "50 marks":  make_preset("50 marks",  units=5, marks_per_unit=10, has_choice=False),
    "50 Marks (Standard)": make_preset("50 Marks (Standard)", units=5, marks_per_unit=10, has_choice=False),
    "25 Marks (Internal Assessment)": make_preset("25 Marks (Internal Assessment)", units=5, marks_per_unit=5, has_choice=False),
    "Custom Blueprint": make_preset("Custom Blueprint", units=5, marks_per_unit=20, has_choice=True),
}


def format_paper(questions: list[dict[str, Any]], has_choice: Optional[bool] = None) -> str:
    """Render question paper into academic unit layout format.

    Format:
    UNIT 1
    Q1.  (a) ...............  [10 marks]  CO1  L2
         (b) ...............  [10 marks]  CO1  L3
                            OR
    Q2.  (a) ...............  [10 marks]  CO2  L2
         (b) ...............  [10 marks]  CO2  L4

    A question without sub-parts prints as one block with its full marks:
    Q1.  ...............  [20 marks]  CO1  L2
    """
    if not questions:
        return ""

    if has_choice is None:
        has_choice = any(bool(q.get("alternative_group")) for q in questions)
        if not has_choice:
            by_u = {}
            for q in questions:
                by_u.setdefault(q.get("unit", 1), set()).add(q.get("question_no", 1))
            if any(len(qs) > 1 for qs in by_u.values()):
                has_choice = True

    # Group by unit, preserving order
    units_order: list[Any] = []
    by_unit: dict[Any, list[dict[str, Any]]] = {}
    for q in questions:
        u = q.get("unit", 1)
        if u not in by_unit:
            units_order.append(u)
            by_unit[u] = []
        by_unit[u].append(q)

    lines: list[str] = []
    for u_idx, u in enumerate(units_order):
        unit_qs = by_unit[u]
        lines.append(f"UNIT {u}")

        # Group by question_no within unit
        q_order: list[int] = []
        by_q_no: dict[int, list[dict[str, Any]]] = {}
        for q in unit_qs:
            q_no = q.get("question_no", 1)
            if q_no not in by_q_no:
                q_order.append(q_no)
                by_q_no[q_no] = []
            by_q_no[q_no].append(q)

        for q_idx, q_no in enumerate(q_order):
            subs = by_q_no[q_no]
            if q_idx > 0 and has_choice:
                lines.append("                     OR")

            has_subparts = len(subs) > 1 or (
                len(subs) == 1
                and subs[0].get("sub_index") is not None
                and "(" in str(subs[0].get("id", ""))
            )

            if not has_subparts:
                rec = subs[0]
                q_text = rec.get("question") or rec.get("desc") or "..."
                marks = rec.get("marks", 20)
                co = rec.get("co", "")
                lvl = rec.get("level", "")
                tag = f"  [{marks} marks]"
                if co:
                    tag += f"  {co}"
                if lvl:
                    tag += f"  {lvl}"
                lines.append(f"Q{q_no}.  {q_text}{tag}")
            else:
                for s_i, rec in enumerate(subs, 1):
                    sub_idx = rec.get("sub_index", s_i)
                    sub_letter = chr(96 + int(sub_idx))
                    q_text = rec.get("question") or rec.get("desc") or "..."
                    marks = rec.get("marks", 10)
                    co = rec.get("co", "")
                    lvl = rec.get("level", "")
                    tag = f"  [{marks} marks]"
                    if co:
                        tag += f"  {co}"
                    if lvl:
                        tag += f"  {lvl}"

                    prefix = f"Q{q_no}.  ({sub_letter}) " if s_i == 1 else f"     ({sub_letter}) "
                    lines.append(f"{prefix}{q_text}{tag}")

        if u_idx < len(units_order) - 1:
            lines.append("")

    return "\n".join(lines)

