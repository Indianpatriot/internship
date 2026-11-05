"""Stub and implementation forwarding for teammate modules.

OWNERSHIP RULES:
- parse_scheme: OWNER: Alice (implemented in scheme.py)
- summarize: OWNER: Alice / Bob (implemented in totals.py)
"""
from scheme import parse_scheme
from totals import summarize

__all__ = ["parse_scheme", "summarize"]
