"""Lightweight timing and profiling utility for pipeline stages."""
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

_TIMINGS: dict[str, float] = {}
_ACTIVE = True


@contextmanager
def timed(stage: str, timings_dict: Optional[dict[str, float]] = None) -> Generator[None, None, None]:
    """Time an execution block and record duration in seconds.

    Args:
        stage: Descriptive name of the stage (e.g. 'pdf_extraction', 'embedding').
        timings_dict: Optional dictionary to record into. If omitted, records into global _TIMINGS.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        target = timings_dict if timings_dict is not None else _TIMINGS
        target[stage] = target.get(stage, 0.0) + elapsed
        print(f"[PERF] {stage}: {elapsed:.3f}s")


def get_timings() -> dict[str, float]:
    """Return a copy of the globally recorded timings."""
    return dict(_TIMINGS)


def clear_timings() -> None:
    """Clear all globally recorded timings."""
    _TIMINGS.clear()


def format_timings_table(timings: Optional[dict[str, float]] = None) -> list[dict[str, Any]]:
    """Format timings dictionary as rows suitable for dataframe / display."""
    data = timings if timings is not None else _TIMINGS
    rows = []
    total = sum(data.values())
    for stage, sec in data.items():
        pct = (sec / total * 100.0) if total > 0 else 0.0
        rows.append({
            "Stage": stage,
            "Duration (s)": round(sec, 3),
            "Share (%)": round(pct, 1),
        })
    return rows
