"""LLM-based unit-to-chapter auto-mapping with confidence gating and validation."""
import json
import logging
import re
from typing import Any, Optional

from config import OLLAMA_MODEL, TEMPERATURE
from generation.cache import get_cached_response, store_cached_response

logger = logging.getLogger(__name__)

MAPPING_PROMPT_TEMPLATE = """You are an expert curriculum assistant.
Your task is to map each unmapped syllabus unit to the most relevant textbook chapter(s) from the provided list.

Available Textbook Chapters:
{chapter_catalog}

Unmapped Syllabus Units to Map:
{unit_catalog}

Rules:
1. For each unit, select one or more matching chapter IDs (e.g. "C1", "C2") from the Available Textbook Chapters.
2. Assign a confidence score between 0.0 and 1.0 based on how closely the chapter content aligns with the unit topics.
3. If no chapter matches a unit with high relevance, return an empty list [] for chapter_ids and confidence 0.0. Do NOT guess.
4. Output strictly a single JSON object mapping unit number strings to their matched chapter IDs and confidence.
5. No conversational preamble, explanation, or markdown formatting outside the JSON object.

Required JSON format:
{{
{format_example}
}}
"""


def extract_available_chapters(library: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract all chapters from library books with metadata and text overviews."""
    books = library.get("books", {})
    chunks = library.get("chunks", [])
    chapters: list[dict[str, Any]] = []
    c_idx = 1

    for bid, binfo in books.items():
        btitle = binfo.get("display_label") or binfo.get("title") or bid
        for ch in binfo.get("chapters", []):
            cid = f"C{c_idx}"
            c_idx += 1

            # Get short overview from first matching content chunks
            ch_chunks = [
                c.get("text", "")
                for c in chunks
                if c.get("book_id") == bid and c.get("chapter") == ch
            ]
            overview = ""
            if ch_chunks:
                first_chunk = ch_chunks[0].strip()
                words = first_chunk.split()
                overview = " ".join(words[:25]) + ("..." if len(words) > 25 else "")

            chapters.append({
                "id": cid,
                "book_id": bid,
                "book_title": btitle,
                "chapter": ch,
                "overview": overview,
            })

    return chapters


def build_mapping_prompt(
    unmapped_units: list[dict[str, Any]],
    available_chapters: list[dict[str, Any]],
) -> str:
    """Build structured deterministic prompt for LLM unit-to-chapter mapping."""
    ch_lines = []
    for c in available_chapters:
        snippet_part = f" | Overview: {c['overview']}" if c.get("overview") else ""
        ch_lines.append(
            f'- ID: "{c["id"]}" | Book: "{c["book_title"]}" | Chapter: "{c["chapter"]}"{snippet_part}'
        )
    ch_catalog = "\n".join(ch_lines) if ch_lines else "None"

    u_lines = []
    ex_lines = []
    for u in unmapped_units:
        u_num = u.get("unit", 1)
        title = u.get("title", f"Unit {u_num}")
        topics = u.get("topics", [])
        top_str = ", ".join(topics) if topics else "General unit content"
        u_lines.append(f'- Unit {u_num}: Title: "{title}" | Topics: {top_str}')
        ex_lines.append(f'  "{u_num}": {{"chapter_ids": ["C1"], "confidence": 0.95}}')

    u_catalog = "\n".join(u_lines) if u_lines else "None"
    fmt_example = ",\n".join(ex_lines) if ex_lines else '  "1": {"chapter_ids": ["C1"], "confidence": 0.95}'

    return MAPPING_PROMPT_TEMPLATE.format(
        chapter_catalog=ch_catalog,
        unit_catalog=u_catalog,
        format_example=fmt_example,
    )


def parse_mapping_response(
    raw_resp: str,
    available_chapters: list[dict[str, Any]],
    min_confidence: float = 0.50,
) -> dict[int, dict[str, Any]]:
    """Parse and validate LLM mapping JSON response against available chapters."""
    clean = raw_resp.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.I)
        clean = re.sub(r"\s*```$", "", clean)

    # Find outermost JSON object
    match = re.search(r"(\{.*\})", clean, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw_resp[:100]}...")

    data = json.loads(match.group(1))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    # Build resolution lookup for chapters
    id_map: dict[str, tuple[str, str]] = {}
    title_map: dict[str, tuple[str, str]] = {}

    for c in available_chapters:
        cid = c["id"].strip().upper()
        bid = c["book_id"]
        chap = c["chapter"]
        id_map[cid] = (bid, chap)
        title_map[chap.strip().lower()] = (bid, chap)
        title_map[f"{c['book_title']} - {chap}".strip().lower()] = (bid, chap)

    result: dict[int, dict[str, Any]] = {}

    for k, v in data.items():
        try:
            u_num = int(k)
        except (ValueError, TypeError):
            continue

        raw_chaps: list[Any] = []
        conf = 0.0

        if isinstance(v, dict):
            raw_chaps = v.get("chapter_ids") or v.get("chapters") or []
            try:
                conf = float(v.get("confidence", 0.0))
            except (ValueError, TypeError):
                conf = 0.0
        elif isinstance(v, list):
            raw_chaps = v
            conf = 0.8  # default when list given without explicit confidence

        if not isinstance(raw_chaps, list):
            raw_chaps = [raw_chaps]

        # Resolve chapters and drop any invalid IDs
        resolved_chapters: dict[str, list[str]] = {}
        for item in raw_chaps:
            item_str = str(item).strip()
            item_upper = item_str.upper()
            item_lower = item_str.lower()

            resolved = None
            if item_upper in id_map:
                resolved = id_map[item_upper]
            elif item_lower in title_map:
                resolved = title_map[item_lower]
            else:
                # Fuzzy check against chapter titles
                for t_key, r in title_map.items():
                    if item_lower == t_key or item_lower in t_key or t_key in item_lower:
                        resolved = r
                        break

            if resolved:
                bid, ch_name = resolved
                resolved_chapters.setdefault(bid, [])
                if ch_name not in resolved_chapters[bid]:
                    resolved_chapters[bid].append(ch_name)

        if conf >= min_confidence and resolved_chapters:
            result[u_num] = {
                "status": "auto_suggested",
                "confidence": round(conf, 2),
                "chapters": resolved_chapters,
                "method": "llm",
                "warning": None,
            }
        else:
            reason = f"Low confidence ({conf:.2f})" if conf < min_confidence else "No matching chapters found"
            result[u_num] = {
                "status": "unmapped",
                "confidence": round(conf, 2),
                "chapters": {},
                "method": "llm",
                "warning": f"Unit {u_num} could not be mapped confidently ({reason}).",
            }

    return result


def auto_map_units_with_llm(
    units: list[dict[str, Any]],
    library: dict[str, Any],
    client: Any = None,
    min_confidence: float = 0.50,
    force_regenerate: bool = False,
    manually_mapped_units: Optional[set[int]] = None,
) -> dict[int, dict[str, Any]]:
    """Propose unit-to-chapter mappings using LLM for unmapped units only.

    Args:
        units: List of syllabus unit dicts (with 'unit', 'title', 'topics').
        library: Loaded library catalog dictionary.
        client: LLM client (OllamaClient or MockClient). If None, defaults to MockClient.
        min_confidence: Confidence threshold required to accept mapping.
        force_regenerate: If True, bypasses generation cache.
        manually_mapped_units: Set of unit integers already mapped manually (never overwritten).

    Returns:
        Mapping of {unit_number: {"status": str, "confidence": float, "chapters": dict, "method": str, "warning": Optional[str]}}
    """
    manual_set = set(manually_mapped_units or set())

    # Filter to only units that are not manually mapped
    unmapped = [u for u in units if u.get("unit") not in manual_set]
    if not unmapped or not library.get("books"):
        return {}

    available_chapters = extract_available_chapters(library)
    if not available_chapters:
        return {
            u.get("unit", idx + 1): {
                "status": "unmapped",
                "confidence": 0.0,
                "chapters": {},
                "method": "llm",
                "warning": "No chapters available in library.",
            }
            for idx, u in enumerate(unmapped)
        }

    prompt = build_mapping_prompt(unmapped, available_chapters)

    if client is None:
        from generation.mock_client import MockClient

        client = MockClient()

    model_name = getattr(client, "model", "mock")

    try:
        # Check cache first for deterministic, fast reruns
        cached_resp = get_cached_response(
            prompt=prompt,
            model=model_name,
            temperature=0.0,
            force_regenerate=force_regenerate,
        )

        if cached_resp:
            raw_resp = cached_resp
        else:
            raw_resp = client.generate(
                prompt=prompt,
                temperature=0.0,
                num_predict=400,
                format="json",
            )
            store_cached_response(
                prompt=prompt,
                model=model_name,
                response=raw_resp,
                temperature=0.0,
            )

        mappings = parse_mapping_response(
            raw_resp=raw_resp,
            available_chapters=available_chapters,
            min_confidence=min_confidence,
        )

        # Fill in any missing unmapped units
        for u in unmapped:
            u_num = u.get("unit")
            if u_num not in mappings:
                mappings[u_num] = {
                    "status": "unmapped",
                    "confidence": 0.0,
                    "chapters": {},
                    "method": "llm",
                    "warning": f"Unit {u_num} was not mapped by the model.",
                }

        return mappings

    except Exception as e:
        logger.warning("LLM auto-mapping failed, falling back safely: %s", e)
        # Fallback gracefully to unmapped status without crashing
        return {
            u.get("unit", idx + 1): {
                "status": "unmapped",
                "confidence": 0.0,
                "chapters": {},
                "method": "llm",
                "warning": f"Auto-mapping error: {e}",
            }
            for idx, u in enumerate(unmapped)
        }
