"""Glossary for replacing outdated technical terms in generated questions and answers."""
import re

OUTDATED_TERM_REPLACEMENTS = [
    (re.compile(r"\bminion\s+nodes\b", re.I), "worker nodes"),
    (re.compile(r"\bminion\s+node\b", re.I), "worker node"),
    (re.compile(r"\bminions\b", re.I), "worker nodes"),
    (re.compile(r"\bminion\b", re.I), "worker node"),
    (re.compile(r"\bmaster\s+nodes\b", re.I), "control plane nodes"),
    (re.compile(r"\bmaster\s+node\b", re.I), "control plane node"),
    (re.compile(r"\bwhitelisted\b", re.I), "allowlisted"),
    (re.compile(r"\bwhitelisting\b", re.I), "allowlisting"),
    (re.compile(r"\bwhitelist\b", re.I), "allowlist"),
    (re.compile(r"\bblacklisted\b", re.I), "blocklisted"),
    (re.compile(r"\bblacklisting\b", re.I), "blocklisting"),
    (re.compile(r"\bblacklist\b", re.I), "blocklist"),
]


def apply_glossary_replacements(text: str) -> str:
    """Replace outdated terminology with modern technical terms."""
    if not text:
        return text
    out = text
    for pattern, repl in OUTDATED_TERM_REPLACEMENTS:
        out = pattern.sub(repl, out)
    return out
