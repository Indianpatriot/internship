"""Mine the sample question paper for the verbs it uses.
   python -m blooms.learn_from_paper sample_paper.pdf
Prints known verbs (with level) and unknown first-words to add manually to verbs.json."""
import re
import sys
from collections import Counter

import fitz

from blooms.verb_bank import verb_level

NUM_PREFIX = re.compile(r"^\s*(?:q\.?\s*)?\(?(?:\d{1,2}|[a-z]|[ivx]{1,4})[.)]\s*", re.I)


def first_words(pdf_path):
    doc = fitz.open(pdf_path)
    c = Counter()
    for page in doc:
        for line in page.get_text().splitlines():
            if not NUM_PREFIX.match(line):
                continue
            rest = NUM_PREFIX.sub("", line).strip()
            m = re.match(r"[A-Za-z]+", rest)
            if m:
                c[m.group(0).lower()] += 1
    return c


def main(path):
    known, unknown = [], []
    for w, n in first_words(path).most_common():
        lvl = verb_level(w)
        (known if lvl else unknown).append((w, n, lvl))
    print("KNOWN:")
    for w, n, lvl in known:
        print(f"  {w:<14} x{n}  {lvl}")
    print("UNKNOWN (review; add real verbs to blooms/verbs.json):")
    for w, n, _ in unknown:
        print(f"  {w:<14} x{n}")


if __name__ == "__main__":
    main(sys.argv[1])
