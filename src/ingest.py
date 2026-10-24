"""
ingest.py - parse module handbook PDFs into module-aware chunks.

Run from the project root:
    python -m src.ingest

Reads  : data/raw/*.pdf
Writes : data/processed/chunks.jsonl   (one chunk per line)

Why module-aware? A module description is a self-contained unit (credits,
prerequisites, exam form, content). Fixed-size splitting would cut across
modules and mix up facts like "5 CP" belonging to different modules.
So we split at module boundaries first, and only sub-split long modules.
"""
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/chunks.jsonl")

MAX_CHARS = 1600      # target max size of one chunk (body only)
OVERLAP_CHARS = 200   # overlap when a long module is split

# ---------------------------------------------------------------------------
# Per-handbook configuration (each university uses a different layout)
# ---------------------------------------------------------------------------
SOURCES = {
    "88-979-H-2025_v1_2025-10-14_en.pdf": {
        "source": "kit",
        "university": "KIT",
        "program": "Computer Science (M.Sc.) 2025",
        "language": "en",
        # "4.12 Module: Autonomous Learning ... [M-INFO-106608]"
        "module_re": re.compile(
            r"^\d+(?:\.\d+)+\s+Module(?: component)?:\s+(?P<name>.+?)\s*\[(?P<id>[MT]-\s*[A-Z]+-\d+)\]\s*$"
        ),
        # long titles wrap over 2-3 lines in the PDF -> glue them back together
        "join_re": re.compile(r"(?m)^(\d+(?:\.\d+)+ Module(?: component)?:[^\]\n]*)\n([^\]\n]*\]?)"),
        "noise_res": [
            re.compile(r"^Computer Science Master 2025 \(Master of Science"),
            re.compile(r"^Module Handbook as of \d{2}/\d{2}/\d{4}"),
            re.compile(r"^\d+ [A-Z][A-Z ]{3,}"),   # running header e.g. "4 MODULES ..."
            re.compile(r"^Module(?: component)?: .+"),   # running header with module name
            re.compile(r"^5 Module components$"),
            re.compile(r"^[A-Z]+-\d+\]$"),            # leftover tail of a wrapped running header
            re.compile(r"^\d{1,3}$"),                # bare page number
            re.compile(r"^[A-Z]$"),                   # side-tab letter
        ],
    },
    "ModHb_MSc_DaSc_2022.pdf": {
        "source": "tu_dortmund",
        "university": "TU Dortmund",
        "program": "Data Science (M.Sc.), updated Nov 2022",
        "language": "en",
        # "Module MD E1-2: Survival Analysis"
        "module_re": re.compile(
            r"^Module\s+(?P<id>MD [A-Za-z0-9\-]+):\s+(?P<name>.+?)\s*$"
        ),
        "noise_res": [],
        "drop_first_line_if_digit": True,  # page number
        "section_res": [re.compile(
            r"^(Elective Courses|\u201cMethods\u201d Elective Courses|\u201cApplications\u201d Elective Courses"
            r"|Possible requirements in case of conditional admission"
            r"|In addition to the above, many German language courses.*)$")],
    },
    "Modulhandbuch_DataScience_PO2_SS26_de.pdf": {
        "source": "tu_braunschweig",
        "university": "TU Braunschweig",
        "program": "Data Science (M.Sc.) PO 2, SS 2026",
        "language": "de/en",
        # PDF text has "Modulname" and the name on SEPARATE lines; same for "Nummer"
        "label_name": "Modulname",
        "label_id": "Nummer",
        "noise_res": [
            re.compile(r"^Technische Universität Braunschweig \| Modulhandbuch"),
            re.compile(r"^Seite \d+ von \d+"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Step 1: extract text page by page
# ---------------------------------------------------------------------------
def clean_text(t: str) -> str:
    t = t.replace("\r", "")
    t = t.replace("\ufffe", "-").replace("\uffff", "-").replace("\u00ad", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t


def extract_pages(pdf_path: Path, cfg: dict):
    """Yield (page_number, [clean lines]) with running headers/footers removed."""
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = clean_text(page.get_text())
        if cfg.get("join_re"):
            for _ in range(3):  # a header may wrap over up to 3 lines
                text = cfg["join_re"].sub(r"\1 \2", text)
        lines = text.split("\n")
        lines = [ln.strip() for ln in lines]
        if cfg.get("drop_first_line_if_digit") and lines and lines[0].isdigit():
            lines = lines[1:]
        lines = [
            ln for ln in lines
            if ln and not re.search(r"\.{4,}", ln)
            and not any(rx.match(ln) for rx in cfg["noise_res"])
        ]
        yield i + 1, lines


# ---------------------------------------------------------------------------
# Step 2: split into modules
# ---------------------------------------------------------------------------
def split_into_modules(pdf_path: Path, cfg: dict):
    """Return list of dicts: {module_id, module_name, pages:set, lines:[...]}.
    Text before the first module header becomes a 'front matter' pseudo-module."""
    modules = [{"module_id": "front-matter", "module_name": "General information",
                "pages": set(), "lines": []}]
    current = modules[0]

    for page_no, lines in extract_pages(pdf_path, cfg):
        for idx, ln in enumerate(lines):
            new_name = new_id = None
            if "module_re" in cfg:  # header on a single line
                m = cfg["module_re"].match(ln)
                if m and not re.search(r"\.{4,}", ln):  # skip table-of-contents lines
                    new_name = re.sub(r"\s+", " ", m.group("name")).strip()
                    new_id = re.sub(r"\s+", "", m.groupdict().get("id") or "")
            elif ln == cfg["label_name"] and idx + 1 < len(lines):  # label / value on 2 lines
                new_name, new_id = re.sub(r"\s+", " ", lines[idx + 1]).strip(), ""

            if not new_name:
                for rx in cfg.get("section_res", []):
                    if rx.match(ln):
                        new_name, new_id = ln, "section"
                        break

            if new_name:
                current = {"module_id": new_id, "module_name": new_name,
                           "pages": {page_no}, "lines": [ln]}
                modules.append(current)
                continue

            if "label_id" in cfg and ln == cfg["label_id"] and not current["module_id"] \
                    and idx + 1 < len(lines) and lines[idx + 1].isdigit():
                current["module_id"] = lines[idx + 1]
            current["pages"].add(page_no)
            current["lines"].append(ln)

    return [m for m in modules if len("\n".join(m["lines"])) > 40]


# ---------------------------------------------------------------------------
# Step 3: chunk
# ---------------------------------------------------------------------------
def split_long_text(text: str, max_chars: int, overlap: int):
    """Split on line boundaries into pieces <= max_chars with a small overlap."""
    if len(text) <= max_chars:
        return [text]
    lines, pieces, buf = text.split("\n"), [], ""
    for ln in lines:
        if buf and len(buf) + len(ln) + 1 > max_chars:
            pieces.append(buf)
            buf = buf[-overlap:] + "\n" + ln  # carry overlap into next piece
        else:
            buf = f"{buf}\n{ln}" if buf else ln
    if buf.strip():
        pieces.append(buf)
    return pieces


def build_chunks(pdf_path: Path, cfg: dict):
    chunks = []
    for mod in split_into_modules(pdf_path, cfg):
        body = "\n".join(mod["lines"])
        pieces = split_long_text(body, MAX_CHARS, OVERLAP_CHARS)
        pages = sorted(mod["pages"])
        mid = mod["module_id"]
        kind = ("front-matter" if mid == "front-matter" else "section" if mid == "section"
                else "component" if mid.startswith("T-") else "module")
        for j, piece in enumerate(pieces):
            # Context header is prepended so every chunk "knows" where it is from.
            header = (f"[{cfg['university']} | {cfg['program']} | "
                      f"{mod['module_name']} ({mod['module_id'] or 'n/a'})]")
            chunks.append({
                "chunk_id": f"{cfg['source']}::{mod['module_id'] or mod['module_name']}::{j}",
                "source": cfg["source"],
                "university": cfg["university"],
                "program": cfg["program"],
                "language": cfg["language"],
                "kind": kind,
                "module_id": mod["module_id"],
                "module_name": mod["module_name"],
                "page_start": pages[0],
                "page_end": pages[-1],
                "part": j,
                "n_parts": len(pieces),
                "text": f"{header}\n{piece}",
            })
    return chunks


# ---------------------------------------------------------------------------
def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for fname, cfg in SOURCES.items():
        path = RAW_DIR / fname
        if not path.exists():
            print(f"[skip] {fname} not found in {RAW_DIR}")
            continue
        chunks = build_chunks(path, cfg)
        n_modules = len({c["module_id"] or c["module_name"] for c in chunks})
        print(f"[ok]   {cfg['source']:16s} modules={n_modules:4d} chunks={len(chunks):4d}")
        all_chunks.extend(chunks)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [len(c["text"]) for c in all_chunks]
    print(f"\nTotal chunks: {len(all_chunks)}  ->  {OUT_FILE}")
    if sizes:
        print(f"Chunk length (chars): min={min(sizes)} "
              f"avg={sum(sizes)//len(sizes)} max={max(sizes)}")
    print("Chunks per source:", dict(Counter(c["source"] for c in all_chunks)))


if __name__ == "__main__":
    main()