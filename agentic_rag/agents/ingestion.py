"""Agent 1 - INGESTION_AGENT.

Converts a raw document into clean, structured, normalized text with preserved
hierarchy. It never chunks or interprets content. Markdown, plain text, and
PDF documents are parsed natively; anything else is flagged as an unparsed
region rather than guessed at.

PDF parsing uses pymupdf (fitz) to extract text spans. Headings are inferred
by comparing each span's font size against the median body font size per page.
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from ..contracts import (
    Document,
    IngestionResult,
    OutlineEntry,
    TableRegion,
    UnparsedRegion,
)

_LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_HEADING_RE = re.compile(r"^#{1,6}(\s|$)")


def ingest_file(path: str | Path) -> IngestionResult:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return _parse_markdown(path)
    if suffix in (".txt", ".text"):
        return _parse_text(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    return IngestionResult(
        stage="ingestion",
        status="fail",
        confidence=0.0,
        document=None,
        reason=f"unsupported format '{suffix}' - provide .md, .txt, or .pdf",
    )


# ------------------------------------------------------------------- markdown

def _parse_markdown(path: Path) -> IngestionResult:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = path.stem
    stack: list[tuple[int, str]] = []
    blocks: list[dict] = []
    tables: list[TableRegion] = []
    buf: list[str] = []
    code: list[str] = []
    in_code = False
    i = 0

    def path_of() -> str:
        return " / ".join([title] + [h for _, h in stack])

    def flush_prose() -> None:
        nonlocal buf
        if buf:
            blocks.append({"type": "prose", "text": "\n".join(buf).strip(), "path": path_of()})
            buf = []

    def push_heading(level: int, heading: str) -> None:
        nonlocal title
        if level == 1:
            title = heading or title
            stack.clear()
            return
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if in_code:
            code.append(line)
            if stripped.startswith("```") and len(code) > 1:
                blocks.append({"type": "prose", "text": "\n".join(code), "path": path_of()})
                in_code, code = False, []
            i += 1
            continue
        if stripped.startswith("```"):
            flush_prose()
            in_code, code = True, [line]
        elif _HEADING_RE.match(stripped):
            flush_prose()
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped.lstrip("#").strip()
            push_heading(level, heading)
            blocks.append({"type": "heading", "level": level, "heading": heading,
                           "text": heading, "path": path_of()})
        elif stripped.startswith("|") and i + 1 < len(lines) and _is_table_sep(lines[i + 1]):
            flush_prose()
            tbl_lines = [line, lines[i + 1]]
            i += 1
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                tbl_lines.append(lines[i])
            md = "\n".join(tbl_lines)
            blocks.append({"type": "table", "text": md, "path": path_of()})
            tables.append(TableRegion(location=path_of(), markdown=md))
        elif _LIST_RE.match(line):
            flush_prose()
            lst = [line]
            while i + 1 < len(lines) and _LIST_RE.match(lines[i + 1]):
                i += 1
                lst.append(lines[i])
            blocks.append({"type": "list", "text": "\n".join(lst), "path": path_of()})
        elif stripped == "":
            flush_prose()
        else:
            buf.append(line)
        i += 1
    if in_code:  # unclosed fence: keep the block instead of guessing
        blocks.append({"type": "prose", "text": "\n".join(code), "path": path_of()})
    flush_prose()

    return _finalize(path, title, "markdown", blocks, tables)


# ----------------------------------------------------------------- plain text

def _parse_text(path: Path) -> IngestionResult:
    text = path.read_text(encoding="utf-8")
    title = path.stem
    blocks = [
        {"type": "prose", "text": para.strip(), "path": title}
        for para in re.split(r"\n\s*\n", text)
        if para.strip()
    ]
    return _finalize(path, title, "text", blocks, [])


# ---------------------------------------------------------------------- pdf

def _parse_pdf(path: Path) -> IngestionResult:
    """Parse a PDF using pymupdf.

    Strategy:
    - For each page, collect all text spans with their font sizes.
    - Compute the median font size for the document (proxy for body text).
    - Any span whose font size is >= 1.25x the median is treated as a heading;
      the heading level is derived from the relative size rank (largest = h1).
    - Everything else becomes prose blocks grouped by page.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        return IngestionResult(
            stage="ingestion", status="fail", confidence=0.0, document=None,
            reason="pymupdf is not installed - run: pip install pymupdf",
        )

    title = path.stem
    all_spans: list[dict[str, Any]] = []

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        return IngestionResult(
            stage="ingestion", status="fail", confidence=0.0, document=None,
            reason=f"could not open PDF: {exc}",
        )

    # ── Pass 1: collect every text span across all pages ──────────────────
    for page_num, page in enumerate(doc, start=1):
        blocks_raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for blk in blocks_raw:
            if blk.get("type") != 0:   # 0 = text block
                continue
            for line in blk.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    all_spans.append({
                        "text": text,
                        "size": round(span["size"], 1),
                        "flags": span.get("flags", 0),
                        "page": page_num,
                    })
    doc.close()

    if not all_spans:
        return IngestionResult(
            stage="ingestion", status="fail", confidence=0.0, document=None,
            reason="PDF contains no extractable text (scanned image PDF?)",
        )

    # ── Pass 2: determine heading size thresholds ─────────────────────────
    sizes = [s["size"] for s in all_spans]
    median_size = statistics.median(sizes)
    heading_threshold = median_size * 1.25

    # Collect unique heading sizes (above threshold), sorted largest → smallest
    heading_sizes = sorted(
        {s for s in sizes if s >= heading_threshold},
        reverse=True
    )

    def heading_level(size: float) -> int:
        """Map a font size to an h1-h4 heading level (1 = largest)."""
        try:
            return min(heading_sizes.index(size) + 1, 4)
        except ValueError:
            return 4

    # ── Pass 3: build block list ──────────────────────────────────────────
    blocks: list[dict] = []
    stack: list[tuple[int, str]] = []    # same stack as markdown parser
    prose_buf: list[str] = []
    current_path = title

    def path_of() -> str:
        return " / ".join([title] + [h for _, h in stack])

    def flush_prose() -> None:
        nonlocal prose_buf
        joined = " ".join(prose_buf).strip()
        if joined:
            blocks.append({"type": "prose", "text": joined, "path": current_path})
        prose_buf = []

    def push_heading(level: int, heading: str) -> None:
        nonlocal title, current_path
        if level == 1 and not blocks:   # use first h1 as doc title
            title = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        current_path = path_of()

    for span in all_spans:
        text = span["text"]
        size = span["size"]
        bold = bool(span["flags"] & 0b10000)   # bit 4 = bold in pymupdf

        if size >= heading_threshold:
            flush_prose()
            level = heading_level(size)
            push_heading(level, text)
            blocks.append({
                "type": "heading",
                "level": level,
                "heading": text,
                "text": text,
                "path": path_of(),
            })
        else:
            prose_buf.append(text)

    flush_prose()

    return _finalize(path, title, "pdf", blocks, [])


# ---------------------------------------------------------------------- shared

def _finalize(path: Path, title: str, source_type: str,
              blocks: list[dict], tables: list[TableRegion]) -> IngestionResult:
    if not blocks:
        return IngestionResult(
            stage="ingestion", status="fail", confidence=0.0, document=None,
            reason="no extractable content",
        )

    # Assemble normalized_text and compute heading offsets in one pass.
    outline: list[OutlineEntry] = []
    parts: list[str] = []
    offset = 0
    for b in blocks:
        if b["type"] == "heading":
            heading_line = "#" * b["level"] + " " + b["heading"]
            outline.append(OutlineEntry(level=b["level"], heading=b["heading"],
                                        start_offset=offset))
            parts.append(heading_line)
            offset += len(heading_line) + 2
        else:
            parts.append(b["text"])
            offset += len(b["text"]) + 2
    normalized_text = "\n\n".join(parts)

    doc_id = re.sub(r"[^a-z0-9_-]+", "-", title.lower()).strip("-")
    document = Document(
        doc_id=doc_id,
        title=title,
        source_type=source_type,
        outline=outline,
        normalized_text=normalized_text,
        tables=tables,
        unparsed_regions=[],
        block_tree=blocks,
    )
    return IngestionResult(
        stage="ingestion",
        status="pass",
        confidence=1.0,
        document=document,
        reason=f"parsed {len(blocks)} blocks, {len(tables)} table(s), {len(outline)} heading(s)",
    )


def _is_table_sep(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    return all(re.fullmatch(r":?-{3,}:?", c) for c in cells)
