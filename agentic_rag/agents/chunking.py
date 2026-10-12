"""Agent 2 - CHUNKING_AGENT.

Splits normalized text into chunks that are coherent standalone units of
meaning - never fixed-size character splits. The chunker is deterministic and
structure-aware (headings, tables, lists, code blocks); the LLM-based quality
gate lives in CHUNK_VALIDATOR_AGENT, whose retry_feedback is applied here.
"""
from __future__ import annotations

import re

from ..contracts import Chunk, ChunkingResult, Document

MAX_TOKENS = 500     # target ceiling per chunk
TINY_TOKENS = 60     # below this, chunks are candidates for merging


def est_tokens(text: str) -> int:
    """Rough English token estimate (words / 0.75); no tokenizer dependency."""
    return max(1, round(len(text.split()) / 0.75))


def chunk_document(doc: Document, retry_feedback: str | None = None) -> ChunkingResult:
    chunks = _blocks_to_chunks(doc.block_tree, doc.doc_id)
    chunks = _merge_tiny(chunks)
    if retry_feedback:
        applied, chunks = _apply_feedback(retry_feedback, chunks, doc.doc_id)
        if applied:
            chunks = _merge_tiny(chunks)
    _renumber(chunks, doc.doc_id)
    return ChunkingResult(
        stage="chunking",
        status="pass" if chunks else "fail",
        confidence=0.9 if chunks else 0.0,
        chunks=chunks,
        reason=f"produced {len(chunks)} chunk(s)"
               + (f" after applying retry_feedback: {retry_feedback[:80]}" if retry_feedback else ""),
    )


def _blocks_to_chunks(blocks: list[dict], doc_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    consumed: set[int] = set()

    def make(text: str, path: str, ctype: str) -> Chunk:
        return Chunk(
            chunk_id="",  # renumbered later
            text=text.strip(),
            section_path=path,
            token_count=est_tokens(text),
            chunk_type=ctype,
        )

    for idx, b in enumerate(blocks):
        if idx in consumed:
            continue
        if b["type"] == "heading":
            heading_text = "#" * b["level"] + " " + b["heading"]
            # Rule: never split a heading from the paragraph it introduces.
            if idx + 1 < len(blocks) and blocks[idx + 1]["type"] == "prose":
                nxt = blocks[idx + 1]
                combined = heading_text + "\n\n" + nxt["text"]
                if est_tokens(combined) <= MAX_TOKENS:
                    chunks.append(make(combined, b["path"], "heading_block"))
                    consumed.add(idx + 1)
                    continue
                # Too big together: keep heading with the first sentence(s).
                sentences = _sentences(nxt["text"])
                head_part = [heading_text]
                tokens = est_tokens(heading_text)
                while sentences and tokens + est_tokens(sentences[0]) <= MAX_TOKENS:
                    head_part.append(sentences.pop(0))
                    tokens = est_tokens("\n\n".join(head_part))
                chunks.append(make("\n\n".join(head_part), b["path"], "heading_block"))
                if sentences:
                    chunks.append(make(" ".join(sentences), nxt["path"], "prose"))
                consumed.add(idx + 1)
                continue
            chunks.append(make(heading_text, b["path"], "heading_block"))
        elif b["type"] == "prose":
            for piece in _split_prose(b["text"]):
                chunks.append(make(piece, b["path"], "prose"))
        elif b["type"] == "table":
            # Rule: never split a table row from its table; keep tables whole.
            chunks.append(make(b["text"], b["path"], "table"))
        elif b["type"] == "list":
            # Rule: never split a list item from its list.
            chunks.append(make(b["text"], b["path"], "list"))
    return chunks


def _split_prose(text: str) -> list[str]:
    """Split prose at sentence boundaries, targeting <= MAX_TOKENS per piece."""
    sentences = _sentences(text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        t = est_tokens(sent)
        if current and current_tokens + t > MAX_TOKENS:
            pieces.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(sent)
        current_tokens += t
    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


def _sentences(text: str) -> list[str]:
    # Never split inside code fences.
    if "```" in text:
        return [text.strip()]
    # Split after sentence-ending punctuation, but not after digits, so
    # "4. Pricing" style fragments and "2.1" numbers stay whole.
    parts = re.split(r"(?<=[.!?])(?<![0-9][.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _merge_tiny(chunks: list[Chunk]) -> list[Chunk]:
    """Merge undersized chunks with a compatible neighbor where size allows."""
    out: list[Chunk] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if c.token_count >= TINY_TOKENS or c.chunk_type in ("table", "list"):
            out.append(c)
            i += 1
            continue
        merged = False
        if out and _mergeable(out[-1], c):
            last = out.pop()
            out.append(_merged(last, c))
            merged = True
            i += 1  # consume the tiny chunk; otherwise it would merge again
        elif i + 1 < len(chunks) and _mergeable(c, chunks[i + 1]):
            out.append(_merged(c, chunks[i + 1]))
            i += 2
            merged = True
        if not merged:
            out.append(c)
            i += 1
    return out


def _mergeable(a: Chunk, b: Chunk) -> bool:
    compatible = {a.chunk_type, b.chunk_type} <= {"prose", "heading_block"}
    return compatible and a.section_path == b.section_path and a.token_count + b.token_count <= MAX_TOKENS


def _merged(a: Chunk, b: Chunk) -> Chunk:
    text = a.text + "\n\n" + b.text
    return Chunk(
        chunk_id="", text=text, section_path=a.section_path,
        token_count=est_tokens(text),
        chunk_type="heading_block" if "heading_block" in (a.chunk_type, b.chunk_type) else "prose",
    )


# The judge may write ids as "c004", "chunk_004", or "doc-c004".
_ID_RE = re.compile(r"(?:chunk[_ ]?|c)(\d{3})", re.IGNORECASE)


def _apply_feedback(feedback: str, chunks: list[Chunk], doc_id: str) -> tuple[bool, list[Chunk]]:
    """Best-effort application of CHUNK_VALIDATOR_AGENT's retry_feedback.

    - "split X": re-split the mentioned chunk(s) at sentence boundaries.
    - any merge-style complaint ("merge", "cuts off", "kept whole", "orphaned"):
      fuse the contiguous run covering the mentioned chunks. A single mentioned
      chunk fuses with its next neighbor. Judge id hallucinations are bounded
      by capping the fused run at 5 chunks / 900 tokens.
    - anything unparseable returns False so the caller can fall back.
    """
    lowered = feedback.lower()
    nums = sorted({int(m) for m in _ID_RE.findall(feedback)})
    by_num = {int(c.chunk_id.rsplit("-c", 1)[1]): c for c in chunks}
    targets = [by_num[n] for n in nums if n in by_num]
    if not targets:
        return False, chunks

    if "split" in lowered:
        rebuilt: list[Chunk] = []
        for c in chunks:
            if c in targets:
                parts = _split_prose(c.text)
                if len(parts) >= 2:
                    rebuilt.extend(Chunk(
                        chunk_id="", text=p, section_path=c.section_path,
                        token_count=est_tokens(p), chunk_type=c.chunk_type,
                    ) for p in parts)
                else:
                    rebuilt.append(c)
            else:
                rebuilt.append(c)
        return True, rebuilt

    lo, hi = min(nums), max(nums)
    if hi - lo + 1 > 5:      # implausibly wide - bound the damage
        hi = lo
    run = [by_num[n] for n in range(lo, hi + 1) if n in by_num]
    if len(run) < 2:
        nxt = by_num.get(lo + 1)
        if nxt and est_tokens(run[0].text) + est_tokens(nxt.text) <= 900:
            run.append(nxt)
    if len(run) < 2:
        return False, chunks
    merged_text = "\n\n".join(c.text for c in run)
    kept = run[0]
    rest = [c for c in chunks if c not in run]
    rest.append(Chunk(
        chunk_id="", text=merged_text, section_path=kept.section_path,
        token_count=est_tokens(merged_text),
        chunk_type="heading_block" if any(t.chunk_type == "heading_block" for t in run) else "prose",
    ))
    return True, rest


def _renumber(chunks: list[Chunk], doc_id: str) -> None:
    for seq, c in enumerate(chunks, start=1):
        c.chunk_id = f"{doc_id}-c{seq:03d}"
