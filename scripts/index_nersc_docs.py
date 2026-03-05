#!/usr/bin/env python3
"""
Index NERSC documentation markdown files into a FAISS vector index for retrieval.

Run once offline before the LLM pipeline:

    python3 scripts/index_nersc_docs.py \\
        --docs-root /mscratch/sd/a/asnaylor/nersc.gitlab.io/docs \\
        --output /mscratch/sd/a/asnaylor/nersc_docs_index

Produces:
    <output>.faiss   — FAISS flat inner-product index (cosine similarity)
    <output>.jsonl   — chunk metadata: path, url, page_title, section_heading, text

Dependencies (install once into your mounted PYTHONUSERBASE):
    pip install --user faiss-cpu sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Markdown cleaning
# ---------------------------------------------------------------------------

# Remove images entirely — no useful text content.
_IMG_RE = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)
# Links: keep display text, drop URL.
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
# HTML comments.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# MkDocs admonition/details markers (e.g. "!!! note", "??? tip") — drop the
# marker line itself; the content lines that follow are kept as-is.
_ADMONITION_MARKER_RE = re.compile(r"^[!?]{3}\+?\s+\w+.*$", re.MULTILINE)
# MkDocs attribute lists e.g. { #anchor .class }.
_ATTR_LIST_RE = re.compile(r"\{[^}\n]+\}")
# Markdown header prefixes (# through ######).
_HEADER_PREFIX_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
# Bold / italic markers (keep the wrapped text).
_EMPHASIS_RE = re.compile(r"\*{1,3}([^*\n]+)\*{1,3}")
# Collapse 3+ consecutive blank lines to 2.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _strip_markdown(text: str) -> str:
    """
    Strip markdown formatting while preserving plain text and code blocks.

    Code blocks (``` ... ```) are intentionally left intact — they contain
    commands, module names, and flags that are directly useful for retrieval.
    """
    text = _IMG_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_COMMENT_RE.sub("", text)
    text = _ADMONITION_MARKER_RE.sub("", text)
    text = _ATTR_LIST_RE.sub("", text)
    text = _HEADER_PREFIX_RE.sub("", text)
    text = _EMPHASIS_RE.sub(r"\1", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------

def _rel_path_to_url(rel_path: Path) -> str:
    """
    Convert a docs-relative path to a docs.nersc.gov URL.

    Examples:
        jobs/index.md                         → https://docs.nersc.gov/jobs/
        development/compilers/wrappers.md     → https://docs.nersc.gov/development/compilers/wrappers/
        systems/perlmutter/running-jobs/index.md → https://docs.nersc.gov/systems/perlmutter/running-jobs/
    """
    parts = list(rel_path.parts)
    if parts[-1] == "index.md":
        parts = parts[:-1]
    elif parts[-1].endswith(".md"):
        parts[-1] = parts[-1][:-3]
    return "https://docs.nersc.gov/" + "/".join(parts) + "/"


# ---------------------------------------------------------------------------
# Page chunking
# ---------------------------------------------------------------------------

_H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_H2_SPLIT_RE = re.compile(r"^## (.+)$", re.MULTILINE)


def _page_title(content: str) -> str:
    m = _H1_RE.search(content)
    return m.group(1).strip() if m else ""


def _chunk_page(
    file_path: Path,
    docs_root: Path,
    max_chunk_chars: int,
) -> list[dict[str, str]]:
    """
    Split one markdown page into chunks at ## section boundaries.

    Each chunk contains:
      - path: docs-relative file path (for debugging / provenance)
      - url: canonical docs.nersc.gov URL for the page
      - page_title: H1 heading of the page
      - section_heading: ## heading of this chunk (page_title for intro chunk)
      - text: cleaned text embedded and stored for retrieval context

    The text field starts with "<page_title> — <section_heading>" so that
    semantic search on the section heading also benefits from page-level context.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    rel_path = file_path.relative_to(docs_root)
    url = _rel_path_to_url(rel_path)
    title = _page_title(content)

    chunks: list[dict[str, str]] = []

    # Split at ## boundaries.
    # _H2_SPLIT_RE.split() produces:
    #   [intro_text, heading_1, body_1, heading_2, body_2, ...]
    parts = _H2_SPLIT_RE.split(content)
    # Remove the H1 line from the intro before cleaning to avoid duplicating
    # the page title (we prepend it explicitly in the text field below).
    intro_clean = _strip_markdown(_H1_RE.sub("", parts[0], count=1))

    # Intro section: text before the first ##
    if len(intro_clean) > 80:
        chunks.append({
            "path": str(rel_path),
            "url": url,
            "page_title": title,
            "section_heading": title,
            "text": f"{title}\n\n{intro_clean}"[:max_chunk_chars],
        })

    # H2 sections
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            break
        heading = parts[i].strip()
        body_clean = _strip_markdown(parts[i + 1])
        if len(body_clean) < 40:
            continue
        chunks.append({
            "path": str(rel_path),
            "url": url,
            "page_title": title,
            "section_heading": heading,
            "text": f"{title} — {heading}\n\n{body_clean}"[:max_chunk_chars],
        })

    return chunks


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    docs_root: Path,
    output_prefix: Path,
    model_name: str,
    max_chunk_chars: int,
) -> None:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError:
        raise SystemExit(
            "faiss not found.\n"
            "Install with: pip install --user faiss-cpu"
        )
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
    except ImportError:
        raise SystemExit(
            "sentence-transformers not found.\n"
            "Install with: pip install --user sentence-transformers"
        )

    # Discover markdown files, skipping image/stylesheet subdirs.
    print(f"Scanning: {docs_root}")
    md_files = sorted(
        f for f in docs_root.rglob("*.md")
        if not any(part in ("images", "img", "stylesheets") for part in f.parts)
    )
    print(f"Found {len(md_files)} markdown files")

    # Chunk all pages.
    all_chunks: list[dict[str, str]] = []
    for md_file in md_files:
        all_chunks.extend(_chunk_page(md_file, docs_root, max_chunk_chars))

    if not all_chunks:
        raise SystemExit("No chunks produced — check --docs-root.")
    print(f"Total chunks: {len(all_chunks)}")

    # Embed.
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Embedding...")
    embeddings = model.encode(
        [c["text"] for c in all_chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # required for cosine similarity via IndexFlatIP
        convert_to_numpy=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Build and save FAISS index.
    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (n={len(all_chunks)}, dim={dim})")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = str(output_prefix) + ".faiss"
    meta_path = str(output_prefix) + ".jsonl"

    faiss.write_index(index, faiss_path)
    print(f"Saved index:    {faiss_path}")

    with open(meta_path, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved metadata: {meta_path}")
    print(f"Done — {len(all_chunks)} chunks, dim={dim}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index NERSC docs markdown into a FAISS vector index.",
    )
    parser.add_argument(
        "--docs-root",
        required=True,
        help="Path to the docs/ directory of the cloned nersc.gitlab.io repo.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output path prefix (no extension). "
            "Creates <prefix>.faiss and <prefix>.jsonl."
        ),
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-small-en-v1.5", #tried with BAAI/bge-large-en-v1.5
        help="Sentence-transformers model for embeddings (default: BAAI/bge-small-en-v1.5).",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=3000,
        help="Maximum characters per chunk (default: 3000).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    build_index(
        docs_root=Path(args.docs_root),
        output_prefix=Path(args.output),
        model_name=args.model,
        max_chunk_chars=args.max_chunk_chars,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
