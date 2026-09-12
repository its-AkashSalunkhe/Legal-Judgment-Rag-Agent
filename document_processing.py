"""
Node 1 + 2: Judgment Input + Document Processing
Takes a judgment/legal PDF, returns a list of paragraphs/sections with metadata
(chapter, section numbers mentioned, paragraph id). Handles text-layer PDFs and
scanned PDFs (OCR fallback).
"""

import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

BOILERPLATE_PATTERNS = [
    r"SCC Online Web Edition.*",
    r"Printed For:.*",
    r"TruePrint.*source.*",
    r"this judgment is protected by the law declared.*",
    r"Modak,\s*\(2008\)\s*1\s*SCC\s*1\s*paras.*",
    r"^-{5,}$",
    r"^Page \d+ .*\d{4}$",
    r"msrlawbooks.*",
    r"^\d+\s?egaP\s?$",           # reversed "Page N" footer artifact
    r"^\s*P\s?T\s?O\s*$",         # "please turn over" footer
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

MAX_CHUNK_CHARS = 1000
OVERLAP_CHARS = 150


def extract_page_text(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


def ocr_page_text(pdf_path):
    images = convert_from_path(pdf_path, dpi=200)
    pages = []
    for img in images:
        text = pytesseract.image_to_string(img)
        pages.append(text)
    return pages


def clean_lines(text):
    lines = text.split("\n")
    kept = [l for l in lines if not BOILERPLATE_RE.search(l.strip())]
    return "\n".join(kept)


def looks_scanned(cleaned_pages):
    total_chars = sum(len(p.strip()) for p in cleaned_pages)
    return total_chars < 200 * len(cleaned_pages)


def load_judgment_text(pdf_path):
    pages = extract_page_text(pdf_path)
    cleaned = [clean_lines(p) for p in pages]
    if looks_scanned(cleaned):
        pages = ocr_page_text(pdf_path)
        cleaned = [clean_lines(p) for p in pages]
    return "\n".join(cleaned)


FRONTMATTER_END_RE = re.compile(r"\n\s*CHAPTER\s+[IVXLC1]+\b", re.IGNORECASE)

SPLIT_RE = re.compile(
    r"(?:\n|^)\s*("
    r"CHAPTER\s+[IVXLC\d]+[^\n]*"          # chapter heading
    r"|Ch\.\s?\d+-\d+\s+[^\n]*"            # sub-chapter heading e.g. "Ch. 1-2 Common roll"
    r"|\d{1,3}[A-Z]{0,2}\.\s*(?=[A-Z(])"   # numbered paragraph / bare-act section marker
    r")"
)

JUNK_PATTERNS = [
    r"^CONTENTS$", r"^QUESTIONS BANK", r"^Chapters\s+Page$",
]
JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)

SUBHEADING_SPLIT_RE = re.compile(r"(?:\n|^)\s*(?:[ivx]{1,4}\)|\d\))\s+(?=[A-Z])")

SECTION_MENTION_RE = re.compile(r"\bS(?:ection|n)\.?\s?(\d{1,3}[A-Za-z]{0,2})\b", re.IGNORECASE)


def extract_sections(text):
    return sorted(set(m.group(1) for m in SECTION_MENTION_RE.finditer(text)))


def split_large_chunk(text, base_id):
    if len(text) <= MAX_CHUNK_CHARS:
        return [(base_id, text)]
    pieces = SUBHEADING_SPLIT_RE.split(text)
    if len(pieces) <= 1:
        # no natural subheadings found - just cut every MAX_CHUNK_CHARS on a space
        out = []
        i = 0
        part = 1
        while i < len(text):
            cut = text.rfind(" ", i, i + MAX_CHUNK_CHARS)
            if cut <= i:
                cut = min(i + MAX_CHUNK_CHARS, len(text))
            out.append((f"{base_id}-{part}", text[i:cut].strip()))
            i = cut
            part += 1
        return [p for p in out if p[1]]
    out = []
    for idx, piece in enumerate(pieces, start=1):
        piece = piece.strip()
        if piece:
            out.append((f"{base_id}-{idx}", piece))
    return out


def split_into_paragraphs(full_text, judgment_name="Judgment"):
    """
    Splits text on chapter headings, sub-chapter headings, and numbered
    paragraph/section markers. Drops obvious front matter (title page, table
    of contents, question banks). Returns list of dicts:
    {judgment, para_no, chapter, sections, text}
    """
    cut = FRONTMATTER_END_RE.search(full_text)
    if cut:
        full_text = full_text[cut.start():]

    matches = list(SPLIT_RE.finditer(full_text))
    raw_chunks = []  # (marker_text_or_None, para_id, body)

    if not matches:
        blocks = [c.strip() for c in full_text.split("\n\n") if c.strip()]
        for i, b in enumerate(blocks, start=1):
            raw_chunks.append((None, str(i), b))
    else:
        preamble = full_text[: matches[0].start()].strip()
        if preamble and not JUNK_RE.search(preamble):
            raw_chunks.append((None, "preamble", preamble))
        for i, m in enumerate(matches):
            marker = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            body = full_text[start:end].strip()
            if body:
                raw_chunks.append((marker, marker, body))

    paragraphs = []
    current_chapter = None
    in_reference_section = False

    for marker, para_id, body in raw_chunks:
        if JUNK_RE.search(body) or len(body) < 15:
            continue

        # a chapter/sub-chapter marker updates the running chapter label, but
        # the body text captured after it is still real content and must
        # still turn into paragraphs below - it must NOT be skipped
        is_heading_marker = marker and (marker.upper().startswith("CHAPTER") or marker.startswith("Ch."))
        if is_heading_marker:
            current_chapter = marker

        if re.search(r"REFERENCE SECTION|SELECTED SECTIONS", body, re.IGNORECASE):
            in_reference_section = True

        # a bare list-number ("1.", "2.") only counts as a real statutory
        # section reference once we're in the reference/bare-act part of the
        # document - otherwise it's just an ordinary numbered list item
        section_no = None
        if marker and not is_heading_marker and in_reference_section:
            m = re.match(r"^(\d{1,3}[A-Z]{0,2})\.", marker)
            if m:
                section_no = m.group(1)

        pieces = split_large_chunk(body, para_id)
        prev_tail = ""  # overlap only stitches pieces cut from the same chunk
        for sub_id, sub_text in pieces:
            context_text = (prev_tail + " " + sub_text).strip() if prev_tail else sub_text
            paragraphs.append({
                "judgment": judgment_name,
                "para_no": sub_id,
                "chapter": current_chapter,
                "sections": extract_sections(sub_text) + ([section_no] if section_no else []),
                "text": context_text,
            })
            prev_tail = sub_text[-OVERLAP_CHARS:] if len(sub_text) > OVERLAP_CHARS else ""

    return paragraphs


def process_judgment(pdf_path, judgment_name):
    text = load_judgment_text(pdf_path)
    return split_into_paragraphs(text, judgment_name)
