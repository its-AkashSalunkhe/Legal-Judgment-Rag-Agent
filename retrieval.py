"""
Node 3: Retrieval
TF-IDF + cosine similarity over paragraphs, with two additions the plain
version didn't have:
- if the question names an exact section number ("Section 30"), chunks that
  are actually tagged with that section get a big score boost, since keyword
  overlap alone under-ranks short, precise statutory text
- if the question names two or more sections (a comparison question, e.g.
  "difference between Section 30 and Section 32"), each side is retrieved
  separately so the weaker side doesn't get crowded out
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CANDIDATE_POOL = 15
SECTION_BOOST = 0.5
MIN_SCORE = 0.06

QUESTION_SECTION_RE = re.compile(r"\bS(?:ection|n)\.?\s?(\d{1,3}[A-Za-z]{0,2})\b", re.IGNORECASE)


def _rank(paragraphs, question, pool_size):
    if not paragraphs:
        return []
    texts = [p["text"] for p in paragraphs]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:pool_size]


def _boost_for_sections(ranked, wanted_sections):
    boosted = []
    for p, score in ranked:
        if wanted_sections and any(s in p.get("sections", []) for s in wanted_sections):
            score = score + SECTION_BOOST
        boosted.append((p, score))
    return sorted(boosted, key=lambda x: x[1], reverse=True)


def get_top_paragraphs(paragraphs, question, k=5):
    wanted_sections = QUESTION_SECTION_RE.findall(question)

    if len(set(wanted_sections)) >= 2:
        return _get_comparison_paragraphs(paragraphs, question, wanted_sections, k)

    ranked = _rank(paragraphs, question, CANDIDATE_POOL)
    ranked = _boost_for_sections(ranked, wanted_sections)

    if not ranked or ranked[0][1] < MIN_SCORE:
        return []

    top = [{**p, "score": round(float(s), 4)} for p, s in ranked[:k] if s > 0]
    return top


def _get_comparison_paragraphs(paragraphs, question, wanted_sections, k):
    per_section_k = max(1, k // len(set(wanted_sections)))
    ranked = _rank(paragraphs, question, CANDIDATE_POOL)
    ranked = _boost_for_sections(ranked, wanted_sections)

    combined = []
    seen = set()
    for section in dict.fromkeys(wanted_sections):
        matches = [(p, s) for p, s in ranked if section in p.get("sections", [])]
        for p, s in matches[:per_section_k]:
            key = (p["judgment"], p["para_no"])
            if key not in seen:
                combined.append((p, s))
                seen.add(key)

    for p, s in ranked:
        if len(combined) >= k:
            break
        key = (p["judgment"], p["para_no"])
        if key not in seen and s > 0:
            combined.append((p, s))
            seen.add(key)

    if not combined:
        return []

    return [{**p, "score": round(float(s), 4)} for p, s in combined[:k]]
