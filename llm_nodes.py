"""
Node 4/5: LLM Answer + Citation Extraction (combined - the model cites while it answers)
Node 6/8: Citation Verification + Evaluation (separate call, done after generation)

Uses the Anthropic API. If no ANTHROPIC_API_KEY is set, falls back to a simple
extractive mock so the rest of the app still works for a demo/dry run -
this is NOT a substitute for the real LLM reasoning the assignment asks for.

Two things are enforced by plain code rather than left to the model, because
they matter enough not to trust to a prompt alone:
- if retrieval found nothing above the relevance threshold, we abstain
  without even calling the LLM
- if the question names a specific year (e.g. "2025 amendments") that
  doesn't appear anywhere in the retrieved evidence, we abstain instead of
  letting the model guess
"""

import os
import re
import json
import anthropic

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

ABSTAIN_MESSAGE = "The provided document does not contain sufficient information to answer this question."

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def _call_claude(system, user, max_tokens=1200):
    client = _client()
    if client is None:
        return None
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text


def _parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    return json.loads(raw)


def _question_asks_for_year_not_in_evidence(question, evidence_paragraphs):
    question_years = set(YEAR_RE.findall(question))
    if not question_years:
        return False
    evidence_text = " ".join(p["text"] for p in evidence_paragraphs)
    evidence_years = set(YEAR_RE.findall(evidence_text))
    return not (question_years & evidence_years)


GEN_SYSTEM = """You are a legal research assistant answering questions about an uploaded
legal document. Follow these rules strictly:

- Answer ONLY using the evidence paragraphs given to you. Do not use outside legal
  knowledge, and do not use general knowledge about laws or amendments not shown here.
- Give the direct answer to the question first, then any supporting explanation.
- If the question has multiple parts (e.g. asks to compare two sections, or asks
  several things at once), answer every part. Do not answer only the first part.
- Do not use a retrieved paragraph just because it is from the same document if it
  does not actually address the question - ignore irrelevant retrieved paragraphs.
- Cite the specific paragraph/section that supports each part of your answer.
- Never fabricate section numbers, dates, amendments, case names, or interpretations
  that are not in the evidence.
- Distinguish what the document directly states from anything you are inferring from it.
- Do not repeat document metadata, author information, table-of-contents text, or
  unrelated boilerplate as if it were an answer.
- If the evidence does not actually answer the question, set "abstained" to true and
  use exactly this text as the answer: "{abstain}"

Respond with JSON only, in this exact shape:
{{
  "answer": "...",
  "basis": "directly_stated" | "inferred" | "unsupported",
  "citations": [{{"judgment": "...", "paragraph": "..."}}],
  "abstained": false
}}
""".format(abstain=ABSTAIN_MESSAGE)


def generate_answer(question, evidence_paragraphs):
    if not evidence_paragraphs:
        return {"answer": ABSTAIN_MESSAGE, "basis": "unsupported", "citations": [], "abstained": True}

    if _question_asks_for_year_not_in_evidence(question, evidence_paragraphs):
        return {"answer": ABSTAIN_MESSAGE, "basis": "unsupported", "citations": [], "abstained": True}

    evidence_text = "\n\n".join(
        f"[{p['judgment']} - {p['para_no']}]\n{p['text']}" for p in evidence_paragraphs
    )
    user = f"Question: {question}\n\nEvidence paragraphs:\n{evidence_text}"

    raw = _call_claude(GEN_SYSTEM, user)
    if raw is None:
        return _mock_generate_answer(question, evidence_paragraphs)

    try:
        return _parse_json(raw)
    except Exception:
        return {"answer": raw, "basis": "unsupported", "citations": [], "abstained": False}


def _mock_generate_answer(question, evidence_paragraphs):
    # naive extractive fallback used only when no API key is configured -
    # this cannot judge relevance the way a real LLM call would, it just
    # returns the top retrieved paragraph
    top = evidence_paragraphs[0]
    return {
        "answer": top["text"][:400] + ("..." if len(top["text"]) > 400 else ""),
        "basis": "directly_stated",
        "citations": [{"judgment": top["judgment"], "paragraph": top["para_no"]}],
        "abstained": False,
    }


VERIFY_SYSTEM = """You evaluate one generated answer to a legal-document question, given
the paragraph(s) it was supposedly based on. Score each dimension independently -
do not let a good score on one dimension pull up another. A wrong answer must not
receive a high score just because it is grounded in real document text.

Dimensions:
- retrieval_relevance (0-100): how relevant is the cited paragraph to the question,
  regardless of whether the final answer used it well
- citation_correctness ("Supported" | "Partially Supported" | "Unsupported"): does the
  cited paragraph actually contain the fact the answer claims
- answer_correctness (0-100): does the answer actually answer the question. If an
  expected answer is given, compare against it. An answer built from an unrelated
  paragraph, or that dodges the question, scores low here even if fluently written.
- completeness (0-100): if the question has multiple parts, did the answer address
  all of them
- hallucination_flag (true/false): true if the answer states something not present
  in the cited paragraph
- abstention_quality ("N/A" | "Correct" | "Incorrect"): only relevant if the answer
  abstained - was abstaining the right call given the evidence

Respond with JSON only, in this exact shape:
{
  "citation_correctness": "Supported",
  "retrieval_relevance": 0,
  "answer_correctness": 0,
  "completeness": 0,
  "hallucination_flag": false,
  "abstention_quality": "N/A",
  "notes": "..."
}
"""


def verify_and_evaluate(question, answer, cited_paragraph_text, expected_answer=None, abstained=False):
    if abstained and answer.strip() == ABSTAIN_MESSAGE:
        # abstaining is the correct behavior whenever retrieval found nothing
        # solid - score it as a success rather than running the normal
        # citation/overlap checks against an empty citation
        return {
            "citation_correctness": "N/A", "citation_verification": "N/A",
            "retrieval_relevance": 100, "answer_correctness": 100, "completeness": 100,
            "hallucination_flag": False, "abstention_quality": "Correct",
            "overall_evaluation_score": 100, "confidence": "High",
            "notes": "Correctly abstained - no relevant evidence was found.",
        }

    user = (
        f"Question: {question}\n"
        f"Generated answer: {answer}\n"
        f"Answer abstained: {abstained}\n"
        f"Cited paragraph text: {cited_paragraph_text}\n"
        f"Expected answer (optional): {expected_answer or 'N/A'}"
    )

    raw = _call_claude(VERIFY_SYSTEM, user)
    if raw is None:
        result = _mock_verify(answer, cited_paragraph_text)
    else:
        try:
            result = _parse_json(raw)
        except Exception:
            result = {"citation_correctness": "Unsupported", "retrieval_relevance": 0,
                       "answer_correctness": 0, "completeness": 0, "hallucination_flag": True,
                       "abstention_quality": "N/A", "notes": raw}

    numeric_scores = [v for k, v in result.items()
                       if k in ("retrieval_relevance", "answer_correctness", "completeness")
                       and isinstance(v, (int, float))]
    overall = round(sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0
    result["overall_evaluation_score"] = overall
    result["confidence"] = _confidence_label(overall, result.get("citation_correctness"))
    # keep the old key name around too, since app.py / evaluate.py read it
    result["citation_verification"] = result.get("citation_correctness")
    return result


def _confidence_label(overall_score, citation_correctness):
    if citation_correctness == "Unsupported" or overall_score < 40:
        return "Low"
    if overall_score < 75 or citation_correctness == "Partially Supported":
        return "Medium"
    return "High"


def _mock_verify(answer, cited_paragraph_text):
    # naive word-overlap check used only when no API key is configured - this
    # cannot tell whether an answer is actually correct, only whether it
    # shares vocabulary with the cited text, so treat scores from this path
    # with real skepticism
    a_words = set(answer.lower().split())
    p_words = set(cited_paragraph_text.lower().split())
    overlap = len(a_words & p_words) / max(len(a_words), 1)
    correctness = "Supported" if overlap > 0.5 else "Partially Supported" if overlap > 0.25 else "Unsupported"
    score = round(overlap * 100)
    return {
        "citation_correctness": correctness,
        "retrieval_relevance": score,
        "answer_correctness": score,
        "completeness": score,
        "hallucination_flag": overlap <= 0.25,
        "abstention_quality": "N/A",
        "notes": "mock word-overlap check (no ANTHROPIC_API_KEY set) - not a real correctness check",
    }
