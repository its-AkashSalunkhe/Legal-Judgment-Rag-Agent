"""
Runs the full pipeline over a set of judgments + a predefined set of
evaluation questions, and writes the generated artefacts to /outputs.

Usage: python evaluate.py
"""

import json
import os

from document_processing import process_judgment
from retrieval import get_top_paragraphs
from llm_nodes import generate_answer, verify_and_evaluate

SUITES = [
    {
        "name": "judgments",
        "judgments": [
            ("sample_data/judgment_1.pdf", "Jaipur_Mineral_v_CIT"),
            ("sample_data/judgment_2.pdf", "Vasudev_Modi_v_Rajabhai"),
        ],
        "eval_questions_file": "sample_data/eval_questions.json",
    },
    {
        "name": "advocates_act",
        "judgments": [
            ("sample_data/advocates_act.pdf", "ADVOCATES_ACT_f.pdf"),
        ],
        "eval_questions_file": "sample_data/eval_questions_advocates_act.json",
    },
]

OUTPUT_DIR = "outputs"


def load_paragraphs(judgments):
    all_paragraphs = []
    for path, name in judgments:
        if os.path.exists(path):
            all_paragraphs.extend(process_judgment(path, name))
    return all_paragraphs


def run_suite(suite):
    paragraphs = load_paragraphs(suite["judgments"])
    print(f"[{suite['name']}] loaded {len(paragraphs)} paragraphs from {len(suite['judgments'])} document(s).")

    with open(suite["eval_questions_file"]) as f:
        eval_questions = json.load(f)

    results = []
    for item in eval_questions:
        question = item["question"]
        evidence = get_top_paragraphs(paragraphs, question, k=5)
        answer = generate_answer(question, evidence)

        cited_text = ""
        if answer.get("citations"):
            first = answer["citations"][0]
            match = next((p for p in evidence
                          if p["judgment"] == first.get("judgment") and p["para_no"] == first.get("paragraph")),
                         None)
            cited_text = match["text"] if match else (evidence[0]["text"] if evidence else "")
        elif evidence:
            cited_text = evidence[0]["text"]

        verification = verify_and_evaluate(question, answer.get("answer", ""), cited_text,
                                            item.get("expected_answer"), abstained=answer.get("abstained", False))

        results.append({
            "question": question,
            "generated_answer": answer.get("answer"),
            "basis": answer.get("basis"),
            "abstained": answer.get("abstained"),
            "citations": answer.get("citations"),
            "retrieved_evidence": [{"judgment": p["judgment"], "paragraph": p["para_no"], "score": p["score"]}
                                    for p in evidence],
            "citation_verification": verification.get("citation_verification"),
            "hallucination_flag": verification.get("hallucination_flag"),
            "retrieval_relevance": verification.get("retrieval_relevance"),
            "answer_correctness": verification.get("answer_correctness"),
            "completeness": verification.get("completeness"),
            "confidence": verification.get("confidence"),
            "overall_evaluation_score": verification.get("overall_evaluation_score"),
        })

    return results


def write_outputs(name, results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(f"{OUTPUT_DIR}/{name}_qna_output.json", "w") as f:
        json.dump(results, f, indent=2)

    avg_score = sum(r["overall_evaluation_score"] for r in results) / len(results) if results else 0
    with open(f"{OUTPUT_DIR}/{name}_evaluation_report.md", "w") as f:
        f.write(f"# Evaluation Report - {name}\n\n")
        f.write(f"Questions evaluated: {len(results)}\n\n")
        f.write(f"Average overall evaluation score: {round(avg_score, 1)}/100\n\n")
        f.write("| Question | Citation verification | Abstained | Hallucination | Overall score |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['question']} | {r['citation_verification']} | {r['abstained']} | "
                     f"{r['hallucination_flag']} | {r['overall_evaluation_score']} |\n")

    print(f"Wrote {OUTPUT_DIR}/{name}_qna_output.json and {OUTPUT_DIR}/{name}_evaluation_report.md")


if __name__ == "__main__":
    for suite in SUITES:
        results = run_suite(suite)
        write_outputs(suite["name"], results)
