import os
import json
import tempfile
import streamlit as st

from document_processing import process_judgment
from retrieval import get_top_paragraphs
from llm_nodes import generate_answer, verify_and_evaluate

st.set_page_config(page_title="Legal Judgment Analysis Agent by Akash Salunkhe", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e6e6e6; }
[data-testid="stSidebar"] { background-color: #161a23; }
h1, h2, h3, h4, p, label, span, div { color: #e6e6e6; }
.stButton>button {
    background-color: #ff4b4b; color: white; border: none;
    border-radius: 8px; padding: 0.5em 1.2em; font-weight: 600;
}
.stButton>button:hover { background-color: #ff6b6b; }
.stTextInput>div>div>input, .stTextArea textarea {
    background-color: #1c1f26; color: #e6e6e6; border: 1px solid #333;
}
[data-testid="stMetricValue"] { color: #00d4aa; }
[data-testid="stExpander"] { background-color: #161a23; border-radius: 8px; }
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 14px;
    font-weight: 600; font-size: 0.85em;
}
.badge-supported { background-color: #1f8a4c; color: white; }
.badge-partial { background-color: #b5860a; color: white; }
.badge-unsupported { background-color: #b5312b; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("Legal Judgment Analysis & Citation Verification Agent by Akash Salunkhe")

with st.sidebar:
    st.header("Setup")
    api_key_input = st.text_input("Anthropic API key (optional)", type="password",
                                   help="If left empty, ANTHROPIC_API_KEY from the environment is used. "
                                        "Without a key the app falls back to a naive mock answer, just so the UI still works.")
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
    top_k = st.slider("Paragraphs to retrieve", 1, 10, 5)


def verification_badge(label):
    css_class = {
        "Supported": "badge-supported",
        "Partially Supported": "badge-partial",
        "Unsupported": "badge-unsupported",
    }.get(label, "badge-partial")
    st.markdown(f'<span class="badge {css_class}">{label}</span>', unsafe_allow_html=True)


if "paragraphs" not in st.session_state:
    st.session_state.paragraphs = []

tab1, tab2 = st.tabs(["Ask a question", "Batch evaluation"])

with tab1:
    st.subheader("1. Upload judgment(s)")
    files = st.file_uploader("Judgment PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Process judgments") and files:
        all_paragraphs = []
        progress = st.progress(0, text="Starting...")
        for i, f in enumerate(files):
            progress.progress((i) / len(files), text=f"Processing {f.name}...")
            tmp_path = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp_path, "wb") as out:
                out.write(f.read())
            paras = process_judgment(tmp_path, judgment_name=f.name)
            all_paragraphs.extend(paras)
        progress.progress(1.0, text="Done")
        st.session_state.paragraphs = all_paragraphs
        st.success(f"Processed {len(files)} judgment(s), extracted {len(all_paragraphs)} paragraphs.")
        st.balloons()

    if st.session_state.paragraphs:
        st.caption(f"{len(st.session_state.paragraphs)} paragraphs loaded and ready to query.")

    st.subheader("2. Ask a question")
    question = st.text_input("Your question about the uploaded judgment(s)")

    if st.button("Get answer") and question and st.session_state.paragraphs:
        with st.spinner("Retrieving evidence and generating answer..."):
            evidence = get_top_paragraphs(st.session_state.paragraphs, question, k=top_k)
            result = generate_answer(question, evidence)

            cited_text = ""
            if result.get("citations"):
                first = result["citations"][0]
                match = next((p for p in evidence
                              if p["judgment"] == first.get("judgment") and p["para_no"] == first.get("paragraph")),
                             None)
                cited_text = match["text"] if match else ""

            verification = verify_and_evaluate(question, result.get("answer", ""), cited_text,
                                                abstained=result.get("abstained", False))

        st.markdown("### Answer")
        if result.get("abstained"):
            st.warning(result.get("answer", ""))
        else:
            st.write(result.get("answer", ""))
        st.caption(f"Basis: {result.get('basis')}  |  Abstained: {result.get('abstained')}")

        st.markdown("### Citations")
        st.json(result.get("citations", []))

        st.markdown("### Retrieved evidence")
        if not evidence:
            st.caption("No evidence cleared the relevance threshold.")
        for p in evidence:
            chapter = f" ({p['chapter']})" if p.get("chapter") else ""
            with st.expander(f"{p['judgment']} - {p['para_no']}{chapter}  (score {p['score']})"):
                st.write(p["text"])

        st.markdown("### Citation verification & evaluation")
        verification_badge(verification.get("citation_verification", "Unsupported"))
        st.caption(f"Confidence: {verification.get('confidence')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval relevance", verification.get("retrieval_relevance"))
        c2.metric("Answer correctness", verification.get("answer_correctness"))
        c3.metric("Completeness", verification.get("completeness"))
        c4.metric("Overall score", verification.get("overall_evaluation_score"))
        st.write("Hallucination flag:", verification.get("hallucination_flag"))
        st.write("Abstention quality:", verification.get("abstention_quality"))
        st.write("Notes:", verification.get("notes"))

with tab2:
    st.subheader("Run a predefined set of evaluation questions")
    eval_file = st.file_uploader("Evaluation questions (JSON list of {question, expected_answer})",
                                  type=["json"], key="eval_upload")

    if st.button("Run evaluation") and eval_file and st.session_state.paragraphs:
        questions = json.load(eval_file)
        rows = []
        progress = st.progress(0, text="Starting evaluation...")
        for i, item in enumerate(questions):
            progress.progress(i / len(questions), text=f"Evaluating question {i+1}/{len(questions)}...")
            evidence = get_top_paragraphs(st.session_state.paragraphs, item["question"], k=top_k)
            result = generate_answer(item["question"], evidence)
            cited_text = evidence[0]["text"] if evidence else ""
            verification = verify_and_evaluate(item["question"], result.get("answer", ""), cited_text,
                                                item.get("expected_answer"), abstained=result.get("abstained", False))
            rows.append({
                "question": item["question"],
                "answer": result.get("answer"),
                "abstained": result.get("abstained"),
                "citation_verification": verification.get("citation_verification"),
                "retrieval_relevance": verification.get("retrieval_relevance"),
                "answer_correctness": verification.get("answer_correctness"),
                "overall_evaluation_score": verification.get("overall_evaluation_score"),
                "confidence": verification.get("confidence"),
                "hallucination_flag": verification.get("hallucination_flag"),
            })
        progress.progress(1.0, text="Done")
        st.dataframe(rows)
        avg = sum(r["overall_evaluation_score"] for r in rows) / len(rows) if rows else 0
        st.metric("Average overall score", round(avg, 1))
        st.download_button("Download results as JSON", json.dumps(rows, indent=2),
                            file_name="evaluation_results.json")
