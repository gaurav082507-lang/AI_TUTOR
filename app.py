from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import streamlit as st

from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

# =========================================================================
# ================  ORIGINAL BACKEND LOGIC (UNCHANGED)  =================
# =========================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    asked_question: str
    query_type: str
    retrieved_context: str
    retrieved_concepts: str
    question_paper: str
    human_response: str
    evaluation_report: str


@st.cache_resource(show_spinner=False)
def build_retriever(pdf_path):
    data = PyPDFLoader(pdf_path)
    docs = data.load()
    embedding_model = MistralAIEmbeddings(model='mistral-embed-2312')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunk_docs = splitter.split_documents(docs)
    vector_store = Chroma.from_documents(
        documents=chunk_docs,
        embedding=embedding_model
    )
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={
            'k': 5,
            'fetch_k': 20,
            'lambda_mult': 0.5
        }
    )
    return retriever


CLASSIFIER_SYSTEM_PROMPT = """You are a strict binary text classifier. Read the user's input and classify it into exactly one of these two labels:

LABEL 1: "Question asked to LLM"
Definition: The user is directly asking the LLM something and expects an answer, explanation, fact, opinion, or solution in response.
Examples:
- "What is the capital of France?"
- "How does photosynthesis work?"
- "Can you explain quantum entanglement?"
- "Why is my code throwing a null pointer exception?"

LABEL 2: "Telling the LLM to give Question"
Definition: The user is instructing the LLM to produce, generate, write, or output one or more questions — the deliverable itself is a question, not an answer.
Examples:
- "Give me 5 interview questions about Python."
- "Ask me a trivia question."
- "Write a quiz question about World War II."
- "Generate 10 exam questions on thermodynamics."
- "Create a question to test someone's knowledge of history."

Classification rules:
1. Judge intent, not just sentence structure. A sentence phrased as a question (ending in "?") can still belong to LABEL 2 if it's asking the LLM to produce a question (e.g., "Can you frame a question about gravity?" → LABEL 2).
2. A sentence phrased as an imperative can still belong to LABEL 1 if it's really asking for information (e.g., "Tell me the capital of France" → LABEL 1).
3. Key signal for LABEL 2: verbs like "give," "generate," "write," "create," "frame," "ask," "prepare," "list" combined with the word "question(s)" as the object being produced.
4. Key signal for LABEL 1: the user wants a fact, explanation, opinion, calculation, or solution — the word "question" (if present) refers to what they're asking, not what they want produced.
5. If genuinely ambiguous, default to LABEL 1, since most inputs to an LLM are direct questions.

Output constraints:
- Output must be exactly one of the two strings below, with no extra characters, punctuation, quotes, or explanation:
Question asked to LLM
Telling the LLM to give Question"""

LLM = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.4)


def classifier(state: State) -> dict:
    question_asked = state['asked_question']
    user_prompt = f"This is the Asked Question :{question_asked}"
    messages = [('system', CLASSIFIER_SYSTEM_PROMPT), ('human', user_prompt)]
    response = LLM.invoke(messages)
    category = response.content
    if "question asked to llm" in category.lower():
        category = 'Question asked to LLM'
    elif "telling the llm to give question" in category.lower():
        category = 'Telling the LLM to give Question'

    return {
        'messages': [('human', user_prompt), ('ai', response.content)],
        'query_type': category
    }


def give_context(state: State):
    question_asked = state['asked_question']
    docs = st.session_state.book_retriever.invoke(question_asked)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {
        'retrieved_context': context
    }


RAG_SYSTEM_PROMPT = """You are a precise question-answering assistant. You will be given a CONTEXT and a QUESTION. Your task is to answer the QUESTION using only the information provided in the CONTEXT, and to cite the parts of the CONTEXT that support your answer.

Rules:
1. Base your answer strictly on the given CONTEXT. Do not use outside knowledge, assumptions, or information not present in the CONTEXT.
2. If the CONTEXT contains the answer, respond clearly and concisely, directly addressing the QUESTION.
3. If the CONTEXT does not contain enough information to answer the QUESTION, respond with: "The provided context does not contain enough information to answer this question." Do not provide citations in this case.
4. Do not fabricate, infer beyond what is stated, or add information not explicitly supported by the CONTEXT.
5. If the QUESTION is ambiguous relative to the CONTEXT, ask for clarification instead of guessing.
6. Keep answers concise and to the point unless the QUESTION explicitly asks for a detailed explanation.

Citation rules:
1. Every factual claim in your answer that comes from the CONTEXT must be followed by a citation marker indicating its source.
2. If the CONTEXT is a single passage without labeled sections, cite using the relevant sentence or line number, e.g., [Line 4].
3. If the CONTEXT is split into multiple labeled sections, documents, or paragraphs, cite using the label and, if available, the sentence/line number, e.g., [Doc 2, Line 3] or [Paragraph 1].
4. Use the minimum number of citations necessary to support each claim — do not over-cite.
5. Do not quote large verbatim chunks of the CONTEXT; paraphrase the content in your own words and cite the source location.
6. If a single claim draws from multiple parts of the CONTEXT, include all relevant citation markers together, e.g., [Line 2, Line 5].
7. Place citation markers immediately after the specific claim they support, not just at the end of the whole answer.

Input format:
CONTEXT: <the passage or document text, ideally with line numbers or section labels>
QUESTION: <the user's question>

Output format:
Answer: <clear, concise answer with inline citation markers after each supported claim>"""


def give_answer(state: State) -> dict:
    context = state['retrieved_context']
    question_asked = state['asked_question']
    user_prompt = (f"The question asked is {question_asked}"
                    f"The context is :{context}"
                    )
    parser = StrOutputParser()
    messages = [('system', RAG_SYSTEM_PROMPT), ('human', user_prompt)]
    chain = LLM | parser
    answer = chain.invoke(messages)

    return {
        'messages': [('human', user_prompt), ('ai', answer)]
    }


CONCEPT_EXTRACTOR_PROMPT = """You are a concept extraction node in a pipeline that generates question papers from a book. You will receive RETRIEVED_CHUNKS (text passages retrieved from the book). Your task is to extract the important concepts from these chunks so a downstream node can generate exam questions from them.

Rules:
1. Extract only concepts explicitly present in the RETRIEVED_CHUNKS. Do not use outside knowledge.
2. A concept can be: a definition, theory, principle, formula, process, named entity, or significant idea important enough to be asked about in an exam.
3. For each concept, include:
   - Concept name
   - A 1-2 line description grounded in the chunk
   - Source chunk id (if provided, else "null")
   - Difficulty level: easy, medium, or hard (based on how foundational vs advanced the concept is)
4. Skip narrative filler, transitional text, or content with no testable value.
5. Merge duplicate/repeated concepts across chunks into a single entry.
6. Preserve technical terminology exactly as used in the source.
7. Do not generate questions yourself — only extract concepts. Question generation is handled by a separate node.
8. If no meaningful concepts are found in the RETRIEVED_CHUNKS, return only: "No significant concepts found in the provided content."

Input format:
RETRIEVED_CHUNKS: <list of retrieved text chunks, each with an id if available>

Output format:
Return the extracted concepts as a single plain text string (no JSON, no markdown code blocks, no extra commentary), formatted as follows:

CONCEPT: <concept name>
DESCRIPTION: <brief grounded description>
SOURCE_CHUNK_ID: <id or null>
DIFFICULTY: <easy | medium | hard>

CONCEPT: <concept name>
DESCRIPTION: <brief grounded description>
SOURCE_CHUNK_ID: <id or null>
DIFFICULTY: <easy | medium | hard>

(Repeat this block for each extracted concept, separated by a blank line.)"""


def concept_extractor(state: State) -> dict:
    docs = st.session_state.book_retriever.invoke(
        "Extract all key concepts, definitions, theories, formulas, important terms, named entities, and core ideas covered in this content, suitable for generating exam or question paper questions.")
    chunks_text = "\n\n".join([f"[chunk_id: {i}]\n{doc.page_content}" for i, doc in enumerate(docs)])
    user_prompt = f"RETRIEVED_CHUNKS: {chunks_text}"
    messages = [('system', CONCEPT_EXTRACTOR_PROMPT), ('human', user_prompt)]
    concepts = LLM.invoke(messages)

    return {
        'messages': [('human', user_prompt), ('ai', concepts.content)],
        'retrieved_concepts': concepts.content
    }


PAPER_MAKEING_PROMPT = """You are a question paper generation assistant. You will be given a plain text string called CONCEPTS_TEXT, containing a list of concepts extracted from a book. Your task is to generate a complete question paper worth a total of 30 MARKS using only these concepts.

CONCEPTS_TEXT format (each concept is a block separated by a blank line):
CONCEPT: <concept name>
DESCRIPTION: <brief grounded description>
SOURCE_CHUNK_ID: <id or null>
DIFFICULTY: <easy | medium | hard>

Rules:
1. Parse CONCEPTS_TEXT to identify each concept block using the CONCEPT, DESCRIPTION, SOURCE_CHUNK_ID, and DIFFICULTY labels.
2. Use only the concepts provided in CONCEPTS_TEXT. Do not introduce topics, facts, or concepts not present in the given text.
3. Use each concept's DESCRIPTION as the grounding source for question content, and DIFFICULTY to decide question type.
4. Distribute marks across a mix of question types to cover different difficulty levels and testing depths. Suggested structure (adjust proportionally if concept count is limited):
   - Very Short Answer / MCQ questions: 1 mark each
   - Short Answer questions: 2-3 marks each
   - Long Answer / Descriptive questions: 5 marks each
5. Ensure the total marks across all questions sum to exactly 30. Do not exceed or fall short.
6. Prioritize concepts with DIFFICULTY "easy" for low-mark questions, "medium" for short-answer questions, and "hard" for long-answer/descriptive questions.
7. Avoid generating multiple questions from the same concept unless the concept is broad enough to justify it (e.g., a major theory) — otherwise, ensure a good spread across distinct concepts.
8. Each question must be answerable strictly from the given concept's DESCRIPTION — do not require outside knowledge.
9. Do not include answer keys or model answers unless explicitly requested — only generate the questions.
10. Number the questions sequentially and clearly indicate the marks allotted to each question.
11. Group questions into sections by type (e.g., Section A, Section B, Section C) for readability.
12. If CONCEPTS_TEXT is empty, malformed, or contains no valid concept blocks, respond only with: "Invalid or empty concepts data provided."
13. If the given concepts are insufficient to reasonably fill 30 marks without repetition, generate as many quality questions as possible and add a note at the end stating that additional concepts are needed rather than padding with irrelevant or repeated questions.

Input format:
CONCEPTS_TEXT: <plain text string of concept blocks as described above>

Output format:
Return the question paper as a single plain text string, formatted as follows (no JSON, no markdown code blocks, no extra commentary):

QUESTION PAPER
Total Marks: 30

4. Use EXACTLY this structure — do not deviate from these counts:
   - Section A (1 Mark Questions): exactly 5 questions × 1 mark = 5 marks
   - Section B (Short Answer Questions): exactly 5 questions × 3 marks = 15 marks
   - Section C (Long Answer Questions): exactly 2 questions × 5 marks = 10 marks
   Total = 5 + 15 + 10 = 30 marks.
   Do not add, remove, or reweight any question. If you cannot fill a section with distinct concepts, reuse the closest related concept rather than changing the count or marks per question.

(Total marks must sum to exactly 30. If concepts are insufficient, add a final line: "Note: Additional concepts required to reach full 30 marks.")"""


def make_paper(state: State):
    concepts = state['retrieved_concepts']
    user_prompt = f"This is the Retrieved Concept :{concepts}"
    messages = [('system', PAPER_MAKEING_PROMPT), ('human', user_prompt)]
    parser = StrOutputParser()
    chains = LLM | parser
    question_paper = chains.invoke(messages)

    return {
        'messages': [('human', user_prompt), ('ai', question_paper)],
        'question_paper': question_paper
    }


def human_answer_node(state: State) -> dict:
    """
    Pauses graph execution and waits for the human to submit answers
    to the generated question paper.
    """
    instructions = """
INSTRUCTIONS TO ANSWER THE QUESTION PAPER
-------------------------------------------
1. Read each question carefully before answering.
2. Answer questions section-wise, in the same order as given (Section A, then B, then C).
3. For Section A (1 Mark Questions): Give short, direct answers (a word, phrase, or one line).
4. For Section B (Short Answer Questions): Answer in 2-4 sentences, covering the key points relevant to the marks allotted.
5. For Section C (Long Answer Questions): Provide detailed, well-structured answers with explanation, examples, or steps where applicable.
6. Clearly label each answer with its corresponding question number (e.g., "A1:", "B2:", "C1:").
7. Do not skip any question. If unsure, write "Not Attempted" against that question number.
8. Submit your final answers as a single plain text response, in the following format:

A1: <your answer>
A2: <your answer>
...
B1: <your answer>
B2: <your answer>
...
C1: <your answer>
C2: <your answer>
...

Please review the QUESTION PAPER below and submit your answers in the format specified above.
"""

    payload = {
        "instructions": instructions.strip(),
        "question_paper": state["question_paper"]
    }

    # Pauses execution here; resumes when Command(resume=<human_input>) is sent
    human_response = interrupt(payload)

    return {
        "human_response": human_response
    }


EVALUATER_SYSTEM_PROMPT = """You are an answer evaluation assistant. You will be given three inputs: the QUESTION_PAPER, the CONCEPTS_TEXT (source material the questions were generated from), and the HUMAN_ANSWERS submitted by the student. Your task is to evaluate the HUMAN_ANSWERS strictly against the QUESTION_PAPER and CONCEPTS_TEXT, assign marks, and provide detailed strength/weakness feedback.

Rules:
1. Use CONCEPTS_TEXT as the source of truth for correctness. Do not use outside knowledge to judge answers — grounding must come only from the DESCRIPTION fields in CONCEPTS_TEXT.
2. Match each answer in HUMAN_ANSWERS to its corresponding question number in QUESTION_PAPER (e.g., "A1" in HUMAN_ANSWERS maps to Question 1 in Section A).
3. For each question, evaluate the answer based on:
   - Correctness: does it align with the concept's description?
   - Completeness: does it cover the key points expected for the marks allotted?
   - Clarity: is the answer well-expressed (minor weight compared to correctness/completeness)?
4. Award marks per question using the following logic:
   - Full marks: answer is correct and complete relative to the concept.
   - Partial marks: answer is partially correct, incomplete, or missing some key points. Award proportionally (e.g., half marks for a half-correct answer).
   - Zero marks: answer is incorrect, irrelevant, or marked "Not Attempted".
5. Do not award marks for information not grounded in CONCEPTS_TEXT, even if factually true in general knowledge — evaluation must stay scoped to the given source material.
6. Provide brief feedback (1-2 lines) for each question explaining the marks awarded, especially for partial or zero-mark answers.
7. Track which CONCEPT(s) each question was based on (via QUESTION_PAPER's "based_on_concept" reference if available, or by matching question content to CONCEPTS_TEXT).
8. Identify STRENGTH AREAS: concepts/topics where the student scored full or near-full marks consistently.
9. Identify WEAKNESS AREAS: concepts/topics where the student scored low, partial, or zero marks, or left answers unattempted. Group weak answers by their underlying concept, not just by question number, so patterns across similar topics are visible.
10. Sum all question-wise marks to compute the total score out of 30.
11. If HUMAN_ANSWERS is missing an answer for a question, treat it as "Not Attempted" and award 0 marks with feedback "No answer submitted," and include that concept under weakness areas.
12. If HUMAN_ANSWERS, QUESTION_PAPER, or CONCEPTS_TEXT is empty or malformed, respond only with: "Invalid or incomplete data provided for evaluation."
13. Do not modify, re-word, or regenerate the original questions — only evaluate the given answers against them.
14. If an answer's core concept is correct but a specific supporting example, factor, or detail is weak, vague, or only partially relevant, deduct at least 1 mark rather than awarding full marks — full marks require both the core concept AND supporting details to be accurate and well-chosen.
Input format:
QUESTION_PAPER: <plain text string of the generated question paper with sections and marks>
CONCEPTS_TEXT: <plain text string of concept blocks with CONCEPT, DESCRIPTION, SOURCE_CHUNK_ID, DIFFICULTY>
HUMAN_ANSWERS: <plain text string of answers labeled by question number, e.g., A1, B2, C1>

Output format:
Return the evaluation as a single plain text string (no JSON, no markdown code blocks, no extra commentary), formatted as follows:

EVALUATION REPORT

Section A
A1: <marks awarded>/1 - <brief feedback>
A2: <marks awarded>/1 - <brief feedback>
...

Section B
B1: <marks awarded>/3 - <brief feedback>
B2: <marks awarded>/3 - <brief feedback>
...

Section C
C1: <marks awarded>/5 - <brief feedback>
C2: <marks awarded>/5 - <brief feedback>
...

TOTAL SCORE: <sum>/30

STRENGTH AREAS:
- <Concept name>: <1-2 line reason, e.g., consistently correct and complete answers>
- <Concept name>: <reason>

WEAKNESS AREAS:
- <Concept name>: <1-2 line reason, e.g., incomplete/incorrect/unattempted answers, what was missing>
- <Concept name>: <reason>

OVERALL FEEDBACK: <2-3 line summary combining strengths, weaknesses, and suggestions for improvement>"""


def evaluater(state: State):
    question_paper = state['question_paper']
    human_answers = state['human_response']
    concepts = state['retrieved_concepts']
    user_prompt = (
        f"This is the concepts using to make the paper {concepts}\n\n"
        f"This is the Question paper :{question_paper}\n\n"
        f"This is the Human Answer :{human_answers}"
    )
    message = [('system', EVALUATER_SYSTEM_PROMPT), ('human', user_prompt)]
    parser = StrOutputParser()
    chains = LLM | parser
    report = chains.invoke(message)

    return {
        'evaluation_report': report
    }


def route_query(state: State):
    category = state['query_type']
    if (category == 'Question asked to LLM'):
        return "Get_Context"
    else:
        return "Concept_Extractor"


@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(State)

    graph.add_node("Classifier", classifier)
    graph.add_node("Get_Context", give_context)
    graph.add_node("Give_Answer", give_answer)
    graph.add_node("Concept_Extractor", concept_extractor)
    graph.add_node("Make Question Paper", make_paper)
    graph.add_node("Human_Answer", human_answer_node)
    graph.add_node("Evaluate_Answer", evaluater)

    graph.add_edge(START, "Classifier")
    graph.add_conditional_edges("Classifier", route_query)

    graph.add_edge("Get_Context", "Give_Answer")
    graph.add_edge("Give_Answer", END)

    graph.add_edge("Concept_Extractor", "Make Question Paper")
    graph.add_edge("Make Question Paper", "Human_Answer")
    graph.add_edge("Human_Answer", "Evaluate_Answer")
    graph.add_edge("Evaluate_Answer", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    return app

# =========================================================================
# =======================  STREAMLIT UI LAYER  ==========================
# =========================================================================

st.set_page_config(
    page_title="AI Tutor — RAG Question Paper Generator",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Global styling ----------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }

    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f5f5f5;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #9aa0a6;
        margin-bottom: 1.5rem;
    }
    .section-card {
        background-color: #161b22;
        border: 1px solid #262b33;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .badge-green { background-color: #113a24; color: #4ade80; }
    .badge-blue  { background-color: #10243d; color: #60a5fa; }
    .badge-amber { background-color: #3a2c11; color: #fbbf24; }

    .paper-box {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        font-family: 'Courier New', monospace;
        font-size: 0.92rem;
        white-space: pre-wrap;
        color: #e6edf3;
        max-height: 520px;
        overflow-y: auto;
    }
    .report-box {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        font-family: 'Courier New', monospace;
        font-size: 0.92rem;
        white-space: pre-wrap;
        color: #e6edf3;
    }
    .sidebar-credit {
        position: fixed;
        bottom: 1.2rem;
        padding: 0.8rem 1rem;
        background-color: #161b22;
        border: 1px solid #262b33;
        border-radius: 10px;
        width: 85%;
    }
    .footer-name {
        font-weight: 700;
        color: #f5f5f5;
        font-size: 0.95rem;
    }
    .footer-link a {
        color: #60a5fa;
        text-decoration: none;
        font-size: 0.85rem;
    }
    .footer-link a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state init ----------
if "graph_app" not in st.session_state:
    st.session_state.graph_app = None
if "book_retriever" not in st.session_state:
    st.session_state.book_retriever = None
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "session_1"
if "stage" not in st.session_state:
    st.session_state.stage = "idle"   # idle -> awaiting_answers -> evaluated
if "question_paper" not in st.session_state:
    st.session_state.question_paper = None
if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None
if "direct_answer" not in st.session_state:
    st.session_state.direct_answer = None
if "last_query_type" not in st.session_state:
    st.session_state.last_query_type = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📘 AI Tutor")
    st.markdown("RAG-powered question paper generator & evaluator")
    st.divider()

    st.markdown("#### 1. Upload your book (PDF)")
    uploaded_pdf = st.file_uploader("Upload a PDF to build the knowledge base", type=["pdf"])

    if uploaded_pdf is not None:
        if st.session_state.pdf_name != uploaded_pdf.name:
            with st.spinner("Indexing your PDF... this may take a moment"):
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, uploaded_pdf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())

                st.session_state.book_retriever = build_retriever(tmp_path)
                st.session_state.graph_app = build_graph()
                st.session_state.pdf_ready = True
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.stage = "idle"
                st.session_state.question_paper = None
                st.session_state.evaluation_report = None
                st.session_state.direct_answer = None

        st.success(f"Indexed: {st.session_state.pdf_name}")
    else:
        st.info("Upload a PDF to get started.")

    st.divider()
    if st.session_state.pdf_ready:
        if st.button("🔄 Reset session", use_container_width=True):
            for key in ["stage", "question_paper", "evaluation_report", "direct_answer", "last_query_type"]:
                st.session_state[key] = None if key != "stage" else "idle"
            st.session_state.thread_id = f"session_{os.urandom(4).hex()}"
            st.rerun()

    st.markdown(
        """
        <div class="sidebar-credit">
            <div class="footer-name">Made by Gaurav Gupta</div>
            <div class="footer-link">
                <a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank">
                    🔗 LinkedIn Profile
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Main header ----------
st.markdown('<div class="app-title">📘 AI Tutor — Ask, Generate & Get Evaluated</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload a book, ask a question directly, or ask the bot to generate a question paper — answer it and receive a detailed evaluation report.</div>',
    unsafe_allow_html=True
)

if not st.session_state.pdf_ready:
    st.warning("👈 Please upload a PDF from the sidebar to begin.")
    st.stop()

# ---------- Stage: idle -> take a query ----------
if st.session_state.stage == "idle":
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Ask a question, or ask for a question paper")
        st.caption("Examples: *\"What is uniform circular motion?\"*  or  *\"Generate a question paper on this chapter.\"*")

        user_question = st.text_area("Your input", height=100, placeholder="Type your question here...")

        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.button("Submit ▶", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if submit and user_question.strip():
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.spinner("Thinking..."):
            result = st.session_state.graph_app.invoke(
                {"asked_question": user_question}, config=config
            )

        if "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            st.session_state.question_paper = interrupt_payload["question_paper"]
            st.session_state.stage = "awaiting_answers"
            st.rerun()
        else:
            st.session_state.direct_answer = result["messages"][-1][1] if result.get("messages") else None
            st.session_state.last_query_type = result.get("query_type")
            st.session_state.stage = "answered"
            st.rerun()

# ---------- Stage: direct answer shown ----------
if st.session_state.stage == "answered":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-blue">Direct Answer</span>', unsafe_allow_html=True)
    st.markdown("#### Response")
    st.markdown(st.session_state.direct_answer or "No answer generated.")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("↩ Ask another question"):
        st.session_state.stage = "idle"
        st.session_state.direct_answer = None
        st.rerun()

# ---------- Stage: awaiting human answers ----------
if st.session_state.stage == "awaiting_answers":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-amber">Question Paper Generated</span>', unsafe_allow_html=True)
    st.markdown("#### 📄 Your Question Paper")
    st.markdown(f'<div class="paper-box">{st.session_state.question_paper}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 How to submit your answers", expanded=True):
        st.markdown("""
- Answer questions section-wise (Section A, then B, then C).
- Label each answer with its question number, e.g. `A1:`, `B2:`, `C1:`.
- If unsure about a question, write `Not Attempted`.
- Paste all your answers together in the box below.
        """)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### ✍️ Submit Your Answers")
    human_answers = st.text_area(
        "Answers",
        height=300,
        placeholder="A1: ...\nA2: ...\n\nB1: ...\nB2: ...\n\nC1: ...\nC2: ..."
    )
    submit_answers = st.button("Submit Answers for Evaluation ✅", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if submit_answers and human_answers.strip():
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.spinner("Evaluating your answers..."):
            final_result = st.session_state.graph_app.invoke(
                Command(resume=human_answers), config=config
            )
        st.session_state.evaluation_report = final_result.get("evaluation_report")
        st.session_state.stage = "evaluated"
        st.rerun()

# ---------- Stage: evaluation report ----------
if st.session_state.stage == "evaluated":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-green">Evaluation Complete</span>', unsafe_allow_html=True)
    st.markdown("#### 📊 Evaluation Report")
    st.markdown(f'<div class="report-box">{st.session_state.evaluation_report}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("↩ Start a new session"):
        st.session_state.thread_id = f"session_{os.urandom(4).hex()}"
        st.session_state.stage = "idle"
        st.session_state.question_paper = None
        st.session_state.evaluation_report = None
        st.rerun()
