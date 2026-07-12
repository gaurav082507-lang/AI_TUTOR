import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq  # noqa: F401 (kept for parity with original pipeline)
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

# =========================================================
# PAGE CONFIG + THEME
# =========================================================
st.set_page_config(page_title="TutorLens AI", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{
    --bg:#0a0d16; --panel:#111528; --panel-2:#151a30;
    --border:#232946; --text:#e7e9f5; --text-dim:#8b93b5;
    --accent1:#8b7bf8; --accent2:#4f8ff7; --green:#34d399; --red:#f56464;
    --grad:linear-gradient(135deg,#8b7bf8,#4f8ff7);
    --mono:'JetBrains Mono','Courier New',monospace;
}
.stApp{ background:var(--bg); color:var(--text); }
section[data-testid="stSidebar"]{
    background:var(--panel); border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] * { color:var(--text); }
#MainMenu{ visibility:hidden; }
footer{ visibility:hidden; }
header[data-testid="stHeader"]{ background:transparent; }

.brand-row{ display:flex; align-items:center; gap:10px; margin-bottom:4px; }
.brand-icon{
    width:32px; height:32px; border-radius:8px; background:var(--grad);
    display:flex; align-items:center; justify-content:center; font-size:16px;
}
.brand-name{ font-size:17px; font-weight:700; }
.tagline{ font-size:12.5px; color:var(--text-dim); margin:2px 0 16px 42px; }

.status-badge{
    display:flex; align-items:center; gap:8px;
    background:var(--panel-2); border:1px solid var(--border); border-radius:8px;
    padding:9px 12px; font-size:12.5px; font-family:var(--mono);
    margin-bottom:18px;
}
.dot{ width:7px; height:7px; border-radius:50%; box-shadow:0 0 8px currentColor; }
.dot-on{ background:var(--green); color:var(--green); }
.dot-off{ background:var(--red); color:var(--red); }

.side-label{
    font-size:12px; font-weight:700; color:var(--text-dim);
    text-transform:uppercase; letter-spacing:.6px; margin:14px 0 6px 0;
}

.kicker{
    font-family:var(--mono); font-size:12px; letter-spacing:3px;
    color:var(--text-dim); text-transform:uppercase; text-align:center; margin-bottom:14px;
}
.hero-title{
    font-size:48px; font-weight:800; text-align:center; margin:0 0 14px 0;
    background:var(--grad); -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent; letter-spacing:-1px;
}
.hero-sub{
    max-width:680px; margin:0 auto 26px auto; text-align:center;
    color:var(--text-dim); font-size:15.5px; line-height:1.7;
}
.pills-row{ display:flex; gap:10px; flex-wrap:wrap; justify-content:center; margin-bottom:12px; }
.pill{
    border:1px solid var(--border); background:var(--panel-2); border-radius:20px;
    padding:7px 15px; font-size:12.5px; color:var(--accent2); font-family:var(--mono);
}

.card{
    background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:22px 26px; margin-bottom:18px;
}
.card h3{
    margin:0 0 12px 0; font-size:13px; text-transform:uppercase; letter-spacing:1px;
    color:var(--accent2); font-family:var(--mono); font-weight:700;
}
.placeholder-card{
    background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:20px 24px; text-align:center; color:var(--text-dim); font-size:14px;
}
.q-row{
    display:flex; justify-content:space-between; gap:14px; padding:9px 0;
    border-bottom:1px solid var(--border); font-size:14px; line-height:1.6;
}
.q-marks{ flex-shrink:0; font-family:var(--mono); font-size:12px; color:var(--text-dim); }
.section-title{
    font-size:13px; font-weight:700; color:var(--accent1);
    text-transform:uppercase; letter-spacing:.6px; margin:14px 0 6px 0;
}
.eval-total{ font-size:34px; font-weight:800; background:var(--grad);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.eval-of{ color:var(--text-dim); font-size:14px; margin-left:6px; }

textarea, input[type="text"]{
    background:var(--panel-2) !important; color:var(--text) !important;
    border:1px solid var(--border) !important; border-radius:9px !important;
}
.stButton>button{
    width:100%; border:none; border-radius:10px; padding:12px;
    font-weight:700; color:#0a0d16; background:var(--grad);
}
.stButton>button:hover{ opacity:0.92; color:#0a0d16; }

.footer{
    margin-top:40px; padding-top:18px; border-top:1px solid var(--border);
    color:#5c6483; font-size:12.5px; text-align:center;
}

.wf-wrap{ max-width:860px; margin:4px auto 32px auto; }
.wf-caption{
    text-align:center; font-family:var(--mono); font-size:11.5px; letter-spacing:.5px;
    color:var(--text-dim); text-transform:uppercase; margin-bottom:14px;
}
.wf-root-row{ display:flex; justify-content:center; margin-bottom:2px; }
.wf-node{
    border:1px solid var(--border); background:var(--panel-2); border-radius:8px;
    padding:9px 16px; font-size:12.5px; color:var(--text-dim); text-align:center;
    white-space:nowrap;
}
.wf-node.active{
    border-color:var(--accent2); color:var(--text);
    box-shadow:0 0 0 1px var(--accent2) inset;
}
.wf-node.done{ border-color:var(--green); color:var(--text); }
.wf-down{ text-align:center; color:var(--border); font-size:15px; line-height:1; margin:2px 0; }
.wf-down.active{ color:var(--accent2); }
.wf-branches{ display:flex; justify-content:center; gap:24px; margin-top:4px; flex-wrap:wrap; }
.wf-col{ display:flex; flex-direction:column; align-items:center; min-width:170px; }
.wf-col-label{
    font-family:var(--mono); font-size:10.5px; color:var(--text-dim);
    text-transform:uppercase; letter-spacing:.5px; margin-bottom:8px; text-align:center;
}
.wf-col .wf-node{ width:100%; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# PIPELINE (same graph as the original CLI tutor)
# =========================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    asked_question: str
    query_type: str
    retrieved_context: str
    retrieved_concepts: str
    question_paper: str
    human_response: str
    evaluation_report: str

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

RAG_SYSTEM_PROMPT = """You are a precise question-answering assistant. Answer the QUESTION using only the CONTEXT provided, citing the supporting parts. If the CONTEXT lacks the answer, say: "The provided context does not contain enough information to answer this question." Keep answers concise unless a detailed explanation is requested."""

CONCEPT_EXTRACTOR_PROMPT = """You are a concept extraction node. From RETRIEVED_CHUNKS, extract testable concepts only from the given text. For each concept output:
CONCEPT: <name>
DESCRIPTION: <1-2 line grounded description>
SOURCE_CHUNK_ID: <id or null>
DIFFICULTY: <easy|medium|hard>
Separate concepts with a blank line. If nothing meaningful is found, return: "No significant concepts found in the provided content." """

PAPER_MAKING_PROMPT = """You are a question paper generator. Using only the given CONCEPTS_TEXT, produce a question paper worth exactly 30 marks with this EXACT structure:
Section A (1 Mark Questions): exactly 5 questions x 1 mark = 5 marks
Section B (Short Answer Questions): exactly 5 questions x 3 marks = 15 marks
Section C (Long Answer Questions): exactly 2 questions x 5 marks = 10 marks
Ground every question strictly in the given concept descriptions. Output format:

QUESTION PAPER
Total Marks: 30

Section A (1 Mark Questions)
1. ...
...

Section B (Short Answer Questions)
1. ...
...

Section C (Long Answer Questions)
1. ...
2. ...
"""

EVALUATOR_SYSTEM_PROMPT = """You are an answer evaluation assistant. Grade HUMAN_ANSWERS strictly against QUESTION_PAPER and CONCEPTS_TEXT (source of truth). Output format:

EVALUATION REPORT

Section A
A1: <marks>/1 - <feedback>
...

Section B
B1: <marks>/3 - <feedback>
...

Section C
C1: <marks>/5 - <feedback>
...

TOTAL SCORE: <sum>/30

STRENGTH AREAS:
- <concept>: <reason>

WEAKNESS AREAS:
- <concept>: <reason>

OVERALL FEEDBACK: <2-3 line summary>"""


@st.cache_resource(show_spinner=False)
def build_retriever(pdf_path: str):
    data = PyPDFLoader(pdf_path)
    docs = data.load()
    embedding_model = MistralAIEmbeddings(model="mistral-embed-2312")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunk_docs = splitter.split_documents(docs)
    vector_store = Chroma.from_documents(documents=chunk_docs, embedding=embedding_model)
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
    )


@st.cache_resource(show_spinner=False)
def build_graph():
    llm = ChatMistralAI(model="mistral-medium-3-5", temperature=0.2)

    def classifier(state: State) -> dict:
        question_asked = state["asked_question"]
        user_prompt = f"This is the Asked Question :{question_asked}"
        messages = [("system", CLASSIFIER_SYSTEM_PROMPT), ("human", user_prompt)]
        response = llm.invoke(messages)
        category = response.content
        if "question asked to llm" in category.lower():
            category = "Question asked to LLM"
        elif "telling the llm to give question" in category.lower():
            category = "Telling the LLM to give Question"
        return {"messages": [("human", user_prompt), ("ai", response.content)], "query_type": category}

    def give_context(state: State) -> dict:
        retriever = st.session_state["retriever"]
        docs = retriever.invoke(state["asked_question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"retrieved_context": context}

    def give_answer(state: State) -> dict:
        context = state["retrieved_context"]
        question_asked = state["asked_question"]
        user_prompt = f"The question asked is {question_asked}\nThe context is :{context}"
        parser = StrOutputParser()
        chain = llm | parser
        messages = [("system", RAG_SYSTEM_PROMPT), ("human", user_prompt)]
        answer = chain.invoke(messages)
        return {"messages": [("human", user_prompt), ("ai", answer)]}

    def concept_extractor(state: State) -> dict:
        retriever = st.session_state["retriever"]
        docs = retriever.invoke(
            "Extract all key concepts, definitions, theories, formulas, important terms, "
            "named entities, and core ideas covered in this content, suitable for generating "
            "exam or question paper questions."
        )
        chunks_text = "\n\n".join(f"[chunk_id: {i}]\n{doc.page_content}" for i, doc in enumerate(docs))
        user_prompt = f"RETRIEVED_CHUNKS: {chunks_text}"
        messages = [("system", CONCEPT_EXTRACTOR_PROMPT), ("human", user_prompt)]
        concepts = llm.invoke(messages)
        return {"messages": [("human", user_prompt), ("ai", concepts.content)], "retrieved_concepts": concepts.content}

    def make_paper(state: State) -> dict:
        concepts = state["retrieved_concepts"]
        user_prompt = f"This is the Retrieved Concept :{concepts}"
        messages = [("system", PAPER_MAKING_PROMPT), ("human", user_prompt)]
        parser = StrOutputParser()
        chain = llm | parser
        question_paper = chain.invoke(messages)
        return {"messages": [("human", user_prompt), ("ai", question_paper)], "question_paper": question_paper}

    def human_answer_node(state: State) -> dict:
        payload = {"question_paper": state["question_paper"]}
        human_response = interrupt(payload)
        return {"human_response": human_response}

    def evaluator(state: State) -> dict:
        question_paper = state["question_paper"]
        human_answers = state["human_response"]
        concepts = state["retrieved_concepts"]
        user_prompt = (
            f"This is the concepts used to make the paper {concepts}\n\n"
            f"This is the Question paper :{question_paper}\n\n"
            f"This is the Human Answer :{human_answers}"
        )
        messages = [("system", EVALUATOR_SYSTEM_PROMPT), ("human", user_prompt)]
        parser = StrOutputParser()
        chain = llm | parser
        report = chain.invoke(messages)
        return {"evaluation_report": report}

    def route_query(state: State):
        return "Get_Context" if state["query_type"] == "Question asked to LLM" else "Concept_Extractor"

    graph = StateGraph(State)
    graph.add_node("Classifier", classifier)
    graph.add_node("Get_Context", give_context)
    graph.add_node("Give_Answer", give_answer)
    graph.add_node("Concept_Extractor", concept_extractor)
    graph.add_node("Make Question Paper", make_paper)
    graph.add_node("Human_Answer", human_answer_node)
    graph.add_node("Evaluate_Answer", evaluator)

    graph.add_edge(START, "Classifier")
    graph.add_conditional_edges("Classifier", route_query)
    graph.add_edge("Get_Context", "Give_Answer")
    graph.add_edge("Give_Answer", END)
    graph.add_edge("Concept_Extractor", "Make Question Paper")
    graph.add_edge("Make Question Paper", "Human_Answer")
    graph.add_edge("Human_Answer", "Evaluate_Answer")
    graph.add_edge("Evaluate_Answer", END)

    return graph.compile(checkpointer=MemorySaver())


NODE_LABELS = {
    "Classifier": "Classifier",
    "Get_Context": "Get context",
    "Give_Answer": "Give answer",
    "Concept_Extractor": "Extract concepts",
    "Make Question Paper": "Make paper",
    "Human_Answer": "Human answers",
    "Evaluate_Answer": "Evaluate answer",
}

def render_workflow(placeholder=None):
    completed = st.session_state.get("completed_nodes", set())
    active = st.session_state.get("active_node")

    def node_html(key):
        label = NODE_LABELS[key]
        if key in completed:
            return f'<div class="wf-node done">✓ {label}</div>'
        if key == active:
            return f'<div class="wf-node active">{label}</div>'
        return f'<div class="wf-node">{label}</div>'

    def arrow_html(from_key):
        state = "active" if (from_key in completed or from_key == active) else ""
        return f'<div class="wf-down {state}">↓</div>'

    html = f"""
    <div class="wf-wrap">
      <div class="wf-caption">How this request is routed</div>
      <div class="wf-root-row">{node_html('Classifier')}</div>
      {arrow_html('Classifier')}
      <div class="wf-branches">
        <div class="wf-col">
          <div class="wf-col-label">In the PDF</div>
          {node_html('Get_Context')}
          {arrow_html('Get_Context')}
          {node_html('Give_Answer')}
        </div>
        <div class="wf-col">
          <div class="wf-col-label">Make a question paper</div>
          {node_html('Concept_Extractor')}
          {arrow_html('Concept_Extractor')}
          {node_html('Make Question Paper')}
          {arrow_html('Make Question Paper')}
          {node_html('Human_Answer')}
          {arrow_html('Human_Answer')}
          {node_html('Evaluate_Answer')}
        </div>
      </div>
    </div>
    """
    target = placeholder if placeholder is not None else st
    target.markdown(html, unsafe_allow_html=True)



if "stage" not in st.session_state:
    st.session_state.stage = "idle"          # idle | paper | evaluated
if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": "streamlit_session"}}
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "question_paper" not in st.session_state:
    st.session_state.question_paper = None
if "evaluation_report" not in st.session_state:
    st.session_state.evaluation_report = None
if "query_type" not in st.session_state:
    st.session_state.query_type = None
if "completed_nodes" not in st.session_state:
    st.session_state.completed_nodes = set()
if "active_node" not in st.session_state:
    st.session_state.active_node = None

mistral_key_present = bool(os.getenv("MISTRAL_API_KEY"))

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-row">
        <div class="brand-icon">📘</div>
        <div class="brand-name">TutorLens AI</div>
    </div>
    <p class="tagline">RAG Q&amp;A · Question papers · AI evaluation</p>
    """, unsafe_allow_html=True)

    dot_class = "dot-on" if mistral_key_present else "dot-off"
    status_text = "Mistral API key connected" if mistral_key_present else "Mistral API key missing"
    st.markdown(f"""
    <div class="status-badge"><span class="dot {dot_class}"></span>{status_text}</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">📚 Study material</div>', unsafe_allow_html=True)
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    pdf_path = "iest104.pdf"
    if uploaded_pdf is not None:
        pdf_path = f"/tmp/{uploaded_pdf.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

    st.markdown('<div class="side-label">💬 Ask your tutor</div>', unsafe_allow_html=True)
    question = st.text_area(
        "question", placeholder="e.g. Explain the working principle of a venturimeter",
        label_visibility="collapsed", height=100,
    )

    run_clicked = st.button("🚀 Ask Tutor")

# =========================================================
# MAIN
# =========================================================
st.markdown('<div class="kicker">Adaptive Learning &amp; Assessment Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">TutorLens AI</div>', unsafe_allow_html=True)
st.markdown("""
<p class="hero-sub">Ask a question and get a grounded answer, or request a paper and get a
full 30-mark exam with instant AI evaluation — built on retrieval, LangGraph routing, and Mistral.</p>
""", unsafe_allow_html=True)
st.markdown("""
<div class="pills-row">
    <div class="pill">🧩 LangGraph</div>
    <div class="pill">🔗 LangChain</div>
    <div class="pill">🎯 Chroma + MMR</div>
    <div class="pill">🤖 Mistral</div>
</div>
""", unsafe_allow_html=True)

workflow_placeholder = st.empty()
render_workflow(workflow_placeholder)

if run_clicked:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        st.session_state["retriever"] = build_retriever(pdf_path)
        graph_app = build_graph()

        st.session_state.completed_nodes = set()
        st.session_state.active_node = None
        st.session_state.query_type = None

        node_outputs = {}
        interrupt_payload = None

        with st.spinner("Routing your request through the tutor pipeline..."):
            for chunk in graph_app.stream(
                {"asked_question": question}, config=st.session_state.config, stream_mode="updates"
            ):
                for node_name, node_output in chunk.items():
                    if node_name == "__interrupt__":
                        interrupt_payload = node_output[0].value
                        st.session_state.active_node = "Human_Answer"
                        continue
                    node_outputs[node_name] = node_output
                    st.session_state.completed_nodes.add(node_name)
                    if node_name == "Classifier":
                        st.session_state.query_type = node_output.get("query_type")
                render_workflow(workflow_placeholder)

        if interrupt_payload is not None:
            st.session_state.question_paper = interrupt_payload["question_paper"]
            st.session_state.stage = "paper"
            st.session_state.evaluation_report = None
        else:
            answer_output = node_outputs.get("Give_Answer", {})
            last_msg = answer_output.get("messages", [None])[-1]
            st.session_state.last_answer = last_msg[1] if isinstance(last_msg, tuple) else getattr(last_msg, "content", "")
            st.session_state.stage = "answer"

content = st.container()
with content:
    if st.session_state.stage == "idle":
        st.markdown("""
        <div class="placeholder-card">👉 Ask a question in the sidebar and click <b>Ask Tutor</b> to get started.</div>
        """, unsafe_allow_html=True)

    elif st.session_state.stage == "answer":
        st.markdown(f"""
        <div class="card"><h3>Question</h3><p>{question}</p></div>
        <div class="card"><h3>Answer</h3><p>{st.session_state.last_answer}</p></div>
        """, unsafe_allow_html=True)

    elif st.session_state.stage == "paper":
        st.markdown(f"""
        <div class="card"><h3>Question paper</h3>
        <pre style="white-space:pre-wrap;font-family:inherit;color:var(--text);font-size:14.5px;">{st.session_state.question_paper}</pre>
        </div>
        """, unsafe_allow_html=True)

        answers = st.text_area(
            "Your answers", placeholder="A1: ...\nA2: ...\n\nB1: ...\n\nC1: ...", height=240,
        )
        if st.button("Submit for evaluation"):
            if not answers.strip():
                st.warning("Write your answers first.")
            else:
                graph_app = build_graph()
                node_outputs = {}
                with st.spinner("Evaluating your answers..."):
                    for chunk in graph_app.stream(
                        Command(resume=answers), config=st.session_state.config, stream_mode="updates"
                    ):
                        for node_name, node_output in chunk.items():
                            if node_name == "__interrupt__":
                                continue
                            node_outputs[node_name] = node_output
                            st.session_state.completed_nodes.add(node_name)
                        render_workflow(workflow_placeholder)
                st.session_state.active_node = None
                st.session_state.evaluation_report = node_outputs.get("Evaluate_Answer", {}).get("evaluation_report")
                st.session_state.stage = "evaluated"
                st.rerun()

    elif st.session_state.stage == "evaluated":
        st.markdown(f"""
        <div class="card"><h3>Question paper</h3>
        <pre style="white-space:pre-wrap;font-family:inherit;color:var(--text);font-size:14.5px;">{st.session_state.question_paper}</pre>
        </div>
        <div class="card"><h3>Evaluation report</h3>
        <pre style="white-space:pre-wrap;font-family:inherit;color:var(--text);font-size:14.5px;">{st.session_state.evaluation_report}</pre>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ask another question"):
            st.session_state.stage = "idle"
            st.session_state.query_type = None
            st.session_state.completed_nodes = set()
            st.session_state.active_node = None
            st.rerun()

st.markdown("""
<div class="footer">Built by Gaurav Gupta &middot;
<a href="https://www.linkedin.com/in/gaurav-gupta-79754a377" target="_blank" style="color:var(--accent2);text-decoration:none;">Connect on LinkedIn</a>
</div>
""", unsafe_allow_html=True)
