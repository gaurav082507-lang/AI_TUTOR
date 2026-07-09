# 📘 AI Tutor — RAG-Powered Question Paper Generator & Evaluator

An intelligent tutoring system that turns any textbook (PDF) into a personalized exam. Ask it a direct question and get a **cited, grounded answer** — or ask it to generate a **question paper**, answer it yourself, and receive a detailed **evaluation report** with strengths and weaknesses, all built entirely from the book's content.

Built with **LangGraph**, **RAG (Retrieval-Augmented Generation)**, and a **human-in-the-loop** workflow, wrapped in a clean **Streamlit** UI.

---

## ✨ Features

- **📄 Upload any PDF** — build a searchable knowledge base from your own textbook or notes.
- **🤖 Smart query classification** — automatically detects whether you're asking a question or requesting a question paper.
- **💬 Grounded Q&A** — answers are generated strictly from retrieved book content, with inline citations.
- **📝 Auto-generated question papers** — a structured 30-mark paper (MCQ/short/long answer sections) built entirely from concepts extracted from the book.
- **🧑‍🎓 Human-in-the-loop answering** — pauses execution using LangGraph's `interrupt()` and waits for you to submit your answers.
- **📊 Detailed evaluation** — grades your answers against the source material, with:
  - Question-by-question marks and feedback
  - Total score out of 30
  - **Strength areas** (concepts you understand well)
  - **Weakness areas** (concepts to review)
  - Overall summary feedback

---

## 🏗️ Architecture

The system is built as a **LangGraph state machine** with conditional routing:

```
START
  │
  ▼
Classifier ──────────────┐
  │                       │
  ▼ (Question)            ▼ (Generate Paper)
Get_Context          Concept_Extractor
  │                       │
  ▼                       ▼
Give_Answer          Make Question Paper
  │                       │
  ▼                       ▼
 END                Human_Answer (⏸ interrupt)
                          │
                          ▼
                   Evaluate_Answer
                          │
                          ▼
                         END
```

### Pipeline stages

| Node | Purpose |
|---|---|
| **Classifier** | Determines if the user wants a direct answer or a question paper. |
| **Get_Context** | Retrieves relevant chunks from the book via MMR-based retrieval. |
| **Give_Answer** | Answers the question using only retrieved context, with citations. |
| **Concept_Extractor** | Pulls key concepts, definitions, and ideas from the book for exam use. |
| **Make Question Paper** | Generates a structured 30-mark paper from extracted concepts. |
| **Human_Answer** | Pauses the graph (`interrupt()`) and waits for the user to submit answers. |
| **Evaluate_Answer** | Grades answers against the source concepts and gives structured feedback. |

---

## 🧰 Tech Stack

- **LangGraph** — stateful multi-node orchestration, conditional routing, human-in-the-loop via `interrupt`/`Command`
- **LangChain** — document loading, text splitting, retrieval chains
- **Groq (Llama 3.3 70B)** — LLM for classification, answering, extraction, paper generation, and evaluation
- **Mistral AI Embeddings** — `mistral-embed-2312` for vectorizing document chunks
- **Chroma** — in-memory vector store for retrieval (MMR search)
- **Streamlit** — web UI layer

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

---

## 🚀 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then in your browser:

1. **Upload a PDF** from the sidebar — the app will index it into a vector store.
2. **Ask a question directly**, e.g. *"What is uniform circular motion?"* → get a grounded, cited answer.
3. **Ask for a question paper**, e.g. *"Generate a question paper on this chapter."* → get a structured 30-mark paper.
4. **Answer the paper** in the provided text box, labeling each answer (e.g. `A1:`, `B2:`, `C1:`).
5. **Submit for evaluation** → receive a full evaluation report with marks, feedback, strengths, and weaknesses.

---

## 📂 Project Structure

```
.
├── app.py               # Streamlit app (UI + LangGraph pipeline)
├── requirements.txt      # Python dependencies
├── .env                  # API keys (not committed)
└── README.md
```

---

## 🔮 Possible Future Improvements

- Multi-query / chapter-wise retrieval for broader concept coverage across large books
- Persistent vector store (avoid re-embedding PDFs on every run)
- Adaptive difficulty — track weak concepts across sessions and prioritize them in future papers
- Support for multiple question paper formats (MCQ-only, custom mark totals, custom difficulty mix)
- Downloadable PDF export of the question paper and evaluation report

---

## 👤 Author

**Made by Gaurav Gupta**
🔗 [LinkedIn](https://www.linkedin.com/in/gaurav-gupta-79754a377)
