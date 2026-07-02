# 🛡️ RAG-Based HR Policy Assistant
### Automated HR Compliance using Dual-Layer RAG Architecture

A production-ready AI system that answers HR policy queries with context-aware, compliance-enforced responses. Built with a novel **Dual-Layer RAG Architecture** combining retrieval-augmented generation with NLI-based hallucination detection.

---

## 🚀 Key Results
- **97% reduction** in HR query resolution time — from ~8 minutes to under 15 seconds
- **Dual-layer verification** — L1 Critic enforces compliance rules, L2 NLI detects hallucinations
- **Zero hallucination tolerance** — contradictions flagged and warned in real-time

---

## 🏗️ Architecture

```
User Query
    ↓
[L0] RAG Retriever — ChromaDB vector search over uploaded HR handbook
    ↓
[L1] Critic Layer — Rewrites answer with mandatory compliance rules injected
    ↓
[L2] NLI Verification — CrossEncoder checks answer against source document
    ↓
Final Verified Response
```

### Why Dual-Layer?
Standard RAG systems retrieve and generate — but they don't verify. This system adds:
- **L1 Critic:** Ensures every response includes company-specific compliance rules
- **L2 NLI Check:** Uses DeBERTa-based Natural Language Inference to detect if the generated answer contradicts the source document

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | Streamlit |
| LLM | LLaMA 3 via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector Store | ChromaDB |
| Document Loading | LangChain PyPDFLoader |
| NLI Verification | CrossEncoder (DeBERTa-v3) |
| RAG Orchestration | LangChain LCEL |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally

### 1. Clone the repository
```bash
git clone https://github.com/GunalKarthikeyanS/RAG-HR-Policy-Assistant.git
cd RAG-HR-Policy-Assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull required Ollama models
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 4. Run the application
```bash
streamlit run Rag_demo.py
```

### 5. Use the app
1. Enter your company name and employee name in the sidebar
2. Set your mandatory compliance rules
3. Upload your HR handbook PDF
4. Ask HR policy questions in the chat

---

## 💡 Features

- **Dynamic Company Configuration** — Adapts to any company's HR handbook
- **Custom Compliance Rules** — Inject mandatory rules into every response
- **Real-time Hallucination Detection** — L2 NLI layer flags contradictions instantly
- **Clean Chat Interface** — Professional UI with custom avatars and toast notifications
- **Session Memory** — Chat history maintained within session

---

## 📁 Project Structure

```
RAG-HR-Policy-Assistant/
├── Rag_demo.py          # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore rules
```

---

## 🎓 Academic Context

Built as part of the MS in Artificial Intelligence program at **Yeshiva University, Katz School of Science and Health** (Spring 2026). Group project with Vishal Balaji.

**Course:** Capstone / Independent Study  
**Focus:** Production-grade GenAI systems with safety and compliance enforcement

---

## 👤 Author

**Gunal Karthikeyan Saravanan**  
MS in Artificial Intelligence — Yeshiva University  
[LinkedIn](https://linkedin.com/in/gunalkarthikeyans) | [GitHub](https://github.com/GunalKarthikeyanS)
