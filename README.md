
# NyayaMitra ⚖️

**NyayaMitra** is an AI-powered legal documentation and chatbot system that assists users in generating legal documents (e.g., contracts, agreements, petitions) and answering law-related queries using Retrieval-Augmented Generation (RAG) and ReAct-based agents. Developed as a hackathon-winning prototype to democratize access to legal support in India.

---

## 🚀 Features

- ✍️ **Legal Document Generation** – Create petitions, agreements, contracts from structured input.
- 🤖 **AI Legal Chatbot** – Interacts with users to resolve legal questions using retrieval-augmented methods.
- 🧠 **ReAct + RAG Agents** – Combines reasoning and retrieval for accurate and context-rich responses.
- 🧾 **PDF Parsing Tools** – Understands and extracts content from uploaded legal documents.
- ⚡ **Fast & Token Efficient** – Optimized for quick inference and deployment using local or open-source models.

---

## 🧱 Tech Stack

| Layer         | Tools & Frameworks                          |
|---------------|---------------------------------------------|
| Frontend      | React, Streamlit (alternative UI)           |
| Backend       | Python, Flask                               |
| AI Models     | LangChain, ReAct, RAG agents, LLMs (e.g., Phi-3 via Ollama) |
| Vector DB     | ChromaDB or Milvus                          |
| PDF Handling  | PyMuPDF, pdf2image                          |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/srinivastls/NyayaMitra.git
cd NyayaMitra
```

### 2. Setup Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Frontend

```bash
cd ../frontend
npm install
npm run build
```

---

## 🧪 Running Locally

### Start Backend (Flask API)

```bash
cd backend
flask run
```

### Start Frontend (React)

```bash
cd frontend
npm start
```

> Optionally, launch `Streamlit` UI if included:
```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **User Inputs a Query or Document Request**
2. **LLM Agent (ReAct + RAG)** processes query and searches vector DB.
3. **Legal Tools** (e.g., PDF parser, case retrieval) help extract and cite references.
4. **Generated Output**: Either a chatbot response or a downloadable legal document.

---

## 🎯 Use Cases

- Drafting rental, employment, or partnership agreements
- Submitting RTI or PIL petitions
- Understanding sections of IPC, CrPC, or Constitution
- Accessing summaries of past Supreme/High Court judgments

---

## 🔒 License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## 🏆 Acknowledgements

- 🏅 Built during **NMIT HACKS 2024** – won **2nd Prize**
- 🔗 Powered by open-source models and LangChain agents
- 🙏 Thanks to mentors, contributors, and reviewers

---

## 📌 Future Work

- Multi-language legal support (Hindi, Telugu, etc.)
- Integration with court cause lists and case management systems
- Legal document OCR and voice-to-text support
- Deployment via Docker, Streamlit Cloud, or Hugging Face Spaces

---

**Empowering justice through technology – NyayaMitra.**
