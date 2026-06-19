# Document Q&A Bot

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions grounded only in their local document library (PDF and DOCX files). The backend uses ChromaDB for vector storage and the Gemini API for embeddings and grounded response generation. A command-line interface, python API, and Streamlit dashboard are provided.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Installation
Clone this repository and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```
Note: The application will check for both `GOOGLE_API_KEY` and `GEMINI_API_KEY`.

---

## Document Ingestion

Place your PDF or DOCX files in the `data/` directory.

To process these files, chunk the text, compute vector embeddings, and save them into the local vector database, run the ingestion script:
```bash
python -m src.ingest
```

This clears any existing database index, chunks documents page-by-page, generates embeddings, and saves them to the Chroma database located at `db/`.

---

## Running the Application

### 1. Web Application (Streamlit)
To run the interactive chat interface and control dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. The web app features:
- A document control panel showing available files.
- A re-ingestion controller.
- A dynamic retrieval size slider (TOP_K settings).
- Chat window showing grounded answers with inline page references and collapsible source citation blocks.

### 2. Command Line Interface (CLI)
You can query the RAG pipeline directly from the command line:
```bash
python -m src.main --query "What does ChromaDB store?"
```

To output the raw JSON containing the grounded answer and source metadata list:
```bash
python -m src.main --query "What does ChromaDB store?" --json
```

### 3. Python REPL Usage
The pipeline is designed to be fully UI-agnostic. You can call the Q&A interface inside a Python REPL or script:
```python
from src.main import answer_question

result = answer_question("What does ChromaDB store?")
print(result["answer"])
print(result["sources"])
```

The output conforms to the schema:
```json
{
  "answer": "Grounded answer text",
  "sources": [
    {
      "file": "source_filename.docx",
      "page": -1,
      "snippet": "Matching context snippet..."
    }
  ]
}
```

---

## Deployment

### Local Tunneling
For temporary testing, you can expose the local Streamlit port (8501) via localtunnel:
```bash
npx localtunnel --port 8501
```

### Production Deployment
For persistent web hosting:
- Host the code on GitHub.
- Deploy via Streamlit Community Cloud (share.streamlit.io).
- Add the `GOOGLE_API_KEY` key in the Advanced Settings / Secrets section.
