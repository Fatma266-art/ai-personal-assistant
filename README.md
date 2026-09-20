# Personal AI Assistant

An AI assistant that doesn't just answer questions about your documents, it **executes tasks** on them: RAG Q&A with sources, HR CV shortlisting to Excel, and PDF report generation.

> From AI Student to AI Engineer 🚀

## Features
- **RAG Q&A**: ask questions over indexed papers and get answers with source + page citations.
- **HR shortlist**: upload CVs (PDF/DOCX/PPTX/TXT), define role + required skills, get a scored `.xlsx` shortlist.
- **PDF report**: turn an uploaded document into a structured PDF report.
- **Upload from the UI**: PDF, DOCX, PPTX and TXT files are ingested on the fly.

## Architecture
```
Upload → Text extraction → Chunking → Embeddings → ChromaDB
                                                      ↓
User task → Router → RAG / HR shortlist / PDF report → LLM (Groq) → Output file / answer
```

**Stack:** Python, FastAPI, ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`), Groq (OpenAI-compatible API), PyMuPDF, ReportLab, pandas, Docker.

## Project structure
```
src/
├── ingestion/   # text extraction + chunking
├── embeddings/  # build the vector store
├── retrieval/   # semantic search over ChromaDB
├── llm/         # prompts + LLM calls
├── actions/     # HR shortlist, PDF report
├── agent/       # router + tools
└── api/         # FastAPI backend
frontend/        # simple web UI
```

## Getting started
1. Copy `.env.example` to `.env` and fill in your values.
2. Put your PDFs in `Data/Raw Data/`, then build the knowledge base:
```bash
   python src/ingestion/extract_text.py
   python src/ingestion/chunk_text.py
   python src/embeddings/create_vectorstore.py
```
3. Run the API with Docker:
```bash
   docker compose up
```
   The API runs at `http://localhost:8000`.
4. Open `frontend/index.html` (e.g. with VS Code Live Server on port 5500) and make sure the `API` constant in it points to `http://localhost:8000`.

## Known limitations (v1.0)
This is a working prototype. Known gaps, planned for v2:
- **No authentication** on the API, and the client sends server file paths (should switch to file IDs).
- **Keyword-based router**: task routing uses simple keyword matching, not real LLM tool-calling.
- **No evaluation or automated tests** yet for retrieval quality or the scoring logic.
- **RAG always answers**: no relevance threshold, so out-of-scope questions still get an answer.
- **English-only embeddings**: `all-MiniLM-L6-v2` is weak for Arabic content.
- **HR scoring** is exact keyword matching on the first 8 extracted skills; it ignores experience and education.
- Shared vector store: no per-user isolation.

## Roadmap v2
- [ ] API authentication + file IDs + rate limiting
- [ ] LLM tool-calling agent
- [ ] Retrieval evaluation set + pytest suite
- [ ] Multilingual embeddings + relevance threshold
- [ ] Structured LLM output with Pydantic validation
- [ ] Proper cloud deployment