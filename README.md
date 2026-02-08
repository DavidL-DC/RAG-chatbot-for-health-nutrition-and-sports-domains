# RAG Chatbot for Health, Nutrition & Sports

This repository contains a prototype implementation of a Retrieval-Augmented Generation (RAG) chatbot
developed as part of a Bachelor's thesis in Applied Artificial Intelligence (B.Sc., IU Internationale Hochschule).

The system focuses on knowledge-sensitive domains such as health, nutrition, and physical activity,
with an emphasis on factual grounding, source transparency, and controlled abstention.

---

## Project Overview

The chatbot combines:
- a curated scientific document corpus (PDF-based),
- embedding-based vector search (Chroma),
- a large language model (LLM),
- and explicit abstention mechanisms to reduce hallucinations.

The system retrieves relevant document chunks and generates answers **strictly based on retrieved sources**.
If the available context is insufficient, the chatbot explicitly abstains from answering.

---

## Key Features

- **PDF ingestion pipeline** (recursive folder scan)
- **Chunking** with overlap for robust retrieval
- **Embeddings** (`text-embedding-3-small`)
- **Vector store**: Chroma (persistent)
- **Retriever**: MMR (diversity-aware retrieval)
- **Prompt-controlled answering** (source-grounded)
- **Abstention mechanism** (retrieval-based + prompt-based)
- **Source aggregation** with page references
- **Mini evaluation** (RAG vs. No-RAG baseline; CSV output)
- **Streamlit UI** for interactive queries

---

## Architecture (High-Level)

1. Ingest PDFs → split into chunks  
2. Create embeddings → store in vector database (Chroma)  
3. For each query: retrieve top chunks (MMR)  
4. Generate answer from retrieved context only  
5. Show sources with file name + page(s)  
6. Abstain if context is insufficient

---

## Repository Structure

- `app/ingest.py` – builds the vector store from PDFs  
- `app/query_core.py` – RAG pipeline (retrieval, prompt, abstention, source formatting)  
- `app/ui_streamlit.py` – Streamlit UI  
- `app/evaluation.py` – mini evaluation script (exports CSV)  
- `data/raw/` – input PDFs (not included)  
- `vectorstore/chroma/` – persisted vector database (generated locally)  
- `eval/` – evaluation outputs (CSV)

---

## Setup (Local)

### 1) Create a virtual environment
```bash
python -m venv .venv
Activate (Windows PowerShell):

.\.venv\Scripts\Activate.ps1
Activate (macOS/Linux):

source .venv/bin/activate
2) Install dependencies
pip install -r requirements.txt
3) Create a .env file
Create a file named .env in the project root with:

OPENAI_API_KEY=your_api_key_here
Usage
1) Ingest documents and build the vector store
Place PDFs in:

data/raw/health/

data/raw/nutrition/

data/raw/sport/

Then run:

python app/ingest.py
2) Run the Streamlit UI
python -m streamlit run app/ui_streamlit.py
3) Run the mini evaluation
python app/evaluation.py
Outputs will be written to eval/ as a timestamped CSV.

Notes
The document corpus is not included due to copyright restrictions (library access).

This prototype is intended for academic demonstration purposes.

The system is not a medical device and must not be used for diagnosis or treatment decisions.

License
MIT License
