# 🧬 AutoACMG — Genetic Variation RAG Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![Model](https://img.shields.io/badge/Embeddings-BioBERT-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A domain-specific **Retrieval-Augmented Generation (RAG)** system for querying PubMed/PMC literature on genetic variants and ACMG classification guidelines. Retrieves semantically relevant biomedical context and generates grounded, source-cited answers using a large language model.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Components](#components)
- [Configuration](#configuration)
- [Known Issues](#known-issues)

---

## Overview

AutoACMG combines biomedical literature retrieval with LLM-based question answering. It is designed for researchers and clinicians who need fast, evidence-backed answers about genetic variant interpretation and ACMG classification criteria.

**Key capabilities:**
- Fetches and parses full-text articles from PubMed/PMC via E-utilities API
- Embeds documents using a BioBERT-based biomedical embedding model
- Stores and retrieves chunks via a persistent ChromaDB vector store
- Generates grounded answers with source attribution via Gemini LLM
- Provides both a CLI and a Streamlit chat interface

---

## Pipeline Architecture

| Step | Component | Tool / Method | Output |
|:----:|-----------|---------------|--------|
| 1️⃣ | **Data Retrieval** | PubMed/PMC E-utilities API (`esearch` + `efetch`) | Raw XML articles |
| 2️⃣ | **Parsing** | `paper_extraction.ipynb` — BeautifulSoup XML parser | `pubmed_papers/*.txt` |
| 3️⃣ | **Chunking** | `RecursiveCharacterTextSplitter` — 500 tokens, 100 overlap | Text chunks |
| 4️⃣ | **Embedding** | `NeuML/biomedbert-base-embeddings` (BioBERT) | Dense vectors |
| 5️⃣ | **Indexing** | `vector.py` → ChromaDB persistent store | `chromadb/` collection |
| 6️⃣ | **Retrieval** | Cosine similarity search — top-k = 5 | Relevant chunks + sources |
| 7️⃣ | **Generation** | Gemini LLM + LangChain prompt template | Grounded answer with citations |

---

## Project Structure

```
.
├── paper_extraction.ipynb   # Fetch & parse PubMed/PMC articles → .txt files
├── vector.py                # Build ChromaDB vector store; expose retriever
├── main.py                  # CLI for interactive terminal Q&A
├── streamlit_app.py        # Streamlit web chat UI
├── pubmed_papers/           # Extracted paper text files (auto-created)
└── chromadb/                # Persistent ChromaDB vector store (auto-created)
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [Google AI API key](https://aistudio.google.com/) (Gemini)
- A [HuggingFace token](https://huggingface.co/settings/tokens)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export HF_TOKEN="your-huggingface-token"
```

### 3. Fetch Papers

Run `paper_extraction.ipynb` to query PubMed/PMC (default query: `variant AND acmg`) and save cleaned article text to `./pubmed_papers/`.

### 4. Build the Vector Store

```bash
python vector.py
```

> This only needs to be run once, or whenever new papers are added.

---

## Usage

### CLI

```bash
python main.py
```

```
Enter your question (or 'quit' to exit): What are the ACMG criteria for pathogenic variants?

[Source PMC1234567] The ACMG/AMP guidelines classify variants using ...
```

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser to use the chat interface.

---

## Components

### `paper_extraction.ipynb`

Queries the NCBI E-utilities API (`esearch` + `efetch`) to retrieve full-text PMC articles in XML. For each research article:
- Strips inline formatting tags (`<italic>`) and citation markers (`<xref>`)
- Extracts title, abstract, and body sections
- Saves output as `./pubmed_papers/<PMCID>.txt`

### `vector.py`

| Step | Detail |
|------|--------|
| **Load** | Reads all `.txt` files from `./pubmed_papers/` |
| **Chunk** | 500-token chunks, 100-token overlap (`RecursiveCharacterTextSplitter`) |
| **Embed** | `NeuML/biomedbert-base-embeddings` (BioBERT fine-tuned for biomedical similarity) |
| **Store** | Persistent ChromaDB collection: `Info_variations` |
| **Retrieve** | Top-5 chunks by cosine similarity |

Device is selected automatically: `CUDA` → `MPS` → `CPU`.

### `main.py`

Interactive CLI loop. Retrieves context chunks, formats them with source labels, and passes them to the Gemini LLM via a LangChain prompt chain.

### `streamlite_app.py`

Streamlit chat UI with session-state conversation history, spinner during retrieval, and markdown-rendered responses.

---

## Configuration

| Parameter | File | Default |
|-----------|------|---------|
| Chunk size | `vector.py` | `500` tokens |
| Chunk overlap | `vector.py` | `100` tokens |
| Top-k retrieval | `vector.py` | `5` chunks |
| Embedding model | `vector.py` | `NeuML/biomedbert-base-embeddings` |
| ChromaDB collection | `vector.py` | `Info_variations` |
| ChromaDB path | `vector.py` | `./chromadb` |
| LLM | `main.py`, `streamlite_app.py` | `gemini-3.5-flash` |

---

## Embedding Model

**[`NeuML/biomedbert-base-embeddings`](https://huggingface.co/NeuML/biomedbert-base-embeddings)** — a BioBERT-based sentence embedding model fine-tuned for biomedical semantic similarity. Chosen for superior retrieval accuracy on clinical and genomic text compared to general-purpose embeddings.
