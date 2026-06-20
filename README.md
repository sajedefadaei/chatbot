# AutoACMG — Genetic Variation RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying PubMed literature on genetic variants and ACMG classification guidelines. The system retrieves semantically relevant context from biomedical papers and generates grounded answers using a large language model.

---

## Project Structure

```
.
├── paper_extraction.ipynb   # Fetches and parses PubMed/PMC articles into .txt files
├── vector.py                # Builds and manages the ChromaDB vector store; exposes retriever
├── main.py                  # CLI interface for interactive Q&A
├── streamlite_app.py        # Streamlit web UI for chat-based Q&A
└── pubmed_papers/           # Directory of extracted paper text files (auto-created)
└── chromadb/                # Persistent ChromaDB vector store (auto-created)
```

---

## Pipeline Overview

```
PubMed/PMC API
     │
     ▼
paper_extraction.ipynb   →   pubmed_papers/*.txt
     │
     ▼
vector.py (chunk + embed + store)   →   chromadb/
     │
     ▼
retriever (top-k semantic search)
     │
     ▼
LLM (Gemini) + prompt template   →   answer with sources
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

Set the following environment variables before running:

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export HF_TOKEN="your-huggingface-token"
```

> **Note:** Do not hard-code API keys in source files. Use environment variables or a `.env` file (with `python-dotenv`).

### 3. Fetch papers

Run `paper_extraction.ipynb` to query PubMed/PMC for articles matching your terms (default: `variant` AND `acmg`) and save cleaned text to `./pubmed_papers/`.

### 4. Build the vector store

Run `vector.py` directly to embed the documents and populate ChromaDB:

```bash
python vector.py
```

This only needs to be done once (or whenever you add new papers).

### 5. Run the app

**CLI:**
```bash
python main.py
```

**Streamlit web UI:**
```bash
streamlit run streamlite_app.py
```

---

## Components

### `paper_extraction.ipynb`
Queries the NCBI E-utilities API (`esearch` + `efetch`) to retrieve full-text PMC articles in XML format. Parses each article with BeautifulSoup, strips citation tags and formatting, and saves title + abstract + body sections as plain text files named by PMC ID.

### `vector.py`
Loads all `.txt` files from `./pubmed_papers/`, splits them into 500-token chunks (100-token overlap) using `RecursiveCharacterTextSplitter`, and embeds them with `NeuML/biomedbert-base-embeddings` — a domain-specific model trained on biomedical text. Stores embeddings in a persistent ChromaDB collection (`Info_variations`). Exposes a `retriever` that returns the top 5 most relevant chunks for a given query.

### `main.py`
A simple command-line loop that accepts a question, retrieves context chunks, formats them with source labels, and passes them to the Gemini LLM via a LangChain prompt chain. Prints the grounded answer to the terminal.

### `streamlite_app.py`
A Streamlit chat UI wrapping the same RAG chain. Maintains conversation history in session state and streams answers with a loading spinner.

---

## Embedding Model

**`NeuML/biomedbert-base-embeddings`** — a BioBERT-based sentence embedding model fine-tuned for biomedical semantic similarity. Chosen for better retrieval accuracy on clinical and genomic text compared to general-purpose embeddings.

Device selection is automatic: CUDA → MPS (Apple Silicon) → CPU.

---

## Configuration

| Parameter | Location | Default |
|---|---|---|
| Chunk size | `vector.py` | 500 tokens |
| Chunk overlap | `vector.py` | 100 tokens |
| Top-k retrieval | `vector.py` | 5 chunks |
| Embedding model | `vector.py` | `NeuML/biomedbert-base-embeddings` |
| LLM | `main.py`, `streamlite_app.py` | `gemini-3.5-flash` |
| ChromaDB collection | `vector.py` | `Info_variations` |

---

## Notes

- The vector store is persistent. Re-running `vector.py` as `__main__` will add documents again — deduplicate or clear the collection if rebuilding from scratch.
- The `results.text` attribute in the Streamlit app assumes a plain text LLM response; update to `results.content` if switching to a different LangChain model wrapper.