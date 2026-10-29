# StudyLens — Conversational Research Paper RAG

A lightweight undergraduate CS project for uploading, studying, questioning, and comparing PDF research papers. Built with **Python + Streamlit + scikit-learn**, with **no Docker, database, or vector server**.

## Quick start (macOS)

```bash
cd /path/to/ResearchPaperRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501. For Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use **Paper library** to add your own text-based PDFs or load two included example PDFs. Open **Ask & follow up** to choose papers, ask a question, then ask natural follow-ups. Open **Compare papers** for a comparison of 2–5 papers.

## Three answer modes (choose in the sidebar)

| Mode | API key? | Extra installation? | Output |
|---|---|---|---|
| **No-key: source extracts** (default) | No | No | Selected relevant **original sentences with citations**. This is retrieval + extractive presentation, *not* LLM generation. |
| **Local AI: Ollama** | No | Yes: Ollama + downloaded model | Plain-language, conversational LLM explanations, grounded on retrieved evidence. |
| **Cloud AI** | Yes, for the provider you pick | No local model | Plain-language, conversational LLM explanations through **OpenAI, Anthropic, Gemini, Groq, OpenRouter, Mistral, DeepSeek, xAI, Together, or any OpenAI-compatible server**; may incur usage costs. |

**Recommended for best no-key experience: local Ollama.** Install Ollama from https://ollama.com/download, then in a terminal run:

```bash
ollama pull llama3.2:3b
```

Start Ollama if it is not already running (e.g. `ollama serve`; if its desktop app already started the server, don't start another). In the StudyLens sidebar, select **Local AI: Ollama** and **llama3.2:3b**. Other available selections include `qwen2.5:3b` and `phi3:mini`; download the chosen model first with `ollama pull MODEL_NAME`. Models need several GB of storage/RAM and initial download; smaller models can make more mistakes than larger ones. The interface runs in your browser; the local model serves on `127.0.0.1:11434`.

## API keys and the `.env` file (cloud providers)

Keys are **never stored in the code**. They are read from a `.env` file next to `app.py`:

```bash
cp .env.example .env        # Windows: copy .env.example .env
# open .env and paste the key(s) you have, e.g.
# ANTHROPIC_API_KEY=your-key-here
python -m pip install -r requirements.txt   # includes python-dotenv
python -m streamlit run app.py              # restart after every .env change
```

In the sidebar choose **Cloud AI**, then a **Provider** (a ✓ marks providers whose key was found in `.env`) and a **model**. If a key is missing you can paste one for the current session only. `.env` is already in `.gitignore`; commit `.env.example`, never `.env`.

| Provider | `.env` variable | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | A ChatGPT subscription is not API credit. |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | Native Messages API. |
| Google Gemini | `GEMINI_API_KEY` | Native Gemini API; free tier available in AI Studio. |
| Groq / OpenRouter / Mistral / DeepSeek / xAI / Together | `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `TOGETHER_API_KEY` | OpenAI-compatible APIs. |
| Custom (OpenAI-compatible) | `CUSTOM_BASE_URL`, optional `CUSTOM_API_KEY`, `CUSTOM_MODELS` | LM Studio, vLLM, llama.cpp server, etc. |
| Ollama (local) | none; optional `OLLAMA_HOST` | See above. |

Optional settings: `DEFAULT_PROVIDER=anthropic` preselects a provider; `OPENAI_MODELS=gpt-4o-mini,gpt-4.1` (and `ANTHROPIC_MODELS`, `GEMINI_MODELS`, ...) replace a model dropdown. Model names change often, so every dropdown also has an **Other (type a model name)** option.

**Add another provider:** if it offers an OpenAI-compatible API, add one `Provider(...)` line to `PROVIDERS` in `providers.py` with `style='openai'`, its base URL, and its env variable name. Selected source passages and recent chat are sent to whichever provider you choose.

**Why aren't all answers automatically written in plain English without installing anything?** Natural-language generation requires an LLM, which either runs somewhere locally or through a provider. There is no bundled model and no secretly embedded ChatGPT key. Source-extract mode works immediately; local Ollama gives key-free generated answers once installed.

## Features

- Upload multiple user-selected PDFs; load two example papers; remove papers.
- File validation: up to 20 MB, 350 PDF pages; encrypted / scanned-only PDFs aren't supported.
- Detect duplicate uploads by content hash.
- Select exactly which papers inform your chat; ask successive follow-up questions in one conversation.
- Follow-up query expansion incorporates the preceding question for lexical retrieval; local/cloud LLM mode also receives limited preceding messages for dialogue context. Every response uses **fresh retrieved evidence**.
- Explain technical content in simple language without dropping units, parameters, qualifications, experimental details, or limitations (instruction-based; **not guaranteed**).
- Show source labels `[S1]`, corresponding original PDF page positions, text excerpts and cosine-similarity scores.
- Compare 2–5 papers using evidence balanced across the selected documents; optional comparison focus; Markdown download.
- Switch between three response modes and multiple local/cloud models without rebuilding or changing code.
- In-session document library and chat history, no database required.

## What RAG concepts are used?

```text
PDF upload → pypdf page text extraction → 230-word chunks (45-word overlap)
          → TF-IDF sparse vector index (in-memory, scikit-learn)
User question + short follow-up context → TF-IDF query vector
          → cosine-similarity ranking → top-k passages filtered by selected papers
          → [S1]/[S2] source IDs + PDF page numbers
          → either (a) verbatim extractive evidence [no-key mode]
             or (b) augmented prompt + retrieved passages → LLM (Ollama or any cloud provider) → grounded answer
```

| Concept | Where | Why |
|---|---|---|
| Ingestion | `extract_pdf()` | Reads uploaded PDF page text. |
| Chunking with overlap | `chunk_pages()` | Helps passages fit in retrieval and retain context across chunk edges. |
| Metadata | `Passage` | Keeps paper title, ID and PDF page with each passage. |
| Sparse vector indexing | `Retriever.__init__()` | TF-IDF vectors without external embedding service or GPU. |
| Similarity retrieval | `Retriever.search()` | Cosine similarity ranks relevant passages for each question. |
| Paper filtering | `Retriever.search(..., paper_ids=...)` | Answers based on chosen papers only. |
| Conversational retrieval | `followup_search_query()` | Supplements short follow-ups with preceding question topic for lexical matching. |
| Augmented prompting | `source_context()`, `llm_answer()` | Provides LLM with numbered retrieved text, rather than relying on general model knowledge. |
| Guardrails | `SYSTEM_PROMPT` | Requests accessible explanations, uncertainty, preserved technical details, and per-claim source citations. |
| Cross-paper evidence | `compare_retrieval()` and app comparison | Searches methods, datasets, results, limits across each selected paper. |
| Human verification | `sources_view()` | Lets users inspect the cited retrieved passages. |

**RAG vs. retrieval:** The no-key mode performs retrieval and shows extractive evidence. The full retrieval-**augmented generation** pipeline runs in local Ollama or cloud-provider mode. TF-IDF is an explainable **sparse retrieval** approach; it can miss synonyms and deep semantic similarity compared with neural embeddings. The index is rebuilt from the current in-session papers on each query (appropriate for small classroom libraries).

## How to demonstrate in a RAG course

1. Load two sample papers from **Paper library**.
2. In **Ask & follow up**, select a paper and ask: *What problem does this study address?* Then ask *How was it evaluated?* Show how the second question uses the prior topic.
3. Expand **Check sources** to demonstrate PDF page provenance.
4. Choose **Local AI: Ollama** (after installing a model) and repeat to demonstrate grounded generation. Compare against **No-key: source extracts**.
5. In **Compare papers**, choose both papers and ask for a comparison of evaluation methods or limitations.
6. Explain TF-IDF, chunk overlap, cosine similarity, top-k, paper filters, prompt augmentation, and hallucination limitations.

## Project files

```text
ResearchPaperRAG/
  app.py                 Streamlit UI: library, conversation, comparison and model choice
  rag_engine.py          PDF ingestion, chunking, TF-IDF retrieval, extractive/LLM answers
  providers.py           Provider registry (OpenAI, Anthropic, Gemini, ...) and .env key loading
  .env.example           Template for your private .env (copy it; never commit .env)
  requirements.txt       Small Python dependency list
  sample_papers/         Two bundled PDFs for testing
  tests/                 Retrieval and validation tests
  .streamlit/config.toml UI config
  README.md              This file
```

## Tests

```bash
python -m pip install pytest
python -m pytest -q tests
```

The automated tests check chunk overlap and page metadata, filtering, follow-up query expansion, extractive answer citations, and basic PDF validation. PDF indexing was also tested with both bundled samples. A live model request is **not** part of automated tests. Use a local manual test for the Streamlit UI and any LLM you choose.

## Limitations & scientific accuracy

- **Never assume a model-generated explanation is scientifically verified.** Verify metrics, numerical findings, interpretations, and citations by expanding original excerpts, and consult the complete paper for high-stakes claims.
- PDF page numbers are file page indices, which may differ from printed numbering.
- PDF extraction may mix columns, footnotes, tables and equations. Scanned/image-only PDFs need OCR not included in this lightweight app.
- Extractive source mode retains original technical writing and sometimes cuts sentences awkwardly due to PDF text extraction; it cannot deliver the natural explanations a generative model can.
- TF-IDF primarily recognizes similar words, not semantic equivalents. Conversational topic expansion is a simple heuristic; pronouns can remain ambiguous.
- Chunk-level citations indicate supporting passages; neither API nor local model has automatic citation verification.
- No persistent document database; uploads and conversation disappear when the session/server is reset.
- Comparisons use retrieved excerpts rather than an exhaustive full-paper analysis.
- Local Ollama uses CPU/RAM/disk and model downloads. Cloud providers require API access and sends excerpts off-device.

## Troubleshooting

- `ModuleNotFoundError: No module named 'pypdf'`: activate `.venv`, run `python -m pip install -r requirements.txt`, and start via `python -m streamlit run app.py`.
- Completely blank Streamlit page: stop with Ctrl+C, restart using the same Python environment, try a private browser window and look for terminal errors.
- Ollama connection refused: make sure Ollama is installed and started, and the named model was downloaded (`ollama list`).
- Ollama model not found: run `ollama pull llama3.2:3b` or pull the model selected in the sidebar.
- `No API key for ...`: add the variable named in the message to `.env`, save, and restart Streamlit (`.env` is read at startup).
- Provider error (401/403/404): check the key, that the model name is valid for your account (use *Other* to type one), billing, and internet connection.
- Weak answers: ask a specific question, increase evidence count, inspect sources, or try a stronger model.

The program is designed as an educational demonstration, not a peer-reviewed scientific analysis tool.
