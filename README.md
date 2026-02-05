# Synapse 2.0

Memory-driven scientific exploration CLI inspired by EverMemOS and OpenScholar.

## Quick start

```bash
# one-click setup + run
bash run.sh
```

## Configuration

Secrets are loaded from `.env` (git-ignored).

```bash
cp .env.example .env
# edit GEMINI_API_KEY in .env
```

Environment variables:
- `GEMINI_API_KEY`: enables Gemini generation and embeddings.
- `GEMINI_MODEL`: model name (default `gemini-2.0-flash`).
- `GEMINI_TEMPERATURE`: generation temperature (default `0.0` for reproducibility).
- `GEMINI_TOP_P`: nucleus sampling parameter.
- `GEMINI_TOP_K`: top-k sampling parameter.
- `GEMINI_MAX_OUTPUT_TOKENS`: output budget per call.
- `GEMINI_EMBEDDING_MODELS`: comma-separated embedding model candidates (auto-fallback).
- `SYNAPSE_AUTO_BACKFILL_EMBEDDINGS`: auto-embed legacy memories missing vectors at startup.
- `SYNAPSE_MEMORY_DIR`: persistent memory directory (default `.synapse_memory`).

## LLM requirement

Synapse now runs in strict LLM mode for scientific workflows:
- `upload`, `ask`, and `explore` require a working LLM connection.
- No offline fallback is used for extraction, reasoning, or exploration planning.

## Audit logs

Every run writes structured JSONL events to:
- `.synapse_memory/runs/run_<timestamp>.jsonl`

Each event contains query/topic metadata, citation coverage, hypothesis counts,
and runtime info for reproducibility and paper traceability.

## Core commands

```text
/memory upload <file>   Upload and extract memory
/memory trace <query>   Trace relevant memory graph
/explore <topic>        Multi-step scientific exploration
/status                 Runtime and memory stats
```

## Notes

- QA evidence retrieval is now purity-filtered to `ATOMIC_FACT` and `EPISODE` only.
- Long documents are processed with chunked extraction and hierarchical summarization.
- If `faiss` is unavailable, retrieval falls back to in-memory cosine search.
