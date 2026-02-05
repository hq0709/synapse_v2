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
- `SYNAPSE_MEMORY_DIR`: persistent memory directory (default `.synapse_memory`).

## LLM requirement

Synapse now runs in strict LLM mode for scientific workflows:
- `upload`, `ask`, and `explore` require a working LLM connection.
- No offline fallback is used for extraction, reasoning, or exploration planning.

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
