# rag-mcp

Semantic search over a folder of markdown documents, built from scratch with Postgres + pgvector and local embeddings. No RAG framework: each piece (chunking, embeddings, vector search) is a small module you can read in a few minutes.

**Status:** ingestion and search work. Next: answer generation with citations, an MCP server exposing `search` to Claude Code, and a retrieval evaluation (hit rate@5).

## Architecture

1. `chunking.py` splits each `.md` file in `docs/` into 800-character chunks with 150 characters of overlap.
2. `embeddings.py` turns chunks into 1024-dimension vectors with `bge-m3`, served locally by Ollama through its OpenAI-compatible API.
3. `ingest.py` stores each chunk with its source file, position and vector in Postgres; re-running it never duplicates rows.
4. `search.py` embeds the question and returns the k closest chunks by cosine distance (`<=>`), with source and distance.
5. Everything runs on the local machine: no API key, and the documents never leave it.

## Run it

Requirements: Docker, Python 3.12+, [Ollama](https://ollama.com).

```bash
# Postgres with pgvector (port 5433 to avoid clashing with a local 5432)
docker run -d --name rag-db -e POSTGRES_PASSWORD=dev -p 5433:5432 \
  -v rag-pgdata:/var/lib/postgresql/data pgvector/pgvector:pg17

# Embedding model
ollama pull bge-m3

# Python environment
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
```

Put your markdown files in `docs/` (the folder is git-ignored), then:

```bash
.venv/bin/python ingest.py --dry-run   # count files and chunks first
.venv/bin/python ingest.py
.venv/bin/python search.py "How do I ship a new release?" -k 3
```

To inspect how a single file is chunked: `.venv/bin/python chunking.py docs/some-file.md`.

## Design decisions

- **Fixed-size chunks with overlap** as a deliberate baseline. The overlap (step = size − overlap) keeps a sentence cut at a boundary whole in at least one chunk. Splitting by markdown heading is the planned change to measure against it.
- **Idempotent ingestion.** `UNIQUE (source, chunk_index)` plus `INSERT ... ON CONFLICT DO UPDATE`, and chunks left over when a file gets shorter are deleted.
- **Batched embedding calls** instead of one request per chunk.
- **Local, multilingual embeddings.** `bge-m3` keeps documents on the machine, which matters for confidential material, and matches questions across languages.
- **Search returns the distance**, so the caller can refuse to answer when the closest chunk is too far away instead of making something up.

## First results

Corpus: 25 markdown files (~181k characters, 237 chunks) of internal docs from one of my apps. Ingestion takes about 13 seconds on an Apple Silicon Mac.

| Question | Top result | Cosine distance |
|---|---|---|
| Quais são os limites do plano gratuito? (Portuguese) | `plans-and-limits.md` | 0.338 |
| How do I ship a new release to the App Store? | `release-process.md` | 0.352 |
| Comment les événements d'analytique sont-ils nommés ? (French) | `observability/events.md` | 0.438 |
| receita de bolo de cenoura (off-topic) | unrelated chunk | 0.665 |

The corpus is in English: the Portuguese and French questions still find the right file. The off-topic question stays far from everything; that gap is where the "too far to answer" threshold will be set, once the evaluation set exists.
