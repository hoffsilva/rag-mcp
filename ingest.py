import argparse
from collections import defaultdict

import psycopg
from pgvector import Vector

from chunking import chunk_text
from config import DOCS_DIR, EMBEDDING_MODEL
from db import connect
from embeddings import embed, make_client

UPSERT = """
    INSERT INTO chunks (source, chunk_index, content, embedding)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (source, chunk_index)
    DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
"""

# A file that got shorter leaves old chunks behind; remove them.
DELETE_STALE = "DELETE FROM chunks WHERE source = %s AND chunk_index >= %s"


def load_chunks() -> list[tuple[str, int, str]]:
    """Return (source, chunk_index, content) for every chunk of every markdown file in docs/."""
    rows: list[tuple[str, int, str]] = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        source = str(path.relative_to(DOCS_DIR))
        for index, content in enumerate(chunk_text(path.read_text())):
            rows.append((source, index, content))
    return rows


def save(conn: psycopg.Connection, rows: list[tuple[str, int, str]], vectors: list[list[float]]) -> None:
    counts: dict[str, int] = defaultdict(int)
    with conn.transaction():
        for (source, index, content), vector in zip(rows, vectors, strict=True):
            conn.execute(UPSERT, (source, index, content, Vector(vector)))
            counts[source] += 1
        for source, count in counts.items():
            conn.execute(DELETE_STALE, (source, count))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk docs/, embed the chunks and store them in Postgres.")
    parser.add_argument("--dry-run", action="store_true", help="only count chunks and estimate tokens")
    args = parser.parse_args()

    rows = load_chunks()
    files = len({source for source, _, _ in rows})
    chars = sum(len(content) for _, _, content in rows)
    print(f"{files} files, {len(rows)} chunks, {chars} chars, model {EMBEDDING_MODEL}")
    if args.dry_run:
        return

    client = make_client()
    vectors = embed(client, [content for _, _, content in rows])
    with connect() as conn:
        save(conn, rows, vectors)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"Stored. SELECT count(*) FROM chunks = {total}")


if __name__ == "__main__":
    main()
