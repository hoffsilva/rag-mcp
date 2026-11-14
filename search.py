import argparse

import psycopg
from openai import OpenAI
from pgvector import Vector

from db import connect
from embeddings import embed, make_client

SEARCH = """
    SELECT source, chunk_index, content, embedding <=> %(q)s AS distance
    FROM chunks
    ORDER BY embedding <=> %(q)s   -- cosine distance: smaller means closer in meaning
    LIMIT %(k)s
"""


def search(conn: psycopg.Connection, client: OpenAI, question: str, k: int = 5) -> list[dict]:
    """Return the k chunks closest in meaning to the question, with their source and distance."""
    query_vector = Vector(embed(client, [question])[0])
    rows = conn.execute(SEARCH, {"q": query_vector, "k": k}).fetchall()
    return [
        {"source": source, "chunk_index": index, "content": content, "distance": float(distance)}
        for source, index, content, distance in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over the ingested docs.")
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    with connect() as conn:
        for hit in search(conn, make_client(), args.question, args.k):
            print(f"{hit['distance']:.3f}  {hit['source']}#{hit['chunk_index']}")
            print("      " + hit["content"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
