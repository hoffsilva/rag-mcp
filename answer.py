import argparse

import psycopg
from openai import OpenAI

from config import CHAT_MODEL, MAX_DISTANCE, TOP_K
from db import connect
from embeddings import make_client
from search import search

NOT_FOUND = "Je ne sais pas : les documents ne contiennent pas cette information."

SYSTEM_PROMPT = f"""You answer questions using only the document excerpts provided.
Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge.
- After each fact, cite its file in square brackets exactly as labelled, e.g. [release-process.md].
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Answer in the language of the question."""


def build_user_message(question: str, hits: list[dict]) -> str:
    """Label each excerpt with its source so the model has something to cite."""
    excerpts = "\n\n".join(f"[{hit['source']}]\n{hit['content']}" for hit in hits)
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def answer(conn: psycopg.Connection, client: OpenAI, question: str) -> dict:
    """Answer from the docs; returns the text and the sources it actually cited."""
    hits = search(conn, client, question, k=TOP_K)
    # First guard: if even the closest chunk is far away, don't call the LLM at all.
    if not hits or hits[0]["distance"] > MAX_DISTANCE:
        return {"answer": NOT_FOUND, "sources": []}

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,  # factual and repeatable
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, hits)},
        ],
    )
    text = completion.choices[0].message.content or ""
    # Sources come from the search results, filtered by what the model cited:
    # a file name the model invents never reaches the list.
    sources = list(dict.fromkeys(h["source"] for h in hits if f"[{h['source']}]" in text))
    return {"answer": text, "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question from the ingested docs.")
    parser.add_argument("question")
    args = parser.parse_args()

    with connect() as conn:
        result = answer(conn, make_client(), args.question)
    print(result["answer"])
    if result["sources"]:
        print("\nSources: " + ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
