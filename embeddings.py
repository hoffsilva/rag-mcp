from itertools import batched

from openai import OpenAI

from config import EMBEDDING_BASE_URL, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


def make_client() -> OpenAI:
    """Client for Ollama's OpenAI-compatible API; Ollama ignores the key but the SDK requires one."""
    return OpenAI(base_url=EMBEDDING_BASE_URL, api_key="ollama")


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed texts in batches; returns one vector per text, in the same order."""
    vectors: list[list[float]] = []
    for batch in batched(texts, EMBEDDING_BATCH_SIZE):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=list(batch))
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    return vectors
