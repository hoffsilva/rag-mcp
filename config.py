import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:dev@localhost:5433/postgres"
)

# Local embeddings served by Ollama through its OpenAI-compatible API.
EMBEDDING_BASE_URL = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:11434/v1/")
EMBEDDING_MODEL = "bge-m3"  # multilingual, 1024 dimensions; must match schema.sql
EMBEDDING_BATCH_SIZE = 32

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Local chat model (same Ollama server) that writes answers from the retrieved chunks.
CHAT_MODEL = "gemma3:12b"
TOP_K = 5
# Refuse when the closest chunk is farther than this. Provisional: between the worst
# on-topic hit (0.438) and an off-topic question (0.665); calibrated by the eval.
MAX_DISTANCE = 0.55
