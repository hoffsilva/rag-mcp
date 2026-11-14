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
