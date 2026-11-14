import sys
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size chunks; consecutive chunks share `overlap` characters."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(piece)
        # The last chunk already reached the end: another step would only repeat its tail.
        if start + size >= len(text):
            break
    return chunks


if __name__ == "__main__":
    # Etapa 2 checkpoint: print the chunks of one file and read them on their own.
    path = Path(sys.argv[1])
    for index, chunk in enumerate(chunk_text(path.read_text())):
        print(f"--- chunk {index} ({len(chunk)} chars) ---")
        print(chunk)
