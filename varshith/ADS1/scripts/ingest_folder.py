import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vector_store import ingest_file


SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".png", ".jpg", ".jpeg"}


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/documents")
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    total = 0
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            count = ingest_file(path)
            total += count
            print(f"Ingested {count} chunks from {path}")

    print(f"Done. Total chunks: {total}")


if __name__ == "__main__":
    main()
