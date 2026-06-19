import os
import shutil
import uuid
from pathlib import Path
import chromadb
import google.generativeai as genai
from docx import Document
from pypdf import PdfReader
from tqdm import tqdm
from chromadb.api.types import Documents, Embeddings
from dotenv import load_dotenv

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    DB_DIR,
    GOOGLE_API_KEY,
    SUPPORTED_EXTENSIONS,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
if api_key:
    genai.configure(api_key=api_key)


class GoogleGenerativeAiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom Google Generative AI Embedding Function for ChromaDB.
    Avoids ClientOptions incompatibility issues present in standard Chroma utility.
    Includes an automatic fallback to models/gemini-embedding-001 if the requested
    model is not supported or not found.
    """
    def __init__(self, api_key: str, model_name: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

        # model availability check
        try:
            genai.embed_content(
                model=self.model_name,
                content="test",
                task_type="retrieval_document"
            )
        except Exception:
            self.model_name = "models/gemini-embedding-001"

    def __call__(self, input: Documents) -> Embeddings:
        response = genai.embed_content(
            model=self.model_name,
            content=input,
            task_type="retrieval_document",
        )
        embeddings = response["embedding"]
        if not embeddings:
            return []
        if isinstance(embeddings[0], (int, float)):
            return [embeddings]
        return embeddings


def load_pdf(path: Path) -> list[tuple[str, int]]:
    reader = PdfReader(str(path))
    pages: list[tuple[str, int]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((text, page_number))

    return pages


def load_docx(path: Path) -> list[tuple[str, int | None]]:
    document = Document(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]

    if not paragraphs:
        return []

    return [("\n".join(paragraphs), None)]


def load_document(path: Path) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)

    raise ValueError(f"Unsupported file type: {path.name}")


def discover_documents(data_dir: Path = DATA_DIR) -> list[Path]:
    if not data_dir.exists():
        return []

    documents = [
        path
        for path in sorted(data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return documents


def chunk_extracted_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """
    Splits page-level documents into smaller, overlapping chunks.
    Ensures that source metadata is carried over to every individual chunk.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        metadata = page["metadata"]

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })

            start += (chunk_size - chunk_overlap)

    return chunks


def save_to_vector_db(chunks: list[dict], db_path: str = str(DB_DIR)):
    """
    Embeds text chunks and saves them into a persistent disk-based ChromaDB.
    """
    # chroma db client
    client = chromadb.PersistentClient(path=db_path)

    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"} # Use Cosine Distance
    )

    # batch data
    ids = [f"id_{i}_{uuid.uuid4().hex[:6]}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully indexed {len(chunks)} chunks in the vector database.")


def ingest_documents(data_dir: Path = DATA_DIR) -> int:
    if not api_key:
        raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY is not set. Add it to your .env file.")

    documents = discover_documents(data_dir)
    if not documents:
        raise FileNotFoundError(
            f"No supported documents found in {data_dir}. "
            f"Add PDF or DOCX files before running ingest."
        )

    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    pages = []
    for document_path in documents:
        document_pages = load_document(document_path)
        for text, page_num in document_pages:
            pages.append({
                "text": text,
                "metadata": {
                    "source": document_path.name,
                    "page": page_num if page_num is not None else -1
                }
            })
        print(f"Loaded {len(document_pages)} pages from {document_path.name}")

    if not pages:
        raise ValueError("No text was extracted from the documents in data/.")

    chunks = chunk_extracted_pages(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    save_to_vector_db(chunks, db_path=str(DB_DIR))

    return len(chunks)


def main() -> None:
    ingest_documents()


if __name__ == "__main__":
    main()
