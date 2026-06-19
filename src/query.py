import os
import google.generativeai as genai
import chromadb
from chromadb.api.types import Documents, Embeddings
from dotenv import load_dotenv

from src.config import (
    COLLECTION_NAME,
    DB_DIR,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    GOOGLE_API_KEY,
)

# Load environment variables
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Configure genai with the API Key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
if api_key:
    genai.configure(api_key=api_key)


class GoogleGenerativeAiEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom Google Generative AI Embedding Function for ChromaDB.
    Avoids ClientOptions incompatibility issues present in standard Chroma utility.
    """
    def __init__(self, api_key: str, model_name: str = EMBEDDING_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        response = genai.embed_content(
            model=self.model_name,
            content=input,
            task_type="retrieval_query",
        )
        embeddings = response["embedding"]
        if not embeddings:
            return []
        if isinstance(embeddings[0], (int, float)):
            return [embeddings]
        return embeddings


def query_rag_pipeline(user_query: str, db_path: str = str(DB_DIR), k: int = 3) -> dict:
    """
    Searches the database, builds a grounded prompt, and queries the LLM.
    """
    current_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
    if not current_api_key:
        raise ValueError(
            "API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in the environment or .env file."
        )
    genai.configure(api_key=current_api_key)

    client = chromadb.PersistentClient(path=db_path)

    # Get collection without overriding its registered embedding function to avoid conflicts.
    collection = client.get_collection(name=COLLECTION_NAME)

    # Embed the query manually using the same embedding model and task type "retrieval_query".
    embed_response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=user_query,
        task_type="retrieval_query"
    )
    query_embedding = embed_response["embedding"]

    # Query collection using the generated query embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # Format the retrieved documents as background context
    context_blocks = []
    sources = []

    if results and 'documents' in results and results['documents'] and results['metadatas']:
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            source_name = meta.get('source', 'Unknown')
            page_num = meta.get('page', -1)
            citation_str = f"Source: {source_name}, Page: {page_num}"

            context_blocks.append(f"[{citation_str}]\nContext: {doc}")
            sources.append({
                "file": source_name,
                "page": page_num,
                "snippet": doc
            })

    context_payload = "\n\n---\n\n".join(context_blocks)

    # Set up grounding system prompt
    system_prompt = (
        "You are a professional, accurate document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Cite the sources (filenames and pages) inline next to facts you cite. "
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do not make up facts or use external knowledge sources."
    )

    prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    # Set up model and generate content
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    answer = response.text.strip()

    return {
        "answer": answer,
        "sources": sources
    }


def answer_question(query: str) -> dict:
    """
    Main interface callable function to answer questions using the RAG pipeline.
    """
    from src.config import TOP_K
    return query_rag_pipeline(user_query=query, k=TOP_K)
