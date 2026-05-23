"""
FAQ Ingestor.
Chunks the synthetic FAQ markdown file, embeds it with Ollama, and stores it in ChromaDB.
"""

import logging
from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag.ingestor")

def load_documents(path: str) -> str:
    """Load raw text from faqs.md"""
    logger.info(f"Loading documents from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_documents(text: str) -> list[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    logger.info("Chunking document content...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)

def embed_and_store(chunks: list[str]) -> int:
    """
    Embed chunks with nomic-embed-text-v2-moe and store in ChromaDB.
    Returns number of chunks stored.
    """
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    
    # Reset collection if exists to avoid duplication
    try:
        client.delete_collection(name=settings.chroma_collection_name)
        logger.info(f"Deleted existing Chroma collection: {settings.chroma_collection_name}")
    except Exception:
        pass
        
    collection = client.create_collection(name=settings.chroma_collection_name)
    
    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url
    )
    
    logger.info(f"Generating embeddings using model '{settings.embedding_model}' from Ollama...")
    vectors = embeddings.embed_documents(chunks)
    
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "faqs.md", "chunk_index": i} for i in range(len(chunks))]
    
    logger.info("Writing vectors to ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas
    )
    
    # Perform verification step
    logger.info("Performing verification query...")
    query_text = "What are the visiting hours?"
    query_vector = embeddings.embed_query(query_text)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )
    
    if results and "documents" in results and results["documents"]:
        top_doc = results["documents"][0][0]
        logger.info(f"Verification query: '{query_text}'")
        logger.info(f"Top matching chunk found:\n{top_doc}")
    else:
        logger.warning("Verification query returned no results!")
        
    return len(chunks)

def ingest(faq_path: str = None) -> None:
    """Full ingestion pipeline. Called as entry point."""
    if faq_path is None:
        faq_path = Path(__file__).parent / "data" / "faqs.md"
    else:
        faq_path = Path(faq_path)
        
    logger.info("Starting FAQ ingestion...")
    text = load_documents(faq_path)
    chunks = chunk_documents(text)
    logger.info(f"Created {len(chunks)} chunks")
    count = embed_and_store(chunks)
    logger.info(f"Stored {count} chunks in ChromaDB at {settings.chroma_persist_dir}")

if __name__ == "__main__":
    ingest()
