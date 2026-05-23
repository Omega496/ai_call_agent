"""
RAG Retriever.
Queries the local ChromaDB database to retrieve the top-k most relevant FAQ chunks.
"""

import logging
from core.config import settings

logger = logging.getLogger(__name__)

class FAQRetriever:
    """
    Singleton retriever that maintains a persistent ChromaDB connection
    and Ollama embedding model for FAQ chunk retrieval.
    """
    
    def __init__(self):
        self._client = None
        self._collection = None
        self._embedder = None
    
    def initialize(self) -> None:
        """
        Initialize ChromaDB client and Ollama embedder.
        Called once during FastAPI lifespan startup.
        """
        try:
            import chromadb
            from langchain_ollama import OllamaEmbeddings
            
            logger.info("Initializing FAQRetriever...")
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            self._collection = self._client.get_or_create_collection(name=settings.chroma_collection_name)
            self._embedder = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url
            )
            logger.info("FAQRetriever initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize FAQRetriever: {e}", exc_info=True)
            # Allow fallback to uninitialized state rather than crashing
            
    def is_initialized(self) -> bool:
        """Return True if ChromaDB connection is ready."""
        return self._collection is not None and self._embedder is not None
        
    def retrieve_chunks(self, query: str, n_results: int = 5) -> list[str]:
        """
        Embed the query and retrieve top-n relevant FAQ chunks.
        
        Args:
            query: The patient's question (raw transcript text)
            n_results: Number of chunks to retrieve (default: 5)
        
        Returns:
            List of chunk text strings, ordered by relevance (most relevant first)
            Returns empty list if retrieval fails (do not raise).
        """
        truncated_query = query[:100] + ("..." if len(query) > 100 else "")
        logger.debug(f"Retrieving chunks for query: '{truncated_query}' (n_results={n_results})")
        
        if not self.is_initialized():
            logger.error("FAQRetriever is not initialized. Cannot retrieve chunks.")
            return []
            
        try:
            # 1. Embed query using OllamaEmbeddings
            embedding = self._embedder.embed_query(query)
            
            # 2. Query ChromaDB collection
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            
            # 3. Extract documents from result["documents"][0]
            if result and "documents" in result and result["documents"]:
                documents = result["documents"][0]
                logger.debug(f"Retrieved {len(documents)} chunks successfully.")
                return documents
                
            logger.warning("Query returned empty documents list.")
            return []
            
        except Exception as e:
            logger.error(f"Error during retrieve_chunks for query '{truncated_query}': {e}", exc_info=True)
            return []

# Module-level singleton
faq_retriever = FAQRetriever()

from langsmith import traceable

# Convenience function used by faq_node
@traceable(name="faq_retrieval", tags=["rag", "chromadb"])
def retrieve_chunks(query: str, n_results: int = 5) -> list[str]:
    return faq_retriever.retrieve_chunks(query, n_results)

def retrieve_faq_context(query: str, k: int = 5) -> str:
    """
    Backward-compatibility wrapper.
    """
    chunks = retrieve_chunks(query, n_results=k)
    return "\n\n".join(chunks)
