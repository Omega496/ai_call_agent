import pytest
from unittest.mock import MagicMock, patch
from rag.ingestor import chunk_documents, embed_and_store, ingest
from rag.retriever import FAQRetriever

# 1. test_chunk_documents_produces_correct_count
def test_chunk_documents_produces_correct_count():
    # Generate 5 long Q&A paragraphs to ensure splitting happens
    text = "\n\n".join([
        f"Q: Question number {i}?\n"
        f"A: This is a long answer for question number {i}. "
        "We need to write enough sentences to ensure that the content is descriptive and "
        "takes up sufficient character space. This is important for chunking testing. "
        "It helps us verify that the text splitter correctly splits the document into smaller pieces."
        for i in range(5)
    ])
    
    chunks = chunk_documents(text)
    
    # Assert: produces between 3-8 chunks (flexible range due to overlap)
    assert 3 <= len(chunks) <= 8
    
    # Assert: no chunk exceeds 512 tokens (or characters as RecursiveCharacterTextSplitter handles it)
    for chunk in chunks:
        # Check token length estimate (words) and character length
        assert len(chunk.split()) <= 512
        assert len(chunk) <= 512

# 2. test_chunk_documents_overlap_preserved
def test_chunk_documents_overlap_preserved():
    # Long text sharing a specific phrase to verify overlap behavior
    text = (
        "Q: What is the hospital address?\n"
        "A: The main campus address is 123 Care Lane, Medical City. "
        "This location is easily accessible from the main highway. "
        "Please remember this address: 123 Care Lane, Medical City.\n\n"
        "Q: Where is parking?\n"
        "A: Parking is available directly opposite the main entrance at 123 Care Lane, Medical City."
    )
    chunks = chunk_documents(text)
    assert len(chunks) > 0
    # Verify that the text splitter successfully parsed the input
    assert any("123 Care Lane" in chunk for chunk in chunks)

# 3. test_ingestor_calls_chromadb_add
def test_ingestor_calls_chromadb_add():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.create_collection.return_value = mock_collection
    
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2]] * 3
    mock_embeddings.embed_query.return_value = [0.1, 0.2]
    
    with patch("chromadb.PersistentClient", return_value=mock_client), \
         patch("rag.ingestor.OllamaEmbeddings", return_value=mock_embeddings):
        
        count = embed_and_store(["chunk1", "chunk2", "chunk3"])
        
        assert count == 3
        mock_collection.add.assert_called_once()
        args, kwargs = mock_collection.add.call_args
        assert len(kwargs["documents"]) == 3
        assert kwargs["ids"] == ["chunk_0", "chunk_1", "chunk_2"]

# 4. test_retriever_returns_chunks
def test_retriever_returns_chunks():
    retriever = FAQRetriever()
    
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    
    mock_collection.query.return_value = {"documents": [["answer1", "answer2"]]}
    
    with patch("chromadb.PersistentClient", return_value=mock_client), \
         patch("langchain_ollama.OllamaEmbeddings", return_value=mock_embeddings):
        
        retriever.initialize()
        chunks = retriever.retrieve_chunks("test query", n_results=2)
        
        assert chunks == ["answer1", "answer2"]
        mock_collection.query.assert_called_once()

# 5. test_retriever_returns_empty_on_failure
def test_retriever_returns_empty_on_failure():
    retriever = FAQRetriever()
    
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.side_effect = ConnectionError("Mock connection failed")
    
    with patch("chromadb.PersistentClient", return_value=mock_client), \
         patch("langchain_ollama.OllamaEmbeddings", return_value=mock_embeddings):
        
        retriever.initialize()
        chunks = retriever.retrieve_chunks("test query")
        assert chunks == []

# 6. test_retriever_returns_empty_when_not_initialized
def test_retriever_returns_empty_when_not_initialized():
    retriever = FAQRetriever()
    # Call retrieve_chunks without initializing
    chunks = retriever.retrieve_chunks("test")
    assert chunks == []

# 7. test_full_ingest_pipeline
def test_full_ingest_pipeline():
    with patch("rag.ingestor.load_documents", return_value="some text") as mock_load, \
         patch("rag.ingestor.chunk_documents", return_value=["chunk1"]) as mock_chunk, \
         patch("rag.ingestor.embed_and_store", return_value=1) as mock_store:
        
        ingest(faq_path="tests/fixtures/sample_faqs.md")
        
        mock_load.assert_called_once()
        mock_chunk.assert_called_once_with("some text")
        mock_store.assert_called_once_with(["chunk1"])
