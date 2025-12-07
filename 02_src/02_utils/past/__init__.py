"""
RAG Utils Module for HuggingFace DailyPapers Chatbot (Simplified)

This package provides core infrastructure for the RAG-based chatbot:
- Data validation (Pydantic models)
- Document loading and conversion
- Text chunking (RecursiveCharacterTextSplitter + pickle cache)
- HuggingFace embeddings (FREE!)
- Vector database operations (ChromaDB)
- Search and retrieval

Example usage:
    >>> from utils import VectorDBManager, EmbeddingManager
    >>> from utils import load_all_documents, get_statistics
    >>> from utils import chunk_documents, chunk_and_save, load_chunks_from_pkl
    >>>
    >>> # Load and chunk documents
    >>> docs = load_all_documents(count=5)
    >>> chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    >>>
    >>> # Or load from cache
    >>> chunks = load_chunks_from_pkl("chunks_all.pkl")
    >>>
    >>> # Initialize VectorDB with embeddings
    >>> db_manager = VectorDBManager()
    >>> db_manager.index_documents(chunks, reset=True)
    >>>
    >>> # Search
    >>> results = db_manager.similarity_search("transformer models", k=5)
"""

# Validation (Pydantic models)
from .validators import (
    PaperDocument,
    PaperMetadata,
    DocIdInfo,
    ValidationReport,
    validate_json_document,
    validate_document_batch,
    validate_doc_id,
    validate_directory,
    validate_all_weeks
)

# Documents (loading and parsing)
from .documents import (
    parse_doc_id,
    normalize_upvote,
    load_json_document,
    json_to_langchain_document,
    load_documents_by_week,
    load_documents_batch,
    load_all_documents,
    get_document_statistics,
    get_statistics,  # Alias
    load_documents_by_filter
)

# Chunking (Text splitting + pickle cache)
from .chunking import (
    chunk_documents,
    save_chunks_to_pkl,
    load_chunks_from_pkl,
    chunk_and_save,
    get_chunk_statistics,
    list_chunk_files
)

# Vector DB (ChromaDB + Search)
from .vectordb import (
    load_vectordb
)


__all__ = [
    # Validation
    "PaperDocument",
    "PaperMetadata",
    "DocIdInfo",
    "ValidationReport",
    "validate_json_document",
    "validate_document_batch",
    "validate_doc_id",
    "validate_directory",
    "validate_all_weeks",

    # Documents
    "parse_doc_id",
    "normalize_upvote",
    "load_json_document",
    "json_to_langchain_document",
    "load_documents_by_week",
    "load_documents_batch",
    "load_all_documents",
    "get_document_statistics",
    "get_statistics",
    "load_documents_by_filter",

    # Chunking
    "chunk_documents",
    "save_chunks_to_pkl",
    "load_chunks_from_pkl",
    "chunk_and_save",
    "get_chunk_statistics",
    "list_chunk_files",

    # Core Classes
    "EmbeddingManager",
    "VectorDBManager",
]


__version__ = "2.0.0-simplified"
__author__ = "SKN20-3rd-2TEAM"