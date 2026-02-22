"""
Document Processing Package

This package handles document ingestion, parsing, chunking, and storage.
- processor.py: Parse and chunk documents (PDF, TXT, MD)
- store.py: ChromaDB vector store for semantic search
"""

from .processor import DocumentProcessor
from .store import DocumentStore

__all__ = ["DocumentProcessor", "DocumentStore"]
