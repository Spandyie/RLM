"""
Document Store - ChromaDB vector storage.

Stores document chunks and enables semantic search.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from .processor import DocumentChunk


@dataclass
class SearchResult:
    """A search result with relevance score."""
    chunk: DocumentChunk
    score: float
    distance: float


class DocumentStore:
    """Vector store for document chunks."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "documents",
    ):
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunk(self, chunk: DocumentChunk) -> str:
        """Add a single chunk."""
        chunk_id = f"{chunk.doc_id}_{chunk.chunk_index}"

        self.collection.add(
            documents=[chunk.text],
            ids=[chunk_id],
            metadatas=[{
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }],
        )
        return chunk_id

    def add_document(self, doc) -> List[str]:
        """Add all chunks from a document."""
        return [self.add_chunk(c) for c in doc.chunks]

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for relevant chunks."""
        where = {"doc_id": doc_id} if doc_id else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            dists = results["distances"][0] if results["distances"] else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                chunk = DocumentChunk(
                    text=doc,
                    doc_id=meta.get("doc_id", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                )
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=1.0 - dist,
                    distance=dist,
                ))

        return search_results

    def list_documents(self) -> List[dict]:
        """List all documents."""
        items = self.collection.get(include=["metadatas"])

        docs = {}
        for meta in items.get("metadatas", []):
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        """Delete a document."""
        items = self.collection.get(where={"doc_id": doc_id})
        ids = items.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def get_document_chunks(self, doc_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        if results["documents"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                chunks.append(DocumentChunk(
                    text=doc,
                    doc_id=doc_id,
                    chunk_index=meta.get("chunk_index", 0),
                    metadata=meta,
                ))

        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Get all chunks across all documents."""
        results = self.collection.get(
            include=["documents", "metadatas"],
        )

        chunks = []
        docs = results.get("documents", []) or []
        metas = results.get("metadatas", []) or []

        for doc, meta in zip(docs, metas):
            chunks.append(DocumentChunk(
                text=doc,
                doc_id=meta.get("doc_id", ""),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            ))

        chunks.sort(key=lambda c: (c.doc_id, c.chunk_index))
        return chunks

    def get_stats(self) -> dict:
        """Get store statistics."""
        items = self.collection.get(include=["metadatas"])
        doc_ids = set(m.get("doc_id") for m in items.get("metadatas", []))

        return {
            "total_chunks": len(items.get("ids", [])),
            "total_documents": len(doc_ids),
        }
