"""
Vector Service — stores and retrieves processed documents in ChromaDB
for semantic search and similarity matching.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import chromadb

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class VectorService:
    """Manages document embeddings in ChromaDB."""

    def __init__(self):
        self._client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            self._client = chromadb.PersistentClient(
                path=settings.chroma_persist_dir,
            )
            self._collection = self._client.get_or_create_collection(
                name=settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "chroma_initialized",
                collection=settings.chroma_collection_name,
                count=self._collection.count(),
            )
        except Exception as e:
            logger.error("chroma_init_failed", error=str(e))

    async def store_document(
        self,
        document_id: str,
        ocr_text: str,
        extracted_data: dict[str, Any],
        document_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a processed document with its extraction results."""
        if not self._collection:
            logger.warning("chroma_not_available")
            return False

        try:
            doc_metadata = {
                "document_type": document_type,
                "extracted_at": datetime.utcnow().isoformat(),
                "field_count": len(
                    [v for v in extracted_data.values() if v is not None]
                ),
            }
            if metadata:
                doc_metadata.update({k: str(v) for k, v in metadata.items()})

            # Store OCR text as the document, extracted data as metadata
            self._collection.upsert(
                ids=[document_id],
                documents=[ocr_text],
                metadatas=[doc_metadata],
            )

            logger.info("document_stored", doc_id=document_id, type=document_type)
            return True
        except Exception as e:
            logger.error("document_store_failed", error=str(e))
            return False

    async def search_similar(
        self,
        query: str,
        n_results: int = 5,
        document_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar documents using semantic similarity."""
        if not self._collection:
            return []

        try:
            where_filter = None
            if document_type:
                where_filter = {"document_type": document_type}

            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
            )

            documents = []
            for i in range(len(results["ids"][0])):
                documents.append(
                    {
                        "id": results["ids"][0][i],
                        "text_preview": results["documents"][0][i][:200] + "...",
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if results.get("distances")
                        else None,
                    }
                )

            return documents
        except Exception as e:
            logger.error("search_failed", error=str(e))
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if not self._collection:
            return {"status": "unavailable"}

        return {
            "status": "connected",
            "total_documents": self._collection.count(),
            "collection_name": settings.chroma_collection_name,
        }


# Singleton
_vector_service: VectorService | None = None


def get_vector_service() -> VectorService:
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
