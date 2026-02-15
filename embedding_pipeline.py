#!/usr/bin/env python3
"""ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only.

Reads parsed text data from NASA space mission folders and creates a persistent
ChromaDB collection with OpenAI embeddings for RAG applications.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_embedding_text_only.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings."""

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize the embedding pipeline.

        Args:
            openai_api_key: OpenAI API key.
            chroma_persist_directory: Directory to persist ChromaDB.
            collection_name: Name of the ChromaDB collection.
            embedding_model: OpenAI embedding model to use.
            chunk_size: Maximum size of text chunks in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.chroma_persist_directory = chroma_persist_directory

        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)

        # Create or get the collection (using OpenAI embedding function)
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key, model_name=embedding_model
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Initialized pipeline: collection='{collection_name}', "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Split text into chunks with metadata.

        Attempts to break at sentence boundaries when possible.  Chunks never
        exceed ``self.chunk_size`` characters and consecutive chunks share
        ``self.chunk_overlap`` characters of overlap.

        Args:
            text: Text to chunk.
            metadata: Base metadata for the text.

        Returns:
            List of (chunk_text, chunk_metadata) tuples.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # If the text fits in a single chunk, return it directly
        if len(text) <= self.chunk_size:
            chunk_meta = {**metadata, "chunk_index": 0, "total_chunks": 1}
            return [(text, chunk_meta)]

        chunks: List[Tuple[str, Dict[str, Any]]] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the very end, try to break at a sentence boundary
            if end < len(text):
                # Look backward from `end` for a sentence-ending punctuation
                search_region = text[start:end]
                best_break = -1
                for punct in (". ", ".\n", "! ", "!\n", "? ", "?\n"):
                    idx = search_region.rfind(punct)
                    if idx != -1 and idx > best_break:
                        best_break = idx + len(punct)

                if best_break > self.chunk_size // 4:
                    end = start + best_break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_meta = {**metadata, "chunk_index": chunk_index}
                chunks.append((chunk_text, chunk_meta))
                chunk_index += 1

            # Advance with overlap
            start = end - self.chunk_overlap
            if start <= (end - self.chunk_size):
                # Safety: always advance at least a bit
                start = end

        # Fill in total_chunks now that we know the final count
        for i, (ct, cm) in enumerate(chunks):
            cm["total_chunks"] = len(chunks)

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        """Check if a document with the given ID already exists in the collection.

        Args:
            doc_id: Document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result["ids"]) > 0
        except Exception:
            return False

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing document in the collection.

        Args:
            doc_id: Document ID to update.
            text: New text content.
            metadata: New metadata.

        Returns:
            True if successful, False otherwise.
        """
        try:
            embedding = self.get_embedding(text)
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """Delete all documents from a specific source.

        Args:
            source_pattern: Pattern to match source names.

        Returns:
            Number of documents deleted.
        """
        try:
            all_docs = self.collection.get()
            ids_to_delete = []
            for i, metadata in enumerate(all_docs["metadatas"]):
                if source_pattern in metadata.get("source", ""):
                    ids_to_delete.append(all_docs["ids"][i])

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} documents matching: {source_pattern}"
                )
                return len(ids_to_delete)
            else:
                logger.info(f"No documents found matching: {source_pattern}")
                return 0
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """Get all document IDs for a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            List of document IDs for the file.
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            all_docs = self.collection.get()
            file_doc_ids = []
            for i, metadata in enumerate(all_docs["metadatas"]):
                if (
                    metadata.get("source") == source
                    and metadata.get("mission") == mission
                ):
                    file_doc_ids.append(all_docs["ids"][i])
            return file_doc_ids
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for a text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text, model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_document_id(
        self, file_path: Path, metadata: Dict[str, Any]
    ) -> str:
        """Generate a stable document ID based on file path and chunk position.

        Format: ``{mission}_{source}_chunk_{chunk_index:04d}``

        Args:
            file_path: Path to source file.
            metadata: Chunk metadata (must contain chunk_index).

        Returns:
            A unique, deterministic document ID string.
        """
        mission = metadata.get("mission", "unknown")
        source = file_path.stem
        chunk_index = metadata.get("chunk_index", 0)
        return f"{mission}_{source}_chunk_{chunk_index:04d}"

    def process_text_file(
        self, file_path: Path
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a plain text file with enhanced metadata extraction.

        Args:
            file_path: Path to text file.

        Returns:
            List of (text, metadata) tuples.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return []

            metadata = {
                "source": file_path.stem,
                "file_path": str(file_path),
                "file_type": "text",
                "content_type": "full_text",
                "mission": self.extract_mission_from_path(file_path),
                "data_type": self.extract_data_type_from_path(file_path),
                "document_category": self.extract_document_category_from_filename(
                    file_path.name
                ),
                "file_size": len(content),
                "processed_timestamp": datetime.now().isoformat(),
            }

            return self.chunk_text(content, metadata)
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Metadata extraction helpers
    # ------------------------------------------------------------------ #

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Extract mission name from file path."""
        path_str = str(file_path).lower()
        if "apollo11" in path_str or "apollo_11" in path_str:
            return "apollo_11"
        elif "apollo13" in path_str or "apollo_13" in path_str:
            return "apollo_13"
        elif "challenger" in path_str:
            return "challenger"
        return "unknown"

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Extract data type from file path."""
        path_str = str(file_path).lower()
        if "transcript" in path_str:
            return "transcript"
        elif "textract" in path_str:
            return "textract_extracted"
        elif "audio" in path_str:
            return "audio_transcript"
        elif "flight_plan" in path_str:
            return "flight_plan"
        return "document"

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Extract document category from filename for better organization."""
        fn = filename.lower()
        if "pao" in fn:
            return "public_affairs_officer"
        elif "cm" in fn:
            return "command_module"
        elif "tec" in fn:
            return "technical"
        elif "flight_plan" in fn:
            return "flight_plan"
        elif "mission_audio" in fn:
            return "mission_audio"
        elif "ntrs" in fn:
            return "nasa_archive"
        elif "19900066485" in fn:
            return "technical_report"
        elif "19710015566" in fn:
            return "mission_report"
        elif "full_text" in fn:
            return "complete_document"
        return "general_document"

    # ------------------------------------------------------------------ #
    # File scanning
    # ------------------------------------------------------------------ #

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """Scan data directories for text files only.

        Args:
            base_path: Base directory path.

        Returns:
            List of text file paths to process.
        """
        base_path = Path(base_path)
        files_to_process: List[Path] = []

        data_dirs = ["apollo11", "apollo13", "challenger"]

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                text_files = list(dir_path.glob("**/*.txt"))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")

        # Filter unwanted files
        filtered = [
            fp
            for fp in files_to_process
            if not fp.name.startswith(".")
            and "summary" not in fp.name.lower()
            and fp.suffix.lower() == ".txt"
        ]

        logger.info(f"Total text files to process: {len(filtered)}")

        mission_counts: Dict[str, int] = {}
        for fp in filtered:
            m = self.extract_mission_from_path(fp)
            mission_counts[m] = mission_counts.get(m, 0) + 1
        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")

        return filtered

    # ------------------------------------------------------------------ #
    # Collection operations
    # ------------------------------------------------------------------ #

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """Add documents to ChromaDB collection in batches with update handling.

        Args:
            documents: List of (text, metadata) tuples.
            file_path: Path to the source file.
            batch_size: Number of documents to process in each batch.
            update_mode: 'skip', 'update', or 'replace'.

        Returns:
            Dict with counts of added, updated, and skipped documents.
        """
        if not documents:
            return {"added": 0, "updated": 0, "skipped": 0}

        stats = {"added": 0, "updated": 0, "skipped": 0}

        # Handle replace mode: delete all existing docs from this file first
        if update_mode == "replace":
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(
                    f"Replaced: deleted {len(existing_ids)} existing chunks for {file_path.name}"
                )

        # Process in batches
        for batch_start in range(0, len(documents), batch_size):
            batch = documents[batch_start : batch_start + batch_size]

            batch_ids: List[str] = []
            batch_texts: List[str] = []
            batch_metas: List[Dict[str, Any]] = []

            for text, meta in batch:
                doc_id = self.generate_document_id(file_path, meta)
                exists = self.check_document_exists(doc_id)

                if exists and update_mode == "skip":
                    stats["skipped"] += 1
                    continue
                elif exists and update_mode == "update":
                    if self.update_document(doc_id, text, meta):
                        stats["updated"] += 1
                    continue

                # New document (or replace mode already deleted)
                batch_ids.append(doc_id)
                batch_texts.append(text)
                batch_metas.append(meta)

            if batch_ids:
                try:
                    self.collection.add(
                        ids=batch_ids,
                        documents=batch_texts,
                        metadatas=batch_metas,
                    )
                    stats["added"] += len(batch_ids)
                except Exception as e:
                    logger.error(f"Error adding batch: {e}")

        return stats

    def process_all_text_data(
        self, base_path: str, update_mode: str = "skip"
    ) -> Dict[str, Any]:
        """Process all text files and add to ChromaDB.

        Args:
            base_path: Base directory containing data folders.
            update_mode: 'skip', 'update', or 'replace'.

        Returns:
            Processing statistics.
        """
        stats: Dict[str, Any] = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }

        files = self.scan_text_files_only(base_path)

        for file_path in files:
            mission = self.extract_mission_from_path(file_path)
            if mission not in stats["missions"]:
                stats["missions"][mission] = {
                    "files": 0,
                    "chunks": 0,
                    "added": 0,
                    "updated": 0,
                    "skipped": 0,
                }

            try:
                documents = self.process_text_file(file_path)
                if not documents:
                    logger.warning(f"No chunks produced for {file_path}")
                    continue

                result = self.add_documents_to_collection(
                    documents, file_path, update_mode=update_mode
                )

                stats["files_processed"] += 1
                stats["documents_added"] += result["added"]
                stats["documents_updated"] += result["updated"]
                stats["documents_skipped"] += result["skipped"]
                stats["total_chunks"] += len(documents)

                stats["missions"][mission]["files"] += 1
                stats["missions"][mission]["chunks"] += len(documents)
                stats["missions"][mission]["added"] += result["added"]
                stats["missions"][mission]["updated"] += result["updated"]
                stats["missions"][mission]["skipped"] += result["skipped"]

                logger.info(
                    f"Processed {file_path.name}: {len(documents)} chunks "
                    f"(added={result['added']}, updated={result['updated']}, "
                    f"skipped={result['skipped']})"
                )
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Error processing {file_path}: {e}")

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        """Get basic information about the ChromaDB collection.

        Returns:
            Dict with collection_name, document_count, and persist_directory.
        """
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.chroma_persist_directory,
        }

    def query_collection(
        self, query_text: str, n_results: int = 5
    ) -> Dict[str, Any]:
        """Query the collection for testing.

        Args:
            query_text: Query text.
            n_results: Number of results to return.

        Returns:
            Query results dict.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text], n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {"error": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the collection.

        Returns:
            Dict with total_documents, missions breakdown, data_types,
            document_categories, and file_types.
        """
        try:
            all_docs = self.collection.get()
            if not all_docs["metadatas"]:
                return {"error": "No documents in collection"}

            stats: Dict[str, Any] = {
                "total_documents": len(all_docs["metadatas"]),
                "missions": {},
                "data_types": {},
                "document_categories": {},
                "file_types": {},
            }

            for metadata in all_docs["metadatas"]:
                for field in ("missions", "data_types", "document_categories", "file_types"):
                    key_name = field.rstrip("s") if field != "missions" else "mission"
                    if field == "data_types":
                        key_name = "data_type"
                    elif field == "document_categories":
                        key_name = "document_category"
                    elif field == "file_types":
                        key_name = "file_type"
                    val = metadata.get(key_name, "unknown")
                    stats[field][val] = stats[field].get(val, 0) + 1

            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="ChromaDB Embedding Pipeline for NASA Data"
    )
    parser.add_argument(
        "--data-path", default=".", help="Path to data directories"
    )
    parser.add_argument(
        "--openai-key", required=True, help="OpenAI API key"
    )
    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db_openai",
        help="ChromaDB persist directory",
    )
    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="Collection name",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Text chunk size"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100, help="Chunk overlap size"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing"
    )
    parser.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="skip",
        help="How to handle existing documents: skip, update, or replace",
    )
    parser.add_argument("--test-query", help="Test query after processing")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show collection statistics",
    )
    parser.add_argument(
        "--delete-source",
        help="Delete all documents from a specific source pattern",
    )

    args = parser.parse_args()

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Handle delete source operation
    if args.delete_source:
        deleted_count = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted_count} documents matching: {args.delete_source}")
        return

    # If stats only, show collection statistics and exit
    if args.stats_only:
        logger.info("Collection Statistics:")
        stats = pipeline.get_collection_stats()
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return

    # Process all data
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()

    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)

    processing_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    logger.info("\nMission breakdown:")
    for mission, mission_stats in stats["missions"].items():
        logger.info(
            f"  {mission}: {mission_stats['files']} files, "
            f"{mission_stats['chunks']} chunks"
        )
        logger.info(
            f"    Added: {mission_stats['added']}, "
            f"Updated: {mission_stats['updated']}, "
            f"Skipped: {mission_stats['skipped']}"
        )

    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(
        f"Total documents in collection: {collection_info.get('document_count', 'N/A')}"
    )

    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and "documents" in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results["documents"][0][:3]):
                logger.info(f"Result {i + 1}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
