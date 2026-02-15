"""RAG Client for NASA Mission Intelligence System.

Handles ChromaDB backend discovery, initialization, semantic retrieval,
and context formatting for the LLM pipeline.
"""

import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory.

    Scans the current directory tree for directories whose names contain
    'chroma' and attempts to connect to each one, listing all collections.

    Returns:
        Dictionary mapping unique keys to backend info dicts containing
        'directory', 'collection_name', 'display_name', and 'doc_count'.
    """
    backends = {}
    current_dir = Path(".")

    # Find directories matching the chroma naming pattern
    chroma_dirs = [
        d for d in current_dir.rglob("*")
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collections = client.list_collections()

            for col in collections:
                key = f"{chroma_dir.name}_{col.name}"
                try:
                    doc_count = col.count()
                except Exception:
                    doc_count = 0

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": col.name,
                    "display_name": f"{col.name} ({chroma_dir.name}) [{doc_count} docs]",
                    "doc_count": doc_count,
                }

        except Exception as e:
            fallback_key = f"{chroma_dir.name}_error"
            backends[fallback_key] = {
                "directory": str(chroma_dir),
                "collection_name": "unknown",
                "display_name": f"{chroma_dir.name} (error: {str(e)[:80]})",
                "doc_count": 0,
            }

    return backends


def initialize_rag_system(
    chroma_dir: str, collection_name: str
) -> Tuple:
    """Initialize the RAG system by connecting to a ChromaDB collection.

    Args:
        chroma_dir: Path to the ChromaDB persistence directory.
        collection_name: Name of the collection to open.

    Returns:
        Tuple of (collection, success_bool, error_message_or_empty_string).
    """
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection(name=collection_name)
        return collection, True, ""
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional mission filtering.

    Args:
        collection: ChromaDB collection object.
        query: User question to search for.
        n_results: Number of top-k results to return.
        mission_filter: If provided and not 'all', restrict results to this mission.

    Returns:
        Query results dict or None on failure.
    """
    where_filter = None

    if mission_filter and mission_filter.lower() not in ("all", "none", ""):
        where_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into a structured context string.

    Deduplicates documents, adds clear separators and source attributions.

    Args:
        documents: List of document text strings.
        metadatas: Corresponding list of metadata dicts.

    Returns:
        Formatted context string ready for the LLM.
    """
    if not documents:
        return ""

    context_parts = ["=== Retrieved Context ===\n"]
    seen_texts = set()

    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        # Deduplicate by checking normalized content
        doc_key = doc.strip()[:200]
        if doc_key in seen_texts:
            continue
        seen_texts.add(doc_key)

        mission = meta.get("mission", "Unknown")
        mission = mission.replace("_", " ").title()

        source = meta.get("source", "Unknown source")

        category = meta.get("document_category", "General")
        category = category.replace("_", " ").title()

        header = f"--- Source {idx}: {mission} | {category} | {source} ---"
        context_parts.append(header)

        # Truncate very long chunks to keep context manageable
        if len(doc) > 2000:
            context_parts.append(doc[:2000] + " [truncated]")
        else:
            context_parts.append(doc)

    return "\n".join(context_parts)
