"""OpenAI-embedded Chroma index and semantic search for policy chunks."""

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from customer_support.policy_knowledge_base import PROJECT_ROOT, load_policy_chunks


EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "customer_support_policies"
DEFAULT_PERSIST_DIRECTORY = PROJECT_ROOT / ".chroma" / "policy-index"
DEFAULT_RESULTS = 4


def create_policy_vector_store(
    *,
    embedding_function: Embeddings | None = None,
    persist_directory: str | Path | None = DEFAULT_PERSIST_DIRECTORY,
) -> Chroma:
    """Create or open the local Chroma collection used for policy chunks.

    If no embedding function is injected, OpenAI credentials are read by
    ``OpenAIEmbeddings`` from the ``OPENAI_API_KEY`` environment variable.
    """

    if embedding_function is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "Set OPENAI_API_KEY in the environment before using OpenAI embeddings."
            )
        embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    persist_path = str(persist_directory) if persist_directory is not None else None
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=persist_path,
    )


def index_policy_knowledge_base(
    *,
    vector_store: Chroma | None = None,
    embedding_function: Embeddings | None = None,
    persist_directory: str | Path | None = DEFAULT_PERSIST_DIRECTORY,
    chunks: Sequence[Document] | None = None,
) -> Chroma:
    """Embed and index policy chunks, preserving their existing metadata."""

    store = vector_store or create_policy_vector_store(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
    policy_chunks = list(chunks) if chunks is not None else load_policy_chunks()
    if not policy_chunks:
        raise ValueError("No policy chunks were provided for indexing.")

    store.add_documents(
        documents=policy_chunks,
        ids=[_document_id(document) for document in policy_chunks],
    )
    return store


def search_policy_chunks(
    query: str,
    *,
    vector_store: Chroma | None = None,
    embedding_function: Embeddings | None = None,
    persist_directory: str | Path | None = DEFAULT_PERSIST_DIRECTORY,
    k: int = DEFAULT_RESULTS,
) -> list[Document]:
    """Return the most relevant policy chunks for a natural-language query."""

    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    store = vector_store or create_policy_vector_store(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
    return store.similarity_search(query, k=k)


def _document_id(document: Document) -> str:
    """Return a stable ID so indexing identical chunks is repeatable."""

    payload = json.dumps(
        {"content": document.page_content, "metadata": document.metadata},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
