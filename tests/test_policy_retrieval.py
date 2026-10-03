"""Tests for the policy Chroma index and semantic retrieval interface."""

import hashlib
import re

import pytest
from langchain_core.embeddings import Embeddings

from customer_support.policy_knowledge_base import load_policy_chunks
from customer_support.policy_retrieval import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    create_policy_vector_store,
    index_policy_knowledge_base,
    search_policy_chunks,
)


class LocalSemanticEmbeddings(Embeddings):
    """Small deterministic test embedder; it makes no external API calls."""

    dimensions = 512
    synonyms = {
        "discounted": "sale",
        "discount": "sale",
        "markdown": "sale",
        "items": "item",
        "returns": "return",
        "returned": "return",
        "returning": "return",
        "days": "day",
        "long": "calendar",
        "how": "calendar",
        "period": "window",
        "customers": "standard",
        "have": "allowed",
    }
    stop_words = {"a", "an", "do", "for", "in", "of", "the", "to"}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            if token in self.stop_words:
                continue
            normalized = self.synonyms.get(token, token)
            slot = int.from_bytes(
                hashlib.sha256(normalized.encode("utf-8")).digest()[:4], "big"
            ) % self.dimensions
            vector[slot] += 1.0
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]


@pytest.fixture
def policy_vector_store(tmp_path):
    store = create_policy_vector_store(
        embedding_function=LocalSemanticEmbeddings(),
        persist_directory=tmp_path / "chroma",
    )
    yield store
    store.delete_collection()


@pytest.fixture
def indexed_policy_vector_store(policy_vector_store):
    return index_policy_knowledge_base(vector_store=policy_vector_store)


def test_vector_store_is_created(policy_vector_store) -> None:
    assert policy_vector_store._collection.name == COLLECTION_NAME
    assert policy_vector_store.get()["ids"] == []


def test_openai_embedding_model_is_configured() -> None:
    assert EMBEDDING_MODEL == "text-embedding-3-small"


def test_policy_chunks_are_indexed(indexed_policy_vector_store) -> None:
    indexed = indexed_policy_vector_store.get()

    assert len(indexed["ids"]) == len(load_policy_chunks()) == 20
    assert all(indexed["documents"])


def test_indexing_preserves_policy_metadata(indexed_policy_vector_store) -> None:
    indexed = indexed_policy_vector_store.get()

    final_sale = next(
        metadata
        for metadata in indexed["metadatas"]
        if metadata.get("subsection") == "Final-Sale Items"
    )
    assert final_sale["source"] == "policies.md"
    assert final_sale["document"] == "Store Policies"
    assert final_sale["section"] == "Sale Items"


def test_semantic_search_returns_documents_with_metadata(
    indexed_policy_vector_store,
) -> None:
    results = search_policy_chunks(
        "Can I return a discounted item?",
        vector_store=indexed_policy_vector_store,
    )

    assert results
    assert all(result.page_content.strip() for result in results)
    assert all("source" in result.metadata for result in results)


def test_sale_item_query_retrieves_final_sale_policy(
    indexed_policy_vector_store,
) -> None:
    results = search_policy_chunks(
        "Can I return a discounted or sale item?",
        vector_store=indexed_policy_vector_store,
        k=3,
    )

    assert any(
        result.metadata.get("subsection") == "Final-Sale Items" for result in results
    )


def test_return_duration_query_retrieves_return_window_policy(
    indexed_policy_vector_store,
) -> None:
    results = search_policy_chunks(
        "How long do customers have to return an item?",
        vector_store=indexed_policy_vector_store,
        k=3,
    )

    assert any(
        result.metadata.get("subsection") == "Return Window" for result in results
    )


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_queries(query: str, policy_vector_store) -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        search_policy_chunks(query, vector_store=policy_vector_store)
