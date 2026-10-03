"""Tests for policy document loading and chunking."""

from langchain_core.documents import Document

from customer_support.policy_knowledge_base import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    POLICY_PATH,
    load_policy_chunks,
    load_policy_documents,
    split_policy_documents,
)


def test_policies_markdown_can_be_loaded() -> None:
    documents = load_policy_documents()

    assert POLICY_PATH.is_file()
    assert len(documents) == 1
    assert "# Store Policies" in documents[0].page_content
    assert documents[0].metadata["source"].endswith("policies.md")


def test_policy_chunks_are_produced_and_contain_content() -> None:
    chunks = load_policy_chunks()

    assert chunks
    assert all(chunk.page_content.strip() for chunk in chunks)


def test_policy_chunks_preserve_source_and_heading_metadata() -> None:
    chunks = load_policy_chunks()
    sale_item_chunk = next(
        chunk
        for chunk in chunks
        if chunk.metadata.get("section") == "Sale Items"
        and "Items marked" in chunk.page_content
    )

    assert all(chunk.metadata["source"] == "policies.md" for chunk in chunks)
    assert sale_item_chunk.metadata["document"] == "Store Policies"
    assert sale_item_chunk.metadata["section"] == "Sale Items"
    assert sale_item_chunk.metadata["subsection"] == "Final-Sale Items"


def test_oversized_section_is_split_recursively() -> None:
    oversized_document = Document(
        page_content="Policy detail. " * 150,
        metadata={"source": "example.md", "document": "Example", "section": "Long"},
    )

    chunks = split_policy_documents([oversized_document])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= DEFAULT_CHUNK_SIZE for chunk in chunks)
    assert all(chunk.metadata["source"] == "example.md" for chunk in chunks)
    assert all(chunk.metadata["section"] == "Long" for chunk in chunks)
    assert DEFAULT_CHUNK_OVERLAP == 100
