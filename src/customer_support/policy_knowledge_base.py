"""Load and chunk the local policy document for a later RAG pipeline."""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "policies.md"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

MARKDOWN_HEADERS = [
    ("#", "document"),
    ("##", "section"),
    ("###", "subsection"),
]


def load_policy_documents(policy_path: str | Path = POLICY_PATH) -> list[Document]:
    """Load the policy Markdown file as LangChain documents."""

    return TextLoader(str(policy_path), encoding="utf-8").load()


def split_policy_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents by Markdown headings, then bound oversized sections."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=True,
    )
    section_documents: list[Document] = []

    for document in documents:
        source = document.metadata.get("source")
        split_sections = markdown_splitter.split_text(document.page_content)
        for section in split_sections:
            section.metadata = {**document.metadata, **section.metadata}
            if source is not None:
                section.metadata["source"] = _source_label(str(source))
            section_documents.append(section)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return size_splitter.split_documents(section_documents)


def load_policy_chunks(
    policy_path: str | Path = POLICY_PATH,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Load and chunk ``policies.md`` while retaining source metadata."""

    documents = load_policy_documents(policy_path)
    return split_policy_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _source_label(source: str) -> str:
    """Use a stable project-relative source where possible."""

    source_path = Path(source)
    try:
        return source_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return source_path.name
