"""Phase 1 + 2 of the loop: ingest -> chunk.

Turn a PDF on disk into a list of overlapping text passages, each of which
remembers the page it came from. That page number is what lets every answer
cite its source later, so we're careful to carry it all the way through.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def load_and_chunk(pdf_path: str) -> list[Document]:
    """Read a PDF and split it into page-tagged chunks ready for embedding."""

    # PyPDFLoader yields ONE Document per page, with the 0-indexed page number
    # in metadata["page"]. Loading per-page is what makes citations possible.
    pages = PyPDFLoader(pdf_path).load()

    # RecursiveCharacterTextSplitter tries to break on natural boundaries first
    # (paragraphs, then lines, then words) instead of mid-word. The overlap
    # means a fact sitting on a chunk boundary still appears whole in one chunk.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Splitting preserves each source page's metadata onto its child chunks, so
    # every chunk still knows which page it belongs to.
    chunks = splitter.split_documents(pages)

    # Normalize to a human-friendly 1-indexed page number under a stable key,
    # so the rest of the app never has to think about 0-indexing again.
    for chunk in chunks:
        zero_indexed = chunk.metadata.get("page", 0)
        chunk.metadata["page_number"] = int(zero_indexed) + 1

    return chunks
