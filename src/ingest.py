"""Load and chunk medical reference documents (.txt, .md, .pdf) for embedding."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(docs_dir: Path) -> list[Document]:
    """Load every supported file under docs_dir (non-recursive) into LangChain Documents."""
    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        else:
            documents.extend(TextLoader(str(path), encoding="utf-8").load())

    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
