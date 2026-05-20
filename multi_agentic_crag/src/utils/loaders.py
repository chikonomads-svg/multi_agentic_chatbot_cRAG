"""Document loaders and preprocessing utilities."""
from typing import List
from pathlib import Path
import logging
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


LOG = logging.getLogger(__name__)


def load_documents_from_dir(path: str, extensions: List[str] = None) -> List[Document]:
    """Load supported documents (.pdf, .docx, .txt) from a directory and return LangChain Documents.

    Uses RecursiveCharacterTextSplitter with chunk_size=1000 and chunk_overlap=200 per spec.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    exts = set(e.lower() for e in (extensions or [".pdf", ".docx", ".txt"]))
    files = [str(f) for f in p.rglob("*") if f.suffix.lower() in exts]
    docs = []
    for f in files:
        if f.lower().endswith(".pdf"):
            loader = PyPDFLoader(f)
        elif f.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(f)
        else:
            loader = TextLoader(f, encoding="utf-8")
        raw = loader.load()
        docs.extend(raw)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = []
    for d in docs:
        chunks = splitter.split_documents([d])
        split_docs.extend(chunks)

    return split_docs
