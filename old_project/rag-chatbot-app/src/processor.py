"""Document processing: loaders and chunking.

This module tries to import langchain loaders and text splitter but
provides lightweight fallbacks so the package can be used in
environments where langchain isn't fully available (e.g. in CI tests).
"""
from typing import List
import os
import logging

LOG = logging.getLogger(__name__)

try:
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False
    LOG.warning("langchain not fully available - using lightweight fallbacks for loaders/splitter")

    # Lightweight Document replacement compatible with langchain.schema.Document
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class TextLoader:
        def __init__(self, path, encoding="utf-8"):
            self.path = path
            self.encoding = encoding

        def load(self):
            with open(self.path, "r", encoding=self.encoding, errors="ignore") as fh:
                return [Document(fh.read(), metadata={"source": self.path})]

    class Docx2txtLoader(TextLoader):
        def load(self):
            try:
                import docx2txt

                text = docx2txt.process(self.path)
                return [Document(text, metadata={"source": self.path})]
            except Exception:
                return super().load()

    class PyPDFLoader(TextLoader):
        def load(self):
            try:
                from pypdf import PdfReader

                reader = PdfReader(self.path)
                pages = []
                for p in reader.pages:
                    pages.append(Document(p.extract_text() or "", metadata={"source": self.path}))
                return pages
            except Exception:
                return super().load()

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> List[str]:
            if not text:
                return []
            chunks = []
            i = 0
            step = self.chunk_size - self.chunk_overlap
            while i < len(text):
                chunks.append(text[i : i + self.chunk_size])
                i += step
            return chunks


def load_documents(filepaths: List[str]) -> List[Document]:
    """Load file paths into Documents with filename metadata.

    Supports .pdf, .docx, .txt, .md. Uses langchain loaders when
    available, otherwise falls back to simple file readers.
    """
    docs: List[Document] = []
    for path in filepaths:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(path)
                loaded = loader.load()
            elif ext == ".docx":
                loader = Docx2txtLoader(path)
                loaded = loader.load()
            elif ext in (".txt", ".md"):
                loader = TextLoader(path, encoding="utf-8")
                loaded = loader.load()
            else:
                LOG.info("Skipping unsupported file type: %s", path)
                continue
        except Exception as e:
            LOG.exception("Failed to load %s: %s", path, e)
            continue

        filename = os.path.basename(path)
        for d in loaded:
            meta = dict(getattr(d, "metadata", {}) or {})
            meta["source_filename"] = filename
            docs.append(Document(page_content=d.page_content, metadata=meta))

    return docs


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into overlapping chunks preserving metadata.

    If langchain splitter exists it will be used, otherwise a simple
    fallback splitter is provided.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    new_docs: List[Document] = []
    for d in documents:
        chunks = splitter.split_text(d.page_content)
        for i, c in enumerate(chunks):
            meta = dict(getattr(d, "metadata", {}) or {})
            meta.setdefault("chunk", i)
            new_docs.append(Document(page_content=c, metadata=meta))

    return new_docs
