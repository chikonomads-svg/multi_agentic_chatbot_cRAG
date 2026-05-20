import tempfile
from src.processor import load_documents, chunk_documents
try:
    from langchain.schema import Document
except Exception:
    # Fallback to the local Document class when langchain.schema is not present
    from src.processor import Document


def test_chunking_txt(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("This is a test. " * 200)
    docs = load_documents([str(p)])
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)
