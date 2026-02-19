from typing import List, Optional, Tuple
from pathlib import Path
import os
import pickle
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import fitz  # PyMuPDF for handling PDFs
from .utils.env import load_env

# Try to import langchain-related classes or fallback components.
HAS_LANGCHAIN = True
try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_openai import AzureOpenAIEmbeddings
except Exception:
    HAS_LANGCHAIN = False
    logging.getLogger(__name__).warning(
        "LangChain/FAISS not available. Vector indexing requires LangChain."
    )

LOG = logging.getLogger(__name__)

# Ensure environment variables from .env loaded.
load_env()

# Function to ensure directory exists
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# Function to read PDF and split into chunks
def split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits a PDF into text chunks."""
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text("text") for page in doc)
    if not full_text:
        return []
    step = max(1, chunk_size - chunk_overlap)
    return [full_text[i:i + chunk_size] for i in range(0, len(full_text), step)]

# Function to process a file and return documents
def process_file(file_path: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[Document], List[str]]:
    """Processes a single file and splits it into text chunks."""
    chunks = split_pdf(file_path, chunk_size, chunk_overlap)
    return [Document(page_content=chunk, metadata={"source": source}) for chunk in chunks], [str(hash(chunk)) for chunk in chunks]

# Function to create a FAISS vector database
def create_and_persist_faiss(docs: List[Document], index_path: str, embedding_model: str = "text-embedding-3-small") -> FAISS:
    """Create and persist a FAISS vector store."""
    ensure_dir(index_path)
    if not HAS_LANGCHAIN:
        raise RuntimeError("LangChain/FAISS support is required for vector DB creation.")
    try:
        model_to_use = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or embedding_model
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        emb = AzureOpenAIEmbeddings(deployment=model_to_use, azure_endpoint=azure_endpoint)
        vector_store = FAISS.from_documents(docs, emb)
        vector_store.save_local(index_path)
        return vector_store
    except Exception as exc:
        LOG.exception("Error creating FAISS vector store: %s", exc)
        raise

# Function to load FAISS vector store
def load_faiss(index_path: str, embedding_model: str = "text-embedding-3-small") -> Optional[FAISS]:
    """Load FAISS vector store from disk."""
    if not os.path.isdir(index_path):
        LOG.error("Index path does not exist: %s", index_path)
        return None
    try:
        model_to_use = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or embedding_model
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        emb = AzureOpenAIEmbeddings(deployment=model_to_use, azure_endpoint=azure_endpoint)
        return FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
    except Exception as exc:
        LOG.exception("Failed to load FAISS index at %s: %s", index_path, exc)
        return None

# Main function to process files in a folder
def process_folder(folder_path: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Optional[FAISS]:
    """Processes all PDF files in a folder into a FAISS vector store."""
    futures = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for root, _, files in os.walk(folder_path):
            for file in (f for f in files if f.endswith(".pdf")):
                file_path = os.path.join(root, file)
                futures.append(
                    executor.submit(process_file, file_path, source, chunk_size, chunk_overlap)
                )
    try:
        vector_store = None
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Documents"):
            docs, _ = future.result()
            if docs:
                vector_store = vector_store or create_and_persist_faiss(docs, "C:\\Users\\Komal\\OneDrive\\Desktop\\Vibe Coding\\multi_agentic_crag\\data")
        return vector_store
    except Exception as exc:
        LOG.exception("Failed to process folder: %s", exc)
        return None