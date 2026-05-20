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

# Try to import langchain-related classes; provide lightweight fallbacks so
# the rest of the app can run (useful for tests and constrained envs).
HAS_LANGCHAIN = True
try:
    # langchain v1+ split schema and vectorstores across packages in some
    # distributions; attempt common import locations used in this project.
    try:
        from langchain_core.documents import Document  # preferred in this repo
    except Exception:
        from langchain.schema import Document

    try:
        from langchain_community.vectorstores import FAISS
    except Exception:
        from langchain.vectorstores import FAISS

    # Embeddings client (Azure/OpenAI mapping used later)
    try:
        from langchain_openai import AzureOpenAIEmbeddings
    except Exception:
        # Some environments will use OpenAIEmbeddings - handle absence later
        AzureOpenAIEmbeddings = None
except Exception:
    HAS_LANGCHAIN = False
    logging.getLogger(__name__).warning(
        "langchain or its submodules are not available - FAISS integration requires langchain"
    )

    class Document:  # lightweight fallback used by other modules
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    FAISS = None
    AzureOpenAIEmbeddings = None

    class FallbackIndex:
        """Simple in-memory fallback index used when langchain/FAISS are missing.

        It implements similarity_search_with_score so callers can continue to
        function in tests or constrained environments.
        """
        def __init__(self, docs: List[Document]):
            self.docs = docs

        def similarity_search_with_score(self, query: str, k: int = 5):
            q_tokens = set(w.strip(".,!?;:\"'()[]{}") .lower() for w in query.split())
            results = []
            for d in self.docs:
                d_tokens = set(w.strip(".,!?;:\"'()[]{}") .lower() for w in d.page_content.split())
                overlap = len(q_tokens.intersection(d_tokens))
                results.append((d, float(-overlap)))
            results.sort(key=lambda x: x[1])
            return results[:k]

    # Export fallback under the FAISS name so code importing FAISS doesn't fail
    # (it will be None but search_faiss/loader functions will handle fallback).

# Ensure FallbackIndex exists even when langchain is available so the
# rest of the code can reliably construct a fallback index from persisted
# docs/index files. Some earlier runs raised NameError because FallbackIndex
# was only defined when langchain imports failed.
if "FallbackIndex" not in globals():
    class FallbackIndex:
        """Simple in-memory fallback index used when langchain/FAISS are missing.

        It implements similarity_search_with_score so callers can continue to
        function in tests or constrained environments.
        """
        def __init__(self, docs: List[Document]):
            self.docs = docs

        def similarity_search_with_score(self, query: str, k: int = 5):
            q_tokens = set(w.strip(".,!?;:\"'()[]{}") .lower() for w in query.split())
            results = []
            for d in self.docs:
                d_tokens = set(w.strip(".,!?;:\"'()[]{}") .lower() for w in d.page_content.split())
                overlap = len(q_tokens.intersection(d_tokens))
                results.append((d, float(-overlap)))
            results.sort(key=lambda x: x[1])
            return results[:k]

LOG = logging.getLogger(__name__)

# Ensure environment variables from .env (including Azure mappings) are loaded
load_env()

# Function to ensure directory exists
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# Function to read PDF and split into chunks
def split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits the PDF into text chunks using a character-based window with overlap.

    Defaults chosen to produce reasonably-sized chunks for embedding (1000 chars)
    with a 200-character overlap to preserve context across boundaries.
    """
    doc = fitz.open(pdf_path)  # Open the PDF
    text_chunks: List[str] = []
    full_text = ""

    # Extract text from each page
    for page in doc:
        full_text += page.get_text("text")

    if not full_text:
        return []

    # Ensure sensible parameters
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        chunk_overlap = 0

    step = max(1, chunk_size - chunk_overlap)

    # Slide a window across the full text producing overlapping chunks
    for i in range(0, len(full_text), step):
        text_chunks.append(full_text[i : i + chunk_size])

    return text_chunks

# Function to process file and return documents and their UUIDs
def process_file(file_path: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> Tuple[List[Document], List[str]]:
    """Processes a single file: split it once and return documents with UUIDs.

    Important: this function splits the file a single time (no repeated work per
    chunk) so callers should submit one future per file rather than one per chunk.
    """
    file_chunks = split_pdf(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents: List[Document] = []
    document_uuids: List[str] = []

    for chunk in file_chunks:
        doc = Document(page_content=chunk, metadata={"source": source})
        documents.append(doc)
        document_uuids.append(str(hash(chunk)))  # Generate unique IDs for each chunk

    return documents, document_uuids

# Function to create and persist FAISS index
def create_and_persist_faiss(docs: List[Document], index_path: str, embedding_model: str = "text-embedding-3-small") -> FAISS:
    """Creates and persists the FAISS vector store."""
    ensure_dir(index_path)
    # If langchain/FAISS aren't available, persist a simple JSON/pickle
    # fallback so the rest of the app can continue to function.
    if not HAS_LANGCHAIN or FAISS is None:
        LOG.warning("LangChain/FAISS not available - creating fallback persisted DB at %s", index_path)
        try:
            docs_out = [{"page_content": d.page_content, "metadata": getattr(d, "metadata", {}) or {}} for d in docs]
            docs_file = os.path.join(index_path, "docs.json")
            with open(docs_file, "w", encoding="utf-8") as fh:
                json.dump(docs_out, fh, ensure_ascii=False)
        except Exception:
            LOG.exception("Failed to write docs.json for fallback index at %s", index_path)

        try:
            pkl_path = os.path.join(index_path, "index.pkl")
            mapping = {i: docs[i].page_content for i in range(len(docs))}
            with open(pkl_path, "wb") as pf:
                pickle.dump(mapping, pf)
        except Exception:
            LOG.exception("Failed to write index.pkl for fallback index at %s", index_path)

        return FallbackIndex(docs)

    # Use Azure/OpenAI embeddings when langchain is present. Prefer AzureOpenAIEmbeddings
    # if available, otherwise attempt to use langchain's OpenAIEmbeddings.
    try:
        model_to_use = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT") or embedding_model
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")

        if AzureOpenAIEmbeddings is not None:
            if not azure_endpoint:
                raise RuntimeError("AZURE_OPENAI_ENDPOINT is required when using Azure OpenAI embeddings")
            emb = AzureOpenAIEmbeddings(deployment=model_to_use, azure_endpoint=azure_endpoint)
        else:
            # Try to use OpenAIEmbeddings if Azure client isn't available
            try:
                from langchain.embeddings import OpenAIEmbeddings
                emb = OpenAIEmbeddings(model=model_to_use)
            except Exception as ie:
                LOG.exception("No embeddings client available: %s", ie)
                raise RuntimeError("No embeddings client available; install langchain_openai or configure OpenAIEmbeddings")

        # Create FAISS vector store from documents
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(index_path)

        # Save mapping as a pickle file for later reference
        pkl_path = os.path.join(index_path, "index.pkl")
        mapping = {i: docs[i].page_content for i in range(len(docs))}

        with open(pkl_path, "wb") as pf:
            pickle.dump(mapping, pf)

        return vs
    except Exception as e:
        LOG.exception("Failed to create and persist FAISS vector store: %s", e)
        raise RuntimeError(f"Error creating and persisting FAISS vector store: {str(e)}")

# Function to load FAISS vector store
def load_faiss(index_path: str, embedding_model: str = "text-embedding-3-small") -> Optional[FAISS]:
    """Loads the FAISS vector store from the disk."""
    if not os.path.isdir(index_path):
        LOG.error("Index path does not exist: %s", index_path)
        return None

    # If langchain/FAISS isn't available, attempt to load the simple fallback
    if not HAS_LANGCHAIN or FAISS is None:
        docs_file = os.path.join(index_path, "docs.json")
        if not os.path.exists(docs_file):
            LOG.error("Fallback docs.json not found at %s", docs_file)
            return None
        try:
            with open(docs_file, "r", encoding="utf-8") as f:
                docs_raw = json.load(f)
            docs = [Document(d.get("page_content", ""), metadata=d.get("metadata", {})) for d in docs_raw]
            return FallbackIndex(docs)
        except Exception as e:
            LOG.exception("Failed to load fallback index from %s: %s", index_path, e)
            return None

    try:
        model_to_use = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT") or embedding_model
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")

        if AzureOpenAIEmbeddings is not None:
            if not azure_endpoint:
                raise RuntimeError("AZURE_OPENAI_ENDPOINT is required when using Azure OpenAI embeddings")
            emb = AzureOpenAIEmbeddings(deployment=model_to_use, azure_endpoint=azure_endpoint)
        else:
            # fallback to OpenAIEmbeddings from langchain
            from langchain.embeddings import OpenAIEmbeddings
            emb = OpenAIEmbeddings(model=model_to_use)

        # Attempt to load the FAISS vector store from the disk. Some older/newer
        # langchain/faiss formats may be incompatible; catch failures and provide
        # a graceful fallback to the persisted docs/index files if present.
        try:
            return FAISS.load_local(index_path, emb, allow_dangerous_deserialization=True)
        except Exception as e:
            LOG.exception("FAISS.load_local failed at %s: %s. Attempting fallback to persisted docs/index files.", index_path, e)
            # Try to load a previously saved docs.json or index.pkl as a fallback
            docs_file = os.path.join(index_path, "docs.json")
            if os.path.exists(docs_file):
                try:
                    with open(docs_file, "r", encoding="utf-8") as f:
                        docs_raw = json.load(f)
                    docs = [Document(d.get("page_content", ""), metadata=d.get("metadata", {})) for d in docs_raw]
                    LOG.info("Loaded fallback docs.json from %s", docs_file)
                    return FallbackIndex(docs)
                except Exception:
                    LOG.exception("Failed to load fallback docs.json at %s", docs_file)

            pkl_path = os.path.join(index_path, "index.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, "rb") as pf:
                        mapping = pickle.load(pf)
                    docs = [Document(text, metadata={}) for _, text in sorted(mapping.items())]
                    LOG.info("Loaded fallback index.pkl from %s", pkl_path)
                    return FallbackIndex(docs)
                except Exception:
                    LOG.exception("Failed to load fallback index.pkl at %s", pkl_path)

            LOG.error("No fallback persisted index found at %s; returning None", index_path)
            return None

    except Exception as e:
        LOG.exception("Failed to initialize embeddings or load FAISS at %s: %s", index_path, e)
        return None


def search_faiss(faiss_index: FAISS, query: str, k: int = 5):
    """Run a similarity search against either a real FAISS index or our fallback."""
    if faiss_index is None:
        return []
    try:
        return faiss_index.similarity_search_with_score(query, k=k)
    except Exception:
        # Some FAISS versions expose `similarity_search_with_score` while others
        # may expose differently named methods; for now try the common one and
        # return empty on failure.
        LOG.exception("Failed to run similarity search on provided index")
        return []

# Main method to process all files in a folder
def process_folder(folder_path: str, source: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """Processes all PDF files in a folder and stores them in a vector store.

    This implementation submits one task per file to a ThreadPoolExecutor. Each
    task splits the file once and returns all documents for that file. This
    avoids re-reading/re-splitting a file for every chunk which previously led
    to quadratic work and very slow ingestion.
    """
    vector_store = None
    futures = []

    # Loop through all files in the folder and process them
    with ThreadPoolExecutor(max_workers=2) as executor:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = file.split('.')[-1].lower()
                
                if ext == 'pdf':
                    print(f"Found file: {file_path}")
                    print(f"Splitting large PDF: {file_path}")
                    # Submit one task per file - process_file will split the file once
                    futures.append(
                        executor.submit(process_file, file_path, source, chunk_size, chunk_overlap)
                    )

        # Process each chunk using threads
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Chunks"):
            try:
                chunk_documents, document_uuids = future.result()
            except Exception as e:
                LOG.exception("Failed to retrieve chunk result from future: %s", e)
                # Skip this future and continue with others
                continue

            retry = 0
            # Derive a sensible index path inside the folder being processed so
            # multiple DBs are not written to a hardcoded path.
            derived_index_path = os.path.join(folder_path, "index")

            while retry < 5:
                try:
                    if chunk_documents:
                        # Add documents to the vector store
                        if vector_store is None:
                            LOG.info("Creating vector store at %s for initial chunk...", derived_index_path)
                            vector_store = create_and_persist_faiss(
                                chunk_documents, derived_index_path, embedding_model="text-embedding-3-small"
                            )
                        else:
                            LOG.debug("Adding %d documents to existing vector store", len(chunk_documents))
                            # If the underlying vector store supports ids argument this
                            # will attach our generated uuids to the vectors.
                            try:
                                vector_store.add_documents(chunk_documents, ids=document_uuids)
                            except TypeError:
                                # Some fallback implementations may not accept ids
                                vector_store.add_documents(chunk_documents)
                    break
                except Exception as e:
                    # Handle rate limit-style errors by retrying with backoff.
                    msg = str(e).lower()
                    if "rate limit" in msg or "rate_limit" in msg or "429" in msg:
                        retry += 1
                        wait_time = 60 * retry
                        LOG.warning("Rate limit hit (%s). Retrying in %s seconds... (attempt %d)", e, wait_time, retry)
                        time.sleep(wait_time)
                        continue
                    LOG.exception("Failed to process chunk: %s", e)
                    break
    
    return vector_store
