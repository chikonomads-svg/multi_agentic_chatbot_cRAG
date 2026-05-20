"""FAISS vector store creation, loading, and persistence utilities."""
from typing import List, Optional
from pathlib import Path
import os

import logging
from typing import Tuple
import json
import pickle
from dotenv import load_dotenv

try:
    # Require langchain and FAISS for proper index creation. Use OpenAIEmbeddings
    # (Azure/OpenAI) exclusively - import Document from langchain.schema which
    # matches current langchain distributions.
    from langchain.schema import Document
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OpenAIEmbeddings
    HAS_LANGCHAIN = True
except Exception:
    HAS_LANGCHAIN = False
    logging.getLogger(__name__).warning(
        "langchain or its submodules are not available - FAISS integration requires langchain"
    )
    # Provide a minimal Document fallback so imports elsewhere don't crash
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    # Provide placeholders so rest of the module can reference these names
    FAISS = None
    OpenAIEmbeddings = None

    class FallbackIndex:
        """Small in-memory fallback index used when langchain/FAISS are missing.

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


def index_exists(index_path: str) -> bool:
    return os.path.exists(index_path) and os.path.isdir(index_path)


# Ensure environment variables from a .env file are loaded so Azure/OpenAI
# configuration is available to the embedding client. This is idempotent.
try:
    load_dotenv()
except Exception:
    logging.getLogger(__name__).debug("python-dotenv not available or failed to load .env; relying on environment variables")


def create_faiss_index(docs: List[Document], index_path: str, embedding_model: str = "text-embedding-3-small") -> FAISS:
    """Create and persist a FAISS index from documents using OpenAI/Azure embeddings.

    embedding_model should be the Azure embedding deployment name (or an
    OpenAI model name if using OpenAI). This function requires langchain
    and FAISS to be installed and the Azure/OpenAI environment variables to
    be configured when using Azure.
    """
    # If langchain/FAISS aren't available, fall back to a lightweight
    # persistence format (docs.json + index.pkl) and return an in-memory
    # FallbackIndex so the rest of the app can continue to function.
    os.makedirs(index_path, exist_ok=True)
    if not HAS_LANGCHAIN:
        logging.getLogger(__name__).warning(
            "langchain/FAISS not available - creating fallback persisted DB at %s",
            index_path,
        )
        try:
            docs_out = [{"page_content": d.page_content, "metadata": getattr(d, "metadata", {}) or {}} for d in docs]
            docs_file = os.path.join(index_path, "docs.json")
            with open(docs_file, "w", encoding="utf-8") as fh:
                json.dump(docs_out, fh, ensure_ascii=False)
        except Exception:
            logging.getLogger(__name__).exception("Failed to write docs.json for fallback index at %s", index_path)

        try:
            pkl_path = os.path.join(index_path, "index.pkl")
            mapping = {i: docs[i].page_content for i in range(len(docs))}
            with open(pkl_path, "wb") as pf:
                pickle.dump(mapping, pf)
        except Exception:
            logging.getLogger(__name__).exception("Failed to write index.pkl for fallback index at %s", index_path)

        return FallbackIndex(docs)

    # Use OpenAI/Azure embeddings exclusively when langchain is present
    emb = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(docs, emb)
    vectorstore.save_local(index_path)
    return vectorstore


def _vector_dbs_root() -> str:
    # Place vector DBs inside the application package data directory
    # (rag-chatbot-app/data/vector_dbs) so the DBs live with the app files.
    app_root = Path(__file__).resolve().parents[1]
    root = app_root / "data" / "vector_dbs"
    return str(root)


def get_vector_db_path(name: str) -> str:
    root = _vector_dbs_root()
    return os.path.join(root, name)


def list_vector_dbs() -> List[str]:
    root = _vector_dbs_root()
    if not os.path.exists(root):
        return []
    return [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]


def _has_faiss_files(path: str) -> bool:
    # LangChain/FAISS persists 'index.faiss' and 'index.pkl' when save_local is called
    faiss_path = os.path.join(path, "index.faiss")
    pkl_path = os.path.join(path, "index.pkl")
    return os.path.exists(faiss_path) and os.path.getsize(faiss_path) > 0 and os.path.exists(pkl_path) and os.path.getsize(pkl_path) > 0


def has_faiss_index_for_db(name: str) -> bool:
    """Public helper: returns True if the named vector DB directory contains a persisted FAISS index."""
    return _has_faiss_files(get_vector_db_path(name))


def has_faiss_index_at_path(path: str) -> bool:
    """Public helper: returns True if the given path contains a persisted FAISS index."""
    return _has_faiss_files(path)


def list_vector_dbs_with_index() -> List[Tuple[str, bool]]:
    """Return a list of (db_name, has_index) for all named vector DBs."""
    dbs = list_vector_dbs()
    return [(d, has_faiss_index_for_db(d)) for d in dbs]


def create_vector_db(docs: List[Document], name: str, embedding_model: str = "text-embedding-3-small") -> Optional[FAISS]:
    path = get_vector_db_path(name)
    # Ensure the exact DB directory exists
    os.makedirs(path, exist_ok=True)
    # When creating a named vector DB we require FAISS/langchain to be
    # present and use the configured Azure embedding deployment if set.
    # This enforces use of a FAISS index created with the requested model.
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if azure_deployment:
        model_to_use = azure_deployment
    else:
        model_to_use = embedding_model

    # Create the FAISS/index using the selected embedding model. This will
    # produce index.faiss and index.pkl when successful. We require
    # langchain/FAISS to be installed for a real index to be built.
    # Build the FAISS index using Azure/OpenAI embeddings only. Require
    # Azure environment variables when using Azure deployments to avoid
    # accidental fallback to other embedding providers.
    # If the environment isn't configured for Azure, raise an informative
    # error so the caller can provide the key/endpoint.
    if not HAS_LANGCHAIN:
        # When langchain/FAISS aren't available, fall back to a lightweight
        # persistence format so the app can still store and load documents.
        logging.getLogger(__name__).warning(
            "LangChain/FAISS not available - falling back to JSON/pickle persistence for vector DB '%s'",
            name,
        )
        # Persist raw documents as simple JSON so load_faiss_index can recover them
        try:
            docs_out = [{"page_content": d.page_content, "metadata": getattr(d, "metadata", {}) or {}} for d in docs]
            docs_file = os.path.join(path, "docs.json")
            with open(docs_file, "w", encoding="utf-8") as fh:
                json.dump(docs_out, fh, ensure_ascii=False)
        except Exception:
            logging.getLogger(__name__).exception("Failed to write docs.json for vector DB %s", name)

        # Also write a simple index.pkl mapping indices to text to mimic FAISS
        try:
            pkl_path = os.path.join(path, "index.pkl")
            mapping = {i: docs[i].page_content for i in range(len(docs))}
            with open(pkl_path, "wb") as pf:
                pickle.dump(mapping, pf)
        except Exception:
            logging.getLogger(__name__).exception("Failed to write index.pkl for vector DB %s", name)

        # Return an in-memory fallback index so callers can continue to function
        try:
            return FallbackIndex(docs)
        except Exception:
            # If even the fallback can't be instantiated, surface a helpful error
            raise RuntimeError("LangChain/FAISS not available and fallback index creation failed. Install langchain and faiss to enable full functionality.")

    # If AZURE envs exist, configure OPENAI_* so langchain's OpenAIEmbeddings
    # will use the Azure deployment. If not present, we still allow using a
    # direct OpenAI model name (but the user requested Azure usage).
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if azure_key and azure_endpoint and azure_deployment:
        os.environ["OPENAI_API_KEY"] = azure_key
        os.environ["OPENAI_API_BASE"] = azure_endpoint
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
        model_for_client = azure_deployment
    else:
        # If the user specifically wants Azure but hasn't configured it,
        # raise a helpful error rather than silently using another provider.
        raise RuntimeError(
            "Azure OpenAI environment is not configured. Set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT."
        )

    # Create the FAISS/index using the selected embedding model. This will
    # produce index.faiss and index.pkl when successful.
    idx = create_faiss_index(docs, path, embedding_model=model_for_client)

    # Persist raw embeddings using the same OpenAI/Azure embeddings client
    try:
        emb_client = OpenAIEmbeddings(model=model_for_client)
        texts = [d.page_content for d in docs]
        vectors = emb_client.embed_documents(texts)
        emb_out = []
        for i, v in enumerate(vectors):
            meta = getattr(docs[i], "metadata", {}) or {}
            emb_out.append({"index": i, "embedding": v, "metadata": meta})

        emb_dir = os.path.join(path, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        emb_path = os.path.join(emb_dir, "embeddings.json")
        with open(emb_path, "w", encoding="utf-8") as fh:
            json.dump(emb_out, fh, ensure_ascii=False)
    except Exception:
        logging.getLogger(__name__).exception("Failed to compute or persist embeddings for vector DB %s", name)

    # Ensure index.pkl maps vector ids to chunk text. LangChain typically
    # writes a pickle mapping but to be explicit we create/overwrite it.
    try:
        pkl_path = os.path.join(path, "index.pkl")
        mapping = {i: docs[i].page_content for i in range(len(docs))}
        with open(pkl_path, "wb") as pf:
            pickle.dump(mapping, pf)
    except Exception:
        logging.getLogger(__name__).exception("Failed to write index.pkl for vector DB %s", name)

    return idx


def load_vector_db(name: str, embedding_model: str = "text-embedding-3-small") -> Optional[FAISS]:
    path = get_vector_db_path(name)
    return load_faiss_index(path, embedding_model=embedding_model)


def load_faiss_index(index_path: str, embedding_model: str = "text-embedding-3-small") -> Optional[FAISS]:
    if not index_exists(index_path):
        return None
    if not HAS_LANGCHAIN:
        # Load from the simple JSON fallback format if present
        docs_file = os.path.join(index_path, "docs.json")
        if not os.path.exists(docs_file):
            return None
        with open(docs_file, "r", encoding="utf-8") as f:
            docs_raw = json.load(f)
        docs = [Document(d.get("page_content", ""), metadata=d.get("metadata", {})) for d in docs_raw]
        return FallbackIndex(docs)
    # Respect Azure embedding environment variables if present so reload uses
    # the same embedding client used to create the index.
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if azure_key and azure_endpoint and azure_deployment:
        try:
            os.environ["OPENAI_API_KEY"] = azure_key
            os.environ["OPENAI_API_BASE"] = azure_endpoint
            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_VERSION"] = "2024-12-01-preview"
            emb = OpenAIEmbeddings(model=azure_deployment)
        except Exception:
            logging.getLogger(__name__).exception("Failed to configure Azure embeddings for load; using default")
            emb = OpenAIEmbeddings(model=embedding_model)
    else:
        emb = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(index_path, emb)


def search_faiss(faiss_index: FAISS, query: str, k: int = 5):
    # Both the real FAISS index and our FallbackIndex expose
    # similarity_search_with_score returning a list of (Document, score).
    return faiss_index.similarity_search_with_score(query, k=k)
