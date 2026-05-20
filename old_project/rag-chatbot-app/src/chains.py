"""RAG chains: retrieval, grading, and Tavily fallback."""
from typing import List, Optional, Tuple
import os

import logging
try:
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    HAS_LANGCHAIN = True
except Exception:
    # Provide minimal fallbacks for test environments
    HAS_LANGCHAIN = False
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class ChatOpenAI:
        def __init__(self, temperature=0):
            pass

        def call_as_llm(self, prompt):
            class R: pass
            r = R()
            r.content = "[]"
            return r

    class PromptTemplate:
        @staticmethod
        def from_template(t):
            return t

from .vector_store import search_faiss, load_faiss_index, index_exists

import tavily

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


GRADER_PROMPT = PromptTemplate.from_template(
    """
You are a grader. Given a query and a set of document snippets, score each snippet for relevance from 1 to 5.
Respond with a JSON array of scores in the same order as the snippets.
Query: {query}
Snippets: {snippets}
"""
)


def grade_snippets(snippets: List[str], query: str, llm=None) -> List[int]:
    if llm is None:
        llm = ChatOpenAI(temperature=0)
    prompt = GRADER_PROMPT.format(query=query, snippets="\n---\n".join(snippets))
    try:
        resp = llm.call_as_llm(prompt)
        text = getattr(resp, "content", "[]")
        # naive parse: look for JSON array
        import json
        scores = json.loads(text)
        if isinstance(scores, list):
            return [int(s) for s in scores]
    except Exception:
        LOG.exception("Grader LLM failed, falling back to neutral scores")
    # fallback: return 3 for all
    return [3] * len(snippets)


def tavily_search(query: str, tavily_api_key: Optional[str] = None) -> List[dict]:
    # Minimal Tavily client usage - assumes tavily-python is configured
    try:
        if tavily_api_key:
            tavily.api_key = tavily_api_key
        # Some tavily.Client implementations may require different args;
        # attempt to call the no-arg constructor and fall back gracefully.
        client = tavily.Client()
        results = client.search(query=query, limit=5)
        return results
    except Exception:
        LOG.exception("Tavily client unavailable or failed; returning local fallback results")
        # Return a minimal fallback response so callers can continue.
        return [{"title": "fallback", "url": "", "snippet": query}]


def retrieve_answer(query: str, faiss_index_path: str, k: int = 5, tavily_api_key: Optional[str] = None) -> Tuple[str, List[Document]]:
    """Main retrieval logic. Returns (answer_text, source_documents).

    If FAISS has relevant results and grader approves, use them to generate an answer.
    Otherwise use Tavily fallback.
    """
    LOG.debug("retrieve_answer called with query=%s, faiss_index_path=%s, k=%s", query, faiss_index_path, k)
    # Load index if present
    try:
        faiss = load_faiss_index(faiss_index_path)
    except Exception:
        LOG.exception("Failed to load FAISS index")
        faiss = None

    if faiss is None:
        LOG.info("No FAISS index available at %s - using Tavily fallback", faiss_index_path)
        web = tavily_search(query, tavily_api_key)
        return ("Tavily fallback: " + str(web), [])

    hits = search_faiss(faiss, query, k=k)
    LOG.debug("search_faiss returned %s hits", len(hits) if hits is not None else 0)
    if not hits:
        LOG.info("FAISS returned no hits for query=%s; using Tavily fallback", query)
        web = tavily_search(query, tavily_api_key)
        return ("Tavily fallback: " + str(web), [])

    snippets = [h[0].page_content for h in hits]
    LOG.debug("Top snippets: %s", snippets)
    scores = grade_snippets(snippets, query)
    if not scores:
        LOG.warning("Grader returned empty scores for snippets; falling back to neutral scores")
        scores = [3] * len(snippets) if snippets else [3]
    try:
        avg = sum(scores) / len(scores)
    except Exception:
        LOG.exception("Failed to compute average score; using neutral average")
        avg = 3
    LOG.debug("Snippet scores=%s avg=%s", scores, avg)
    if avg < 3:
        web = tavily_search(query, tavily_api_key)
        return ("Tavily fallback: " + str(web), [h[0] for h in hits])

    # Synthesize a complete, conversational answer from the top snippets.
    raw_docs = [h[0] for h in hits]
    top_texts = [d.page_content.strip() for d in raw_docs[:5] if getattr(d, "page_content", "").strip()]

    # Try to use the ChatOpenAI LLM (langchain) to synthesize a concise answer.
    # The prompt explicitly asks for a short, complete paragraph and to avoid
    # reproducing raw chunks verbatim. It requests a final short answer and
    # then a one-line summary of how many sources were used.
    synthesis_prompt = (
        "You are an efficient assistant. Given a user query and several document snippets, write a brief, polished, and self-contained answer in 2-4 sentences. "
        "Do NOT print filenames or raw chunk text. Instead, synthesize the information and cite sources only by number (e.g., [1], [2]). If the evidence is inconclusive, say you are uncertain and suggest next steps.\n\n"
        "Query: {query}\n\nSnippets:\n{snippets}\n\n"
        "Answer (2-4 sentences):"
    )

    answer = None
    try:
        if HAS_LANGCHAIN:
            llm = ChatOpenAI(temperature=0)
            prompt = synthesis_prompt.format(query=query, snippets="\n---\n".join(top_texts))
            resp = llm.call_as_llm(prompt)
            answer = getattr(resp, "content", None)
    except Exception:
        LOG.exception("LLM synthesis failed, falling back to clipped concatenation")

    if not answer or not answer.strip():
        # Fallback: concatenate the top snippets into a clipped answer but
        # ensure it's a complete sentence/paragraph.
        combined = "\n\n".join(top_texts)
        if not combined:
            answer = "No relevant content found in the index."
        else:
            # Ensure it ends with a period.
            answer = combined.strip()
            if not answer.endswith(('.', '!', '?')):
                answer = answer + '.'
            # Clip to reasonable length
            if len(answer) > 1200:
                answer = answer[:1197] + '...'

    # Return sanitized source documents (with filenames removed/replaced).
    # To maintain privacy, do not include any filename or other identifying
    # metadata in the returned sources — only include the chunk text.
    sanitized_sources = [Document(page_content=d.page_content, metadata={}) for d in raw_docs]
    return (answer, sanitized_sources)
