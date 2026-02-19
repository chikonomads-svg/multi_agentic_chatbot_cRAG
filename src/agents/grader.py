"""Grader Agent: evaluates retrieved chunks for relevance and decides whether to accept or trigger fallback."""
from typing import Dict, Any, List, Tuple
import logging
import os

from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

from ..utils.env import load_env

LOG = logging.getLogger(__name__)


def grade_chunks(state: Dict[str, Any], chunks: List[Tuple[Document, float]]) -> float:
    """Use a small LLM prompt to grade relevance of retrieved chunks.

    Returns a float in [0,1] where higher indicates better relevance.
    Falls back to a simple distance-based heuristic if LLM not configured.
    """
    # Basic fallback: if no chunks return 0
    if not chunks:
        return 0.0

    # Build a short prompt summarizing chunks and question
    question = state.get("user_message", "")
    snippet_texts = "\n---\n".join([f"Source: {getattr(d, 'page_content', str(d))}" for d, _ in chunks[:5]])
    prompt = (
        "You are a relevance grader. Given a user question and extracted source snippets,\n"
        "score how well the snippets address the question from 0 (not relevant) to 1 (highly relevant).\n"
        f"Question: {question}\nSources:\n{snippet_texts}\n\nProvide only a numeric score between 0 and 1."
    )

    # If there's no OpenAI key, fallback to heuristic: invert scores if numeric
    if not os.getenv("OPENAI_API_KEY"):
        LOG.debug("No OPENAI_API_KEY set; using distance heuristic for grading")
        scores = []
        for d, s in chunks:
            try:
                sc = float(s)
            except Exception:
                sc = 0.0
            scores.append(max(0.0, min(1.0, -sc)))
        avg = sum(scores) / len(scores)
        LOG.debug("Heuristic grader avg=%.4f", avg)
        return avg

    # Ensure Azure environment mappings are loaded and required vars exist.
    load_env()
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT")
    if not azure_endpoint or not deployment:
        LOG.debug("Azure OpenAI not fully configured; falling back to heuristic grader")
        scores = []
        for d, s in chunks:
            try:
                sc = float(s)
            except Exception:
                sc = 0.0
            scores.append(max(0.0, min(1.0, -sc)))
        avg = sum(scores) / len(scores)
        LOG.debug("Heuristic grader avg=%.4f", avg)
        return avg

    # Use AzureChatOpenAI (chat endpoint) for modern Azure chat models
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    try:
        # Use a safe default temperature (1) to avoid compatibility issues
        llm = AzureChatOpenAI(azure_deployment=deployment, azure_endpoint=azure_endpoint, temperature=1, openai_api_version=openai_api_version)
    except TypeError:
        llm = AzureChatOpenAI(deployment=deployment, azure_endpoint=azure_endpoint, temperature=1, openai_api_version=openai_api_version)

    try:
        # Chat models expect a list-of-message-lists; wrap prompt in HumanMessage
        res = llm.generate([[HumanMessage(content=prompt)]])
        text = res.generations[0][0].text.strip()
        # Try parse a float from text
        try:
            val = float(text)
            val = max(0.0, min(1.0, val))
            LOG.debug("LLM grader returned %s", val)
            return val
        except Exception:
            LOG.debug("LLM grader returned non-numeric '%s', falling back to heuristic", text)
            # fallback heuristic
            scores = []
            for d, s in chunks:
                try:
                    sc = float(s)
                except Exception:
                    sc = 0.0
                scores.append(max(0.0, min(1.0, -sc)))
            avg = sum(scores) / len(scores)
            return avg
    except Exception:
        LOG.exception("LLM grader failed; falling back to heuristic")
        scores = []
        for d, s in chunks:
            try:
                sc = float(s)
            except Exception:
                sc = 0.0
            scores.append(max(0.0, min(1.0, -sc)))
        avg = sum(scores) / len(scores)
        return avg
