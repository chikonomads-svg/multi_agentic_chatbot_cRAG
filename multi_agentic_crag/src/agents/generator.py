"""Generator Agent: synthesizes final grounded answer with citations using an LLM."""
from typing import Dict, Any, List
import logging
import os

from langchain_openai import AzureChatOpenAI
from src.utils.env import load_env
from langchain_core.messages import HumanMessage
import openai

LOG = logging.getLogger(__name__)


def generate_answer(state: Dict[str, Any], sources: List[str]) -> str:
    """Generate a final answer given the user message and sources.

    This uses a simple OpenAI LLM call. In production you'd stream and format.
    """
    prompt = f"Answer the question based on the following sources:\nSources:\n{sources}\nQuestion:\n{state.get('user_message')}\nProvide a concise, cited answer."
    # Ensure Azure environment mappings are loaded and required vars exist.
    load_env()
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT")
    if not azure_endpoint or not deployment:
        LOG.debug("Azure OpenAI not fully configured; returning placeholder answer")
        return "(Azure OpenAI not configured) I couldn't generate a final answer."
    LOG.debug("Calling OpenAI to generate answer for question=%s", state.get("user_message"))
    # If configured for Azure, prefer the AzureOpenAI LLM wrapper when
    # available so requests go to the Azure endpoint/deployment instead of
    # api.openai.com.
    # Use AzureOpenAI only
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    # Recent langchain/openai SDKs expect Azure deployment to be passed as
    # azure_deployment (or model) rather than a bare 'deployment' kwarg which
    # can end up forwarded to the underlying OpenAI client and cause
    # TypeError: Completions.create() got an unexpected keyword argument 'deployment'.
    # Pass azure_deployment explicitly to avoid that.
    try:
        llm = AzureChatOpenAI(azure_deployment=deployment, azure_endpoint=azure_endpoint, temperature=1, openai_api_version=openai_api_version)
    except TypeError:
        # Fallback for older langchain wrappers that still accept 'deployment'
        llm = AzureChatOpenAI(deployment=deployment, azure_endpoint=azure_endpoint, temperature=1, openai_api_version=openai_api_version)
    try:
        # Newer langchain chat LLMs expect a list-of-message-lists. Wrap the
        # prompt in a HumanMessage and pass as [[HumanMessage(...)]].
        res = llm.generate([[HumanMessage(content=prompt)]])
        # res.generations is a list (one per input) of lists (one per model
        # generation). Use the first generation's text.
        return res.generations[0][0].text
    except Exception:
        # Some Azure deployments correspond to chat-only models (gpt-\*) and
        # will return an OperationNotSupported error when the completions
        # endpoint is used. As a pragmatic fallback, call the Chat
        # completions endpoint directly using the openai SDK which works with
        # both Azure and OpenAI settings when configured via environment
        # variables. If that fails, log and return a safe message.
        LOG.exception("LLM generation failed; attempting chat-completion fallback")
        try:
            # Use the new openai Azure client wrapper (openai>=1.0) to call
            # the chat completions endpoint. This avoids using the removed
            # ChatCompletion helper from older SDKs.
            from openai import AzureOpenAI as OpenAIClient

            client = OpenAIClient(
                api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
                api_version=openai_api_version,
                azure_endpoint=azure_endpoint,
            )

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            # Some Azure models disallow temperature=0; omit temperature to
            # use the model default or set to 1. Omit for compatibility.
            resp = client.chat.completions.create(model=deployment, messages=messages)
            return resp.choices[0].message.content
        except Exception:
            LOG.exception("Chat-completion fallback failed")
            return "(LLM generation failed)"


def generate_answer_stream(state: Dict[str, Any], sources: List[str], chunk_size: int = 100):
    """Yield the generated answer as small text chunks suitable for streaming UI.

    This implementation uses the same LLM call as generate_answer but yields
    the final text in chunks. It's a pragmatic approach that provides a
    streaming experience in Streamlit without relying on async callbacks.
    """
    prompt = f"Answer the question based on the following sources:\nSources:\n{sources}\nQuestion:\n{state.get('user_message')}\nProvide a concise, cited answer."
    if not os.getenv("OPENAI_API_KEY"):
        yield "(OpenAI key not set) I couldn't generate a final answer."
        return
    load_env()
    # choose llm same as generate_answer
    if os.getenv("OPENAI_API_TYPE") == "azure":
        if AzureChatOpenAI is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT")
            openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
            # If langchain's AzureChatOpenAI wrapper isn't present, fall back to
            # a generic OpenAI-compatible client. We try to avoid this path
            # where possible since AzureChatOpenAI is preferred.
            from langchain_openai import OpenAI
            llm = OpenAI(openai_api_base=azure_endpoint, openai_api_type="azure", openai_api_version=openai_api_version, model=deployment, temperature=1)
        else:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")
            deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_DEPLOYMENT")
            try:
                llm = AzureChatOpenAI(azure_deployment=deployment, azure_endpoint=azure_endpoint, temperature=1)
            except TypeError:
                llm = AzureChatOpenAI(deployment=deployment, azure_endpoint=azure_endpoint, temperature=1)
    else:
        llm = OpenAI(temperature=0)

    try:
        # For chat LLMs, pass a message list
        try:
            res = llm.generate([[HumanMessage(content=prompt)]])
            text = res.generations[0][0].text or ""
        except Exception:
            # Fallback: some non-chat LLMs may accept a simple string input
            res = llm.generate([prompt])
            text = res.generations[0][0].text or ""
        # yield in chunks
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
    except Exception:
        LOG.exception("LLM generation failed (stream); attempting chat fallback")
        try:
            from openai import AzureOpenAI as OpenAIClient
            client = OpenAIClient(
                api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE"),
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            # Omit temperature for compatibility with some Azure models
            resp = client.chat.completions.create(model=deployment, messages=messages)
            text = resp.choices[0].message.content or ""
            for i in range(0, len(text), chunk_size):
                yield text[i : i + chunk_size]
        except Exception:
            LOG.exception("Chat-completion fallback (stream) failed")
            yield "(LLM generation failed)"
