"""Environment loader that loads .env from the parent Vibe Coding folder if present.

This helper uses dotenv.find_dotenv to search upward for a .env file and loads it
without writing any secrets to the repository. It is safe to call multiple times.
"""
from dotenv import load_dotenv, find_dotenv
import logging
from typing import Optional
import os

LOG = logging.getLogger(__name__)


def load_env(silent: bool = True) -> Optional[str]:
    """Search for a .env file in parent directories and load it.

    Returns the path loaded or None if nothing was found.
    """
    path = find_dotenv()
    if not path:
        if not silent:
            LOG.warning("No .env file found in parent directories")
        return None
    load_dotenv(path)
    LOG.debug("Loaded environment variables from %s", path)
    # Support Azure-flavored env vars by mapping them to the OpenAI client
    # environment variables expected by the openai/python client and
    # downstream libraries. This keeps keys/config in .env while letting
    # the OpenAI client find them automatically.
    # Map AZURE_OPENAI_API_KEY -> OPENAI_API_KEY
    if os.getenv("AZURE_OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    # Note: do NOT set OPENAI_API_BASE for Azure. The OpenAI client uses
    # azure_endpoint together with deployment names for Azure usage. Setting
    # OPENAI_API_BASE while using deployments can cause client validation
    # errors. We therefore avoid mapping AZURE_OPENAI_ENDPOINT -> OPENAI_API_BASE.
    # Indicate to the OpenAI client that we're using Azure
    if os.getenv("AZURE_OPENAI_API_KEY") and not os.getenv("OPENAI_API_TYPE"):
        os.environ["OPENAI_API_TYPE"] = "azure"
    if os.getenv("AZURE_OPENAI_API_VERSION") and not os.getenv("OPENAI_API_VERSION"):
        os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
    # Expose deployment names so embedding/llm code can pick them up if needed.
    # Support both older names and the canonical names used by this project
    # (AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME).
    if os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") and not os.getenv("OPENAI_EMBEDDING_DEPLOYMENT"):
        os.environ["OPENAI_EMBEDDING_DEPLOYMENT"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    elif os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") and not os.getenv("OPENAI_EMBEDDING_DEPLOYMENT"):
        os.environ["OPENAI_EMBEDDING_DEPLOYMENT"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") and not os.getenv("OPENAI_DEPLOYMENT"):
        os.environ["OPENAI_DEPLOYMENT"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    elif os.getenv("AZURE_OPENAI_DEPLOYMENT") and not os.getenv("OPENAI_DEPLOYMENT"):
        os.environ["OPENAI_DEPLOYMENT"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    return path
