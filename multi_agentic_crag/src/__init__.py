"""Multi-Agent Corrective RAG - package initializer

Ensure environment variables from a parent .env are loaded early so that
modules which create OpenAI/Azure clients at import time pick up the
configuration correctly.
"""
from .utils.env import load_env

# Load environment variables as early as possible when the package is
# imported. This maps any AZURE_* variables into the OPENAI_* variables
# consumers (see src.utils.env.load_env for mapping rules).
load_env()

__all__ = ["agents", "utils", "graph", "app"]
