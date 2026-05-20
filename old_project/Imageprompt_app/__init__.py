"""
Imageprompt_app - A Streamlit application for researching trending topics
and generating Instagram-optimized AI image prompts using Tavily API.

Modules:
    - logic: Contains the ImagePromptResearcher class for research and prompt generation
    - app: Streamlit UI components

Features:
    - Research 10 trending topics from a keyword
    - Select 1 topic for deep research
    - Generate 10 Instagram-optimized prompts for a carousel post
    - Each prompt covers: What, Why, How, Benefits, Examples, Tips, Future, etc.
"""

__version__ = "1.0.0"
__author__ = "Image Prompt Researcher"

from .logic import (
    ImagePromptResearcher, 
    get_recent_logs, 
    read_log_file,
    setup_logging
)

__all__ = [
    "ImagePromptResearcher", 
    "get_recent_logs", 
    "read_log_file",
    "setup_logging"
]