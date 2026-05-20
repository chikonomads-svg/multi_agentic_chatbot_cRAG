import sys
from pathlib import Path

# Ensure the rag-chatbot-app/src folder is importable as 'src' when tests run
ROOT = Path(__file__).resolve().parents[1]
# Add project root so 'src' package (rag-chatbot-app/src) is importable
sys.path.insert(0, str(ROOT))
