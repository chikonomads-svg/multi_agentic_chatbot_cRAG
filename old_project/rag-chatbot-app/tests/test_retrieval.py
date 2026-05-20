from unittest.mock import MagicMock, patch
from src.chains import grade_snippets, retrieve_answer


def test_grade_snippets_mock():
    mock_llm = MagicMock()
    # Simulate an LLM returning a JSON array
    class Resp:
        content = "[5,4,3]"

    mock_llm.call_as_llm.return_value = Resp()
    scores = grade_snippets(["a","b","c"], "query", llm=mock_llm)
    assert scores == [5,4,3]


@patch("src.chains.load_faiss_index")
@patch("src.chains.search_faiss")
def test_retrieve_answer_fallback(mock_search, mock_load):
    # No index -> tavily fallback; we patch tavily_search to avoid network
    mock_load.return_value = None
    ans, sources = retrieve_answer("q", "nonexistent_index", tavily_api_key=None)
    assert "Tavily fallback" in ans
