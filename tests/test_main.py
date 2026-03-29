import uuid
from unittest.mock import MagicMock, patch
from custom_types import RAGChunkAndSrc, RAGUpsertResult


# ---------------------------------------------------------------------------
# _llm_answer tests
# ---------------------------------------------------------------------------

def test_llm_answer_uses_first_provider_on_success():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "  answer from openai  "

    with patch("main.OpenAI") as MockOpenAI:
        MockOpenAI.return_value.chat.completions.create.return_value = mock_response
        from main import _llm_answer
        result = _llm_answer("some context")

    assert result == "answer from openai"
    assert MockOpenAI.call_count == 1  # only first provider tried


def test_llm_answer_falls_back_to_second_provider():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "answer from gemini"

    with patch("main.OpenAI") as MockOpenAI:
        first = MagicMock()
        first.chat.completions.create.side_effect = Exception("quota exceeded")
        second = MagicMock()
        second.chat.completions.create.return_value = mock_response
        MockOpenAI.side_effect = [first, second]

        from main import _llm_answer
        result = _llm_answer("some context")

    assert result == "answer from gemini"
    assert MockOpenAI.call_count == 2


def test_llm_answer_raises_if_all_providers_fail():
    with patch("main.OpenAI") as MockOpenAI:
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("all down")
        MockOpenAI.return_value = client

        from main import _llm_answer
        try:
            _llm_answer("some context")
            assert False, "should have raised"
        except Exception as e:
            assert str(e) == "all down"


# ---------------------------------------------------------------------------
# _upsert tests
# ---------------------------------------------------------------------------

def test_upsert_step_returns_correct_ingested_count():
    chunks_and_src = RAGChunkAndSrc(chunks=["chunk1", "chunk2", "chunk3"], source_id="test.pdf")

    with patch("main.embed_text", return_value=[[0.1] * 768] * 3), \
         patch("main.QdrantStorage") as MockStorage:
        MockStorage.return_value.upsert.return_value = None
        from main import _upsert
        result = _upsert(chunks_and_src)

    assert result == RAGUpsertResult(ingested=3)


def test_upsert_step_generates_deterministic_ids():
    chunks_and_src = RAGChunkAndSrc(chunks=["a", "b"], source_id="src")

    with patch("main.embed_text", return_value=[[0.0] * 768, [0.0] * 768]), \
         patch("main.QdrantStorage") as MockStorage:
        MockStorage.return_value.upsert.return_value = None
        from main import _upsert
        _upsert(chunks_and_src)
        _upsert(chunks_and_src)

    # Both calls should have used the same IDs (deterministic UUID5)
    first_call_ids = MockStorage.return_value.upsert.call_args_list[0][0][0]
    second_call_ids = MockStorage.return_value.upsert.call_args_list[1][0][0]
    assert first_call_ids == second_call_ids


# ---------------------------------------------------------------------------
# _load tests
# ---------------------------------------------------------------------------

def test_load_step_returns_ragchunkandsrc():
    with patch("main.load_and_chunk_pdf", return_value=["chunk1", "chunk2"]):
        from main import _load
        result = _load("some.pdf", "some.pdf")

    assert isinstance(result, RAGChunkAndSrc)
    assert result.chunks == ["chunk1", "chunk2"]
    assert result.source_id == "some.pdf"


def test_load_step_passes_none_source_id():
    with patch("main.load_and_chunk_pdf", return_value=["chunk1"]):
        from main import _load
        result = _load("some.pdf", None)

    assert result.source_id is None
