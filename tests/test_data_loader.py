from unittest.mock import MagicMock, patch


def _fake_doc(text: str):
    doc = MagicMock()
    doc.text = text
    return doc


def test_load_and_chunk_pdf_returns_chunks():
    fake_docs = [_fake_doc("This is a sentence. " * 50)]
    with patch("data_loader.PDFReader") as MockReader:
        MockReader.return_value.load_data.return_value = fake_docs
        from data_loader import load_and_chunk_pdf
        chunks = load_and_chunk_pdf("fake.pdf")
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_load_and_chunk_pdf_skips_empty_docs():
    fake_docs = [_fake_doc(""), _fake_doc(None), _fake_doc("Real content. " * 20)]
    with patch("data_loader.PDFReader") as MockReader:
        MockReader.return_value.load_data.return_value = fake_docs
        from data_loader import load_and_chunk_pdf
        chunks = load_and_chunk_pdf("fake.pdf")
    # Only the doc with real content should produce chunks
    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


def test_embed_text_returns_correct_shape():
    fake_embeddings = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    with patch("data_loader.embed") as mock_embed:
        mock_embed.text.return_value = {"embeddings": fake_embeddings}
        from data_loader import embed_text
        result = embed_text(["a", "b", "c"])
    assert len(result) == 3
    assert all(len(v) == 768 for v in result)


def test_embed_text_calls_api_with_search_document_task():
    with patch("data_loader.embed") as mock_embed:
        mock_embed.text.return_value = {"embeddings": [[0.0] * 768]}
        from data_loader import embed_text
        embed_text(["hello"])
    mock_embed.text.assert_called_once()
    _, kwargs = mock_embed.text.call_args
    assert kwargs.get("task_type") == "search_document"


def test_get_embedding_uses_search_query_task():
    with patch("data_loader.embed") as mock_embed:
        mock_embed.text.return_value = {"embeddings": [[0.0] * 768]}
        from data_loader import get_embedding
        get_embedding("test query")
    _, kwargs = mock_embed.text.call_args
    assert kwargs.get("task_type") == "search_query"
