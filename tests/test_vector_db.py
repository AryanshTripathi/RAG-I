from unittest.mock import MagicMock, patch, call
from qdrant_client.models import PointStruct


def _make_storage(exists=True):
    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = exists
        from vector_db import QdrantStorage
        storage = QdrantStorage()
    return storage, instance


def test_init_skips_creation_if_collection_exists():
    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = True
        from vector_db import QdrantStorage
        QdrantStorage()
    instance.create_collection.assert_not_called()


def test_init_creates_collection_if_missing():
    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = False
        from vector_db import QdrantStorage
        QdrantStorage()
    instance.create_collection.assert_called_once()


def test_upsert_creates_correct_number_of_points():
    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = True
        from vector_db import QdrantStorage
        storage = QdrantStorage()
        storage.upsert(
            ids=["id1", "id2"],
            vectors=[[0.1] * 768, [0.2] * 768],
            payloads=[{"text": "a", "source": "s1"}, {"text": "b", "source": "s1"}],
        )
    kwargs = instance.upsert.call_args.kwargs
    assert len(kwargs["points"]) == 2  # two PointStructs


def test_search_returns_contexts_and_sources():
    fake_point = MagicMock()
    fake_point.payload = {"text": "some context", "source": "doc.pdf"}

    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = True
        instance.query_points.return_value.points = [fake_point]
        from vector_db import QdrantStorage
        storage = QdrantStorage()
        result = storage.search([0.1] * 768, k=5)

    assert result["context"] == ["some context"]
    assert result["sources"] == ["doc.pdf"]


def test_search_filters_empty_sources():
    fake_point = MagicMock()
    fake_point.payload = {"text": "some text", "source": None}

    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = True
        instance.query_points.return_value.points = [fake_point]
        from vector_db import QdrantStorage
        storage = QdrantStorage()
        result = storage.search([0.1] * 768)

    assert result["context"] == ["some text"]
    assert result["sources"] == []


def test_search_filters_empty_text():
    fake_point = MagicMock()
    fake_point.payload = {"text": "", "source": "doc.pdf"}

    with patch("vector_db.QdrantClient") as MockClient:
        instance = MockClient.return_value
        instance.collection_exists.return_value = True
        instance.query_points.return_value.points = [fake_point]
        from vector_db import QdrantStorage
        storage = QdrantStorage()
        result = storage.search([0.1] * 768)

    assert result["context"] == []
    assert result["sources"] == []
