from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from logger import get_logger

logger = get_logger("vector_db")


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=768):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim

        if not self.client.collection_exists(collection_name=self.collection):
            logger.info("creating collection", collection=collection, dim=dim)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
        else:
            logger.debug("collection exists", collection=collection)

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("upserted points", collection=self.collection, count=len(points))

    def search(self, vector, k: int = 5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            with_payload=True,
            limit=k,
        ).points
        contexts = []
        sources = set()
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source") or ""
            if text:
                contexts.append(text)
                if source:
                    sources.add(source)
        logger.info("search results", collection=self.collection, num_results=len(results), num_contexts=len(contexts))
        return {"context": contexts, "sources": list(sources)}
