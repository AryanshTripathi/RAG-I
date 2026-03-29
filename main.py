from fastapi import FastAPI
from openai import OpenAI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
from logger import configure_logging, get_logger
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()
configure_logging()
logger = get_logger("main")

inngest_client = inngest.Inngest(
    app_id="rag_app",
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


# ---------------------------------------------------------------------------
# Extracted step helpers (module-level for testability)
# ---------------------------------------------------------------------------

def _load(pdf_path: str, source_id: str | None) -> RAGChunkAndSrc:
    logger.debug("loading pdf", pdf_path=pdf_path, source_id=source_id)
    chunks = load_and_chunk_pdf(pdf_path)
    logger.info("pdf chunked", pdf_path=pdf_path, num_chunks=len(chunks))
    return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    chunks = chunks_and_src.chunks
    source_id = chunks_and_src.source_id
    logger.debug("embedding chunks", num_chunks=len(chunks), source_id=source_id)
    vecs = embed_text(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
    payloads = [{"text": chunks[i], "source": source_id} for i in range(len(chunks))]
    storage = QdrantStorage()
    storage.upsert(ids, vecs, payloads)
    logger.info("upsert complete", source_id=source_id, ingested=len(chunks))
    return RAGUpsertResult(ingested=len(chunks))


def _search(question: str, top_k: int = 5) -> RAGSearchResult:
    logger.debug("embedding query", question=question, top_k=top_k)
    query_vec = embed_text([question])[0]
    store = QdrantStorage()
    found = store.search(query_vec, k=top_k)
    logger.info("search complete", num_contexts=len(found["context"]), num_sources=len(found["sources"]))
    return RAGSearchResult(contexts=found["context"], sources=found["sources"])


def _llm_answer(user_content: str) -> str:
    providers = [
        {"api_key": os.getenv("OPENAI_API_KEY"), "base_url": None, "model": "gpt-4o-mini"},
        {"api_key": os.getenv("GEMINI_API_KEY"), "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.0-flash-lite"},
        {"api_key": os.getenv("GROQ_API_KEY"), "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    ]
    messages = [
        {"role": "system", "content": "You answer questions using only the provided content."},
        {"role": "user", "content": user_content},
    ]
    last_error = None
    for p in providers:
        try:
            logger.debug("trying llm provider", model=p["model"])
            client = OpenAI(api_key=p["api_key"], base_url=p["base_url"], max_retries=0)
            response = client.chat.completions.create(model=p["model"], max_tokens=1024, temperature=0.2, messages=messages)
            logger.info("llm answer received", model=p["model"])
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("llm provider failed", model=p["model"], error=str(e))
            last_error = e
    raise last_error


# ---------------------------------------------------------------------------
# Inngest functions
# ---------------------------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    pdf_path = ctx.events[0].data["pdf_path"]
    source_id = ctx.events[0].data.get("source_id")
    logger.info("ingest started", pdf_path=pdf_path, source_id=source_id)

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(pdf_path, source_id),
        output_type=RAGChunkAndSrc,
    )
    ingested = await ctx.step.run(
        "upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )
    logger.info("ingest finished", source_id=source_id, ingested=ingested.ingested)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.events[0].data["question"]
    top_k = int(ctx.events[0].data.get("top_k", 5))
    logger.info("query started", question=question, top_k=top_k)

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely using the context. If you cannot answer, respond with 'I don't know'."
    )

    answer = await ctx.step.run("llm-answer", lambda: _llm_answer(user_content))
    logger.info("query finished", question=question, num_sources=len(found.sources))

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts),
    }


app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
