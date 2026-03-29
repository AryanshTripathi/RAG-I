from nomic import embed
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()

logger = get_logger("data_loader")

EMBED_MODEL = "nomic-embed-text-v1.5"
EMBED_DIMENSION = 768

splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)


def get_embedding(text: str) -> list[float]:
    logger.debug("embedding single text", length=len(text))
    output = embed.text(texts=[text], model=EMBED_MODEL, task_type="search_query")
    return output["embeddings"][0]


def load_and_chunk_pdf(file_path: str) -> list[str]:
    logger.debug("loading pdf", file_path=file_path)
    docs = PDFReader().load_data(file_path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    logger.info("pdf loaded and chunked", file_path=file_path, num_pages=len(texts), num_chunks=len(chunks))
    return chunks


def embed_text(texts: list[str]) -> list[list[float]]:
    logger.debug("embedding batch", batch_size=len(texts))
    output = embed.text(texts=texts, model=EMBED_MODEL, task_type="search_document")
    logger.info("batch embedded", batch_size=len(texts), dim=EMBED_DIMENSION)
    return output["embeddings"]
