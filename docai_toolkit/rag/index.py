from pathlib import Path
from typing import Iterable, List, Optional

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings.base import Embeddings
    from docai_toolkit.hf_client import HuggingFaceClient
except ImportError as _langchain_exc:  # pragma: no cover - optional dependency
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]
    FAISS = None  # type: ignore[assignment]
    Embeddings = object  # type: ignore[assignment]
    HuggingFaceClient = None  # type: ignore[assignment]
    _LANGCHAIN_IMPORT_ERROR = _langchain_exc
else:
    _LANGCHAIN_IMPORT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[misc,assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _load_embedding_model(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed") from _IMPORT_ERROR
    return SentenceTransformer(model_name)


class RemoteEmbeddings(Embeddings):
    """Call a remote embedding endpoint that accepts JSON {\"inputs\": text}."""

    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        self.client = HuggingFaceClient(api_key, default_endpoint=endpoint)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self.client.post_json({"inputs": texts})
            return self._extract_batch(resp, len(texts))
        except Exception:
            vectors: List[List[float]] = []
            for text in texts:
                single = self.client.post_json({"inputs": text})
                vectors.append(self._extract_vector(single))
            return vectors

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.post_json({"inputs": text})
        return self._extract_vector(resp)

    @staticmethod
    def _extract_vector(response):
        if isinstance(response, list) and response and isinstance(response[0], list):
            return response[0]
        if isinstance(response, list):
            return response
        raise ValueError("Unexpected embedding response format.")

    @staticmethod
    def _extract_batch(response, expected: int) -> List[List[float]]:
        if isinstance(response, list) and response and isinstance(response[0], list):
            return response
        if isinstance(response, list) and len(response) == expected and isinstance(response[0], dict) and "embedding" in response[0]:
            return [item["embedding"] for item in response]
        raise ValueError("Unexpected batch embedding response format.")


def build_index_from_markdown(
    markdown_files: Iterable[Path],
    embedding_model: str = "all-mpnet-base-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    persist_path: Optional[Path] = None,
    embedding_endpoint: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
):
    if RecursiveCharacterTextSplitter is None or FAISS is None:
        raise RuntimeError("langchain is required for RAG. Install langchain and langchain-community.")

    texts: List[str] = []
    metadatas: List[dict] = []

    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        texts.append(content)
        metadatas.append({"source": str(path)})

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents(texts, metadatas=metadatas)

    if embedding_endpoint:
        embeddings = RemoteEmbeddings(embedding_endpoint, api_key=embedding_api_key)
    else:
        embeddings = _load_embedding_model(embedding_model)
    db = FAISS.from_documents(docs, embeddings)
    if persist_path:
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        db.save_local(str(persist_path))
    return db


def load_index(persist_path: Path):
    if not persist_path.exists():
        raise FileNotFoundError(f"Persisted index not found at {persist_path}")
    if FAISS is None:
        raise RuntimeError("langchain-community is required to load indexes.")
    embeddings = None  # embeddings are restored from disk
    return FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=False)
