from typing import List, Optional

from docai_toolkit.hf_client import HuggingFaceClient

try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline
except ImportError as _chat_import_error:  # pragma: no cover - optional dependency
    HuggingFacePipeline = None  # type: ignore[assignment]
    pipeline = None  # type: ignore[assignment]
    _CHAT_IMPORT_ERROR = _chat_import_error
else:
    _CHAT_IMPORT_ERROR = None


def chat_over_corpus(
    db,
    query: str,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    max_new_tokens: int = 256,
) -> str:
    """Simple retrieve-then-generate over a FAISS db."""
    docs = db.similarity_search(query, k=4)
    context_blocks: List[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        context_blocks.append(f"Source: {src}\n{doc.page_content}")
    context = "\n\n".join(context_blocks)

    prompt = f"Answer the question using only the provided context.\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    if endpoint:
        client = HuggingFaceClient(api_key, default_endpoint=endpoint)
        resp = client.post_json(
            {
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_new_tokens},
            }
        )
        if isinstance(resp, list) and resp and isinstance(resp[0], dict) and "generated_text" in resp[0]:
            return resp[0]["generated_text"]
        if isinstance(resp, dict) and "generated_text" in resp:
            return resp["generated_text"]
        if isinstance(resp, str):
            return resp
        raise ValueError("Unexpected response from generation endpoint.")

    if pipeline is None or HuggingFacePipeline is None:
        raise RuntimeError("transformers/langchain-community required for local HF generation.")

    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm(prompt)
