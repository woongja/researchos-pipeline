# -*- coding: utf-8 -*-
"""Ask a question over the local vault index — fully offline RAG.

Embeds the question with bge-m3, retrieves the closest chunks from the
local Chroma index, and asks a local Ollama model to answer using only
that context. Requires `ollama serve` running and `index_vault.py` to
have been run at least once.

Usage:
    python ask_vault.py "질문 내용"
    python ask_vault.py "질문 내용" --top-k 8 --model qwen2.5:7b-instruct-q4_K_M
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

INDEX_DIR = Path(__file__).resolve().parent / "_local_rag_index"
COLLECTION = "vault"
EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_LLM = "coolsoon/kanana-1.5-8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "너는 사용자의 개인 연구 노트(Obsidian vault)를 기반으로 답하는 조수다. "
    "아래 제공된 컨텍스트에 있는 내용만 근거로 답하고, 컨텍스트에 없으면 "
    "모른다고 명확히 말해라. 답변 끝에 참고한 노트 경로를 나열해라."
)


def retrieve(question: str, top_k: int):
    model = SentenceTransformer(EMBED_MODEL)
    query_emb = model.encode([question], normalize_embeddings=True).tolist()
    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    collection = client.get_or_create_collection(COLLECTION)
    if collection.count() == 0:
        print("index is empty — run index_vault.py first", file=sys.stderr)
        sys.exit(1)
    result = collection.query(query_embeddings=query_emb, n_results=top_k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    return list(zip(docs, metas))


def ask_ollama(question: str, context_blocks, model: str):
    context = "\n\n---\n\n".join(
        f"[출처: {m['path']}]\n{doc}" for doc, m in context_blocks
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"컨텍스트:\n{context}\n\n질문: {question}"},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Ollama 호출 실패 — `ollama serve` 켜져 있는지, 모델 pull 됐는지 확인: {e}", file=sys.stderr)
        sys.exit(1)
    return body.get("message", {}).get("content", "(빈 응답)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--model", default=DEFAULT_LLM)
    args = ap.parse_args()

    hits = retrieve(args.question, args.top_k)
    print(f"[검색된 노트 {len(hits)}개]")
    for _, m in hits:
        print(f"  - {m['path']}")
    print()

    answer = ask_ollama(args.question, hits, args.model)
    print(answer)


if __name__ == "__main__":
    main()
