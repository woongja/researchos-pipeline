# -*- coding: utf-8 -*-
"""Incremental embedding index of the vault for local RAG.

Walks every .md file in the vault, chunks it, embeds chunks with KURE-v1
(Korean retrieval-tuned), and upserts into a local persistent Chroma
collection. Re-running only re-embeds files whose mtime changed since the
last run (tracked in state.json) — safe to run on a schedule or by hand.

Nothing here leaves the machine: the model runs locally, the index is a
local directory, no network calls except the one-time HF model download.
"""
import json
import os
import re
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

VAULT = Path(os.environ.get("RESEARCH_VAULT", str(Path(__file__).resolve().parents[1])))
INDEX_DIR = Path(__file__).resolve().parent / "_local_rag_index"
STATE_FILE = INDEX_DIR / "state.json"
COLLECTION = "vault"
EMBED_MODEL = "BAAI/bge-m3"                 # doc60: 다국어+롱컨텍스트(한국어 노트+영어 논문 혼합). 이전 KURE(한국어 전용)

EXCLUDE_DIRS = {
    ".obsidian", ".git", ".trash", "_trash", ".omc", "node_modules",
    "_attachments", "_export", "local_rag", "_local_rag_index",
    ".claude",                              # 세션 로그 = RAG 노이즈(doc60)
}
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# 청킹 v2(doc60, code-graph-rag 구조 인지 전이): 헤딩·문단 경계 우선.
# 점진 적용 — mtime 변경 파일부터 새 청킹(혼재는 검색 무해). 전체 재적용 원하면 state.json 삭제.
CHUNKER_VERSION = 2
_HEADING = re.compile(r"^#{1,6}\s")


def iter_md_files():
    for p in VAULT.rglob("*.md"):
        if any(part in EXCLUDE_DIRS for part in p.relative_to(VAULT).parts):
            continue
        if p.name.startswith("_"):             # 목록/메타 파일(_unfetched 등) = RAG 노이즈, 스킵
            continue
        yield p


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def _sliding(text, size, overlap):
    """v1 폴백: 경계 무시 슬라이딩(초대형 단일 블록 전용)."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """헤딩·문단 경계 우선 청킹(v2) — 청크가 섹션 중간에서 잘리는 것 방지(스니펫 품질↑).
    블록(헤딩 시작·빈 줄 구분) 단위로 size까지 패킹, 초대형 블록만 슬라이딩 폴백."""
    text = text.strip()
    if not text:
        return []
    # 1) 헤딩 라인 앞에서 분리 → 2) 각 조각을 빈 줄(문단)로 재분리
    blocks, cur = [], []
    for line in text.splitlines():
        if _HEADING.match(line) and cur:
            blocks.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    paras = []
    for b in blocks:
        if len(b) <= size:
            paras.append(b)
        else:
            paras.extend(p for p in re.split(r"\n\s*\n", b) if p.strip())
    # 3) 패킹
    chunks, buf = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) > size:                        # 초대형 단일 문단 → 슬라이딩
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_sliding(p, size, overlap))
        elif len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    print(f"vault: {VAULT}")
    print(f"index: {INDEX_DIR}")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    collection = client.get_or_create_collection(COLLECTION)

    state = load_state()
    state.pop("_chunker", None)              # 메타 키 — 파일 목록 비교에서 제외
    on_disk = {}
    changed_files = []

    for path in iter_md_files():
        rel = path.relative_to(VAULT).as_posix()
        mtime = path.stat().st_mtime
        on_disk[rel] = mtime
        if state.get(rel) != mtime:
            changed_files.append((rel, path))

    deleted = [rel for rel in state if rel not in on_disk]

    if not changed_files and not deleted:
        print("up to date — nothing to re-index")
        return 0

    print(f"changed/new: {len(changed_files)} | deleted: {len(deleted)}")

    for rel in deleted:
        collection.delete(where={"path": rel})
        print(f"  removed: {rel}")

    if changed_files:
        print(f"loading embedding model ({EMBED_MODEL}) — first run downloads it, be patient")
        model = SentenceTransformer(EMBED_MODEL)

        for rel, path in changed_files:
            collection.delete(where={"path": rel})
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  skip (read error) {rel}: {e}")
                continue
            body = strip_frontmatter(raw)
            chunks = chunk_text(body)
            if not chunks:
                continue
            embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
            ids = [f"{rel}::{i}" for i in range(len(chunks))]
            metadatas = [{"path": rel, "chunk": i} for i in range(len(chunks))]
            collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
            print(f"  indexed: {rel} ({len(chunks)} chunks)")

    save_state({**on_disk, "_chunker": CHUNKER_VERSION})
    print(f"done. collection size: {collection.count()} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
