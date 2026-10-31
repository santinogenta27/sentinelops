"""Hybrid policy retriever: ChromaDB dense search + BM25 sparse search fused with RRF."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.models.schemas import PolicyClause
from app.rag.embeddings import Embedder, tokenize

logger = logging.getLogger("sentinelops.rag")

_HEADER_RE = re.compile(r"^##\s+(POL-[A-Z]+-\d{3})\s*\|\s*(.+)$")
_CATEGORY_RE = re.compile(r"^category:\s*([A-Z_]+)\s*$")


def load_policies(path: Path) -> list[PolicyClause]:
    clauses: list[PolicyClause] = []
    current: dict[str, str] | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            clauses.append(
                PolicyClause(
                    clause_id=current["id"],
                    title=current["title"],
                    category=current.get("category", "GENERAL"),
                    text=" ".join(body).strip(),
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        if m := _HEADER_RE.match(line.strip()):
            flush()
            current, body = {"id": m.group(1), "title": m.group(2).strip()}, []
        elif current is not None and (c := _CATEGORY_RE.match(line.strip())):
            current["category"] = c.group(1)
        elif current is not None and line.strip():
            body.append(line.strip())
    flush()
    return clauses


@dataclass(frozen=True, slots=True)
class RetrievedClause:
    clause: PolicyClause
    score: float
    dense_rank: int | None
    sparse_rank: int | None

    def render(self) -> str:
        return f"[{self.clause.clause_id}] {self.clause.title}: {self.clause.text}"


class PolicyRetriever:
    RRF_K = 60

    def __init__(self, clauses: list[PolicyClause], embedder: Embedder, *, persist_path: Path | None = None) -> None:
        if not clauses:
            raise ValueError("policy corpus is empty")
        self.clauses = clauses
        self.by_id = {c.clause_id: c for c in clauses}
        self.embedder = embedder
        chroma_settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        if persist_path is not None:
            persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_path), settings=chroma_settings)
        else:
            self._client = chromadb.EphemeralClient(settings=chroma_settings)
        self._collection_name = "policies"
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        docs = [self._doc_text(c) for c in clauses]
        self._collection.add(
            ids=[c.clause_id for c in clauses],
            documents=docs,
            embeddings=self.embedder.embed(docs),  # type: ignore[arg-type]
            metadatas=[{"category": c.category, "title": c.title} for c in clauses],
        )
        self._bm25 = BM25Okapi([tokenize(d) for d in docs])
        logger.info("indexed %d policy clauses (embedder=%s)", len(clauses), embedder.name)

    @staticmethod
    def _doc_text(c: PolicyClause) -> str:
        return f"{c.title}. {c.text}"

    def get(self, clause_id: str) -> PolicyClause | None:
        return self.by_id.get(clause_id)

    def _dense_ranking(self, query: str, n: int) -> list[str]:
        try:
            res = self._collection.query(query_embeddings=self.embedder.embed([query]), n_results=n)  # type: ignore[arg-type]
            ids = res.get("ids") or [[]]
            return list(ids[0])
        except Exception as exc:  # noqa: BLE001 - dense side failing must not break retrieval
            logger.warning("dense retrieval failed, using BM25 only: %s", exc)
            return []

    def _sparse_ranking(self, query: str) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.clauses)), key=lambda i: float(scores[i]), reverse=True)
        return [self.clauses[i].clause_id for i in order if float(scores[i]) > 0]

    def search(self, query: str, *, k: int = 4, category_hint: str | None = None) -> list[RetrievedClause]:
        n = len(self.clauses)
        dense = self._dense_ranking(query, n)
        sparse = self._sparse_ranking(query)
        fused: dict[str, float] = {}
        for rank, cid in enumerate(dense, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
        for rank, cid in enumerate(sparse, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
        if category_hint:
            for cid, clause in self.by_id.items():
                if clause.category == category_hint:
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        d_pos = {cid: i for i, cid in enumerate(dense, start=1)}
        s_pos = {cid: i for i, cid in enumerate(sparse, start=1)}
        return [
            RetrievedClause(clause=self.by_id[cid], score=round(score, 5), dense_rank=d_pos.get(cid), sparse_rank=s_pos.get(cid))
            for cid, score in ranked
        ]
