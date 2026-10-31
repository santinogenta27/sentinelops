"""Embedding backends.

``HashingEmbedder`` is a deterministic, zero-network, zero-cost embedder (signed feature
hashing over normalised unigrams + bigrams). It keeps CI reproducible and needs no
model download. ``GeminiEmbedder`` uses ``gemini-embedding-001`` and falls back to
hashing if the API is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

from google import genai

logger = logging.getLogger("sentinelops.rag")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or of to for in on at by with is are be been was were it this that as from my me i we our you your "
    "they their there please can could would should will just".split()
)
_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied", "ed", "es", "s")


def stem(token: str) -> str:
    for suf in _SUFFIXES:
        if len(token) > len(suf) + 3 and token.endswith(suf):
            return token[: -len(suf)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "local-hashing-v1"

    def __init__(self, dims: int = 768) -> None:
        self.dims = dims

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feat in features:
            digest = hashlib.blake2b(feat.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dims
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.5 if "_" in feat else 1.0)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self.name = f"gemini:{model}"
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._fallback = HashingEmbedder()
        self._degraded = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._degraded:
            try:
                resp = self._client.models.embed_content(model=self._model, contents=texts)
                vectors = [list(e.values or []) for e in (resp.embeddings or [])]
                if len(vectors) == len(texts) and all(vectors):
                    return vectors
            except Exception as exc:  # noqa: BLE001 - any provider failure degrades gracefully
                logger.warning("gemini embeddings unavailable, falling back to local hashing: %s", str(exc)[:200])
            self._degraded = True
            self.name = self._fallback.name
        return self._fallback.embed(texts)
