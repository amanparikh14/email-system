from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.schema import Row
from src.logging import get_logger

logger = get_logger(__name__)


class Fallback:
    """Sentinel returned by search() when no store row is similar enough."""


FALLBACK = Fallback()


class RetrievalStore:
    def __init__(self, rows: list[Row], model_name: str) -> None:
        self.rows = rows
        self._model = SentenceTransformer(model_name)
        texts = [row.email for row in rows]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        self._embeddings = np.asarray(embeddings)

    def search(self, email: str, k: int, threshold: float) -> list[Row] | Fallback:
        query = self._model.encode([email], normalize_embeddings=True)[0]
        similarities = self._embeddings @ query
        top_indices = np.argsort(-similarities)[:k]
        top_similarity = float(similarities[top_indices[0]])

        if top_similarity < threshold:
            logger.info(f"retrieval fallback fired (top_similarity={top_similarity:.3f} < {threshold})")
            return FALLBACK

        return [self.rows[i] for i in top_indices]
