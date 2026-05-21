"""Embedding model — sentence-transformers wrapper.

Provides :class:`EmbeddingModel` that wraps a `sentence-transformers`_
model to produce fixed-size vector embeddings for arbitrary text
inputs.  Used primarily by :class:`~stateguard.core.tier1.EmbeddingValidator`.

.. _sentence-transformers: https://www.sbert.net/
"""

from __future__ import annotations

from typing import Any

import numpy as np


class EmbeddingModel:
    """Wrapper around a ``sentence-transformers`` model.

    Lazily loads the model on first call to :meth:`encode`.  Supports
    CPU and CUDA execution.

    Usage::

        model = EmbeddingModel("all-MiniLM-L6-v2", device="cpu")
        vectors = model.encode(["hello world", "another text"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        """Initialise the embedding model (model is *not* loaded yet).

        Args:
            model_name: Name or path of a sentence-transformers model.
            device:     Target device (``"cpu"`` or ``"cuda"``).
        """
        self._model_name = model_name
        self._device = device
        self._model: Any = None  # SentenceTransformer instance

    @property
    def model_name(self) -> str:
        """Name or path of the underlying SentenceTransformer model."""
        return self._model_name

    @property
    def device(self) -> str:
        """Device the model is running on."""
        return self._device

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying model has been loaded into memory."""
        return self._model is not None

    def load(self) -> None:
        """Explicitly load the model into memory.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode a list of sentences into embedding vectors.

        Args:
            sentences:           Texts to embed.
            batch_size:          Number of texts per inference batch.
            normalize_embeddings: Whether to L2-normalise the output vectors.
            **kwargs:            Additional arguments passed to
                                 ``SentenceTransformer.encode``.

        Returns:
            A 2-D NumPy array of shape ``(len(sentences), embedding_dim)``.

        Raises:
            ValueError: If *sentences* is empty.
            RuntimeError: If the model fails to encode.
        """
        if not sentences:
            raise ValueError("Cannot encode an empty list of sentences.")

        if self._model is None:
            self.load()

        try:
            embeddings = self._model.encode(
                sentences,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                **kwargs,
            )
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"Embedding encoding failed: {e}") from e

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings.

        Uses normalised dot product (equivalent to cosine similarity
        when both inputs are L2-normalised).

        Args:
            a: Embedding array, shape ``(N, dim)``.
            b: Embedding array, shape ``(M, dim)``.

        Returns:
            Similarity matrix of shape ``(N, M)`` with values in [-1, 1].
        """
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)

        # Guard against zero vectors that would produce NaN from division by zero
        if np.any(norm_a == 0) or np.any(norm_b == 0):
            return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

        a_norm = a / norm_a
        b_norm = b / norm_b
        return np.dot(a_norm, b_norm.T)
