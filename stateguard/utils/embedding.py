"""Embedding model — sentence-transformers wrapper.

Provides :class:`EmbeddingModel` that wraps a `sentence-transformers`_
model to produce fixed-size vector embeddings for arbitrary text
inputs.  Used primarily by :class:`~stateguard.dimensions.semantic.SemanticValidator`.

.. _sentence-transformers: https://www.sbert.net/
"""

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
        ...

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
        """
        ...

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of embeddings.

        Args:
            a: Embedding array, shape ``(N, dim)``.
            b: Embedding array, shape ``(M, dim)``.

        Returns:
            Similarity matrix of shape ``(N, M)`` with values in [-1, 1].
        """
        ...
