"""Pytest configuration and shared fixtures for StateGuard tests.

Provides mock fixtures that replace external dependencies
(sentence-transformers, sklearn models) so tests run without
downloading real models.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from stateguard.models.enums import ValidationDimension


# ── Embedding Model Mock ───────────────────────────────────────────


@pytest.fixture
def mock_embedding_model():
    """Return a MagicMock that replaces sentence-transformers.

    The mock returns a fixed embedding vector so tests don't need
    a real model download. Usage::

        from stateguard.utils.embedding import EmbeddingModel
        model = EmbeddingModel()  # uses mock under the hood
    """
    mock = MagicMock()
    mock.encode.return_value = [0.1] * 384  # typical embedding dim
    mock.get_sentence_embedding_dimension.return_value = 384
    return mock


@pytest.fixture
def auto_patch_embedding():
    """Automatically patch SentenceTransformer for all tests.

    Any test that imports SentenceTransformer will get the mock
    instead of a real model download.
    """
    patcher = patch("sentence_transformers.SentenceTransformer")
    mock_st = patcher.start()
    instance = mock_st.return_value
    instance.encode.return_value = [0.1] * 384
    yield
    patcher.stop()


# ── Validation Context Fixtures ────────────────────────────────────


@pytest.fixture
def sample_validation_context() -> dict[str, Any]:
    """Return a standard validation context dict for testing.

    Contains prompt, expected output, and metadata that validators
    can use during validation.
    """
    return {
        "prompt": "What is the capital of France?",
        "expected_output": "Paris",
        "domain": "geography",
        "difficulty": "easy",
        "metadata": {"source": "test", "version": "1.0"},
    }


@pytest.fixture
def sample_dimension_scores() -> dict[ValidationDimension, float]:
    """Return sample scores across all five validation dimensions."""
    return {
        ValidationDimension.STRUCTURAL: 95.0,
        ValidationDimension.SEMANTIC: 88.0,
        ValidationDimension.QUANTITATIVE: 72.0,
        ValidationDimension.BEHAVIORAL: 90.0,
        ValidationDimension.SECURITY: 100.0,
    }
