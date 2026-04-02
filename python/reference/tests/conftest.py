"""
Pytest conftest for the python/reference test suite.

Provides shared fixtures and marks for the PyTorch reference implementation tests.
Tests are categorized as:
  - Unit tests: Fast, no model/file dependencies (test_reference.py, test_dequantize.py)
  - Integration tests: Require GGUF model files on disk (test_gguf_*.py, test_dequantization_validation.py)
"""

import sys
from pathlib import Path

import pytest

# Ensure the workspace root is on sys.path so `from python.reference import ...` works
# regardless of the working directory pytest is invoked from.
_workspace_root = Path(__file__).resolve().parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def models_dir() -> Path:
    """Path to the models/ directory at the workspace root."""
    return _workspace_root / "models"


@pytest.fixture
def qwen2_q4_model(models_dir) -> Path:
    """Path to the Qwen2.5-0.5B Q4_0 GGUF model, skipping if unavailable."""
    p = models_dir / "qwen2.5-0.5b-instruct-q4_0.gguf"
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    return p


@pytest.fixture
def qwen2_q8_model(models_dir) -> Path:
    """Path to the Qwen2.5-0.5B Q8_0 GGUF model, skipping if unavailable."""
    p = models_dir / "qwen2.5-0.5b-instruct-q8_0.gguf"
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    return p


@pytest.fixture
def llama_q4_model(models_dir) -> Path:
    """Path to the Llama-3.2-1B Q4_0 GGUF model, skipping if unavailable."""
    p = models_dir / "Llama-3.2-1B-Instruct-Q4_0.gguf"
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    return p


@pytest.fixture
def qwen3_q8_model(models_dir) -> Path:
    """Path to the Qwen3-0.6B Q8_0 GGUF model, skipping if unavailable."""
    p = models_dir / "Qwen3-0.6B-Q8_0.gguf"
    if not p.exists():
        pytest.skip(f"Model not found: {p}")
    return p
