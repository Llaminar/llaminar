"""Public exact-only workflow boundaries, with historical readers preserved.

These tests call the real argument parsers in process. They never compile,
load accelerators, fit policies or manufacture a bypass into retired modes.
Internal reader/emitter arithmetic retains its separate existing unit tests.
"""
from __future__ import annotations

import contextlib
from functools import lru_cache
import importlib.util
import importlib
import io
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[4]
KERNELS = ROOT / "tests/v2/performance/kernels"
sys.path.insert(0, str(KERNELS))
ANALYZERS = (
    "cpu/analyze_cpu_native_quantised_decode_trainer.py",
    "cpu/analyze_cpu_native_quantised_verifier_trainer.py",
    "cpu/analyze_cpu_native_quantised_prefill_trainer.py",
    "cuda/gemm/analyze_cuda_native_quantised_decode_trainer.py",
    "rocm/analyze_rocm_native_quantised_decode_trainer.py",
)


@lru_cache(maxsize=None)
def load(relative: str):
    """Import actual source owners without entering their executable workflow."""
    path = KERNELS / relative
    if relative.startswith("native_quantised_dispatch/"):
        return importlib.import_module("native_quantised_dispatch." + path.stem)
    name = "public_exact_boundary_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", ANALYZERS)
def test_analyzer_generic_fit_is_retired_before_inputs(relative):
    """An explicit old flag cannot reach input I/O, discovery or a learner."""
    module = load(relative)
    errors = io.StringIO()
    with patch.object(sys, "argv", [relative, "--freeze-generic"]), contextlib.redirect_stderr(errors):
        with pytest.raises(SystemExit) as caught:
            module.main()
    assert caught.value.code == 2
    assert "generic fitting/CV/seal controls are retired" in errors.getvalue()


@pytest.mark.parametrize("relative", ANALYZERS)
def test_analyzer_help_advertises_exact_and_readers_only(relative):
    """Keeping internal historical code does not advertise executable fitting."""
    module = load(relative)
    output = io.StringIO()
    with patch.object(sys, "argv", [relative, "--help"]), contextlib.redirect_stdout(output):
        with pytest.raises(SystemExit) as caught:
            module.main()
    assert caught.value.code == 0
    text = output.getvalue()
    assert "--freeze-generic" not in text
    assert "--certify-generic" not in text
    assert "--generic-max-leaves" not in text
    assert "--plan-sealed-pairs" not in text
    assert "--frozen-policy-json" not in text
    assert "Exact overlay" in text


@pytest.mark.parametrize("relative", ANALYZERS)
def test_implicit_generic_emission_is_retired(relative):
    """A default old analyzer invocation cannot silently generate generic rules."""
    module = load(relative)
    errors = io.StringIO()
    with patch.object(sys, "argv", [relative, "--output", "/must-not-write"]), contextlib.redirect_stderr(errors):
        with pytest.raises(SystemExit) as caught:
            module.main()
    assert caught.value.code == 2
    assert "implicit generic generation is retired" in errors.getvalue()


@pytest.mark.parametrize("relative", (
    "native_quantised_dispatch/paired_requests.py",
    "native_quantised_dispatch/Perf__NativeQuantisedPolicyCrossValidation.py",
    "native_quantised_dispatch/Perf__NativeQuantisedPolicyScorer.py",
))
def test_legacy_pipeline_modules_cannot_execute_new_fits(relative):
    """Former executable paired/CV transactions are retired, not hidden aliases."""
    module = load(relative)
    errors = io.StringIO()
    with patch.object(sys, "argv", [relative, "--output", "/must-not-write"]), contextlib.redirect_stderr(errors):
        assert module.main() == 2
    assert "workflow is retired" in errors.getvalue()
    assert "exact overlays" in errors.getvalue()


def test_historical_artifact_and_exact_emitters_remain_importable():
    """The cutover preserves import-only historical and current exact owners."""
    for relative in ANALYZERS:
        module = load(relative)
        assert callable(module.main)
    for relative in ANALYZERS[:3]:
        module = load(relative)
        assert callable(module.validate_complete)
        assert callable(module.select_entries)
        assert callable(module.generate_include)
