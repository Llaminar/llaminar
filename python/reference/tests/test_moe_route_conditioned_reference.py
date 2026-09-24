"""Device-free proof of independent expert slicing, cache identity, and fail-closed publication.

No production tensors or models are loaded. Tiny FP32 equations exercise the
same generator as real checkpoints, and every GGUF format exercises the shared
expert-axis slice without introducing a format-specific production oracle.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from python.reference.generate_moe_route_conditioned_reference import (
    authenticate_canonical_sum, evaluate_experts, expert_matrix, generate,
    load_operands, model_identity,
    authenticate_normalization, generate_normalization, normalization_parameters,
)
from python.reference.loaders.gguf_parser import GGUFParser, GGUFTensorInfo, GGUFTensorType


@pytest.fixture
def reference(tmp_path):
    """Create a two-row canonical HF pack and its independent unweighted bank."""
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"test descriptor only")
    bank = np.array([[[1., 0], [2, 0]], [[0, 1], [0, 2]], [[0, -1], [0, -2]]], dtype=np.float32)
    router = np.tile(np.array([.5, .251, .249], np.float32), (2, 1))
    weights = router[:, :2] / router[:, :2].sum(1, keepdims=True)
    output = np.stack([bank[0, row] * weights[row, 0] + bank[1, row] * weights[row, 1] for row in range(2)])
    operands = dict(FFN_NORM=np.array([[1, 2], [2, 4]], np.float32),
                    MOE_ROUTING_INDICES=np.tile(np.array([0, 1], np.int64), (2, 1)),
                    MOE_ROUTING_WEIGHTS=weights, MOE_ROUTER_OUTPUT=router,
                    MOE_EXPERT_OUTPUT=output)
    for name, value in operands.items():
        np.save(tmp_path / f"row_{name}.npy", value)
    return model, bank, tmp_path / "cache.npy", operands


def test_cache_reuse_and_small_input_identity(reference):
    model, bank, output, _ = reference
    calls = []
    def evaluator(hidden, ids):
        calls.append((hidden.copy(), list(ids)))
        return bank.copy()
    args = (model, model.parent, "row_", "blk.48", [0, 1, 2], output)
    assert not generate(*args, evaluator=evaluator)
    assert generate(*args, evaluator=evaluator)
    assert len(calls) == 1
    hidden = np.load(model.parent / "row_FFN_NORM.npy")
    np.save(model.parent / "row_FFN_NORM.npy", hidden * 2)
    assert not generate(*args, evaluator=evaluator)
    assert len(calls) == 2
    model.write_bytes(b"changed descriptor")
    assert not generate(*args, evaluator=evaluator)
    assert len(calls) == 3


@pytest.mark.parametrize("defect", ["bank", "missing_bank", "metadata"])
def test_committed_corrupt_cache_is_not_silently_regenerated(reference, defect):
    model, bank, output, _ = reference
    args = (model, model.parent, "row_", "blk.48", [0, 1, 2], output)
    generate(*args, evaluator=lambda *_: bank)
    if defect == "bank":
        np.save(output, bank * 2)
    elif defect == "missing_bank":
        output.unlink()
    else:
        output.with_suffix(".json").write_text("{broken")
    with pytest.raises((ValueError, FileNotFoundError)):
        generate(*args, evaluator=lambda *_: pytest.fail("must not regenerate committed corruption"))


@pytest.mark.parametrize("ids", [[], [0, 2], [0, 1, 1], [1, 0, 2], [-1, 0, 1], [0, 1, 3]])
def test_invalid_bank_identity_fails(reference, ids):
    model, bank, output, _ = reference
    with pytest.raises(ValueError, match="sorted unique"):
        generate(model, model.parent, "row_", "blk.48", ids, output, evaluator=lambda *_: bank)


def test_canonical_equation_and_operand_corruption_fail(reference):
    model, bank, output, operands = reference
    with pytest.raises(ValueError, match="canonical HF"):
        authenticate_canonical_sum(bank * 2, [0, 1, 2], operands)
    bank[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid independent"):
        authenticate_canonical_sum(bank, [0, 1, 2], operands)
    np.save(model.parent / "row_MOE_ROUTING_WEIGHTS.npy", np.full((2, 2), .5, np.float32))
    with pytest.raises(ValueError, match="not normalized"):
        load_operands(model.parent, "row_")
    with pytest.raises(ValueError, match="basename"):
        load_operands(model.parent, "../row_")


def test_split_model_identity_includes_each_shard(tmp_path):
    paths = [tmp_path / f"model-{index:05d}-of-00002.gguf" for index in (1, 2)]
    for path in paths:
        path.write_bytes(b"fixture")
    before = model_identity(paths[0])
    paths[1].write_bytes(b"changed second shard")
    assert model_identity(paths[0]) != before
    paths[1].unlink()
    with pytest.raises(FileNotFoundError):
        model_identity(paths[0])


@pytest.mark.parametrize("dtype", list(GGUFTensorType))
def test_expert_slice_forwards_every_format_without_repacking(monkeypatch, dtype):
    # A format-agnostic stride contract: the canonical dequantizer, not this
    # helper, decides format support and byte interpretation.
    import python.reference.loaders.dequantize as module
    raw = memoryview(bytes(range(96)))
    tensor = GGUFTensorInfo("expert", [3, 4, 8], dtype, 0)
    def decoder(data, observed_dtype, shape):
        assert bytes(data) == bytes(raw[32:64])
        assert observed_dtype == dtype
        assert shape == (4, 8)
        return np.ones(shape, np.float32)
    monkeypatch.setattr(module, "dequantize", decoder)
    parser = SimpleNamespace(read_tensor_data=lambda _: raw)
    assert expert_matrix(parser, tensor, 1).shape == (4, 8)
    with pytest.raises(ValueError, match="leading expert axis"):
        expert_matrix(parser, tensor, 3)


@pytest.mark.parametrize("dtype", [GGUFTensorType.F32, GGUFTensorType.F16, GGUFTensorType.BF16])
def test_floating_expert_equation_matches_hf_swiglu(dtype):
    tensors, payloads, decoded = [], {}, {}
    for name, values in {
        "gate": np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 8,
        "up": np.arange(12, 24, dtype=np.float32).reshape(3, 2, 2) / 8,
        "down": np.arange(24, 36, dtype=np.float32).reshape(3, 2, 2) / 8,
    }.items():
        tensor = GGUFTensorInfo(f"blk.1.ffn_{name}_exps.weight", [3, 2, 2], dtype, 0)
        tensors.append(tensor)
        if dtype == GGUFTensorType.F32:
            raw = values.tobytes()
        elif dtype == GGUFTensorType.F16:
            raw = values.astype(np.float16).tobytes()
        else:
            raw = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
        payloads[tensor.name] = memoryview(raw)
        decoded[name] = values
    parser = SimpleNamespace(tensors=tensors, read_tensor_data=lambda t: payloads[t.name])
    hidden = np.array([[.25, -.75], [.5, .5]], np.float32)
    bank = evaluate_experts(parser, "blk.1", [0, 1, 2], hidden)
    x = torch.from_numpy(hidden)
    for expert in range(3):
        gate = x @ torch.from_numpy(decoded["gate"][expert]).T
        up = x @ torch.from_numpy(decoded["up"][expert]).T
        expected = (torch.nn.functional.silu(gate) * up) @ torch.from_numpy(decoded["down"][expert]).T
        np.testing.assert_allclose(bank[expert], expected.numpy(), rtol=1e-6, atol=1e-6)


@pytest.fixture
def norm_reference(tmp_path):
    """Independent tiny HF equation with two rows and explicit model parameters."""
    model = tmp_path / "fixture.gguf"
    model.write_bytes(b"normalization descriptor")
    parameters = np.array([1., .75, 1e-6], np.float32)
    residual = np.array([[1., 2.], [3., -4.]], np.float32)
    x = torch.from_numpy(residual)
    normalized = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + parameters[-1])
                  * torch.from_numpy(parameters[:-1])).numpy()
    np.save(tmp_path / "row_FFN_RESIDUAL.npy", residual)
    np.save(tmp_path / "row_FINAL_NORM.npy", normalized)
    return model, parameters, residual, normalized, tmp_path / "cache.norm.npy"


def test_normalization_cache_is_independent_and_authenticates_hf(norm_reference):
    model, parameters, residual, normalized, output = norm_reference
    args = model, model.parent, "row_", "blk.48", output
    assert not generate_normalization(*args, evaluator=lambda: parameters)
    assert generate_normalization(*args, evaluator=lambda: pytest.fail("cache hit must not reload"))
    authenticate_normalization(parameters, residual, normalized)
    # A changed canonical operand cannot reuse a stale parameter receipt or
    # silently accept a plausible-looking output from another normalization.
    np.save(model.parent / "row_FINAL_NORM.npy", normalized * 2)
    with pytest.raises(ValueError, match="canonical HF"):
        generate_normalization(*args, evaluator=lambda: parameters)


@pytest.mark.parametrize("defect", ["parameters", "missing", "epsilon", "scale", "geometry", "nonfinite"])
def test_normalization_corruption_fails_closed(norm_reference, defect):
    model, parameters, residual, normalized, output = norm_reference
    args = model, model.parent, "row_", "blk.48", output
    generate_normalization(*args, evaluator=lambda: parameters)
    if defect == "parameters":
        np.save(output, parameters * 2)
    elif defect == "missing":
        output.unlink()
    else:
        bad = parameters.copy()
        if defect == "epsilon": bad[-1] = 0
        if defect == "scale": bad[0] *= 2
        if defect == "geometry": bad = bad[:-1]
        if defect == "nonfinite": bad[0] = np.nan
        with pytest.raises(ValueError):
            authenticate_normalization(bad, residual, normalized)
        return
    with pytest.raises((ValueError, FileNotFoundError)):
        generate_normalization(*args, evaluator=lambda: pytest.fail("must not mask corrupted receipt"))


@pytest.mark.parametrize("dtype", list(GGUFTensorType))
def test_norm_parameters_use_canonical_dequantizer_for_every_format(monkeypatch, dtype):
    import python.reference.loaders.dequantize as module
    raw = memoryview(bytes(range(32)))
    tensor = GGUFTensorInfo("blk.48.nextn.shared_head_norm.weight", [2], dtype, 0)
    gamma = np.array([.03, .75], np.float32)
    def decoder(data, observed_dtype, shape):
        assert bytes(data) == bytes(raw)
        assert observed_dtype == dtype
        assert shape == (2,)
        return gamma
    monkeypatch.setattr(module, "dequantize", decoder)
    parser = SimpleNamespace(tensors=[tensor], read_tensor_data=lambda _: raw,
                             metadata={"general.architecture": "qwen35moe",
                                       "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6})
    actual = normalization_parameters(parser, "blk.48")
    np.testing.assert_array_equal(actual[:-1], (gamma - np.float32(1)) + np.float32(1))
    assert actual[-1] == np.float32(1e-6)
