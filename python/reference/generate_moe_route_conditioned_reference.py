#!/usr/bin/env python3
"""Independent CPU/FP32 expert equations on an authenticated HF checkpoint.

Only expert IDs are conditioned on production. Inputs and weights come from the
HF pack and exact GGUF, never from production activations or expert outputs.
The output is an unweighted [expert, row, hidden] bank, so the consumer must
separately prove top-k selection and normalized routing weights. This additive
oracle retains the original HF comparison. A separately authenticated terminal
normalization publication permits the consumer to reconstruct only the bounded
combined/residual/norm suffix; it does not author logits or recursive inputs.

Cache identity covers every small input and every split GGUF's file identity;
it never hashes model payloads. Callers hold ReferenceGenerationLease while
checking/publishing. A committed but malformed cache is an error, not a reason
to silently regenerate evidence. Metadata is published last after fsync.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np


SCHEMA = 1
OPERANDS = ("FFN_NORM", "MOE_ROUTING_INDICES", "MOE_ROUTING_WEIGHTS",
            "MOE_ROUTER_OUTPUT", "MOE_EXPERT_OUTPUT")


def model_identity(model: Path) -> list[dict]:
    """Stat every split shard without reading or hashing model payloads."""
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", model.name)
    shards = [model] if match is None else [
        model.with_name(f"{match[1]}-{index:05d}-of-{int(match[3]):05d}.gguf")
        for index in range(1, int(match[3]) + 1)
    ]
    if not shards:
        raise ValueError("empty GGUF shard set")
    return [dict(name=path.name, size=(stat := path.stat()).st_size,
                 device=stat.st_dev, inode=stat.st_ino,
                 mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
            for path in shards]


def load_operands(directory: Path, prefix: str) -> dict[str, np.ndarray]:
    """Read only named immutable HF operands, rejecting malformed geometry."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", prefix):
        raise ValueError("reference prefix must be one checkpoint basename")
    operands = {name: np.load(directory / f"{prefix}{name}.npy", allow_pickle=False)
                for name in OPERANDS}
    hidden = operands["FFN_NORM"]
    if hidden.dtype != np.float32 or hidden.ndim != 2 or 0 in hidden.shape:
        raise ValueError("HF expert input must be nonempty rank-2 FP32")
    rows = hidden.shape[0]
    for name, value in operands.items():
        if value.ndim != 2 or value.shape[0] != rows or not np.isfinite(value).all():
            raise ValueError(f"invalid HF operand {name}")
    indices = operands["MOE_ROUTING_INDICES"]
    weights = operands["MOE_ROUTING_WEIGHTS"]
    router = operands["MOE_ROUTER_OUTPUT"]
    if (weights.shape != indices.shape or indices.shape[1] == 0
            or operands["MOE_EXPERT_OUTPUT"].shape != hidden.shape
            or not np.equal(indices, np.floor(indices)).all()
            or (indices < 0).any() or (indices >= router.shape[1]).any()
            or (router < 0).any() or (weights < 0).any()):
        raise ValueError("invalid HF routing geometry or values")
    ids = indices.astype(np.int64)
    if any(len(set(row)) != len(row) for row in ids):
        raise ValueError("duplicate HF expert IDs")
    selected = np.take_along_axis(router, ids, axis=1).astype(np.float64)
    sums = selected.sum(axis=1, keepdims=True)
    if (sums <= 0).any() or not np.allclose(weights, selected / sums, rtol=0, atol=2e-6):
        raise ValueError("HF routing weights are not normalized selected probabilities")
    return operands


def expert_matrix(parser, tensor, expert: int) -> np.ndarray:
    """Decode one whole expert with the canonical all-format GGUF dispatcher.

    GGUFParser already supplies PyTorch dimension order and owns split-shard
    mmap lifetimes. Slicing only the expert axis preserves the codebook blocks;
    never transpose/repack or assume an activation/weight quantization format.
    """
    from python.reference.loaders.dequantize import dequantize

    if len(tensor.shape) != 3 or not 0 <= expert < tensor.shape[0]:
        raise ValueError("expert tensor must have a valid leading expert axis")
    raw = parser.read_tensor_data(tensor)
    if len(raw) % tensor.shape[0]:
        raise ValueError("packed expert tensor has a fractional expert stride")
    stride = len(raw) // tensor.shape[0]
    return dequantize(raw[expert * stride:(expert + 1) * stride],
                      tensor.type, tensor.shape[1:])


def evaluate_experts(parser, tensor_prefix: str, ids: list[int],
                     hidden: np.ndarray) -> np.ndarray:
    """Evaluate HF's SwiGLU equation independently in CPU PyTorch FP32.

    One expert's three matrices are live at a time; no full model, GPU state,
    native output, native activation, or backend kernel is imported here.
    """
    import torch
    import torch.nn.functional as functional

    tensors = {tensor.name: tensor for tensor in parser.tensors}
    gate, up, down = [tensors[f"{tensor_prefix}.ffn_{name}_exps.weight"]
                      for name in ("gate", "up", "down")]
    expected = (gate.shape[0], hidden.shape[1], gate.shape[1])
    if (len(gate.shape) != 3 or up.shape != gate.shape or down.shape != expected
            or gate.shape[2] != hidden.shape[1]):
        raise ValueError("GGUF SwiGLU expert geometry disagrees with HF input")
    x = torch.from_numpy(hidden)
    outputs = []
    with torch.inference_mode():
        for expert in ids:
            matrices = [expert_matrix(parser, tensor, expert) for tensor in (gate, up, down)]
            # F32 decoding can retain a read-only mmap view. Give PyTorch a
            # writable owner for that case; quantized decoders already own
            # their FP32 buffers, so they need no additional copy.
            g, u, d = [torch.from_numpy(value if value.flags.writeable else value.copy())
                       for value in matrices]
            value = functional.linear(
                functional.silu(functional.linear(x, g)) * functional.linear(x, u), d)
            outputs.append(value.numpy().copy())
    result = np.stack(outputs)
    if result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("independent FP32 expert equation is nonfinite")
    return result


def authenticate_canonical_sum(bank: np.ndarray, ids: list[int], operands: dict) -> None:
    """Prove the same bank reproduces the existing HF routed output tightly."""
    hidden = operands["FFN_NORM"]
    if bank.dtype != np.float32 or bank.shape != (len(ids), *hidden.shape) or not np.isfinite(bank).all():
        raise ValueError("invalid independent expert bank")
    slots = {expert: slot for slot, expert in enumerate(ids)}
    actual = np.zeros_like(hidden, dtype=np.float64)
    for row, routes in enumerate(operands["MOE_ROUTING_INDICES"]):
        for index, expert in enumerate(routes):
            actual[row] += bank[slots[int(expert)], row].astype(np.float64) * float(
                operands["MOE_ROUTING_WEIGHTS"][row, index])
    reference = operands["MOE_EXPERT_OUTPUT"].astype(np.float64)
    error = np.linalg.norm(actual - reference)
    norm = np.linalg.norm(reference)
    if (error != 0 if norm == 0 else error / norm > 2e-5):
        raise ValueError("independent expert bank does not reproduce canonical HF output")


def atomic_publish(path: Path, writer) -> None:
    """Fsync one temporary file and atomically install it in its owned directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalization_parameters(parser, tensor_prefix: str) -> np.ndarray:
    """Decode model-owned HF RMS parameters through the canonical format dispatcher.

    Qwen's GGUF stores gamma, while the HF loader stores gamma-1 and its RMS
    operation adds one. Preserve that FP32 round trip, independent of backend
    and of the expert weight format. The last packed scalar is model epsilon.
    """
    from python.reference.loaders.dequantize import dequantize

    tensor = next(t for t in parser.tensors
                  if t.name == f"{tensor_prefix}.nextn.shared_head_norm.weight")
    if len(tensor.shape) != 1 or not tensor.shape[0]:
        raise ValueError("terminal HF RMS scale must be a nonempty vector")
    scale = dequantize(parser.read_tensor_data(tensor), tensor.type, tensor.shape)
    scale = (scale.astype(np.float32) - np.float32(1)) + np.float32(1)
    architecture = parser.metadata["general.architecture"]
    epsilon = parser.metadata[f"{architecture}.attention.layer_norm_rms_epsilon"]
    return np.concatenate((scale, np.array([epsilon], np.float32)))


def authenticate_normalization(parameters: np.ndarray, residual: np.ndarray,
                               normalized: np.ndarray) -> None:
    """Prove GGUF parameters independently reconstruct the original HF RMS row."""
    if (residual.dtype != np.float32 or residual.ndim != 2 or 0 in residual.shape
            or normalized.dtype != np.float32 or normalized.shape != residual.shape
            or parameters.dtype != np.float32 or parameters.shape != (residual.shape[1] + 1,)
            or not all(np.isfinite(value).all() for value in (parameters, residual, normalized))
            or parameters[-1] <= 0):
        raise ValueError("invalid HF normalization geometry or parameters")
    x = residual.astype(np.float64)
    inverse = (1 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + parameters[-1])).astype(np.float32)
    actual = (residual * inverse) * parameters[:-1]
    error = np.linalg.norm(actual.astype(np.float64) - normalized)
    norm = np.linalg.norm(normalized.astype(np.float64))
    if (error != 0 if norm == 0 else error / norm > 2e-5):
        raise ValueError("GGUF normalization does not reproduce canonical HF output")


def generate_normalization(model: Path, reference_dir: Path, prefix: str,
                           tensor_prefix: str, output: Path, evaluator=None) -> bool:
    """Publish a bounded independent norm receipt without regenerating expert banks.

    The cache binds exact HF operands, all GGUF shard identities and parameter
    bytes. The optional evaluator is only a tiny device-free test seam. No
    production activation/output or fitted normalization enters this interface.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", prefix) or not re.fullmatch(r"blk\.\d+", tensor_prefix):
        raise ValueError("invalid normalization checkpoint identity")
    operands = [np.load(reference_dir / f"{prefix}{suffix}.npy", allow_pickle=False)
                for suffix in ("FFN_RESIDUAL", "FINAL_NORM")]
    identity = dict(schema=1, equation="qwen-hf-1p-rms", model=model_identity(model),
                    prefix=prefix, tensor_prefix=tensor_prefix,
                    operands=[dict(shape=list(value.shape), dtype=str(value.dtype),
                                   sha256=hashlib.sha256(value.tobytes()).hexdigest())
                              for value in operands])
    metadata = output.with_suffix(".json")
    if metadata.exists():
        stored = json.loads(metadata.read_text())
        if stored.get("identity") == identity:
            parameters = np.load(output, allow_pickle=False)
            if hashlib.sha256(parameters.tobytes()).hexdigest() != stored.get("parameters_sha256"):
                raise ValueError("committed normalization checksum mismatch")
            authenticate_normalization(parameters, *operands)
            return True
    if evaluator is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from python.reference.loaders.gguf_parser import GGUFParser
        with GGUFParser(model) as parser:
            parser.parse()
            parameters = normalization_parameters(parser, tensor_prefix)
    else:
        parameters = evaluator()
    authenticate_normalization(parameters, *operands)
    if model_identity(model) != identity["model"]:
        raise ValueError("GGUF identity changed while generating normalization")
    atomic_publish(output, lambda stream: np.save(stream, parameters, allow_pickle=False))
    receipt = dict(identity=identity, parameters_sha256=hashlib.sha256(parameters.tobytes()).hexdigest())
    atomic_publish(metadata, lambda stream: stream.write(json.dumps(receipt, sort_keys=True).encode()))
    return False


def generate(model: Path, reference_dir: Path, prefix: str, tensor_prefix: str,
             ids: list[int], output: Path, evaluator=None) -> bool:
    """Validate/reuse or publish one immutable bank; return whether it was reused.

    The optional evaluator is a device-free test seam, not a command-line mode.
    Changed small-input/model identity invalidates the old cache. Matching
    metadata plus corrupted/missing bank bytes is a fatal publication failure.
    """
    operands = load_operands(reference_dir, prefix)
    if (not ids or ids != sorted(set(ids)) or ids[0] < 0
            or ids[-1] >= operands["MOE_ROUTER_OUTPUT"].shape[1]
            or not set(operands["MOE_ROUTING_INDICES"].astype(int).flat) <= set(ids)
            or not re.fullmatch(r"blk\.\d+", tensor_prefix)):
        raise ValueError("bank requires sorted unique valid IDs including canonical HF routes")
    identity = dict(schema=SCHEMA, model=model_identity(model), prefix=prefix,
                    tensor_prefix=tensor_prefix, expert_ids=ids,
                    operands={name: dict(shape=list(value.shape), dtype=str(value.dtype),
                              sha256=hashlib.sha256(value.tobytes()).hexdigest())
                              for name, value in operands.items()})
    metadata = output.with_suffix(".json")
    if metadata.exists():
        stored = json.loads(metadata.read_text())
        if stored.get("identity") == identity:
            bank = np.load(output, allow_pickle=False)
            if hashlib.sha256(bank.tobytes()).hexdigest() != stored.get("bank_sha256"):
                raise ValueError("committed expert bank checksum mismatch")
            authenticate_canonical_sum(bank, ids, operands)
            return True

    if evaluator is None:
        # Import the heavyweight reference stack only on an actual cache miss.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        import torch
        from python.reference.loaders.gguf_parser import GGUFParser

        # Use physical cores available to this generator, never inherited CTest
        # single-thread settings or every SMT sibling as an independent core.
        import psutil
        torch.set_num_threads(min(len(os.sched_getaffinity(0)), psutil.cpu_count(logical=False) or 1))
        parser = GGUFParser(model)
        parser.parse()
        bank = evaluate_experts(parser, tensor_prefix, ids, operands["FFN_NORM"])
    else:
        bank = evaluator(operands["FFN_NORM"], ids)
    authenticate_canonical_sum(bank, ids, operands)
    if model_identity(model) != identity["model"]:
        raise ValueError("GGUF identity changed while generating the expert bank")
    atomic_publish(output, lambda stream: np.save(stream, bank, allow_pickle=False))
    receipt = dict(identity=identity, bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest())
    atomic_publish(metadata, lambda stream: stream.write(json.dumps(receipt, sort_keys=True).encode()))
    return False


def main() -> None:
    """Generate a bounded real-weight bank while the C++ caller owns its lease."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--reference-prefix", required=True)
    parser.add_argument("--tensor-prefix", required=True)
    parser.add_argument("--expert-ids", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reused = generate(args.model, args.reference_dir, args.reference_prefix,
                      args.tensor_prefix, [int(x) for x in args.expert_ids.split(",")], args.output)
    normalization_reused = generate_normalization(
        args.model, args.reference_dir, args.reference_prefix, args.tensor_prefix,
        args.output.with_suffix(".norm.npy"))
    print(f"route_conditioned_hf schema={SCHEMA} reused={int(reused)} output={args.output}")
    print(f"route_conditioned_hf_normalization reused={int(normalization_reused)}")


if __name__ == "__main__":
    main()
