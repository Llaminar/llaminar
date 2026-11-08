"""Exact CPU/PyTorch oracle for the pinned ruquant W4A4/W4A4H export.

The graph boundary is FP32, but matrix operands are signed int4. A per-token
unit, group-64 asymmetric activation scales, integer zero-point correction,
and a per-group arithmetic shift precede the FP32 epilogue. Dequantizing the
weights and calling ordinary FP32 linear would implement different arithmetic.

W4A4H applies the stored block-diagonal Paley/Sylvester transform before the
activation quantizer. Its accumulation order is explicit; torch.matmul is not
a substitute for its ordered FP32 Paley sum. Integer group dots may use FP32
BLAS because every product and every partial sum is an exactly representable
integer with magnitude at most 4096. This accelerates the independent oracle
without importing production kernels or their reduction policy.

Source contracts are pinned below. No environment variable may change this
reference's group size, clipping or integer shift after weight admission.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch


MODEL_REPOSITORY = "raspbfox/Qwen3.8-27B-W4A4-GGUF"
MODEL_REVISION = "174067dc0522d3c69912487e799b4e9541cb296c"
MODEL_SHA256 = "b1015b212068a2838f23a68985065352e337d9e4e0f40abab3832e610a6a5ef8"
KERNEL_REVISION = "bfc4c5dc3c96df652f42924370ef49b728d585ad"
CONVERTER_REVISION = "7851ba11eddf3de44adc9dba0ccb24472b9ba30e"
ARITHMETIC_VERSION = "ruquant-w4a4-g64-shift4-v1"
TRANSFORMERS_REVISION = "89e65121e562303cd909bde6e7b395da5d0db30b"
TRANSFORMERS_QWEN35_SHA256 = "aa81cfc05324e41536f9e6b43852f92463b5375587e4f12462c8ef41996d7946"
PYTORCH_VERSION = "2.15.0+cpu"
BLOCK_VALUES = 256
BLOCK_BYTES = 144
GROUP_VALUES = 64
TERM_SHIFT = 4
ACTIVATION_MULTIPLIER_MAX = 255


@lru_cache(maxsize=64)
def hadamard_factor(length: int) -> tuple[int, int]:
    """Return the first admitted Paley order and power-of-two Sylvester order.

    The ascending Paley search is part of arithmetic identity: a different
    factorization changes the transform and can change quantization decisions.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("Hadamard length must be a positive integer")
    for paley in range(1, min(128, length) + 1):
        prime = paley - 1
        if paley > 1 and (
            paley % 4 != 0
            or any(prime % divisor == 0 for divisor in range(2, math.isqrt(prime) + 1))
        ):
            continue
        if length % paley:
            continue
        sylvester = length // paley
        if sylvester & (sylvester - 1) == 0:
            return paley, sylvester
    raise ValueError(f"Hadamard length {length} has no supported Paley/Sylvester factorization")


@lru_cache(maxsize=16)
def _paley_signs(order: int) -> tuple[tuple[int, ...], ...]:
    """Construct immutable Paley-I signs; callers cannot mutate a shared tensor."""
    prime = order - 1
    residues = {value * value % prime for value in range(1, prime)}
    rows = []
    for row in range(order):
        values = []
        for column in range(order):
            if row == column:
                sign = 1
            elif row == 0:
                sign = 1
            elif column == 0:
                sign = -1
            else:
                sign = 1 if (column - row) % prime in residues else -1
            values.append(sign)
        rows.append(tuple(values))
    return tuple(rows)


def _require_fp32_cpu(values: torch.Tensor) -> None:
    """Reject device, precision and nonfinite inputs outside the oracle contract."""
    if values.device.type != "cpu" or values.dtype != torch.float32 or torch.is_autocast_enabled('cpu'):
        raise ValueError("W4A4 reference requires CPU FP32 activations without autocast")
    if values.ndim < 1 or not torch.isfinite(values).all().item():
        raise ValueError("W4A4 activations must have a finite, non-scalar shape")


def hadamard(values: torch.Tensor, rotation_length: int, *, transpose: bool = False) -> torch.Tensor:
    """Rotate independent contiguous blocks in the native FP32 arithmetic order.

    The input is not modified. Sylvester butterflies visit increasing strides;
    Paley columns accumulate source rows from zero upward with explicit add or
    subtract. Normalization rounds sqrt and reciprocal separately to FP32.
    """
    _require_fp32_cpu(values)
    paley, sylvester = hadamard_factor(rotation_length)
    if values.shape[-1] == 0 or values.shape[-1] % rotation_length:
        raise ValueError("Hadamard length must divide the complete activation row")
    transformed = values.contiguous().clone().reshape(-1, paley, sylvester)
    stride = 1
    while stride < sylvester:
        pairs = transformed.reshape(-1, paley, sylvester // (2 * stride), 2, stride)
        lower, upper = pairs[..., 0, :].clone(), pairs[..., 1, :].clone()
        pairs[..., 0, :] = lower + upper
        pairs[..., 1, :] = lower - upper
        stride *= 2
    if paley > 1:
        signs = torch.tensor(_paley_signs(paley), dtype=torch.int8)
        result = torch.zeros_like(transformed)
        for source_row in range(paley):
            positive = (signs[:, source_row] if transpose else signs[source_row, :]) > 0
            column = transformed[:, source_row:source_row + 1, :]
            result = torch.where(positive.reshape(1, paley, 1), result + column, result - column)
        transformed = result
    scale = torch.tensor(1.0, dtype=torch.float32) / torch.sqrt(torch.tensor(rotation_length, dtype=torch.float32))
    return (transformed * scale).reshape(values.shape)


@dataclass(frozen=True)
class A4Operands:
    """One independently prepared batch: signed codes plus row/group metadata."""

    codes: torch.Tensor
    unit: torch.Tensor
    multipliers: torch.Tensor
    zero_points: torch.Tensor


def quantize_activations(values: torch.Tensor, clip: float = 1.0) -> A4Operands:
    """Prepare exact group-64 A4 operands, preserving all FP32 rounding points.

    Each group range includes zero. One token-wide maximum selects the shared
    unit; ceil selects integer group multipliers and ties-to-even selects zero
    points/codes. The all-zero row uses range two, exactly as the pinned source.
    """
    _require_fp32_cpu(values)
    width = values.shape[-1]
    if width == 0 or width % GROUP_VALUES:
        raise ValueError("A4 activation width must be a positive multiple of 64")
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("A4 clipping must be finite and positive")
    groups = values.reshape(-1, width // GROUP_VALUES, GROUP_VALUES)
    lower = torch.minimum(groups.amin(dim=-1), torch.zeros((), dtype=torch.float32)) * clip
    upper = torch.maximum(groups.amax(dim=-1), torch.zeros((), dtype=torch.float32)) * clip
    ranges = upper - lower
    widest = ranges.amax(dim=-1, keepdim=True)
    widest = torch.where(widest == 0, torch.full_like(widest, 2.0), widest)
    unit = widest / float(15 * ACTIVATION_MULTIPLIER_MAX)
    multipliers = torch.ceil((ranges * float(ACTIVATION_MULTIPLIER_MAX)) / widest).clamp(1, ACTIVATION_MULTIPLIER_MAX)
    inverse = 1.0 / (multipliers * unit)
    zero_points = torch.round(-lower * inverse).clamp(0, 15)
    codes = (torch.round(groups * inverse.unsqueeze(-1)) + zero_points.unsqueeze(-1)).clamp(0, 15) - 8
    if not torch.isfinite(unit).all().item() or not torch.isfinite(inverse).all().item():
        raise ValueError("A4 activation range is not representable by the FP32 policy")
    return A4Operands(codes.to(torch.int8), unit.squeeze(-1), multipliers.to(torch.int32), zero_points.to(torch.int32))


@dataclass(frozen=True)
class W4A4Weights:
    """Checked native source rows, retained without a dequantized FP32 matrix.

    The oracle owns its decoded int8 code copy. Unit scales stay FP32 and group
    multipliers remain integers; they must never be absorbed into FP16 scales.
    Stored rotation length and clip belong to matrix identity, not dispatch.
    """

    codes: torch.Tensor
    units: torch.Tensor
    multipliers: torch.Tensor
    type_id: int
    rotation_length: int
    clip: float

    @classmethod
    def from_bytes(cls, raw: bytes | memoryview, shape: tuple[int, int], type_id: int) -> "W4A4Weights":
        """Authenticate one complete GGUF matrix and decode its compact fields.

        ``shape`` is row-major (N,K). Source type 44 is W4A4 and 45 is W4A4H.
        Checked geometry excludes int32 overflow even for worst-case admitted
        codes/scales. Uniform repeated header fields are checked bitwise.
        """
        if type_id not in (44, 45) or len(shape) != 2:
            raise ValueError("W4A4 source requires type 44/45 and a rank-two matrix")
        rows, width = shape
        if rows <= 0 or width <= 0 or width % BLOCK_VALUES:
            raise ValueError("W4A4 matrix requires positive rows and K divisible by 256")
        max_group_term = (15 * ACTIVATION_MULTIPLIER_MAX * 8 * 15 * GROUP_VALUES) >> TERM_SHIFT
        if width // GROUP_VALUES * max_group_term > np.iinfo(np.int32).max:
            raise ValueError("W4A4 input width exceeds the proven int32 accumulation range")
        blocks = width // BLOCK_VALUES
        if len(raw) != rows * blocks * BLOCK_BYTES:
            raise ValueError("W4A4 payload must contain exactly the declared native matrix")
        source = np.frombuffer(raw, dtype=np.uint8).reshape(rows, blocks, BLOCK_BYTES)
        header = source[:, :, :16].copy().view('<u4').reshape(rows, blocks, 4)
        packed = source[:, :, 16:].copy()
        # Validation exceptions must not retain an exported mmap view through
        # their traceback. All fields below are owned before validation begins.
        del source
        units = header[:, :, 0].copy().view('<f4')
        if not np.isfinite(units).all() or (units < 0).any():
            raise ValueError("W4A4 weight units must be finite and nonnegative")
        if not np.all(header[:, :, 0] == header[:, :1, 0]):
            raise ValueError("W4A4 weight unit must be identical across each source row")
        rotation = int(header[0, 0, 2])
        if not np.all(header[:, :, 2] == rotation):
            raise ValueError("W4A4 stored row/rotation length must be uniform")
        if type_id == 45:
            hadamard_factor(rotation)
            if width % rotation:
                raise ValueError("W4A4H rotation does not divide the source row")
        elif rotation != width:
            raise ValueError("W4A4 stored row length differs from its source geometry")
        clip_bits = int(header[0, 0, 3])
        if not np.all(header[:, :, 3] == clip_bits):
            raise ValueError("W4A4 activation clip must be identical across the matrix")
        clip = struct.unpack('<f', struct.pack('<I', clip_bits))[0]
        if not math.isfinite(clip) or clip < 0:
            raise ValueError("W4A4 stored clip must be finite and nonnegative")
        # Zero is the exporter's explicit unset representation, resolved to one.
        clip = clip if clip > 0 else 1.0
        descriptor = header[:, :, 1]
        group64 = descriptor >> 29 == 3
        group256 = descriptor >> 30 == 1
        multipliers = np.stack([
            np.where(group64, (descriptor >> (4 * group)) & 15,
                     np.where(group256, descriptor & 15, 1))
            for group in range(4)
        ], axis=-1).astype(np.int32).reshape(rows, width // GROUP_VALUES)
        if (multipliers == 0).any():
            raise ValueError("W4A4 weight multipliers must be in 1..15")
        nibbles = np.stack((packed & 15, packed >> 4), axis=-1).reshape(rows, width).astype(np.int8)
        codes = np.where(nibbles >= 8, nibbles - 16, nibbles).astype(np.int8)
        return cls(torch.from_numpy(codes.copy()), torch.from_numpy(units[:, 0].copy()),
                   torch.from_numpy(multipliers), type_id, rotation, clip)

    def linear(self, values: torch.Tensor, *, output_tile: int = 128) -> torch.Tensor:
        """Apply the exact native arithmetic with bounded batched-BLAS scratch.

        Each group dot is an exact small integer despite using FP32 BLAS. The
        subsequent correction, multiplier product, arithmetic shift and sum
        use int32. Scaling to FP32 happens only after the complete integer sum.
        Output tiling changes neither group membership nor arithmetic order.
        """
        _require_fp32_cpu(values)
        rows, width = self.codes.shape
        if values.shape[-1] != width or output_tile <= 0:
            raise ValueError("W4A4 linear input width or output tile is invalid")
        source = hadamard(values, self.rotation_length) if self.type_id == 45 else values
        operands = quantize_activations(source, self.clip)
        tokens = operands.codes.shape[0]
        groups = width // GROUP_VALUES
        activations = operands.codes.to(torch.float32).permute(1, 0, 2)
        result = torch.empty((tokens, rows), dtype=torch.float32)
        for begin in range(0, rows, output_tile):
            end = min(rows, begin + output_tile)
            weight = self.codes[begin:end].reshape(end - begin, groups, GROUP_VALUES)
            dots = torch.bmm(activations, weight.to(torch.float32).permute(1, 2, 0)).to(torch.int32).permute(1, 2, 0)
            sums = weight.to(torch.int32).sum(dim=-1, dtype=torch.int32)
            corrected = dots + (8 - operands.zero_points[:, None, :]) * sums[None, :, :]
            terms = corrected * self.multipliers[None, begin:end, :] * operands.multipliers[:, None, :]
            total = torch.bitwise_right_shift(terms, TERM_SHIFT).sum(dim=-1, dtype=torch.int32)
            scaled = total.to(torch.float32) * float(1 << TERM_SHIFT)
            scaled = scaled * operands.unit[:, None]
            result[:, begin:end] = scaled * self.units[None, begin:end]
        return result.reshape(*values.shape[:-1], rows)


def require_pinned_hf_runtime() -> dict[str, str]:
    """Authenticate the actual HF/PyTorch graph, not only a mutable package tag.

    The pinned HF fork owns dense Qwen3.5's GGUF head ordering. A different
    upstream version is not an interchangeable numerical reference. Small
    arithmetic-only tests do not import or depend on that model implementation.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5

    distribution = importlib.metadata.distribution('transformers')
    receipt = json.loads(distribution.read_text('direct_url.json') or '{}')
    revision = receipt.get('vcs_info', {}).get('commit_id')
    module_digest = hashlib.sha256(Path(modeling_qwen3_5.__file__).read_bytes()).hexdigest()
    if (revision != TRANSFORMERS_REVISION or module_digest != TRANSFORMERS_QWEN35_SHA256
            or torch.__version__ != PYTORCH_VERSION):
        raise RuntimeError(
            'W4A4 HF reference requires its pinned torch and Transformers source; '
            'install python/reference/quantization/w4a4-requirements.txt. '
            f'Observed torch={torch.__version__}, transformers_revision={revision}, '
            f'qwen35_sha256={module_digest}'
        )
    return {'arithmetic_version': ARITHMETIC_VERSION, 'torch': torch.__version__,
            'transformers_revision': revision, 'qwen35_sha256': module_digest}


class W4A4Linear(torch.nn.Module):
    """A native-operand HF projection with no ordinary floating weight parameter.

    Buffers are the sole owned operand storage. Registering them keeps module
    inspection/state publication honest; precision/device changes are rejected
    before mutating any buffer. The FP32 interface is inference-only.
    """

    def __init__(self, weights: W4A4Weights):
        """Take ownership of admitted operands without creating a FP32 mirror."""
        super().__init__()
        self.register_buffer('codes', weights.codes)
        self.register_buffer('units', weights.units)
        self.register_buffer('multipliers', weights.multipliers)
        self.type_id = weights.type_id
        self.rotation_length = weights.rotation_length
        self.clip = weights.clip
        self.out_features, self.in_features = weights.codes.shape

    def _apply(self, function, recurse=True):
        """Reject module-wide conversion that would change the oracle contract."""
        probe = function(torch.empty(0, dtype=torch.float32, device='cpu'))
        if probe.device.type != 'cpu' or probe.dtype != torch.float32:
            raise ValueError('W4A4 reference modules retain CPU FP32 scale storage')
        return super()._apply(function, recurse=recurse)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Execute exact online A4 preparation and the native integer projection."""
        weights = W4A4Weights(self.codes, self.units, self.multipliers,
                              self.type_id, self.rotation_length, self.clip)
        return weights.linear(values)


def install_native_linear(model: torch.nn.Module, name: str, weights: W4A4Weights,
                          parameters: dict, loaded: set[str]) -> None:
    """Replace one declared HF linear while retiring its unused FP32 allocation.

    This is not a forward hook over a dequantized matrix: the old parameter is
    removed, and every forward consumes the native operand representation.
    Names, geometry, bias and duplicate publication are checked before mutation.
    """
    if not name.endswith('.weight') or name in loaded or name not in parameters:
        raise ValueError(f'Unbound or duplicate native weight: {name}')
    module_name = name.removesuffix('.weight')
    original = model.get_submodule(module_name)
    if (not isinstance(original, torch.nn.Linear) or original.bias is not None
            or tuple(original.weight.shape) != tuple(weights.codes.shape)
            or original.weight is not parameters[name]):
        raise ValueError(f'Native W4A4 projection does not match HF linear: {name}')
    model.set_submodule(module_name, W4A4Linear(weights))
    del parameters[name]
    loaded.add(name)
