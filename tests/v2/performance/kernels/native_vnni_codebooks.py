#!/usr/bin/env python3
"""Canonical NativeVNNI format ids used by perf sweep generators.

The ids mirror ``NativeVnniFormatInfo::codebook_id`` returned by the tensor
classes. Keep aliases here so CUDA/ROCm/CPU sweep analyzers do not each carry
their own stale format table.
"""

from collections import OrderedDict


FORMAT_TO_CODEBOOK = OrderedDict(
    [
        ("Q4_0", 0),
        ("IQ4_NL", 4),
        ("IQ4_XS", 4),
        ("Q4_1", 5),
        ("Q4_K", 5),
        ("Q5_0", 6),
        ("Q5_1", 7),
        ("Q5_K", 7),
        ("Q6_K", 8),
        ("Q3_K", 9),
        ("Q2_K", 10),
        ("IQ3_S", 11),
        ("IQ3_XXS", 12),
        ("IQ2_S", 13),
        ("IQ2_XS", 14),
        ("IQ2_XXS", 15),
        ("IQ1_S", 16),
        ("IQ1_M", 17),
        ("Q8_0", 19),
        ("Q8_1", 20),
    ]
)

CODEBOOK_TO_FORMAT = {
    0: "Q4_0",
    4: "IQ4_NL/IQ4_XS",
    5: "Q4_1/Q4_K",
    6: "Q5_0",
    7: "Q5_1/Q5_K",
    8: "Q6_K",
    9: "Q3_K",
    10: "Q2_K",
    11: "IQ3_S",
    12: "IQ3_XXS",
    13: "IQ2_S",
    14: "IQ2_XS",
    15: "IQ2_XXS",
    16: "IQ1_S",
    17: "IQ1_M",
    19: "Q8_0",
    20: "Q8_1",
}

CODEBOOK_PAYLOAD_BYTES = {
    0: 16,
    4: 16,
    5: 16,
    6: 20,
    7: 20,
    8: 24,
    9: 12,
    10: 8,
    11: 13,
    12: 12,
    13: 9,
    14: 9,
    15: 8,
    16: 6,
    17: 6,
    19: 32,
    20: 32,
}


def infer_format_from_filename(path):
    """Infer a format name from a sweep CSV path, preferring longer aliases."""
    name = path.stem.lower().replace("_", "")
    for candidate in sorted(FORMAT_TO_CODEBOOK, key=len, reverse=True):
        if candidate.lower().replace("_", "") in name:
            return candidate
    return None
