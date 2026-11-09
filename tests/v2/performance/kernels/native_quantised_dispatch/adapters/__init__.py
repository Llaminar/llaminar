"""Backend trainer adapters that emit the strict common observation schema."""

from .cpu_verifier import CPUVerifierAdapterContext, adapt_cpu_verifier_csv
from .rocm_decode import ROCmDecodeAdapterContext, adapt_rocm_decode_csv
from .rocm_moe import ROCmMoEAdapterContext, adapt_rocm_moe_csv

__all__ = (
    "ROCmDecodeAdapterContext",
    "ROCmMoEAdapterContext",
    "CPUVerifierAdapterContext",
    "adapt_cpu_verifier_csv",
    "adapt_rocm_decode_csv",
    "adapt_rocm_moe_csv",
)
