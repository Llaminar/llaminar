"""Shared NativeVNNI dispatch training and certification framework.

Backend perf harnesses own candidate execution. This package owns the common
observation contract, correctness eligibility, robust exact winner selection,
generic regret learner, policy IR, and held-out certification semantics.
"""

from .schema import (  # noqa: F401
    AspectBucket,
    Backend,
    ExecutionMode,
    NativeVNNIObservation,
    SemanticContract,
)
from .candidate_registry import candidate_registry_digest  # noqa: F401
from .format_registry import registry_digest  # noqa: F401
