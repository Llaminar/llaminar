#!/usr/bin/env python3
"""Compatibility facade for the versioned NativeVNNI format registry.

New code should import ``native_vnni_dispatch.format_registry`` directly. This
module remains while backend analyzers migrate, but no longer owns a duplicate
format table. In particular, Q8_K/source codebook 21 is now part of the shared
inventory and GPU callers can query the distinct normalized execution codebook.
"""

from native_vnni_dispatch.format_registry import (  # noqa: F401
    CODEBOOK_PAYLOAD_BYTES,
    CODEBOOK_TO_FORMAT,
    FORMAT_BY_LABEL,
    FORMAT_SPECS,
    FORMAT_TO_CODEBOOK,
    GPU_FORMAT_TO_EXECUTION_CODEBOOK,
    format_spec,
    infer_format_from_filename,
    registry_digest,
    runtime_aliases,
)
