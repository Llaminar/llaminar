#!/usr/bin/env bash
# Runtime uses the same signed repository and package pin as the compiler.
# HIP and gfx906 rocBLAS binaries/kernel libraries are copied from the builder
# by Dockerfile after this minimal host-library installation. No source build,
# foreign kernel graft, compiler or driver installation happens here.
set -euo pipefail
MODE=runtime exec bash "$(dirname -- "${BASH_SOURCE[0]}")/install-rocm.sh"
