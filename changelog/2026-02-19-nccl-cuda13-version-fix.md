# Fix: NCCL "CUDA driver is a stub library" — CUDA 13.0 Version Mismatch

## Summary

Resolved the persistent NCCL runtime failure (`CUDA driver is a stub library`) that blocked real NCCL collective operations for LocalTP multi-GPU inference. The root cause was an NCCL library version incompatibility with CUDA 13.0.

## Root Cause

The installed NCCL package (`libnccl2 2.18.5-1-2`) was compiled against **CUDA 12.0** (confirmed by runtime banner: `NCCL version 2.18.3+cuda12.0`). The dev container environment runs **CUDA 13.0** (toolkit V13.0.88, driver 580.126.09).

NCCL's internal stub detection in `misc/strongstream.cc:60` failed when the CUDA driver API reported version 13000, which is beyond what NCCL 2.18 was designed to handle. Critically:
- The CUDA driver itself was real and functional (verified via `cuInit`, `cuDeviceGetCount`, `cuStreamCreate`, `cuMemAllocAsync` — all returning success)
- NCCL communicator initialization (`ncclCommInitAll`) succeeded
- The error surfaced only during the first `ncclAllReduce` call
- The error code was `CUDA_ERROR_STUB_LIBRARY` (34), a misleading error name for what was actually a version incompatibility

## Fix

### NCCL Upgrade

```bash
sudo apt-get install libnccl2=2.28.9-1+cuda13.0 libnccl-dev=2.28.9-1+cuda13.0
```

Upgraded from `2.18.5-1-2` (built for CUDA 12.0) to `2.28.9-1+cuda13.0` (built for CUDA 13.0), sourced from the NVIDIA CUDA apt repository (`https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/`).

### NCCLDynamicLoader Enum Fix

Fixed `ncclRedOp_t` enum values in `src/v2/collective/backends/NCCLDynamicLoader.h`:

| Enum Value | Old (Wrong) | New (Correct, matches NCCL ABI) |
|-----------|-------------|-------------------------------|
| `ncclMax` | 3 | **2** |
| `ncclMin` | 2 | **3** |

This was a latent bug — `ncclSum` (value 0) is the only reduction op currently used in LocalTP allreduce, so the swap never caused observable failures. However, any future use of Min or Max reductions would have produced silently incorrect results.

Also added NCCL 2.28 data types to the enum:
- `ncclFloat8e4m3 = 10`
- `ncclFloat8e5m2 = 11`
- `ncclNumTypes = 12` (was 10)

## Files Changed

- `src/v2/collective/backends/NCCLDynamicLoader.h` — Fixed `ncclRedOp_t` enum values, added FP8 data types
- `changelog/2026-02-17-enable-real-nccl-collectives-project-plan.md` — Updated blocker status to RESOLVED

## Verification

### Standalone Test
```
NCCL INFO NCCL version 2.28.9+cuda13.0
AllReduce SUCCEEDED!
```

### Integration Tests (all PASSED)
- `V2_Integration_LocalTP_BackendBehavior` — 16/17 passed, 1 skipped (expected: NCCL works, so failure injection isn't reproducible)
- `V2_Integration_NCCLBackend` — PASSED
- `V2_Integration_TPAllreduceStage_LocalNCCL` — PASSED
- `V2_Integration_NCCLCoordinator` — PASSED
- `V2_Integration_LocalTP_MultiDevice` — PASSED

## Diagnostic Journey

The investigation ruled out several false leads before identifying the version mismatch:
1. **Stub library on LD_LIBRARY_PATH** — No. Only real driver at `/usr/lib/x86_64-linux-gnu/libcuda.so.580.126.09`
2. **Binary linking to stub** — No. `ldd` resolves to real driver
3. **CUDA driver API failures** — No. All driver API calls (including `cuMemAllocAsync`, `cuGetProcAddress`) succeed
4. **NCCL loading the wrong libcuda** — No. NCCL doesn't statically link to any CUDA library
5. **Root cause**: NCCL 2.18.3+cuda12.0 internal version checks incompatible with CUDA 13.0 driver API

## Environment

- CUDA Toolkit: 13.0 (V13.0.88)
- NVIDIA Driver: 580.126.09 (Open Kernel Module)
- GPUs: 2x NVIDIA GeForce RTX 3090
- Container: Ubuntu 24.04 dev container
- NCCL: 2.18.5 → **2.28.9+cuda13.0**
