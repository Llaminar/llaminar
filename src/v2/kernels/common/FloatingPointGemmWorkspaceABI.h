/**
 * @file FloatingPointGemmWorkspaceABI.h
 * @brief Shared graph-workspace cardinalities for floating-point GEMM batches.
 *
 * CUDA and ROCm floating projection wrappers submit bounded batches from fused
 * QKV, GDN, and grouped-verifier stages. The pointer arrays and CUDA mapped
 * output redirect are one captured ABI, so runtime and metadata-only memory
 * admission must consume this single constant rather than parallel literals.
 * BLAS and BLASLt share one mutually exclusive stage scratch region, declared
 * here for both runtime bindings and metadata-only physical-memory admission.
 */

#pragma once

#include <cstddef>

namespace llaminar2::floating_gemm_abi
{

/** Maximum projections submitted by one captured floating-point GEMM batch. */
inline constexpr std::size_t kMaxBatchedProjections = 8;

/** Persistent per-concurrent-stage scratch; ordinary BLAS and Lt reuse it. */
inline constexpr std::size_t kBlasMatmulWorkspaceBytes = 4 * 1024 * 1024;
/** Canonical CUDA arena identity used by both BLAS operation families. */
inline constexpr const char *kCudaBlasMatmulWorkspace = "cuda_blas_matmul_workspace";
/** Canonical ROCm arena identity used by both BLAS operation families. */
inline constexpr const char *kROCmBlasMatmulWorkspace = "rocm_blas_matmul_workspace";

} // namespace llaminar2::floating_gemm_abi
