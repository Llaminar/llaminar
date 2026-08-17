/**
 * @file FloatingPointGemmWorkspaceABI.h
 * @brief Shared graph-workspace cardinalities for floating-point GEMM batches.
 *
 * CUDA and ROCm floating projection wrappers submit bounded batches from fused
 * QKV, GDN, and grouped-verifier stages. The pointer arrays and CUDA mapped
 * output redirect are one captured ABI, so runtime and metadata-only memory
 * admission must consume this single constant rather than parallel literals.
 */

#pragma once

#include <cstddef>

namespace llaminar2::floating_gemm_abi
{

/** Maximum projections submitted by one captured floating-point GEMM batch. */
inline constexpr std::size_t kMaxBatchedProjections = 8;

} // namespace llaminar2::floating_gemm_abi
