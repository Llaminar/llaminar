/**
 * @file CUDATensorPrefillKernels.h
 * @brief Typed native entry and shared-memory contract for warp-owned prefill.
 *
 * The host graph controller borrows a native function handle with the canonical
 * phase ABI. No device function crosses this translation-unit boundary. Native
 * Q/K/V and compact FP32 summaries remain caller-owned persistent arena views;
 * kernel selection happens from immutable capture geometry, never live state.
 */
#pragma once
#include <cuda_runtime.h>
#include "../../attention/AttentionDeviceParams.h"
#include "CUDATensorPrefillGeometry.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2::cuda::tensor_prefill {
inline constexpr int kWarps = 8; ///< Sixteen query rows per complete warp.
inline constexpr int kThreads = kWarps * 32; ///< Exact native CTA extent.
inline constexpr int kRows = kWarps * 16; ///< Complete shared query tile.
inline constexpr int kKeys = 16; ///< One bounded PV reduction tile.
inline constexpr int kWidth = 256; ///< Exact admitted head width.
inline constexpr int kStageHalves = 2 * kKeys * kWidth; ///< Native K plus V.
inline constexpr int kOperandBytes = (kRows * kWidth + 2 * kStageHalves) * sizeof(std::uint16_t);
inline constexpr int kSharedBytes = kOperandBytes + 2 * kWarps * sizeof(unsigned);
inline constexpr int kSplitLimit = 4; ///< Contiguous live summaries at long context.
/** Canonical captured phase ABI; unused generic fields retain their positions. */
using Kernel = void (*)(const float *, const void *, const void *, float *,
    int,int,int,int,int,int,float,bool,int,int,
    const attention::AttentionDeviceParams *, const float *,int,int,int,int,int,int,
    float *,float *,float *,int,int,int,int);
/**
 * @brief Borrow the one native numerical entry for an admitted geometry.
 * @param geometry Complete immutable local or replicated head mapping.
 * @return Native function pointer, owned by the loaded CUDA backend artifact.
 * @throws std::invalid_argument If an invalid enum reaches this typed boundary.
 * The function performs no CUDA operation or allocation; capture owns launch
 * streams, shared-memory admission, parameter storage and completion events.
 */
[[nodiscard]] Kernel kernel(HeadGeometry geometry);
} // namespace llaminar2::cuda::tensor_prefill
