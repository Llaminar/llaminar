/**
 * @file GPUHostLoadPreflight.h
 * @brief Host-memory sizing rules for bounded GPU model loading.
 *
 * GPU model loading has two materially different host-memory regimes. Mapped
 * GGUF tensors can be read directly into the bounded pinned upload ring, while
 * CPU targets and explicit non-mmap loads materialize ordinary host storage.
 * Keeping the arithmetic in this small pure helper makes the preflight contract
 * independently testable and prevents UI or runner configuration from becoming
 * a second source of truth.
 */

#pragma once

#include <algorithm>
#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Compute the transient host bytes required by weight loading.
     *
     * A positive staging budget is a hard cap only for a GPU whose source
     * tensors remain file-backed. In every other mode the eager tensor bytes
     * describe real anonymous or retained host storage and must be checked in
     * full. A model smaller than the ring naturally requires only its own size.
     *
     * @param eager_weight_bytes Sum of source GGUF tensor payload bytes.
     * @param target_is_gpu True when weights are being prepared for a GPU.
     * @param uses_mmap True when source tensor payloads remain file-backed.
     * @param staging_budget_bytes Configured pinned upload-ring budget; zero
     *        means that loading is intentionally unbounded.
     * @return Host bytes that must be available before weight loading starts.
     */
    [[nodiscard]] inline size_t gpuHostLoadWorkingSetBytes(
        size_t eager_weight_bytes,
        bool target_is_gpu,
        bool uses_mmap,
        size_t staging_budget_bytes) noexcept
    {
        if (!target_is_gpu || !uses_mmap || staging_budget_bytes == 0)
            return eager_weight_bytes;

        return std::min(eager_weight_bytes, staging_budget_bytes);
    }
} // namespace llaminar2
