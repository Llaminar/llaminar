/**
 * @file NativeVnniPrimaryPlaneUpload.h
 * @brief The single host-to-device uploader body for prepared primary scale planes.
 *
 * Every CUDA/ROCm packed-weight uploader (dense CUDA, both dense ROCm
 * branches, and the batched CUDA/ROCm uploaders) runs this body. The
 * backend supplies only two primitives: allocate N bytes, and submit one
 * host-to-device copy. The body derives the plan itself from the plane
 * geometry, so a call site cannot choose a different transfer extent.
 *
 * Allocation keeps the rounded word-aligned storage stride, so per-chunk
 * device pointers stay aligned. Only live bytes are submitted: an odd
 * one-byte E8M0 plane never sends the unused storage byte after each chunk,
 * which is never read. See NativeVnniPrimaryPlaneUploadPlan.
 *
 * Ownership: the allocated device pointer is published to the caller
 * immediately after allocation succeeds. When a copy fails, the caller's
 * existing cleanup releases it; this body never frees, so it cannot
 * double-free alongside that cleanup.
 */
#pragma once

#include "tensors/NativeVnniScalePlane.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Outcome of one primary-plane upload. */
    enum class NativeVnniPrimaryPlaneUploadStatus : std::uint8_t
    {
        Uploaded,         ///< Allocation and every planned copy succeeded (or the plane is empty).
        AllocationFailed, ///< No device memory was published.
        CopyFailed,       ///< Device memory was published to the caller; a copy failed.
    };

    /**
     * @brief Allocate and upload a host-packed primary plane with exact live extents.
     *
     * @param physical_codebook Execution codebook selecting the plane width.
     * @param blocks_per_chunk Live blocks per matrix (rows * K / 32).
     * @param chunks Number of matrices sharing the slab (1 for a dense weight).
     * @param host Host slab with the planned stride layout.
     * @param host_bytes Size of @p host in bytes; it must equal the planned capacity.
     * @param device Receives the device slab as soon as it is allocated.
     * @param allocate `void *(size_t bytes)`: allocate device memory, or return nullptr.
     * @param copy `bool(void *dst, const void *src, size_t bytes)`: submit one H2D copy.
     * @throws std::runtime_error if the host storage does not match the plan.
     */
    template <typename Allocate, typename Copy>
    [[nodiscard]] NativeVnniPrimaryPlaneUploadStatus uploadNativeVnniPrimaryPlane(
        std::uint8_t physical_codebook,
        std::size_t blocks_per_chunk,
        std::size_t chunks,
        const void *host,
        std::size_t host_bytes,
        void **device,
        Allocate &&allocate,
        Copy &&copy)
    {
        *device = nullptr;
        const auto plan = nativeVnniPrimaryPlaneUploadPlan(physical_codebook, blocks_per_chunk, chunks);
        if (host_bytes != plan.allocation_bytes)
        {
            throw std::runtime_error(
                "Native-VNNI primary plane host storage does not match its upload plan");
        }
        if (plan.allocation_bytes == 0)
            return NativeVnniPrimaryPlaneUploadStatus::Uploaded;
        void *const slab = allocate(plan.allocation_bytes);
        if (slab == nullptr)
            return NativeVnniPrimaryPlaneUploadStatus::AllocationFailed;
        *device = slab;
        for (std::size_t index = 0; index < plan.copies; ++index)
        {
            // The same offset applies to host and device (identical stride layout).
            const std::size_t offset = plan.copyOffset(index);
            if (!copy(static_cast<std::uint8_t *>(slab) + offset,
                      static_cast<const std::uint8_t *>(host) + offset,
                      plan.copy_bytes))
                return NativeVnniPrimaryPlaneUploadStatus::CopyFailed;
        }
        return NativeVnniPrimaryPlaneUploadStatus::Uploaded;
    }
} // namespace llaminar2
