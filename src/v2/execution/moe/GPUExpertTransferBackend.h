/**
 * @file GPUExpertTransferBackend.h
 * @brief Private accelerator-runtime boundary for packed expert transfers.
 *
 * The public transfer API is backend-neutral and may be compiled with CUDA and
 * ROCm enabled together.  Runtime headers cannot safely coexist in one C++
 * translation unit, so this boundary owns runtime-specific submission and
 * ambient-device state access behind ordinary C++ declarations.
 */

#pragma once

#include "GPUExpertTransfer.h"

namespace llaminar2::detail
{
#ifdef HAVE_CUDA
    /** @brief Prepared capture-safe UVA byte body; expert migration keeps its explicit peer API.
     * @param destination Original destination pointer. @param destination_ordinal Exact destination GPU.
     * @param source Original source pointer. @param source_ordinal Exact source GPU.
     * @param bytes Live byte count. @param destination_stream Exact non-null destination-owned stream.
     * @return Native capture-safe enqueue success; no eager or alternate path is attempted.
     */
    bool copyPreparedDevicePeerBytesCUDABackend(void* destination,int destination_ordinal,const void* source,int source_ordinal,std::size_t bytes,void* destination_stream);
    /** @return Calling thread's current CUDA ordinal, or -1 on failure. */
    int currentCUDADeviceOrdinal() noexcept;

    /**
     * @brief Restore a calling thread's CUDA runtime device.
     * @return True when the exact ordinal became current.
     */
    bool restoreCUDADeviceOrdinal(int ordinal) noexcept;

    /**
     * @brief Authenticate and enable one directed CUDA peer-memory edge.
     *
     * This setup-only operation selects the destination because its transfer
     * stream will address the source allocation. It restores the caller's
     * ambient CUDA device before returning.
     *
     * @param source Distinct CUDA allocation owner.
     * @param destination CUDA stream/allocation owner.
     * @return True only when the driver exposes and enables direct access.
     */
    bool prepareDirectPeerAccessCUDABackend(
        const DeviceId &source,
        const DeviceId &destination) noexcept;

    /** Submit one separated NativeQuantised expert copy on a CUDA stream. */
    bool transferExpertCUDABackend(
        const GPUExpertPointers &src_ptrs,
        const GPUExpertPointers &dst_ptrs,
        const DeviceId &src_device,
        const DeviceId &dst_device,
        size_t vnni_bytes,
        size_t scales_bytes,
        size_t mins_bytes,
        size_t emins_bytes,
        void *stream);

#endif

#ifdef HAVE_ROCM
    /** @brief Prepared capture-safe UVA byte body; expert migration keeps its explicit peer API.
     * @param destination Original destination pointer. @param destination_ordinal Exact destination GPU.
     * @param source Original source pointer. @param source_ordinal Exact source GPU.
     * @param bytes Live byte count. @param destination_stream Exact non-null destination-owned stream.
     * @return Native capture-safe enqueue success; no eager or alternate path is attempted.
     */
    bool copyPreparedDevicePeerBytesROCmBackend(void* destination,int destination_ordinal,const void* source,int source_ordinal,std::size_t bytes,void* destination_stream);
    /** @return Calling thread's current ROCm ordinal, or -1 on failure. */
    int currentROCmDeviceOrdinal() noexcept;

    /**
     * @brief Restore ROCm runtime and HipDeviceGuard state together.
     * @return True when the exact ordinal became current.
     */
    bool restoreROCmDeviceOrdinal(int ordinal) noexcept;

    /**
     * @brief Authenticate and enable one directed ROCm peer-memory edge.
     *
     * HIP runtime state and `HipDeviceGuard` state are restored together, so
     * setup cannot redirect later inference launches on the calling thread.
     *
     * @param source Distinct ROCm allocation owner.
     * @param destination ROCm stream/allocation owner.
     * @return True only when the driver exposes and enables direct access.
     */
    bool prepareDirectPeerAccessROCmBackend(
        const DeviceId &source,
        const DeviceId &destination) noexcept;

    /** Submit one separated NativeQuantised expert copy on a ROCm stream. */
    bool transferExpertROCmBackend(
        const GPUExpertPointers &src_ptrs,
        const GPUExpertPointers &dst_ptrs,
        const DeviceId &src_device,
        const DeviceId &dst_device,
        size_t vnni_bytes,
        size_t scales_bytes,
        size_t mins_bytes,
        size_t emins_bytes,
        void *stream);

#endif
}
