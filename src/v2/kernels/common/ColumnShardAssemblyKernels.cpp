/**
 * @file ColumnShardAssemblyKernels.cpp
 * @brief Backend-neutral binding to the existing column deinterleave implementation.
 *
 * Reusing the established KV gather kernel avoids introducing a second device
 * mapping or peer-by-peer copy loop for expert outputs. This bridge validates
 * ranges before enqueue; it owns no allocation, synchronization or coherence.
 */
#include "ColumnShardAssemblyKernels.h"
#include <cstdint>
#include <limits>

#ifdef HAVE_CUDA
extern "C" bool cudaTPKV_deinterleave_rank_major_fp32(
    const float *, float *, int, int, int, int, void *);
extern "C" bool cudaTPKV_deinterleave_live_rank_major_fp32(
    const float *, float *, llaminar2::DeviceRowRange, int, int, int, void *);
#endif
#ifdef HAVE_ROCM
extern "C" bool rocmTPKV_deinterleave_rank_major_fp32(
    const float *, float *, int, int, int, int, void *);
extern "C" bool rocmTPKV_deinterleave_live_rank_major_fp32(
    const float *, float *, llaminar2::DeviceRowRange, int, int, int, void *);
#endif

namespace llaminar2
{
    namespace
    {
        /** @brief Authenticate disjoint capacity-sized banks before either assembly launch. */
        bool validBanks(DeviceId device, const float *rank_major, float *row_major,
            int rows, int participants, int columns, void *stream)
        {
            if (!device.is_gpu() || !stream || !rank_major || !row_major || rows <= 0 ||
                participants < 1 || columns <= 0 || columns > std::numeric_limits<int>::max() / participants)
                return false;
            const auto width = std::size_t(participants) * columns;
            if (width > std::numeric_limits<std::size_t>::max() / sizeof(float) / rows) return false;
            const auto bytes = width * rows * sizeof(float);
            const auto source = reinterpret_cast<std::uintptr_t>(rank_major);
            const auto target = reinterpret_cast<std::uintptr_t>(row_major);
            return source <= std::numeric_limits<std::uintptr_t>::max() - bytes &&
                target <= std::numeric_limits<std::uintptr_t>::max() - bytes &&
                !(source < target + bytes && target < source + bytes);
        }
    }

    bool assembleColumnShardsFP32(DeviceId device, const float *rank_major, float *row_major,
        int rows, int participants, int columns, void *stream)
    {
        if (!validBanks(device, rank_major, row_major, rows, participants, columns, stream)) return false;
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cudaTPKV_deinterleave_rank_major_fp32(
            rank_major, row_major, rows, participants, columns, device.ordinal, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocmTPKV_deinterleave_rank_major_fp32(
            rank_major, row_major, rows, participants, columns, device.ordinal, stream);
#endif
        return false;
    }

    bool assembleLiveColumnShardsFP32(DeviceId device, const float *rank_major, float *row_major,
        DeviceRowRange rows, int participants, int columns, void *stream)
    {
        if (rows.physicalRows() != rows.capacity() ||
            !validBanks(device, rank_major, row_major, rows.capacity(), participants, columns, stream)) return false;
#ifdef HAVE_CUDA
        if (device.is_cuda()) return cudaTPKV_deinterleave_live_rank_major_fp32(
            rank_major, row_major, rows, participants, columns, device.ordinal, stream);
#endif
#ifdef HAVE_ROCM
        if (device.is_rocm()) return rocmTPKV_deinterleave_live_rank_major_fp32(
            rank_major, row_major, rows, participants, columns, device.ordinal, stream);
#endif
        return false;
    }
}
