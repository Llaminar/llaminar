/**
 * @file CUDAFixedKHalfProjection.cu
 * @brief Compiled schedule inventory and launch bridge for CUDA half-RNE fixed-K projections.
 *
 * Owns the instantiations of CUDAFixedKHalfProjection.cuh. The device-free
 * policy in CUDAFixedKProjectionLaunch.h selects among them: narrow groups
 * (GDN alpha/beta and their tensor-parallel shards) spread K partitions over
 * CTAs and fold in a second kernel; wide groups (explicit full-FP16 QKV/Z)
 * walk K per tile. Schedule choice is economy only: the fixed-K contract makes
 * every schedule byte-identical, which the preflight regression proves.
 */
#include "kernels/cuda/gemm/CUDAFixedKProjectionLaunch.h"
#include "kernels/cuda/gemm/CUDAFixedKHalfProjection.cuh"

#include <array>
#include <stdexcept>

namespace llaminar2::cuda
{
    namespace
    {
        namespace half = fixed_k_half;

        constexpr std::array kHalfSchedules{
            FixedKHalfSchedule::Partitioned64x64, FixedKHalfSchedule::Tiled32x64W4,
            FixedKHalfSchedule::Tiled64x32W4, FixedKHalfSchedule::Tiled64x64,
            FixedKHalfSchedule::Tiled64x128W8, FixedKHalfSchedule::Tiled128x64W8,
            FixedKHalfSchedule::Tiled128x64, FixedKHalfSchedule::Tiled128x128W8};

        /**
         * @brief Submit one instantiation; grids cover exactly the live row and compact column tiles.
         * @tparam Mode Tiled register fold, or Partial publication followed by the fold kernel.
         */
        template<int BM, int BN, int WM, int WN, int STAGES, half::FixedKMode Mode>
        void launch(const FixedHalfProjectionGroup &group, int m, int k, cudaStream_t stream, float *partials)
        {
            constexpr int bytes = half::fixedKSharedBytes<BM, BN, STAGES>;
            auto *kernel = k % fixed_k_projection::kRequiredKAlignment == 0
                ? half::fixedKProjectionKernel<BM, BN, WM, WN, STAGES, Mode, true>
                : half::fixedKProjectionKernel<BM, BN, WM, WN, STAGES, Mode, false>;
            // Raising the dynamic shared-memory limit is a host-side function
            // attribute: idempotent and capture-safe, no allocation or sync.
            if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess)
                throw std::runtime_error("Fixed-K half projection cannot reserve its shared-memory stages");
            const bool partial = Mode == half::FixedKMode::Partial;
            const unsigned row_tiles = (m + BM - 1) / BM, col_tiles = (group.totalColumns() + BN - 1) / BN;
            // Tiled mode decodes a banded 1-D raster; partial mode is (row, column, partition).
            const dim3 grid = partial ? dim3(row_tiles, col_tiles, fixed_k_projection::kPartitions)
                                      : dim3(row_tiles * col_tiles, 1, 1);
            kernel<<<grid, WM * WN * 32, bytes, stream>>>(group, m, k, partials);
            if (partial)
            {
                const size_t outputs = static_cast<size_t>(m) * group.totalColumns();
                half::foldPartialsKernel<<<static_cast<unsigned>((outputs + 255) / 256), 256, 0, stream>>>(
                    group, m, partials);
            }
        }
    } // namespace

    std::span<const FixedKHalfSchedule> fixedKHalfSchedules() noexcept { return kHalfSchedules; }

    std::string_view fixedKHalfScheduleName(FixedKHalfSchedule schedule) noexcept
    {
        switch (schedule)
        {
        case FixedKHalfSchedule::Partitioned64x64: return "partitioned_64x64";
        case FixedKHalfSchedule::Tiled32x64W4: return "tiled_32x64_w4";
        case FixedKHalfSchedule::Tiled64x32W4: return "tiled_64x32_w4";
        case FixedKHalfSchedule::Tiled64x64: return "tiled_64x64";
        case FixedKHalfSchedule::Tiled64x128W8: return "tiled_64x128_w8";
        case FixedKHalfSchedule::Tiled128x64W8: return "tiled_128x64_w8";
        case FixedKHalfSchedule::Tiled128x64: return "tiled_128x64";
        case FixedKHalfSchedule::Tiled128x128W8: return "tiled_128x128_w8";
        }
        return "invalid";
    }

    bool launchFixedKHalfProjection(const FixedHalfProjectionGroup &group, int m, int k,
        int device_id, void *stream, FixedKHalfSchedule schedule, float *partials, std::size_t partial_bytes)
    {
        if (!stream)
            throw std::invalid_argument("Fixed-K half projection requires an exact non-null stream");
        if (m <= 0 || k <= 0)
            throw std::invalid_argument("Fixed-K half projection requires positive rows and K");
        if (fixedKHalfScheduleNeedsPartials(schedule) &&
            (!partials || partial_bytes < fixedKHalfPartialBytes(m, group.totalColumns())))
            throw std::invalid_argument("Partitioned fixed-K half projection lacks its admitted partial workspace");
        if (cudaSetDevice(device_id) != cudaSuccess)
            return false;
        const auto native = static_cast<cudaStream_t>(stream);
        constexpr auto Partial = half::FixedKMode::Partial;
        constexpr auto Tiled = half::FixedKMode::Tiled;
        switch (schedule)
        {
        case FixedKHalfSchedule::Partitioned64x64: launch<64, 64, 2, 2, 3, Partial>(group, m, k, native, partials); break;
        case FixedKHalfSchedule::Tiled32x64W4: launch<32, 64, 1, 4, 4, Tiled>(group, m, k, native, nullptr); break;
        case FixedKHalfSchedule::Tiled64x32W4: launch<64, 32, 4, 1, 4, Tiled>(group, m, k, native, nullptr); break;
        case FixedKHalfSchedule::Tiled64x64: launch<64, 64, 2, 2, 3, Tiled>(group, m, k, native, nullptr); break;
        case FixedKHalfSchedule::Tiled64x128W8: launch<64, 128, 2, 4, 3, Tiled>(group, m, k, native, nullptr); break;
        case FixedKHalfSchedule::Tiled128x64W8: launch<128, 64, 4, 2, 3, Tiled>(group, m, k, native, nullptr); break;
        case FixedKHalfSchedule::Tiled128x64: launch<128, 64, 2, 2, 2, Tiled>(group, m, k, native, nullptr); break;
        case FixedKHalfSchedule::Tiled128x128W8: launch<128, 128, 2, 4, 2, Tiled>(group, m, k, native, nullptr); break;
        default: throw std::invalid_argument("Unknown fixed-K half projection schedule");
        }
        return cudaGetLastError() == cudaSuccess;
    }

    bool launchFixedKHalfProjection(const FixedHalfProjectionGroup &group, int m, int k,
        int device_id, void *stream, float *partials, std::size_t partial_bytes)
    {
        return launchFixedKHalfProjection(group, m, k, device_id, stream,
            selectFixedKHalfSchedule(m, group.totalColumns(), k), partials, partial_bytes);
    }
} // namespace llaminar2::cuda
