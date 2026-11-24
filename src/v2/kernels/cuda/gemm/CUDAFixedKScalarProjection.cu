/**
 * @file CUDAFixedKScalarProjection.cu
 * @brief Compiled strict-FP32 fixed-K schedules and their launch bridge on CUDA.
 *
 * Narrow groups use the shared CUDA/HIP partitioned body
 * (FixedKScalarProjectionBody.cuh) plus its ordered fold; wide groups use the
 * SGEMM-class register-blocked kernel (CUDAFixedKScalarProjection.cuh). Both
 * evaluate the scalar fixed-K contract, so every schedule is byte-identical
 * and equal to a host IEEE-FMA reference.
 */
#include "kernels/cuda/gemm/CUDAFixedKProjectionLaunch.h"
#include "kernels/cuda/gemm/CUDAFixedKScalarProjection.cuh"
#include "kernels/common/FixedKScalarProjectionBody.cuh"

#include <array>
#include <stdexcept>

namespace llaminar2::cuda
{
    namespace
    {
        namespace body = ::llaminar2::fixed_k_scalar;
        namespace sgemm = fixed_k_scalar_cuda;

        constexpr std::array kFP32Schedules{
            FixedKFP32Schedule::Partitioned64x32, FixedKFP32Schedule::Partitioned64x64,
            FixedKFP32Schedule::Sgemm64x64, FixedKFP32Schedule::Sgemm128x128};

        /** @brief Partition-chain launch of the shared body plus the ordered fold. */
        template<int BM, int BN, int TM, int TN, int BK>
        void launchPartitioned(const FixedStrictFP32ProjectionGroup &group, int m, int k, cudaStream_t stream,
                               float *partials)
        {
            const dim3 grid((m + BM - 1) / BM, (group.totalColumns() + BN - 1) / BN, fixed_k_projection::kPartitions);
            body::scalarProjectionKernel<float, BM, BN, TM, TN, BK, body::Mode::Partial, false>
                <<<grid, (BM / TM) * (BN / TN), 0, stream>>>(group, m, k, partials);
            const size_t outputs = static_cast<size_t>(m) * group.totalColumns();
            body::foldScalarPartialsKernel<float><<<static_cast<unsigned>((outputs + 255) / 256), 256, 0, stream>>>(
                group, m, partials);
        }

        /** @brief Raise one instantiation's shared-memory limit (host attribute, capture safe). */
        template<class Kernel>
        void reserveShared(Kernel *kernel, int bytes)
        {
            if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess)
                throw std::runtime_error("Fixed-K FP32 projection cannot reserve its shared-memory stages");
        }

        /** @brief Submit raster tiles [first, first+count) as SPLIT row slices each. */
        template<int BM, int BN, int TM, int TN, int BK, int STAGES, int SPLIT>
        void submitSgemm(const FixedStrictFP32ProjectionGroup &group, int m, int k, cudaStream_t stream,
                         int first, int count)
        {
            if (count <= 0) return;
            constexpr int bytes = sgemm::sgemmSharedBytes<BM, BN, BK, STAGES>;
            auto *kernel = sgemm::sgemmAligned<BK>(k)
                ? sgemm::sgemmProjectionKernel<BM, BN, TM, TN, BK, STAGES, true, SPLIT>
                : sgemm::sgemmProjectionKernel<BM, BN, TM, TN, BK, STAGES, false, SPLIT>;
            reserveShared(kernel, bytes);
            kernel<<<static_cast<unsigned>(count * SPLIT), (BM / TM) * (BN / TN), bytes, stream>>>(group, m, k, first);
        }

        /** @brief Resident CTAs of the 128x128 schedule on one device (cached; host queries only). */
        int sgemmWaveSlots(int device_id)
        {
            static thread_local int cached_device = -1, cached_slots = 0;
            if (device_id != cached_device)
            {
                int sms = 0, per_sm = 0;
                constexpr int bytes = sgemm::sgemmSharedBytes<128, 128, 16, 3>;
                auto *kernel = sgemm::sgemmProjectionKernel<128, 128, 8, 8, 16, 3, true, 1>;
                reserveShared(kernel, bytes);
                if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id) != cudaSuccess ||
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, 256, bytes) != cudaSuccess ||
                    sms <= 0 || per_sm <= 0)
                    throw std::runtime_error("Fixed-K FP32 projection cannot read device wave geometry");
                cached_device = device_id;
                cached_slots = sms * per_sm;
            }
            return cached_slots;
        }

        /**
         * @brief 128x128 SGEMM with a wave-balanced tail.
         * @details Full waves run whole raster tiles; the remaining tiles are cut
         * into 2 or 4 row slices when that shortens the last wave. Arithmetic per
         * output is unchanged, so this is purely an economy decision.
         */
        void launchSgemmBalanced(const FixedStrictFP32ProjectionGroup &group, int m, int k, int device_id,
                                 cudaStream_t stream)
        {
            const int tiles = ((m + 127) / 128) * ((group.totalColumns() + 127) / 128);
            const int slots = sgemmWaveSlots(device_id);
            const int full = tiles >= slots ? tiles - tiles % slots : 0;
            const int split = full ? sgemm::tailSplit(tiles - full, slots) : 1;
            const int head = split == 1 ? tiles : full;
            submitSgemm<128, 128, 8, 8, 16, 3, 1>(group, m, k, stream, 0, head);
            if (split == 2) submitSgemm<64, 128, 4, 8, 16, 3, 2>(group, m, k, stream, head, tiles - head);
            if (split == 4) submitSgemm<32, 128, 2, 8, 16, 3, 4>(group, m, k, stream, head, tiles - head);
        }
    } // namespace

    std::span<const FixedKFP32Schedule> fixedKFP32Schedules() noexcept { return kFP32Schedules; }

    std::string_view fixedKFP32ScheduleName(FixedKFP32Schedule schedule) noexcept
    {
        switch (schedule)
        {
        case FixedKFP32Schedule::Partitioned64x32: return "fp32_partitioned_64x32";
        case FixedKFP32Schedule::Partitioned64x64: return "fp32_partitioned_64x64";
        case FixedKFP32Schedule::Sgemm64x64: return "fp32_sgemm_64x64";
        case FixedKFP32Schedule::Sgemm128x128: return "fp32_sgemm_128x128";
        }
        return "invalid";
    }

    bool launchFixedKFP32Projection(const FixedStrictFP32ProjectionGroup &group, int m, int k,
        int device_id, void *stream, FixedKFP32Schedule schedule, float *partials, std::size_t partial_bytes)
    {
        if (!stream)
            throw std::invalid_argument("Fixed-K FP32 projection requires an exact non-null stream");
        if (m <= 0 || k <= 0)
            throw std::invalid_argument("Fixed-K FP32 projection requires positive rows and K");
        if (fixedKFP32ScheduleNeedsPartials(schedule) &&
            (!partials || partial_bytes < fixed_k_projection::partialBytes(m, group.totalColumns())))
            throw std::invalid_argument("Partitioned fixed-K FP32 projection lacks its admitted partial workspace");
        if (cudaSetDevice(device_id) != cudaSuccess)
            return false;
        const auto native = static_cast<cudaStream_t>(stream);
        switch (schedule)
        {
        case FixedKFP32Schedule::Partitioned64x32: launchPartitioned<64, 32, 4, 2, 32>(group, m, k, native, partials); break;
        case FixedKFP32Schedule::Partitioned64x64: launchPartitioned<64, 64, 4, 4, 16>(group, m, k, native, partials); break;
        case FixedKFP32Schedule::Sgemm64x64:
            submitSgemm<64, 64, 4, 4, 16, 4, 1>(group, m, k, native, 0,
                ((m + 63) / 64) * ((group.totalColumns() + 63) / 64));
            break;
        case FixedKFP32Schedule::Sgemm128x128: launchSgemmBalanced(group, m, k, device_id, native); break;
        default: throw std::invalid_argument("Unknown fixed-K FP32 projection schedule");
        }
        return cudaGetLastError() == cudaSuccess;
    }
} // namespace llaminar2::cuda
