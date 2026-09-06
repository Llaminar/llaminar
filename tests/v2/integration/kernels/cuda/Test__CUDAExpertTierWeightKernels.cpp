/**
 * @file Test__CUDAExpertTierWeightKernels.cpp
 * @brief Byte-exact CUDA parity for streamed ExpertOverlay weight conversion.
 *
 * The test uses the device-free CPU oracle to construct complete expected
 * execution bytes, then converts the same projection in arbitrary two-unit
 * chunks on a non-default CUDA stream. It covers every reversible physical CPU
 * encoding, non-64-aligned N padding, non-zero chunk origins, both directions,
 * and strict launcher rejection before the performance harness is allowed to
 * measure these kernels.
 */

#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "execution/moe/GpuExpertSlotPool.h"
#include "execution/moe/MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/common/NativeVNNIGroupedDecodePolicy.h"
#include "kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#include "tensors/TensorKernels.h"
#include "tensors/VnniPackContext.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ExpertTierFusedQKVStreamPoolHarness.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C"
{
    /** Initialize immutable CUDA IQ lookup tables used by production decode. */
    bool cudaNativeVNNIInitIQGridTables_tuned();
    void cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(int enabled);
    int cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config();

    /** Production M=1 NativeVNNI inference kernel used by the overlap proof. */
    bool cudaNativeVNNIGemvTuned_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const std::uint16_t *d_scales,
        const std::uint16_t *d_mins,
        const std::uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        std::uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot);

    /** Physical decoder plus independent source arithmetic identity. */
    bool cudaNativeVNNIGemvTuned_fp32_withPolicy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const std::uint16_t *d_scales,
        const std::uint16_t *d_mins,
        const std::uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        std::uint8_t codebook_id,
        std::uint8_t arithmetic_policy_codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot);

    /** Production verifier-depth NativeVNNI kernel used by promotion parity. */
    bool cudaNativeVNNIGemvTuned_small_m_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const std::uint16_t *d_scales,
        const std::uint16_t *d_mins,
        const std::uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        std::uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot);

    /** Grouped physical decoder plus independent source arithmetic identity. */
    bool cudaNativeVNNIGemvTuned_small_m_fp32_withPolicy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const std::uint16_t *d_scales,
        const std::uint16_t *d_mins,
        const std::uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        std::uint8_t codebook_id,
        std::uint8_t arithmetic_policy_codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot);

    /** Production dense-prefill NativeVNNI kernel used by promotion parity. */
    bool cudaNativeVNNIPrefill_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const std::uint16_t *d_scales,
        const std::uint16_t *d_mins,
        const std::uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const std::int32_t *d_sums_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        std::uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAPrefillContext *prefill_ctx);

    /** Execute promoted bytes while preserving their source arithmetic tree. */
    bool cudaNativeVNNIPrefill_fp32_withPolicy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const std::uint16_t *d_scales,
        const std::uint16_t *d_mins,
        const std::uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const std::int32_t *d_sums_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        std::uint8_t codebook_id,
        std::uint8_t arithmetic_policy_codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAPrefillContext *prefill_ctx);
}

namespace llaminar2
{
    namespace
    {
        /** @brief Enable route evidence for one test without leaking env state. */
        class ScopedPerfStats final
        {
        public:
            /** Save the process setting, enable collection, and reset records. */
            ScopedPerfStats()
            {
                if (const char *old =
                        std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
                {
                    had_old_value_ = true;
                    old_value_ = old;
                }
                setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            /** Restore the process setting and discard isolated records. */
            ~ScopedPerfStats()
            {
                if (had_old_value_)
                {
                    setenv(
                        "LLAMINAR_PERF_STATS_SUMMARY",
                        old_value_.c_str(),
                        1);
                }
                else
                {
                    unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
                }
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            ScopedPerfStats(const ScopedPerfStats &) = delete;
            ScopedPerfStats &operator=(const ScopedPerfStats &) = delete;

        private:
            bool had_old_value_ = false;
            std::string old_value_;
        };

        /** @return One PerfStats tag value, or empty when absent. */
        std::string perfTag(
            const PerfStatRecord &record,
            const std::string &name)
        {
            const auto found = record.tags.find(name);
            return found == record.tags.end() ? std::string{} : found->second;
        }

        /** Keep the public serial-row decode policy active for an inference proof. */
        class ScopedDecodeEquivalentM1 final
        {
        public:
            ScopedDecodeEquivalentM1()
                : previous_(
                      cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config())
            {
                cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(1);
            }

            ~ScopedDecodeEquivalentM1()
            {
                cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(previous_);
            }

            ScopedDecodeEquivalentM1(const ScopedDecodeEquivalentM1 &) = delete;
            ScopedDecodeEquivalentM1 &operator=(
                const ScopedDecodeEquivalentM1 &) = delete;

        private:
            int previous_ = 0;
        };

        /** Small immutable format selector used by the parameterized test. */
        struct FormatCase
        {
            std::uint8_t codebook = 0;
            std::uint8_t payload_bytes = 0;
            bool asymmetric = false;
            bool superblock = false;
            const char *name = nullptr;
        };

        /**
         * @brief Own one non-default CUDA stream for a test scope.
         *
         * Construction throws on runtime failure so a test can never continue
         * with CUDA's legacy null stream after stream creation failed.
         */
        class TestCUDAStream final
        {
        public:
            /** @brief Create one non-blocking, non-default stream. */
            TestCUDAStream()
            {
                if (cudaStreamCreateWithFlags(
                        &stream_, cudaStreamNonBlocking) != cudaSuccess)
                {
                    throw std::runtime_error("failed to create CUDA test stream");
                }
            }

            /** @brief Destroy the owned stream after its test work completes. */
            ~TestCUDAStream()
            {
                if (stream_ != nullptr)
                    (void)cudaStreamDestroy(stream_);
            }

            TestCUDAStream(const TestCUDAStream &) = delete;
            TestCUDAStream &operator=(const TestCUDAStream &) = delete;

            /** @return Native CUDA stream used by test-only copies. */
            [[nodiscard]] cudaStream_t native() const noexcept { return stream_; }

            /** @return Opaque non-null stream expected by production launchers. */
            [[nodiscard]] void *opaque() const noexcept
            {
                return reinterpret_cast<void *>(stream_);
            }

            /**
             * @brief Wait for test-only observation of asynchronous results.
             * @return CUDA status from the explicit stream synchronization.
             */
            [[nodiscard]] cudaError_t synchronize() const noexcept
            {
                return cudaStreamSynchronize(stream_);
            }

        private:
            cudaStream_t stream_ = nullptr;
        };

        /**
         * @brief Own one typed CUDA allocation, including the valid empty case.
         * @tparam T Element type exposed to launch descriptors.
         */
        template <typename T>
        class TestCUDABuffer final
        {
        public:
            /**
             * @brief Allocate `elements` device elements.
             * @param elements Exact logical element capacity; zero stays null.
             */
            explicit TestCUDABuffer(std::size_t elements)
                : elements_(elements)
            {
                if (elements_ != 0 &&
                    cudaMalloc(&pointer_, bytes()) != cudaSuccess)
                {
                    throw std::runtime_error("failed to allocate CUDA test buffer");
                }
            }

            /** @brief Release the device allocation, if one exists. */
            ~TestCUDABuffer()
            {
                if (pointer_ != nullptr)
                    (void)cudaFree(pointer_);
            }

            TestCUDABuffer(const TestCUDABuffer &) = delete;
            TestCUDABuffer &operator=(const TestCUDABuffer &) = delete;

            /** @return Writable device pointer, or null for zero elements. */
            [[nodiscard]] T *data() noexcept { return pointer_; }

            /** @return Read-only device pointer, or null for zero elements. */
            [[nodiscard]] const T *data() const noexcept { return pointer_; }

            /** @return Exact allocation capacity in bytes. */
            [[nodiscard]] std::size_t bytes() const noexcept
            {
                return elements_ * sizeof(T);
            }

            /**
             * @brief Upload a complete host vector with a synchronous test copy.
             * @param source Host elements whose size must equal the allocation.
             * @return CUDA status; empty-to-empty uploads succeed without an API call.
             */
            [[nodiscard]] cudaError_t upload(const std::vector<T> &source) noexcept
            {
                if (source.size() != elements_)
                    return cudaErrorInvalidValue;
                if (elements_ == 0)
                    return cudaSuccess;
                return cudaMemcpy(
                    pointer_, source.data(), bytes(), cudaMemcpyHostToDevice);
            }

            /**
             * @brief Download the complete allocation for byte comparison.
             * @param destination Receives exactly `elements_` host elements.
             * @return CUDA status; empty downloads succeed without an API call.
             */
            [[nodiscard]] cudaError_t download(
                std::vector<T> &destination) const noexcept
            {
                destination.resize(elements_);
                if (elements_ == 0)
                    return cudaSuccess;
                return cudaMemcpy(
                    destination.data(), pointer_, bytes(), cudaMemcpyDeviceToHost);
            }

        private:
            T *pointer_ = nullptr;
            std::size_t elements_ = 0;
        };

        /**
         * @brief Construct deterministic common-GPU bytes with awkward geometry.
         * @param format Packed codebook and metadata selector.
         * @param N Logical output width; defaults to an awkward padded shape.
         * @param K Reduction width, divisible by 32.
         * @return Valid deterministic projection in the requested geometry.
         */
        HostGpuExpertPackedProjection makeProjection(
            const FormatCase &format,
            int N = 70,
            int K = 96)
        {
            HostGpuExpertPackedProjection projection;
            projection.N = N;
            projection.K = K;
            projection.blocks_per_row = static_cast<std::uint32_t>(K / 32);
            projection.source_codebook_id = format.codebook;
            projection.codebook_id = format.codebook;
            projection.payload_bytes_per_block = format.payload_bytes;
            projection.is_asymmetric = format.asymmetric;
            projection.is_superblock = format.superblock;

            const std::size_t blocks =
                static_cast<std::size_t>(projection.N) *
                projection.blocks_per_row;
            projection.payload.resize(blocks * format.payload_bytes);
            projection.scales.resize(blocks);
            if (format.asymmetric)
                projection.mins.resize(blocks);

            // Deterministic non-periodic-looking bytes expose transpose mistakes.
            for (std::size_t block = 0; block < blocks; ++block)
            {
                for (std::size_t byte = 0; byte < format.payload_bytes; ++byte)
                {
                    std::uint8_t value = static_cast<std::uint8_t>(
                        (block * 37u + byte * 19u + 11u) & 0xffu);
                    if (format.codebook == 19 ||
                        format.codebook ==
                            kNativeVnniExpandedInt8MinCodebook)
                    {
                        value = static_cast<std::uint8_t>(
                            static_cast<std::int8_t>(
                                static_cast<int>(
                                    (block * 13u + byte * 7u) % 127u) -
                                63));
                    }
                    projection.payload[
                        block * format.payload_bytes + byte] = value;
                }
                projection.scales[block] = static_cast<std::uint16_t>(
                    0x2400u + (block % 0x0800u));
                if (format.asymmetric)
                {
                    projection.mins[block] = static_cast<std::uint16_t>(
                        0xa000u + (block % 0x0800u));
                }
            }
            return projection;
        }

        /**
         * @brief Pack a real quantized tensor into the common accelerator
         * representation consumed by production GEMM and MoE kernels.
         * @param tensor Source tensor implementing `IINT8Unpackable`.
         * @return Complete host projection with original source provenance.
         * @throws std::invalid_argument when the tensor is not NativeVNNI or
         *         its K dimension is not a whole execution block.
         */
        HostGpuExpertPackedProjection packProductionGpuProjection(
            const TensorBase &tensor)
        {
            const auto *unpackable =
                dynamic_cast<const IINT8Unpackable *>(&tensor);
            if (unpackable == nullptr || unpackable->vnniFormatInfo() == nullptr)
            {
                throw std::invalid_argument(
                    "all-format tier test requires a NativeVNNI tensor");
            }
            const NativeVnniFormatInfo &format =
                *unpackable->vnniFormatInfo();
            const int N = static_cast<int>(tensor.rows());
            const int K = static_cast<int>(tensor.cols());
            if (N <= 0 || K <= 0 || (K % 32) != 0)
                throw std::invalid_argument("invalid all-format tier geometry");

            HostGpuExpertPackedProjection projection;
            projection.N = N;
            projection.K = K;
            projection.blocks_per_row = static_cast<std::uint32_t>(K / 32);
            projection.source_codebook_id = format.codebook_id;
            projection.codebook_id =
                canonicalDeviceVnniCodebookId(format.codebook_id);
            projection.payload_bytes_per_block =
                static_cast<std::uint8_t>(format.payload_bytes);
            projection.is_asymmetric = format.is_asymmetric;
            projection.is_superblock = format.is_superblock;
            projection.has_emins = format.has_emins;

            const std::size_t blocks =
                static_cast<std::size_t>(N) * projection.blocks_per_row;
            projection.payload.resize(blocks * format.payload_bytes);
            projection.scales.resize(blocks);
            if (format.is_asymmetric)
                projection.mins.resize(blocks);
            if (format.has_emins)
                projection.emins.resize(blocks);

            VnniPackContext context{};
            context.N = N;
            context.K = K;
            context.blocks_per_row = K / 32;
            context.payload_bytes = format.payload_bytes;
            context.payload_array = projection.payload.data();
            context.scales_array = projection.scales.data();
            context.mins_array =
                format.is_asymmetric ? projection.mins.data() : nullptr;
            context.emins_array =
                format.has_emins ? projection.emins.data() : nullptr;
            // `packVnniBlock` is the same production source-format authority
            // used by CUDAWeightPacker and ROCmWeightPacker.
            for (int n = 0; n < N; ++n)
            {
                for (int kb = 0; kb < context.blocks_per_row; ++kb)
                    unpackable->packVnniBlock(context, n, n, kb);
            }
            return projection;
        }

        /**
         * @brief Build a read-only device view from owned test allocations.
         * @param payload Device payload owner.
         * @param scales Device scale owner.
         * @param mins Optional device minimum/secondary-scale owner.
         * @return Exact-capacity source view.
         */
        ExpertTierGpuConstProjectionView makeConstView(
            const TestCUDABuffer<std::uint8_t> &payload,
            const TestCUDABuffer<std::uint16_t> &scales,
            const TestCUDABuffer<std::uint16_t> &mins)
        {
            return ExpertTierGpuConstProjectionView{
                .payload = payload.data(),
                .scales = scales.data(),
                .mins = mins.data(),
                .emins = nullptr,
                .payload_bytes = payload.bytes(),
                .scales_bytes = scales.bytes(),
                .mins_bytes = mins.bytes(),
                .emins_bytes = 0,
            };
        }

        /**
         * @brief Build a writable device view from owned test allocations.
         * @param payload Device payload owner.
         * @param scales Device scale owner.
         * @param mins Optional device minimum/secondary-scale owner.
         * @return Exact-capacity destination view.
         */
        ExpertTierGpuMutableProjectionView makeMutableView(
            TestCUDABuffer<std::uint8_t> &payload,
            TestCUDABuffer<std::uint16_t> &scales,
            TestCUDABuffer<std::uint16_t> &mins)
        {
            return ExpertTierGpuMutableProjectionView{
                .payload = payload.data(),
                .scales = scales.data(),
                .mins = mins.data(),
                .emins = nullptr,
                .payload_bytes = payload.bytes(),
                .scales_bytes = scales.bytes(),
                .mins_bytes = mins.bytes(),
                .emins_bytes = 0,
            };
        }

        /**
         * @brief Poll one production tier lane to completion without blocking.
         * @param lane Materialized lane with one active transfer.
         * @return Terminal progress, or `Pending` when the test deadline expires.
         */
        ExpertTierWeightTransferProgress pollLaneToCompletion(
            ExpertTierWeightTransferLane &lane)
        {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            std::string error;
            while (std::chrono::steady_clock::now() < deadline)
            {
                const auto progress = lane.poll(&error);
                if (progress != ExpertTierWeightTransferProgress::Pending)
                {
                    EXPECT_TRUE(error.empty()) << error;
                    return progress;
                }
                std::this_thread::yield();
            }
            return ExpertTierWeightTransferProgress::Pending;
        }

        /** Parameterized CUDA correctness fixture for one physical encoding. */
        class CUDAExpertTierWeightKernelsTest
            : public ::testing::TestWithParam<FormatCase>
        {
        protected:
            /** @brief Skip cleanly when the configured host has no CUDA device. */
            void SetUp() override
            {
                int devices = 0;
                if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                    GTEST_SKIP() << "No CUDA device available";
                ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
                ASSERT_TRUE(cudaNativeVNNIInitIQGridTables_tuned())
                    << "production IQ decode tables must be initialized";
            }
        };

        /**
         * @test TransferEngine exposes exactly the requested reusable CUDA queues.
         *
         * Pointer identity is the contract authority here. PerfStats remains
         * diagnostic evidence and cannot manufacture or validate a live lane.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            PersistentTransferExecutionPoolIsBoundedAndIdempotent)
        {
            constexpr std::size_t lane_count = 4u;
            const DeviceId device = DeviceId::cuda(0);
            const std::string pool_name =
                "cuda_expert_tier_execution_pool_contract";

            const auto first = TransferEngine::instance()
                                   .allocatePersistentTransferExecutionLanes(
                                       lane_count,
                                       device,
                                       pool_name);
            ASSERT_EQ(first.size(), lane_count);
            for (std::size_t lane = 0u; lane < first.size(); ++lane)
            {
                EXPECT_TRUE(first[lane].valid());
                EXPECT_EQ(first[lane].device(), device);
                EXPECT_EQ(first[lane].laneIndex(), lane);
                EXPECT_NE(first[lane].stream(), nullptr);
                for (std::size_t other = lane + 1u;
                     other < first.size();
                     ++other)
                {
                    EXPECT_NE(first[lane].stream(), first[other].stream());
                }
            }

            const auto reused = TransferEngine::instance()
                                    .allocatePersistentTransferExecutionLanes(
                                        lane_count,
                                        device,
                                        pool_name);
            ASSERT_EQ(reused.size(), lane_count);
            for (std::size_t lane = 0u; lane < reused.size(); ++lane)
            {
                EXPECT_TRUE(reused[lane].valid());
                EXPECT_EQ(reused[lane].laneIndex(), lane);
                EXPECT_EQ(reused[lane].stream(), first[lane].stream());
            }
        }

        /**
         * @test Independent CUDA lane events remain correct on one pooled stream.
         *
         * The two submissions race from separate host threads, use disjoint
         * staging slices, and then advance chunk-by-chunk in an interleaved
         * poll loop. This is the exact safe-sharing boundary used when several
         * logical ExpertOverlay operations map to one physical cycle queue.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            SharedPersistentExecutionLanePreservesConcurrentTransfersByteExactly)
        {
            constexpr std::size_t bytes = 1u << 20u;
            constexpr std::size_t staging_bytes = 4096u;
            constexpr int n = 1;
            constexpr int k =
                static_cast<int>(bytes / sizeof(std::uint16_t));
            const DeviceId device = DeviceId::cuda(0);

            const auto staging = TransferEngine::instance()
                                     .allocatePersistentTransferStagingSlices(
                                         staging_bytes,
                                         2u,
                                         device);
            const auto execution = TransferEngine::instance()
                                       .allocatePersistentTransferExecutionLanes(
                                           1u,
                                           device,
                                           "cuda_shared_execution_race_contract")
                                       .front();
            ExpertTierWeightTransferLane first({
                .device = device,
                .staging = staging[0],
                .execution = execution,
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cuda_shared_execution_first",
                .perf_device = "cuda:0",
            });
            ExpertTierWeightTransferLane second({
                .device = device,
                .staging = staging[1],
                .execution = execution,
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cuda_shared_execution_second",
                .perf_device = "cuda:0",
            });
            std::string setup_error;
            ASSERT_TRUE(first.materialize(&setup_error)) << setup_error;
            ASSERT_TRUE(second.materialize(&setup_error)) << setup_error;

            std::array<std::vector<std::uint8_t>, 2> source{
                std::vector<std::uint8_t>(bytes),
                std::vector<std::uint8_t>(bytes),
            };
            for (std::size_t byte = 0u; byte < bytes; ++byte)
            {
                source[0][byte] = static_cast<std::uint8_t>(
                    (byte * 29u + 7u) & 0xffu);
                source[1][byte] = static_cast<std::uint8_t>(
                    (byte * 131u + 19u) & 0xffu);
            }
            TestCUDABuffer<std::uint8_t> first_destination(bytes);
            TestCUDABuffer<std::uint8_t> second_destination(bytes);
            const std::array<ContiguousFloatingPointWeightDescriptor, 2>
                destinations{
                    ContiguousFloatingPointWeightDescriptor{
                        .data = first_destination.data(),
                        .type = TensorType::FP16,
                        .n = n,
                        .k = k,
                        .bytes = bytes,
                    },
                    ContiguousFloatingPointWeightDescriptor{
                        .data = second_destination.data(),
                        .type = TensorType::BF16,
                        .n = n,
                        .k = k,
                        .bytes = bytes,
                    },
                };
            ASSERT_TRUE(destinations[0].valid());
            ASSERT_TRUE(destinations[1].valid());

            std::array<ExpertTierWeightTransferLane *, 2> lanes{
                &first,
                &second,
            };
            std::array<bool, 2> started{false, false};
            std::array<std::string, 2> errors;
            std::thread first_submitter(
                [&]
                {
                    started[0] = lanes[0]->startCpuToGpuContiguous(
                        source[0], destinations[0], &errors[0]);
                });
            std::thread second_submitter(
                [&]
                {
                    started[1] = lanes[1]->startCpuToGpuContiguous(
                        source[1], destinations[1], &errors[1]);
                });
            first_submitter.join();
            second_submitter.join();
            ASSERT_TRUE(started[0]) << errors[0];
            ASSERT_TRUE(started[1]) << errors[1];

            std::array<ExpertTierWeightTransferProgress, 2> progress{
                lanes[0]->progress(),
                lanes[1]->progress(),
            };
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while ((progress[0] == ExpertTierWeightTransferProgress::Pending ||
                    progress[1] == ExpertTierWeightTransferProgress::Pending) &&
                   std::chrono::steady_clock::now() < deadline)
            {
                for (std::size_t lane = 0u; lane < lanes.size(); ++lane)
                {
                    if (progress[lane] ==
                        ExpertTierWeightTransferProgress::Pending)
                    {
                        progress[lane] = lanes[lane]->poll(&errors[lane]);
                    }
                }
                std::this_thread::yield();
            }
            ASSERT_EQ(progress[0], ExpertTierWeightTransferProgress::Ready)
                << errors[0];
            ASSERT_EQ(progress[1], ExpertTierWeightTransferProgress::Ready)
                << errors[1];

            std::vector<std::uint8_t> first_observed;
            std::vector<std::uint8_t> second_observed;
            ASSERT_EQ(
                first_destination.download(first_observed), cudaSuccess);
            ASSERT_EQ(
                second_destination.download(second_observed), cudaSuccess);
            EXPECT_EQ(first_observed, source[0]);
            EXPECT_EQ(second_observed, source[1]);
            for (const auto *lane : lanes)
            {
                const auto stats = lane->stats();
                EXPECT_EQ(stats.transfers_started, 1u);
                EXPECT_EQ(stats.transfers_completed, 1u);
                EXPECT_GT(stats.chunks_submitted, 1u);
                EXPECT_EQ(stats.inference_stream_waits, 0u);
                EXPECT_EQ(stats.blocking_synchronizations, 0u);
            }
        }

        /**
         * @test Both CUDA directions match the host oracle for arbitrary chunks.
         */
        TEST_P(
            CUDAExpertTierWeightKernelsTest,
            ArbitraryOrderedChunksMatchCpuOracleByteForByte)
        {
            const HostGpuExpertPackedProjection source =
                makeProjection(GetParam());
            ASSERT_TRUE(source.valid());

            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(
                source, expected_cpu, &error)) << error;
            const NativeVnniFormatInfo *source_format =
                native_vnni_formats::forSourceIdentity(
                    source.source_codebook_id, source.is_superblock);
            ASSERT_NE(source_format, nullptr);
            const ExpertTierWeightStreamManifest manifest =
                makeGpuToCpuExpertTierWeightStreamManifest(
                    *source_format,
                    source.N,
                    source.K,
                    17,
                    4,
                    9,
                    ExpertTierWeightProjection::Up,
                    2);
            const ExpertTierWeightDeviceLayout layout = manifest.deviceLayout();
            ASSERT_TRUE(layout.valid());
            ASSERT_EQ(layout.unit_count, 6u);

            TestCUDABuffer<std::uint8_t> source_payload(source.payload.size());
            TestCUDABuffer<std::uint16_t> source_scales(source.scales.size());
            TestCUDABuffer<std::uint16_t> source_mins(source.mins.size());
            ASSERT_EQ(source_payload.upload(source.payload), cudaSuccess);
            ASSERT_EQ(source_scales.upload(source.scales), cudaSuccess);
            ASSERT_EQ(source_mins.upload(source.mins), cudaSuccess);
            const auto source_view =
                makeConstView(source_payload, source_scales, source_mins);
            ASSERT_TRUE(source_view.validFor(layout));

            const std::size_t maximum_chunk_bytes = layout.chunkBytes(2);
            TestCUDABuffer<std::uint8_t> device_cpu_chunk(maximum_chunk_bytes);
            TestCUDAStream stream;

            // A null/default stream and an over-capacity range fail before launch.
            EXPECT_FALSE(launchGpuToCpuExpertTierChunkCUDA(
                source_view, layout, 0, 1, device_cpu_chunk.data(),
                device_cpu_chunk.bytes(), nullptr));
            EXPECT_FALSE(launchGpuToCpuExpertTierChunkCUDA(
                source_view, layout, 0, 3, device_cpu_chunk.data(),
                device_cpu_chunk.bytes(), stream.opaque()));

            // Compare each produced chunk at its non-zero global source offset.
            std::vector<std::uint8_t> observed_chunk(maximum_chunk_bytes);
            for (std::uint32_t first_unit = 0; first_unit < layout.unit_count;
                 first_unit += 2)
            {
                const std::uint32_t units =
                    std::min<std::uint32_t>(2, layout.unit_count - first_unit);
                const std::size_t bytes = layout.chunkBytes(units);
                ASSERT_TRUE(launchGpuToCpuExpertTierChunkCUDA(
                    source_view, layout, first_unit, units,
                    device_cpu_chunk.data(), device_cpu_chunk.bytes(),
                    stream.opaque()));
                ASSERT_EQ(cudaMemcpyAsync(
                    observed_chunk.data(), device_cpu_chunk.data(), bytes,
                    cudaMemcpyDeviceToHost, stream.native()), cudaSuccess);
                ASSERT_EQ(stream.synchronize(), cudaSuccess);

                const auto expected_begin =
                    expected_cpu.native_interleaved.begin() +
                    static_cast<std::ptrdiff_t>(
                        layout.chunkBytes(first_unit));
                EXPECT_TRUE(std::equal(
                    observed_chunk.begin(), observed_chunk.begin() + bytes,
                    expected_begin));
            }

            // Consume the same CPU bytes into fresh separated GPU allocations.
            const ExpertTierWeightStreamManifest promotion_manifest =
                makeCpuToGpuExpertTierWeightStreamManifest(
                    expected_cpu,
                    18,
                    4,
                    9,
                    ExpertTierWeightProjection::Up,
                    2);
            const ExpertTierWeightDeviceLayout promotion_layout =
                promotion_manifest.deviceLayout();
            ASSERT_TRUE(promotion_layout.valid());
            ASSERT_EQ(
                promotion_layout.direction,
                ExpertTierWeightConversionDirection::CpuToGpu);
            TestCUDABuffer<std::uint8_t> destination_payload(source.payload.size());
            TestCUDABuffer<std::uint16_t> destination_scales(source.scales.size());
            TestCUDABuffer<std::uint16_t> destination_mins(source.mins.size());
            auto destination_view = makeMutableView(
                destination_payload, destination_scales, destination_mins);
            ASSERT_TRUE(destination_view.validFor(promotion_layout));
            EXPECT_FALSE(launchCpuToGpuExpertTierChunkCUDA(
                device_cpu_chunk.data(), device_cpu_chunk.bytes(),
                promotion_layout,
                0, 1, destination_view, nullptr));

            for (std::uint32_t first_unit = 0;
                 first_unit < promotion_layout.unit_count;
                 first_unit += 2)
            {
                const std::uint32_t units =
                    std::min<std::uint32_t>(
                        2, promotion_layout.unit_count - first_unit);
                const std::size_t bytes = promotion_layout.chunkBytes(units);
                const std::uint8_t *host_source =
                    expected_cpu.native_interleaved.data() +
                    promotion_layout.chunkBytes(first_unit);
                ASSERT_EQ(cudaMemcpyAsync(
                    device_cpu_chunk.data(), host_source, bytes,
                    cudaMemcpyHostToDevice, stream.native()), cudaSuccess);
                ASSERT_TRUE(launchCpuToGpuExpertTierChunkCUDA(
                    device_cpu_chunk.data(), bytes, promotion_layout,
                    first_unit, units, destination_view, stream.opaque()));
            }
            ASSERT_EQ(stream.synchronize(), cudaSuccess);

            HostGpuExpertPackedProjection observed = source;
            observed.payload.clear();
            observed.scales.clear();
            observed.mins.clear();
            ASSERT_EQ(destination_payload.download(observed.payload), cudaSuccess);
            ASSERT_EQ(destination_scales.download(observed.scales), cudaSuccess);
            ASSERT_EQ(destination_mins.download(observed.mins), cudaSuccess);
            EXPECT_EQ(observed.payload, source.payload);
            EXPECT_EQ(observed.scales, source.scales);
            EXPECT_EQ(observed.mins, source.mins);
        }

        INSTANTIATE_TEST_SUITE_P(
            EveryReversiblePreparedEncoding,
            CUDAExpertTierWeightKernelsTest,
            ::testing::Values(
                FormatCase{0, 16, false, false, "Q4_0"},
                FormatCase{4, 16, false, true, "IQ4"},
                FormatCase{5, 16, true, true, "Q4K"},
                FormatCase{8, 24, true, true, "Q6K"},
                FormatCase{19, 32, false, false, "ExpandedInt8"}),
            [](const ::testing::TestParamInfo<FormatCase> &info)
            {
                return info.param.name;
            });

        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            EverySourceFormatMatchesProductionCpuPackerByteForByte)
        {
            constexpr int N = 70;
            constexpr int K = 256;
            constexpr std::uint32_t units_per_chunk = 4;
            ASSERT_EQ(test::quantizedVerifierFormats().size(), 21u);

            TestCUDAStream stream;
            for (std::size_t format_index = 0;
                 format_index < test::quantizedVerifierFormats().size();
                 ++format_index)
            {
                const auto &format =
                    test::quantizedVerifierFormats()[format_index];
                SCOPED_TRACE(format.label);
                auto tensor = format.create(
                    {static_cast<std::size_t>(N),
                     static_cast<std::size_t>(K)},
                    static_cast<std::uint32_t>(73001u + format_index));
                ASSERT_NE(tensor, nullptr);

                HostGpuExpertPackedProjection source =
                    packProductionGpuProjection(*tensor);
                std::string error;
                ASSERT_TRUE(source.valid(&error)) << error;
                cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
                ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                    tensor.get(), expected_cpu));

                const NativeVnniFormatInfo *source_format =
                    native_vnni_formats::forSourceIdentity(
                        source.source_codebook_id,
                        source.is_superblock);
                ASSERT_NE(source_format, nullptr);
                const auto manifest =
                    makeGpuToCpuExpertTierWeightStreamManifest(
                        *source_format,
                        N,
                        K,
                        301,
                        2,
                        5,
                        ExpertTierWeightProjection::Down,
                        units_per_chunk);
                const ExpertTierWeightDeviceLayout layout =
                    manifest.deviceLayout();
                ASSERT_TRUE(layout.valid());
                ASSERT_EQ(
                    expected_cpu.native_interleaved.size(),
                    manifest.total_stream_bytes);

                TestCUDABuffer<std::uint8_t> source_payload(
                    source.payload.size());
                TestCUDABuffer<std::uint16_t> source_scales(
                    source.scales.size());
                TestCUDABuffer<std::uint16_t> source_mins(
                    source.mins.size());
                TestCUDABuffer<std::uint32_t> source_emins(
                    source.emins.size());
                ASSERT_EQ(source_payload.upload(source.payload), cudaSuccess);
                ASSERT_EQ(source_scales.upload(source.scales), cudaSuccess);
                ASSERT_EQ(source_mins.upload(source.mins), cudaSuccess);
                ASSERT_EQ(source_emins.upload(source.emins), cudaSuccess);
                const ExpertTierGpuConstProjectionView source_view{
                    .payload = source_payload.data(),
                    .scales = source_scales.data(),
                    .mins = source_mins.data(),
                    .emins = source_emins.data(),
                    .payload_bytes = source_payload.bytes(),
                    .scales_bytes = source_scales.bytes(),
                    .mins_bytes = source_mins.bytes(),
                    .emins_bytes = source_emins.bytes(),
                };
                ASSERT_TRUE(source_view.validFor(layout));

                TestCUDABuffer<std::uint8_t> device_chunk(
                    layout.chunkBytes(units_per_chunk));
                std::vector<std::uint8_t> observed(
                    layout.chunkBytes(units_per_chunk));
                for (std::uint32_t first_unit = 0;
                     first_unit < layout.unit_count;
                     first_unit += units_per_chunk)
                {
                    const std::uint32_t unit_count =
                        std::min<std::uint32_t>(
                            units_per_chunk,
                            layout.unit_count - first_unit);
                    const std::size_t bytes =
                        layout.chunkBytes(unit_count);
                    ASSERT_TRUE(launchGpuToCpuExpertTierChunkCUDA(
                        source_view,
                        layout,
                        first_unit,
                        unit_count,
                        device_chunk.data(),
                        device_chunk.bytes(),
                        stream.opaque()));
                    ASSERT_EQ(cudaMemcpyAsync(
                        observed.data(),
                        device_chunk.data(),
                        bytes,
                        cudaMemcpyDeviceToHost,
                        stream.native()), cudaSuccess);
                    ASSERT_EQ(stream.synchronize(), cudaSuccess);
                    const auto expected_begin =
                        expected_cpu.native_interleaved.begin() +
                        static_cast<std::ptrdiff_t>(
                            layout.chunkBytes(first_unit));
                    const auto mismatch = std::mismatch(
                        observed.begin(),
                        observed.begin() + bytes,
                        expected_begin);
                    ASSERT_EQ(mismatch.first, observed.begin() + bytes)
                        << "first_unit=" << first_unit
                        << " byte="
                        << std::distance(observed.begin(), mismatch.first)
                        << " observed="
                        << static_cast<unsigned>(*mismatch.first)
                        << " expected="
                        << static_cast<unsigned>(*mismatch.second);
                }
            }
        }

        /**
         * @test The production background lane round-trips exact prepared bytes.
         *
         * This covers its persistent auxiliary stream, source event edge,
         * event-polled chunk pump, pinned DMA storage, and both GPU conversion
         * directions without invoking a stream synchronization inside the lane.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            PersistentBackgroundLaneRoundTripsWithoutBlockingSynchronization)
        {
            const FormatCase format{19, 32, false, false, "ExpandedInt8"};
            const HostGpuExpertPackedProjection source =
                makeProjection(format, 192, 256);
            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(
                source, expected_cpu, &error)) << error;

            const auto *source_format =
                native_vnni_formats::forSourceIdentity(19, false);
            ASSERT_NE(source_format, nullptr);
            const auto demotion_manifest =
                makeGpuToCpuExpertTierWeightStreamManifest(
                    *source_format,
                    source.N,
                    source.K,
                    701,
                    3,
                    11,
                    ExpertTierWeightProjection::Gate,
                    2);
            const auto demotion_layout = demotion_manifest.deviceLayout();

            TestCUDABuffer<std::uint8_t> source_payload(source.payload.size());
            TestCUDABuffer<std::uint16_t> source_scales(source.scales.size());
            TestCUDABuffer<std::uint16_t> source_mins(source.mins.size());
            ASSERT_EQ(source_payload.upload(source.payload), cudaSuccess);
            ASSERT_EQ(source_scales.upload(source.scales), cudaSuccess);
            const auto source_view =
                makeConstView(source_payload, source_scales, source_mins);

            IBackend *backend = getCUDABackend();
            ASSERT_NE(backend, nullptr);
            TestCUDAStream producer_stream;
            void *source_ready = backend->createEvent(0);
            ASSERT_NE(source_ready, nullptr);
            ASSERT_TRUE(backend->recordEvent(
                source_ready, 0, producer_stream.opaque()));

            ExpertTierWeightTransferLane lane({
                .device = DeviceId::cuda(0),
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   demotion_layout.chunkBytes(2),
                                   1u,
                                   DeviceId::cuda(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::cuda(0),
                                     "cuda_tier_round_trip")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cuda_tier_round_trip",
                .perf_device = "cuda:0",
            });
            ASSERT_TRUE(lane.materialize(&error)) << error;
            std::vector<std::uint8_t> observed_cpu(
                expected_cpu.native_interleaved.size());
            ASSERT_TRUE(lane.startGpuToCpu(
                demotion_layout,
                source_view,
                observed_cpu,
                ExpertTierSourceReadiness::producerEvent(source_ready),
                &error)) << error;
            ASSERT_EQ(
                pollLaneToCompletion(lane),
                ExpertTierWeightTransferProgress::Ready);
            EXPECT_EQ(
                observed_cpu.size(),
                expected_cpu.native_interleaved.size());
            EXPECT_TRUE(std::equal(
                observed_cpu.begin(),
                observed_cpu.end(),
                expected_cpu.native_interleaved.begin()));
            backend->destroyEvent(source_ready, 0);

            const auto promotion_manifest =
                makeCpuToGpuExpertTierWeightStreamManifest(
                    expected_cpu,
                    702,
                    3,
                    11,
                    ExpertTierWeightProjection::Gate,
                    2);
            const auto promotion_layout = promotion_manifest.deviceLayout();
            TestCUDABuffer<std::uint8_t> destination_payload(
                source.payload.size());
            TestCUDABuffer<std::uint16_t> destination_scales(
                source.scales.size());
            TestCUDABuffer<std::uint16_t> destination_mins(
                source.mins.size());
            auto destination_view = makeMutableView(
                destination_payload,
                destination_scales,
                destination_mins);
            ASSERT_TRUE(lane.startCpuToGpu(
                promotion_layout,
                expected_cpu.native_interleaved,
                destination_view,
                &error)) << error;
            ASSERT_EQ(
                pollLaneToCompletion(lane),
                ExpertTierWeightTransferProgress::Ready);

            HostGpuExpertPackedProjection observed_gpu = source;
            observed_gpu.payload.clear();
            observed_gpu.scales.clear();
            observed_gpu.mins.clear();
            ASSERT_EQ(
                destination_payload.download(observed_gpu.payload),
                cudaSuccess);
            ASSERT_EQ(
                destination_scales.download(observed_gpu.scales),
                cudaSuccess);
            EXPECT_EQ(observed_gpu.payload, source.payload);
            EXPECT_EQ(observed_gpu.scales, source.scales);

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_started, 2u);
            EXPECT_EQ(stats.transfers_completed, 2u);
            EXPECT_GT(stats.chunks_submitted, 2u);
            EXPECT_GT(stats.bytes_submitted, 0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
        }

        /**
         * @test A shared maintenance thread restores the lane's exact CUDA device.
         *
         * The source arrays and auxiliary stream belong to device one, while
         * the calling thread is deliberately switched back to device zero
         * immediately before submission.  This is the production shape when a
         * rank progresses several GPU participants.  The remote projection
         * lane must select its own device without synchronizing and emit the
         * same final CPU bytes as the device-free packer.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            SecondaryDeviceRemoteProjectionRestoresOwningDeviceBeforeRepack)
        {
            int device_count = 0;
            ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
            if (device_count < 2)
                GTEST_SKIP() << "Secondary-device projection requires two CUDA devices";

            constexpr int device_ordinal = 1;
            constexpr int N = 70;
            constexpr int K = 96;
            constexpr std::uint32_t units_per_chunk = 2;
            ASSERT_EQ(cudaSetDevice(device_ordinal), cudaSuccess);

            const auto &format = test::quantizedVerifierFormats().at(18);
            ASSERT_STREQ(format.label, "Q8_0");
            auto tensor = format.create({N, K}, 0xc0da51ecu);
            ASSERT_NE(tensor, nullptr);
            const HostGpuExpertPackedProjection source =
                packProductionGpuProjection(*tensor);
            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                tensor.get(), expected_cpu));
            const auto *source_format =
                native_vnni_formats::forSourceIdentity(
                    source.source_codebook_id,
                    source.is_superblock);
            ASSERT_NE(source_format, nullptr);
            const auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format,
                N,
                K,
                /*epoch=*/7701,
                /*layer_idx=*/4,
                /*expert_id=*/9,
                ExpertTierWeightProjection::Up,
                units_per_chunk);
            const auto layout = manifest.deviceLayout();
            ASSERT_TRUE(layout.valid());

            TestCUDABuffer<std::uint8_t> source_payload(source.payload.size());
            TestCUDABuffer<std::uint16_t> source_scales(source.scales.size());
            TestCUDABuffer<std::uint16_t> source_mins(source.mins.size());
            TestCUDABuffer<std::uint32_t> source_emins(source.emins.size());
            ASSERT_EQ(source_payload.upload(source.payload), cudaSuccess);
            ASSERT_EQ(source_scales.upload(source.scales), cudaSuccess);
            ASSERT_EQ(source_mins.upload(source.mins), cudaSuccess);
            ASSERT_EQ(source_emins.upload(source.emins), cudaSuccess);
            const ExpertTierGpuConstProjectionView source_view{
                .payload = source_payload.data(),
                .scales = source_scales.data(),
                .mins = source_mins.data(),
                .emins = source_emins.data(),
                .payload_bytes = source_payload.bytes(),
                .scales_bytes = source_scales.bytes(),
                .mins_bytes = source_mins.bytes(),
                .emins_bytes = source_emins.bytes(),
            };
            ASSERT_TRUE(source_view.validFor(layout));

            auto lane = std::make_shared<MoEOverlayGpuRemoteProjectionLane>(
                MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = DeviceId::cuda(device_ordinal),
                    .staging = TransferEngine::instance()
                                   .allocatePersistentTransferStagingSlices(
                                       layout.chunkBytes(units_per_chunk),
                                       1u,
                                       DeviceId::cuda(device_ordinal))
                                   .front(),
                    .execution = TransferEngine::instance()
                                     .allocatePersistentTransferExecutionLanes(
                                         1u,
                                         DeviceId::cuda(device_ordinal),
                                         "cuda_secondary_device_remote_projection")
                                     .front(),
                    .progress = BackgroundTransferProgressBinding::nativeStream(),
                    .lane_name = "cuda_secondary_device_remote_projection",
                    .perf_device = "cuda:1",
                });
            std::string error;
            ASSERT_TRUE(lane->materialize(&error)) << error;
            const int owner = 1;
            ASSERT_TRUE(lane->tryAcquire(&owner));
            ASSERT_TRUE(lane->bindSourceReadiness(
                &owner,
                ExpertTierSourceReadiness::publishedResidencyBank(7701),
                &error)) << error;

            /* Reproduce a maintenance thread last used by another device. */
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            const std::uint32_t unit_count = std::min<std::uint32_t>(
                units_per_chunk, layout.unit_count);
            const std::size_t bytes = layout.chunkBytes(unit_count);
            EXPECT_TRUE(lane->submitGpuToCpuRepack(
                &owner,
                layout,
                source_view,
                /*first_unit=*/0,
                unit_count,
                &error)) << error;

            auto progress = MoEOverlayGpuRemoteLaneProgress::Pending;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (progress == MoEOverlayGpuRemoteLaneProgress::Pending &&
                   std::chrono::steady_clock::now() < deadline)
            {
                progress = lane->poll(&owner, &error);
                std::this_thread::yield();
            }
            EXPECT_EQ(progress, MoEOverlayGpuRemoteLaneProgress::Ready)
                << error;
            if (progress == MoEOverlayGpuRemoteLaneProgress::Ready)
            {
                const auto observed = lane->pinnedOutput(&owner, bytes);
                ASSERT_EQ(observed.size(), bytes);
                EXPECT_TRUE(std::equal(
                    observed.begin(),
                    observed.end(),
                    expected_cpu.native_interleaved.begin()));
            }
            int selected_device = -1;
            ASSERT_EQ(cudaGetDevice(&selected_device), cudaSuccess);
            EXPECT_EQ(selected_device, device_ordinal);
            EXPECT_TRUE(lane->release(&owner, &error)) << error;

            const auto stats = lane->stats();
            EXPECT_EQ(stats.gpu_to_cpu_repack_chunks, 1u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
        }

        /**
         * @test Every floating precision uses the event-polled contiguous lane.
         *
         * An awkward 257-byte staging capacity forces multiple chunks for
         * FP16, BF16, and FP32. The producer upload is ordered by an explicit
         * event on a different non-default stream; both directions must retain
         * every source byte and publish production PerfStats evidence.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            ContiguousFloatingLaneRoundTripsAllPrecisionsWithoutInferenceWaits)
        {
            ScopedPerfStats perf_stats;
            constexpr int n = 19;
            constexpr int k = 23;
            constexpr std::size_t staging_bytes = 257;
            const std::array<std::pair<TensorType, std::size_t>, 3> formats{
                std::pair{TensorType::FP16, sizeof(std::uint16_t)},
                std::pair{TensorType::BF16, sizeof(std::uint16_t)},
                std::pair{TensorType::FP32, sizeof(float)},
            };

            IBackend *backend = getCUDABackend();
            ASSERT_NE(backend, nullptr);
            ExpertTierWeightTransferLane lane({
                .device = DeviceId::cuda(0),
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   staging_bytes,
                                   1u,
                                   DeviceId::cuda(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::cuda(0),
                                     "cuda_floating_tier_round_trip")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cuda_floating_tier_round_trip",
                .perf_device = "cuda:0",
            });
            std::string error;
            ASSERT_TRUE(lane.materialize(&error)) << error;

            for (const auto [type, element_bytes] : formats)
            {
                const std::size_t bytes =
                    static_cast<std::size_t>(n) * k * element_bytes;
                std::vector<std::uint8_t> source(bytes);
                for (std::size_t byte = 0; byte < source.size(); ++byte)
                {
                    source[byte] = static_cast<std::uint8_t>(
                        (byte * 131u + element_bytes * 17u) & 0xffu);
                }

                TestCUDABuffer<std::uint8_t> gpu_source(bytes);
                TestCUDAStream producer_stream;
                ASSERT_EQ(
                    cudaMemcpyAsync(
                        gpu_source.data(),
                        source.data(),
                        bytes,
                        cudaMemcpyHostToDevice,
                        producer_stream.native()),
                    cudaSuccess);
                void *source_ready = backend->createEvent(0);
                ASSERT_NE(source_ready, nullptr);
                ASSERT_TRUE(backend->recordEvent(
                    source_ready, 0, producer_stream.opaque()));

                const ContiguousFloatingPointWeightDescriptor
                    source_descriptor{
                        .data = gpu_source.data(),
                        .type = type,
                        .n = n,
                        .k = k,
                        .bytes = bytes,
                    };
                std::vector<std::uint8_t> observed_cpu(bytes);
                ASSERT_TRUE(lane.startGpuToCpuContiguous(
                    source_descriptor,
                    observed_cpu,
                    ExpertTierSourceReadiness::producerEvent(source_ready),
                    &error)) << error;
                ASSERT_EQ(
                    pollLaneToCompletion(lane),
                    ExpertTierWeightTransferProgress::Ready);
                EXPECT_EQ(observed_cpu, source);
                backend->destroyEvent(source_ready, 0);

                TestCUDABuffer<std::uint8_t> gpu_destination(bytes);
                const ContiguousFloatingPointWeightDescriptor
                    destination_descriptor{
                        .data = gpu_destination.data(),
                        .type = type,
                        .n = n,
                        .k = k,
                        .bytes = bytes,
                    };
                ASSERT_TRUE(lane.startCpuToGpuContiguous(
                    source, destination_descriptor, &error)) << error;
                ASSERT_EQ(
                    pollLaneToCompletion(lane),
                    ExpertTierWeightTransferProgress::Ready);
                std::vector<std::uint8_t> observed_gpu;
                ASSERT_EQ(
                    gpu_destination.download(observed_gpu), cudaSuccess);
                EXPECT_EQ(observed_gpu, source);
            }

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_started, 6u);
            EXPECT_EQ(stats.transfers_completed, 6u);
            EXPECT_GT(stats.chunks_submitted, stats.transfers_completed);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);

            const auto records = PerfStatsCollector::snapshot(
                {"moe_overlay_residency.tier_transfers_completed"});
            const auto observed_direction = [&](const char *direction)
            {
                return std::any_of(
                    records.begin(),
                    records.end(),
                    [&](const PerfStatRecord &record)
                    {
                        return perfTag(record, "direction") == direction &&
                               record.value == 3.0;
                    });
            };
            EXPECT_TRUE(observed_direction("gpu_to_cpu"));
            EXPECT_TRUE(observed_direction("cpu_to_gpu"));
        }

        /**
         * @test Promoted asymmetric CPU weights remain executable on CUDA.
         *
         * Q5_1 loses its compact 5-bit representation in the CPU cold tier.
         * Promotion must therefore publish execution codebook 23: signed INT8
         * payload plus the original FP16 minimum. This test streams those real
         * CPU-native bytes through the persistent production lane, then compares
         * CUDA decode and verifier-depth output words against the original
         * compact codebook-7 matrix. N=512,K=2048 deliberately selects a
         * different generated serial split-K count for source codebook 7 and
         * physical codebook 23 on Ampere, proving that promotion retains the
         * source arithmetic policy rather than merely decoding equal values.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            RemoteCpuGpuEndpointsStreamQ51WithoutInferenceWaits)
        {
            ScopedPerfStats perf_stats;
            constexpr int N = 70;
            constexpr int K = 96;
            constexpr std::uint64_t promoted_epoch = 9201;
            const auto &format = test::quantizedVerifierFormat("Q5_1");
            auto tensor = format.create({N, K}, 0xc0da51u);
            ASSERT_NE(tensor, nullptr);

            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu_weights;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                tensor.get(), cpu_weights));
            ASSERT_TRUE(cpu_weights.usesExpandedInt8());
            ASSERT_TRUE(cpu_weights.is_asymmetric);
            HostGpuExpertPackedProjection expected_gpu;
            std::string error;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(
                cpu_weights, expected_gpu, &error))
                << error;
            ASSERT_EQ(
                expected_gpu.codebook_id,
                kNativeVnniExpandedInt8MinCodebook);

            const MoEOverlayRemoteProjectionIdentity promotion_identity{
                .expected_epoch = promoted_epoch - 1,
                .candidate_epoch = promoted_epoch,
                .execution_fingerprint = {
                    .low = 0xc0da5101u,
                    .high = 0xc0da5102u,
                },
                .migration_index = 3,
                .layer_idx = 2,
                .expert_id = 19,
                .projection = ExpertTierWeightProjection::Down,
                .source_participant = 0,
                .destination_participant = 1,
                .source_world_rank = 0,
                .destination_world_rank = 1,
                .source_device = DeviceId::cpu(),
                .destination_device = DeviceId::cuda(0),
            };
            ASSERT_TRUE(promotion_identity.valid());
            const auto promotion_stream =
                makeCpuToGpuExpertTierWeightStreamManifest(
                    cpu_weights,
                    promotion_identity.candidate_epoch,
                    promotion_identity.layer_idx,
                    promotion_identity.expert_id,
                    promotion_identity.projection,
                    /*maximum_units_per_chunk=*/2);
            const auto promotion_manifest =
                makeMoEOverlayRemoteCpuProjectionManifest(
                    promotion_identity,
                    promotion_stream,
                    /*maximum_chunk_bytes=*/1u << 20u);
            ASSERT_TRUE(promotion_manifest.valid(&error)) << error;

            TestCUDABuffer<std::uint8_t> destination_payload(
                expected_gpu.payload.size());
            TestCUDABuffer<std::uint16_t> destination_scales(
                expected_gpu.scales.size());
            TestCUDABuffer<std::uint16_t> destination_mins(
                expected_gpu.mins.size());
            TestCUDABuffer<std::uint32_t> destination_emins(
                expected_gpu.emins.size());
            const GpuExpertPackedDescriptor destination_descriptor{
                .ptrs = {
                    .d_vnni = destination_payload.data(),
                    .d_scales = destination_scales.data(),
                    .d_mins = destination_mins.data(),
                    .d_emins = destination_emins.data(),
                },
                .n = N,
                .k = K,
                .blocks_per_row = static_cast<std::uint32_t>(K / 32),
                .codebook_id = expected_gpu.codebook_id,
                .payload_bytes_per_block =
                    expected_gpu.payload_bytes_per_block,
                .is_asymmetric = expected_gpu.is_asymmetric,
                .has_emins = expected_gpu.has_emins,
                .vnni_bytes = destination_payload.bytes(),
                .scales_bytes = destination_scales.bytes(),
                .mins_bytes = destination_mins.bytes(),
                .emins_bytes = destination_emins.bytes(),
            };
            ASSERT_TRUE(destination_descriptor.valid());

            auto slot_lifetime = std::make_shared<int>(51);
            auto lane = std::make_shared<
                MoEOverlayGpuRemoteProjectionLane>(
                MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = DeviceId::cuda(0),
                    .staging = TransferEngine::instance()
                                   .allocatePersistentTransferStagingSlices(
                                       promotion_manifest.maximum_chunk_bytes,
                                       1u,
                                       DeviceId::cuda(0))
                                   .front(),
                    .execution = TransferEngine::instance()
                                     .allocatePersistentTransferExecutionLanes(
                                         1u,
                                         DeviceId::cuda(0),
                                         "cuda_remote_q51_roundtrip")
                                     .front(),
                    .progress = BackgroundTransferProgressBinding::nativeStream(),
                    .lane_name = "cuda_remote_q51_roundtrip",
                    .perf_device = "cuda:0",
                });
            ASSERT_TRUE(lane->materialize(&error)) << error;

            const NativeVnniSourceIdentity source_identity{
                .codebook_id = cpu_weights.codebook_id,
                .is_superblock = cpu_weights.is_superblock,
                .present = true,
            };
            MoEOverlayGpuRemoteProjectionDestination destination(
                promotion_identity,
                lane,
                [&, slot_lifetime](
                    const MoEOverlayRemoteProjectionManifest &manifest,
                    MoEOverlayGpuRemoteProjectionDestinationBinding *binding,
                    std::string *factory_error) -> bool
                {
                    if (!binding ||
                        manifest.gpu_codebook_id !=
                            destination_descriptor.codebook_id)
                    {
                        if (factory_error)
                            *factory_error =
                                "CUDA destination factory received an unexpected physical format";
                        return false;
                    }
                    auto engine = std::make_shared<
                        cuda::CUDAQuantisedGemmKernel>(
                        N,
                        K,
                        0,
                        destination_descriptor.ptrs.d_vnni,
                        static_cast<std::uint16_t *>(
                            destination_descriptor.ptrs.d_scales),
                        static_cast<std::uint16_t *>(
                            destination_descriptor.ptrs.d_mins),
                        static_cast<std::uint32_t *>(
                            destination_descriptor.ptrs.d_emins),
                        destination_descriptor.codebook_id,
                        destination_descriptor.blocks_per_row,
                        slot_lifetime,
                        source_identity);
                    *binding = {
                        .descriptor = destination_descriptor,
                        .engine = std::move(engine),
                    };
                    if (factory_error)
                        factory_error->clear();
                    return true;
                },
                slot_lifetime);
            ASSERT_TRUE(destination.beginManifest(
                promotion_manifest, &error))
                << error;

            const std::array<std::span<const std::uint8_t>, 4>
                cpu_regions{
                    std::span<const std::uint8_t>(
                        cpu_weights.native_interleaved.data(),
                        cpu_weights.native_interleaved.size()),
                    std::span<const std::uint8_t>{},
                    std::span<const std::uint8_t>{},
                    std::span<const std::uint8_t>{},
                };
            MoEOverlayRemoteProjectionChunkCursor cursor(
                promotion_manifest, cpu_regions);
            std::uint64_t promotion_chunks = 0;
            while (auto chunk = cursor.takeNext())
            {
                auto progress = destination.beginChunk(
                    chunk->header, chunk->payload, &error);
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::seconds(10);
                while (progress == MoEOverlayResidencyWaveProgress::Pending &&
                       std::chrono::steady_clock::now() < deadline)
                {
                    progress = destination.pollChunk(&error);
                    std::this_thread::yield();
                }
                ASSERT_EQ(progress, MoEOverlayResidencyWaveProgress::Ready)
                    << error;
                ++promotion_chunks;
            }
            ASSERT_TRUE(cursor.complete());
            ASSERT_TRUE(destination.complete());
            ASSERT_TRUE(destination.publishFinal(&error)) << error;
            ASSERT_NE(destination.preparedEngine(), nullptr);

            std::vector<std::uint8_t> observed_payload;
            std::vector<std::uint16_t> observed_scales;
            std::vector<std::uint16_t> observed_mins;
            std::vector<std::uint32_t> observed_emins;
            ASSERT_EQ(
                destination_payload.download(observed_payload), cudaSuccess);
            ASSERT_EQ(
                destination_scales.download(observed_scales), cudaSuccess);
            ASSERT_EQ(destination_mins.download(observed_mins), cudaSuccess);
            ASSERT_EQ(destination_emins.download(observed_emins), cudaSuccess);
            EXPECT_EQ(observed_payload, expected_gpu.payload);
            EXPECT_EQ(observed_scales, expected_gpu.scales);
            EXPECT_EQ(observed_mins, expected_gpu.mins);
            EXPECT_EQ(observed_emins, expected_gpu.emins);

            const auto *source_format =
                native_vnni_formats::forSourceIdentity(
                    cpu_weights.codebook_id,
                    cpu_weights.is_superblock);
            ASSERT_NE(source_format, nullptr);
            const MoEOverlayRemoteProjectionIdentity demotion_identity{
                .expected_epoch = promoted_epoch,
                .candidate_epoch = promoted_epoch + 1,
                .execution_fingerprint = {
                    .low = 0xc0da5201u,
                    .high = 0xc0da5202u,
                },
                .migration_index = 4,
                .layer_idx = promotion_identity.layer_idx,
                .expert_id = promotion_identity.expert_id,
                .projection = promotion_identity.projection,
                .source_participant = 1,
                .destination_participant = 0,
                .source_world_rank = 1,
                .destination_world_rank = 0,
                .source_device = DeviceId::cuda(0),
                .destination_device = DeviceId::cpu(),
            };
            const auto demotion_stream =
                makeGpuToCpuExpertTierWeightStreamManifest(
                    *source_format,
                    destination_descriptor,
                    demotion_identity.expected_epoch,
                    demotion_identity.layer_idx,
                    demotion_identity.expert_id,
                    demotion_identity.projection,
                    /*maximum_units_per_chunk=*/2);
            const auto demotion_manifest =
                makeMoEOverlayRemoteCpuProjectionManifest(
                    demotion_identity,
                    demotion_stream,
                    /*maximum_chunk_bytes=*/1u << 20u);
            auto source_lifetime = destination.preparedEngine();
            ASSERT_NE(source_lifetime, nullptr);
            MoEOverlayGpuRemoteProjectionSource source(
                demotion_manifest,
                lane,
                destination_descriptor,
                ExpertTierSourceReadiness::publishedResidencyBank(
                    promoted_epoch),
                source_lifetime);
            MoEOverlayRemoteProjectionChunkValidator validator(
                demotion_manifest);
            std::vector<std::uint8_t> redemoted_cpu_bytes(
                cpu_weights.native_interleaved.size(), 0xa5u);
            std::uint64_t demotion_chunks = 0;
            while (!validator.complete())
            {
                MoEOverlayRemoteProjectionChunkView chunk;
                auto progress = source.pollNextChunk(&chunk, &error);
                const auto deadline = std::chrono::steady_clock::now() +
                                      std::chrono::seconds(10);
                while (progress == MoEOverlayResidencyWaveProgress::Pending &&
                       std::chrono::steady_clock::now() < deadline)
                {
                    progress = source.pollNextChunk(&chunk, &error);
                    std::this_thread::yield();
                }
                ASSERT_EQ(progress, MoEOverlayResidencyWaveProgress::Ready)
                    << error;
                ASSERT_TRUE(validator.accept(
                    chunk.header, chunk.payload, &error))
                    << error;
                std::memcpy(
                    redemoted_cpu_bytes.data() + chunk.header.region_offset,
                    chunk.payload.data(),
                    chunk.payload.size());
                ASSERT_TRUE(source.acknowledgeChunkSent(
                    chunk.header, &error))
                    << error;
                ++demotion_chunks;
            }
            EXPECT_EQ(redemoted_cpu_bytes.size(),
                      cpu_weights.native_interleaved.size());
            EXPECT_TRUE(std::equal(
                redemoted_cpu_bytes.begin(),
                redemoted_cpu_bytes.end(),
                cpu_weights.native_interleaved.begin()));
            EXPECT_GT(promotion_chunks, 1u);
            EXPECT_GT(demotion_chunks, 1u);

            const auto stats = lane->stats();
            EXPECT_EQ(stats.cpu_to_gpu_repack_chunks, promotion_chunks);
            EXPECT_EQ(stats.gpu_to_cpu_repack_chunks, demotion_chunks);
            EXPECT_EQ(stats.host_to_device_submissions, promotion_chunks);
            EXPECT_EQ(stats.device_to_host_submissions, demotion_chunks);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
            EXPECT_EQ(stats.published_bank_sources, demotion_chunks);
        }

        /**
         * @brief Prove one real asymmetric source format across promotion and
         *        every production row regime.
         * @param format_name Quantized verifier format used to create real
         *                    source bytes.
         * @param N Logical projection output width.
         * @param K Logical projection reduction width.
         * @param seed Deterministic source-weight seed.
         *
         * Superblock formats cannot be represented by arbitrary separated GPU
         * bytes.  Starting from the real tensor packer is therefore part of the
         * proof: the source arithmetic identity must survive CPU preparation,
         * streamed expansion, redemotion, and promoted execution.
         */
        void provePromotedAsymmetricWeightsExecuteByteExactly(
            const char *format_name,
            int N,
            int K,
            std::uint32_t seed)
        {
            ScopedPerfStats perf_stats;
            constexpr int verifier_rows = 4;
            constexpr int prefill_rows = 32;
            const auto &format = test::quantizedVerifierFormat(format_name);
            auto tensor = format.create(
                {static_cast<std::size_t>(N), static_cast<std::size_t>(K)},
                seed);
            ASSERT_NE(tensor, nullptr);
            const HostGpuExpertPackedProjection compact =
                packProductionGpuProjection(*tensor);
            ASSERT_EQ(compact.codebook_id, 7u);
            ASSERT_TRUE(compact.is_asymmetric);

            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu_weights;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                tensor.get(), cpu_weights));
            ASSERT_TRUE(cpu_weights.usesExpandedInt8());
            ASSERT_TRUE(cpu_weights.is_asymmetric);

            const auto probe_manifest =
                makeCpuToGpuExpertTierWeightStreamManifest(
                cpu_weights,
                7301,
                2,
                19,
                ExpertTierWeightProjection::Down,
                1);
            const auto probe_layout = probe_manifest.deviceLayout();
            ASSERT_TRUE(probe_layout.valid());
            const std::size_t production_staging_bytes =
                MoEOverlayPhysicalResidencyFabric::Config{}
                    .staging_capacity_bytes;
            const auto production_units_per_chunk =
                static_cast<std::uint32_t>(std::min<std::size_t>(
                    probe_layout.unit_count,
                    production_staging_bytes /
                        probe_layout.cpu_block_stride));
            ASSERT_GT(production_units_per_chunk, 1u);
            const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu_weights,
                7301,
                2,
                19,
                ExpertTierWeightProjection::Down,
                production_units_per_chunk);
            const auto layout = manifest.deviceLayout();
            ASSERT_TRUE(layout.valid());
            ASSERT_EQ(
                layout.gpu_codebook_id,
                kNativeVnniExpandedInt8MinCodebook);
            const std::size_t blocks =
                static_cast<std::size_t>(N) * (K / 32);

            TestCUDABuffer<std::uint8_t> promoted_payload(
                blocks * layout.gpu_payload_bytes_per_block);
            TestCUDABuffer<std::uint16_t> promoted_scales(blocks);
            TestCUDABuffer<std::uint16_t> promoted_mins(blocks);
            ExpertTierGpuMutableProjectionView promoted_view{
                .payload = promoted_payload.data(),
                .scales = promoted_scales.data(),
                .mins = promoted_mins.data(),
                .emins = nullptr,
                .payload_bytes = promoted_payload.bytes(),
                .scales_bytes = promoted_scales.bytes(),
                .mins_bytes = promoted_mins.bytes(),
                .emins_bytes = 0,
            };
            ASSERT_TRUE(promoted_view.validFor(layout));

            std::string error;
            ExpertTierWeightTransferLane lane({
                .device = DeviceId::cuda(0),
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   production_staging_bytes,
                                   1u,
                                   DeviceId::cuda(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::cuda(0),
                                     "cuda_asymmetric_promotion_execution")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cuda_asymmetric_promotion_execution",
                .perf_device = "cuda:0",
            });
            ASSERT_TRUE(lane.materialize(&error)) << error;
            ASSERT_TRUE(lane.startCpuToGpu(
                layout,
                cpu_weights.native_interleaved,
                promoted_view,
                &error)) << error;
            ASSERT_EQ(
                pollLaneToCompletion(lane),
                ExpertTierWeightTransferProgress::Ready);

            /*
             * Q5_1 no longer exists as compact five-bit bytes after the first
             * promotion: the live GPU bank is the normalized asymmetric INT8
             * representation.  Exercise the next migration hop from that
             * exact published representation.  Reconstructing the original
             * CPU execution bytes here proves that an arbitrary
             * CPU -> GPU -> CPU tier cycle does not depend on stale compact
             * source storage or a host shadow of the GPU bank.
             */
            const auto *source_format =
                native_vnni_formats::forSourceIdentity(
                    compact.source_codebook_id,
                    compact.is_superblock);
            ASSERT_NE(source_format, nullptr);
            const GpuExpertPackedDescriptor promoted_descriptor{
                .ptrs = {
                    .d_vnni = promoted_payload.data(),
                    .d_scales = promoted_scales.data(),
                    .d_mins = promoted_mins.data(),
                    .d_emins = nullptr,
                },
                .n = N,
                .k = K,
                .blocks_per_row = static_cast<std::uint32_t>(K / 32),
                .codebook_id = layout.gpu_codebook_id,
                .payload_bytes_per_block =
                    layout.gpu_payload_bytes_per_block,
                .is_asymmetric = true,
                .has_emins = false,
                .vnni_bytes = promoted_payload.bytes(),
                .scales_bytes = promoted_scales.bytes(),
                .mins_bytes = promoted_mins.bytes(),
                .emins_bytes = 0,
            };
            ASSERT_TRUE(promoted_descriptor.valid());

            /*
             * Production does not publish the transfer view directly.  It
             * retains an ITensorGemm alias over the inactive slot and later
             * exports that alias into the immutable runtime bank.  Exercise
             * that exact handoff here so a wrapper that reallocates, drops
             * provenance, or silently rebinds a pointer cannot pass the raw
             * repack proof while poisoning live sparse execution.
             */
            auto slot_lifetime = std::make_shared<int>(1);
            auto promoted_engine = std::make_shared<
                cuda::CUDAQuantisedGemmKernel>(
                N,
                K,
                0,
                promoted_payload.data(),
                promoted_scales.data(),
                promoted_mins.data(),
                nullptr,
                layout.gpu_codebook_id,
                static_cast<std::uint32_t>(K / 32),
                slot_lifetime,
                NativeVnniSourceIdentity{
                    .codebook_id = compact.source_codebook_id,
                    .is_superblock = compact.is_superblock,
                    .present = true,
                },
                NativeVnniReusableDeviceAllocationFormat{
                    .payload_bytes_per_block =
                        layout.gpu_payload_bytes_per_block,
                    .has_mins = layout.gpu_is_asymmetric != 0,
                    .has_emins = layout.gpu_has_emins != 0,
                });
            DeviceNativeVNNIMatrixDesc published_descriptor;
            ASSERT_TRUE(promoted_engine->exportNativeVNNIMatrixDesc(
                published_descriptor));
            ASSERT_TRUE(published_descriptor.valid());
            EXPECT_EQ(published_descriptor.payload, promoted_payload.data());
            EXPECT_EQ(published_descriptor.scales, promoted_scales.data());
            EXPECT_EQ(published_descriptor.mins, promoted_mins.data());
            EXPECT_EQ(published_descriptor.emins, nullptr);
            EXPECT_EQ(published_descriptor.n, N);
            EXPECT_EQ(published_descriptor.k, K);
            EXPECT_EQ(
                published_descriptor.codebook_id,
                layout.gpu_codebook_id);
            EXPECT_EQ(
                published_descriptor.arithmeticPolicyCodebookId(),
                compact.source_codebook_id);
            const auto redemotion_manifest =
                makeGpuToCpuExpertTierWeightStreamManifest(
                    *source_format,
                    promoted_descriptor,
                    7301,
                    2,
                    19,
                    ExpertTierWeightProjection::Down,
                    1);
            const auto redemotion_layout =
                redemotion_manifest.deviceLayout();
            ASSERT_TRUE(redemotion_layout.valid());
            ASSERT_EQ(
                redemotion_layout.gpu_codebook_id,
                kNativeVnniExpandedInt8MinCodebook);
            const ExpertTierGpuConstProjectionView promoted_source_view{
                .payload = promoted_payload.data(),
                .scales = promoted_scales.data(),
                .mins = promoted_mins.data(),
                .emins = nullptr,
                .payload_bytes = promoted_payload.bytes(),
                .scales_bytes = promoted_scales.bytes(),
                .mins_bytes = promoted_mins.bytes(),
                .emins_bytes = 0,
            };
            ASSERT_TRUE(promoted_source_view.validFor(redemotion_layout));
            std::vector<std::uint8_t> redemoted_cpu_bytes(
                cpu_weights.native_interleaved.size(),
                0xa5u);
            ASSERT_TRUE(lane.startGpuToCpu(
                redemotion_layout,
                promoted_source_view,
                redemoted_cpu_bytes,
                ExpertTierSourceReadiness::publishedResidencyBank(7301),
                &error)) << error;
            ASSERT_EQ(
                pollLaneToCompletion(lane),
                ExpertTierWeightTransferProgress::Ready);
            ASSERT_EQ(
                redemoted_cpu_bytes.size(),
                cpu_weights.native_interleaved.size());
            EXPECT_TRUE(std::equal(
                redemoted_cpu_bytes.begin(),
                redemoted_cpu_bytes.end(),
                cpu_weights.native_interleaved.begin()));

            TestCUDABuffer<std::uint8_t> compact_payload(compact.payload.size());
            TestCUDABuffer<std::uint16_t> compact_scales(compact.scales.size());
            TestCUDABuffer<std::uint16_t> compact_mins(compact.mins.size());
            ASSERT_EQ(compact_payload.upload(compact.payload), cudaSuccess);
            ASSERT_EQ(compact_scales.upload(compact.scales), cudaSuccess);
            ASSERT_EQ(compact_mins.upload(compact.mins), cudaSuccess);

            const std::size_t activation_elements =
                static_cast<std::size_t>(prefill_rows) * K;
            const std::size_t activation_blocks_per_row =
                static_cast<std::size_t>(K) / 32u;
            const std::size_t activation_block_elements =
                static_cast<std::size_t>(prefill_rows) *
                activation_blocks_per_row;
            std::vector<std::int8_t> host_activation(activation_elements);
            std::vector<std::int32_t> host_activation_sums(
                activation_block_elements, 0);
            for (std::size_t index = 0; index < activation_elements; ++index)
            {
                host_activation[index] = static_cast<std::int8_t>(
                    static_cast<int>((index * 11u + 3u) % 29u) - 14);
                const std::size_t row = index / static_cast<std::size_t>(K);
                const std::size_t column =
                    index % static_cast<std::size_t>(K);
                const std::size_t block = column / 32u;
                host_activation_sums[
                    row * activation_blocks_per_row + block] +=
                    host_activation[index];
            }
            std::vector<float> host_activation_scales(
                activation_block_elements);
            for (int row = 0; row < prefill_rows; ++row)
            {
                for (std::size_t block = 0;
                     block < activation_blocks_per_row;
                     ++block)
                {
                    host_activation_scales[
                        static_cast<std::size_t>(row) *
                            activation_blocks_per_row +
                        block] =
                        0.03137f + static_cast<float>(row) * 0.00019f +
                        static_cast<float>(block) * 0.000003f;
                }
            }
            TestCUDABuffer<std::int8_t> activation(activation_elements);
            TestCUDABuffer<float> activation_scales(
                activation_block_elements);
            TestCUDABuffer<std::int32_t> activation_sums(
                activation_block_elements);
            ASSERT_EQ(activation.upload(host_activation), cudaSuccess);
            ASSERT_EQ(
                activation_scales.upload(host_activation_scales),
                cudaSuccess);
            ASSERT_EQ(activation_sums.upload(host_activation_sums), cudaSuccess);

            TestCUDAStream stream;
            ScopedDecodeEquivalentM1 decode_equivalent_scope;
            CUDAGemvContext *gemv_context = cudaGemvContext_create(0);
            ASSERT_NE(gemv_context, nullptr);
            TestCUDABuffer<float> partials(
                static_cast<std::size_t>(
                    NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
                verifier_rows * N);
            cudaGemvContext_bindWorkspace(
                gemv_context,
                partials.data(),
                partials.bytes());

            CUDAPrefillContext *prefill_context =
                cudaPrefillContext_create(0);
            ASSERT_NE(prefill_context, nullptr);
            std::size_t compact_prefill_workspace_bytes = 0;
            std::size_t promoted_prefill_workspace_bytes = 0;
            int planned_partitions = 1;
            ASSERT_TRUE(cudaNativeVNNIPrefill_getWorkspacePlanWithPolicy(
                compact.codebook_id,
                compact.source_codebook_id,
                prefill_rows,
                N,
                K,
                0,
                &compact_prefill_workspace_bytes,
                &planned_partitions));
            ASSERT_TRUE(cudaNativeVNNIPrefill_getWorkspacePlanWithPolicy(
                layout.gpu_codebook_id,
                layout.gpu_source_codebook_id,
                prefill_rows,
                N,
                K,
                0,
                &promoted_prefill_workspace_bytes,
                &planned_partitions));
            const std::size_t prefill_workspace_bytes = std::max(
                compact_prefill_workspace_bytes,
                promoted_prefill_workspace_bytes);
            TestCUDABuffer<float> prefill_partials(
                (prefill_workspace_bytes + sizeof(float) - 1u) /
                sizeof(float));
            cudaPrefillContext_bindWorkspace(
                prefill_context,
                prefill_partials.data(),
                prefill_partials.bytes());

            for (const int rows : {1, verifier_rows, prefill_rows})
            {
                SCOPED_TRACE("rows=" + std::to_string(rows));
                TestCUDABuffer<float> compact_output(
                    static_cast<std::size_t>(rows) * N);
                TestCUDABuffer<float> promoted_output(
                    static_cast<std::size_t>(rows) * N);
                const auto launch = [&](
                    const std::uint8_t *payload,
                    const std::uint16_t *scales,
                    const std::uint16_t *mins,
                    std::uint8_t codebook,
                    std::uint8_t arithmetic_policy_codebook,
                    float *output)
                {
                    if (rows == 1)
                    {
                        return cudaNativeVNNIGemvTuned_fp32_withPolicy(
                            activation.data(), payload, scales, mins, nullptr,
                            output, activation_scales.data(), N, K,
                            1.0f, 0.0f, nullptr, nullptr, codebook,
                            arithmetic_policy_codebook, 0,
                            stream.opaque(), gemv_context, nullptr);
                    }
                    if (rows == verifier_rows)
                    {
                        return cudaNativeVNNIGemvTuned_small_m_fp32_withPolicy(
                            activation.data(), payload, scales, mins, nullptr,
                            output, activation_scales.data(), rows, N, K,
                            1.0f, 0.0f, nullptr, nullptr, codebook,
                            arithmetic_policy_codebook, 0,
                            stream.opaque(), gemv_context, nullptr);
                    }
                    return cudaNativeVNNIPrefill_fp32_withPolicy(
                        activation.data(), payload, scales, mins, nullptr,
                        output, activation_scales.data(), activation_sums.data(),
                        rows, N, K, 1.0f, 0.0f, nullptr, nullptr, codebook,
                        arithmetic_policy_codebook, 0, stream.opaque(),
                        prefill_context);
                };
                ASSERT_TRUE(launch(
                    compact_payload.data(),
                    compact_scales.data(),
                    compact_mins.data(),
                    compact.codebook_id,
                    compact.source_codebook_id,
                    compact_output.data()));
                ASSERT_TRUE(launch(
                    published_descriptor.payload,
                    static_cast<const std::uint16_t *>(
                        published_descriptor.scales),
                    static_cast<const std::uint16_t *>(
                        published_descriptor.mins),
                    published_descriptor.codebook_id,
                    published_descriptor.arithmeticPolicyCodebookId(),
                    promoted_output.data()));
                ASSERT_EQ(stream.synchronize(), cudaSuccess);

                std::vector<float> expected;
                std::vector<float> actual;
                ASSERT_EQ(compact_output.download(expected), cudaSuccess);
                ASSERT_EQ(promoted_output.download(actual), cudaSuccess);
                ASSERT_EQ(expected.size(), actual.size());
                ASSERT_TRUE(std::any_of(
                    expected.begin(), expected.end(),
                    [](float value) { return value != 0.0f; }));
                EXPECT_EQ(
                    std::memcmp(
                        expected.data(), actual.data(),
                        expected.size() * sizeof(float)),
                    0) << "CUDA promoted asymmetric execution changed FP32 words";
            }

            const auto decode_records = PerfStatsCollector::snapshot(
                {"kernel.cuda_native_vnni_gemv_dispatch"});
            const auto prefill_records = PerfStatsCollector::snapshot(
                {"kernel.cuda_native_vnni_prefill_calls"});
            const auto has_policy_record = [](
                const std::vector<PerfStatRecord> &records,
                int rows,
                int execution_codebook,
                int arithmetic_policy_codebook)
            {
                return std::any_of(
                    records.begin(),
                    records.end(),
                    [&](const PerfStatRecord &record)
                    {
                        return perfTag(record, "m") ==
                                   std::to_string(rows) &&
                               perfTag(record, "codebook") ==
                                   std::to_string(execution_codebook) &&
                               perfTag(
                                   record,
                                   "arithmetic_policy_codebook") ==
                                   std::to_string(
                                       arithmetic_policy_codebook);
                    });
            };
            EXPECT_TRUE(has_policy_record(
                decode_records, 1, compact.codebook_id, compact.codebook_id));
            EXPECT_TRUE(has_policy_record(
                decode_records,
                1,
                layout.gpu_codebook_id,
                compact.codebook_id));
            EXPECT_TRUE(has_policy_record(
                decode_records,
                verifier_rows,
                layout.gpu_codebook_id,
                compact.codebook_id));
            EXPECT_TRUE(has_policy_record(
                prefill_records,
                prefill_rows,
                layout.gpu_codebook_id,
                compact.codebook_id));
            EXPECT_FALSE(has_policy_record(
                decode_records, 1, layout.gpu_codebook_id, 19));
            EXPECT_FALSE(has_policy_record(
                prefill_records, prefill_rows, layout.gpu_codebook_id, 19));

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_completed, 2u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            cudaPrefillContext_destroy(prefill_context);
            cudaGemvContext_destroy(gemv_context);
        }

        /**
         * @test Promoted Q5 asymmetric formats preserve exact production math.
         *
         * Q5_1 retains the original gate/up-oriented regression. Q5_K uses the
         * Qwen down-projection orientation implicated by the real-model
         * Dynamic failure, where a broken down projection zeros the complete
         * expert contribution even when gate and up are valid.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            PromotedAsymmetricWeightsExecuteDecodeAndVerifierByteExactly)
        {
            ASSERT_NO_FATAL_FAILURE(
                provePromotedAsymmetricWeightsExecuteByteExactly(
                    "Q5_1", 512, 2048, 99173u));
            ASSERT_NO_FATAL_FAILURE(
                provePromotedAsymmetricWeightsExecuteByteExactly(
                    "Q5_K", 2048, 512, 99179u));
        }

        /**
         * @test A real shadow-slot allocation receives every promoted Q5_K byte.
         *
         * The lower-level repack proof owns tightly sized test allocations. The
         * production physical fabric instead writes into a model-lifetime
         * `GpuExpertSlotPool`: each NativeVNNI projection reserves the largest
         * supported payload/min/emins family and exposes only the arriving
         * format's logical regions. This regression closes that allocation
         * boundary so offset aliasing, capacity accounting, or slot-pool layout
         * changes cannot turn a successfully completed promotion into an
         * all-zero executable expert.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            ProductionShadowSlotReceivesPromotedQ5KBytesExactly)
        {
            constexpr int N = 512;
            constexpr int K = 2048;
            constexpr std::uint64_t candidate_epoch = 81u;
            const auto &format = test::quantizedVerifierFormat("Q5_K");
            auto tensor = format.create({N, K}, 0xc0da5b1u);
            ASSERT_NE(tensor, nullptr);

            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu_weights;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                tensor.get(), cpu_weights));
            ASSERT_TRUE(cpu_weights.usesExpandedInt8());
            ASSERT_TRUE(cpu_weights.is_asymmetric);

            HostGpuExpertPackedProjection expected;
            std::string error;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(
                cpu_weights, expected, &error))
                << error;
            ASSERT_EQ(
                expected.codebook_id,
                kNativeVnniExpandedInt8MinCodebook);

            const NativeVnniSourceIdentity source_identity{
                .codebook_id = cpu_weights.codebook_id,
                .is_superblock = cpu_weights.is_superblock,
                .present = true,
            };
            IBackend *backend = getCUDABackend();
            ASSERT_NE(backend, nullptr);
            auto pool = GpuExpertSlotPool::createForTest(
                backend,
                DeviceId::cuda(0),
                /*device_ordinal=*/0,
                /*layer_idx=*/7,
                /*active_capacity=*/1,
                {GpuExpertSlotPool::ProjectionSpec{
                    .label = "gate",
                    .N = N,
                    .K = K,
                    .payload_bytes_per_block = 32,
                    .is_asymmetric = true,
                    .has_emins = true,
                    .codebook_id = expected.codebook_id,
                    .format = ExpertWeightFormat::nativeVnni(
                        source_identity),
                }},
                /*transfer_capacity=*/0);
            ASSERT_NE(pool, nullptr);
            auto lease = pool->acquire(/*expert_id=*/123, candidate_epoch);
            ASSERT_TRUE(lease.has_value());
            ASSERT_EQ(lease->projections.size(), 1u);
            const auto &slot = lease->projections.front().slot;

            const auto probe = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu_weights,
                candidate_epoch,
                /*layer_idx=*/7,
                /*expert_id=*/123,
                ExpertTierWeightProjection::Gate,
                /*maximum_units_per_chunk=*/1);
            const auto probe_layout = probe.deviceLayout();
            const std::size_t staging_bytes =
                MoEOverlayPhysicalResidencyFabric::Config{}
                    .staging_capacity_bytes;
            const auto maximum_units = static_cast<std::uint32_t>(
                std::min<std::size_t>(
                    probe_layout.unit_count,
                    staging_bytes / probe_layout.cpu_block_stride));
            ASSERT_GT(maximum_units, 0u);
            const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu_weights,
                candidate_epoch,
                /*layer_idx=*/7,
                /*expert_id=*/123,
                ExpertTierWeightProjection::Gate,
                maximum_units);
            const auto layout = manifest.deviceLayout();
            ASSERT_TRUE(layout.valid());

            const std::size_t logical_blocks =
                static_cast<std::size_t>(N) * (K / 32);
            const std::size_t logical_payload_bytes =
                logical_blocks * layout.gpu_payload_bytes_per_block;
            const std::size_t logical_scales_bytes =
                logical_blocks * sizeof(std::uint16_t);
            const std::size_t logical_mins_bytes =
                logical_blocks * sizeof(std::uint16_t);
            ASSERT_GE(slot.payload_bytes, logical_payload_bytes);
            ASSERT_GE(slot.scales_bytes, logical_scales_bytes);
            ASSERT_GE(slot.mins_bytes, logical_mins_bytes);

            ExpertTierGpuMutableProjectionView destination{
                .payload = slot.d_native_vnni_payload,
                .scales = static_cast<std::uint16_t *>(
                    slot.d_native_vnni_scales),
                .mins = static_cast<std::uint16_t *>(
                    slot.d_native_vnni_mins),
                .emins = static_cast<std::uint32_t *>(
                    slot.d_native_vnni_emins),
                // The production descriptor exposes exact logical regions
                // over the deliberately larger reusable slot allocation.
                .payload_bytes = logical_payload_bytes,
                .scales_bytes = logical_scales_bytes,
                .mins_bytes = logical_mins_bytes,
                .emins_bytes = 0u,
            };
            ASSERT_TRUE(destination.validFor(layout));

            ExpertTierWeightTransferLane lane({
                .device = DeviceId::cuda(0),
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   staging_bytes,
                                   1u,
                                   DeviceId::cuda(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::cuda(0),
                                     "cuda_production_shadow_slot_q5k")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cuda_production_shadow_slot_q5k",
                .perf_device = "cuda:0",
            });
            ASSERT_TRUE(lane.materialize(&error)) << error;
            ASSERT_TRUE(lane.startCpuToGpu(
                layout,
                cpu_weights.native_interleaved,
                destination,
                &error)) << error;
            ASSERT_EQ(
                pollLaneToCompletion(lane),
                ExpertTierWeightTransferProgress::Ready);

            std::vector<std::uint8_t> observed_payload(
                expected.payload.size());
            std::vector<std::uint16_t> observed_scales(
                expected.scales.size());
            std::vector<std::uint16_t> observed_mins(
                expected.mins.size());
            ASSERT_EQ(
                cudaMemcpy(
                    observed_payload.data(),
                    slot.d_native_vnni_payload,
                    observed_payload.size(),
                    cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(
                    observed_scales.data(),
                    slot.d_native_vnni_scales,
                    observed_scales.size() * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(
                    observed_mins.data(),
                    slot.d_native_vnni_mins,
                    observed_mins.size() * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost),
                cudaSuccess);
            EXPECT_EQ(observed_payload, expected.payload);
            EXPECT_EQ(observed_scales, expected.scales);
            EXPECT_EQ(observed_mins, expected.mins);
            EXPECT_EQ(pool->slotForExpert(123, candidate_epoch), 0);
            EXPECT_EQ(lane.stats().blocking_synchronizations, 0u);
            EXPECT_EQ(lane.stats().inference_stream_waits, 0u);
        }

        /**
         * @test Real NativeVNNI inference completes unchanged during migration.
         *
         * The baseline and concurrent launches use the same production GEMV,
         * inputs, weights, and explicit inference stream. A 24-chunk demotion
         * runs on the separate persistent transfer lane. The test requires an
         * inference completion event while the residency work is still pending
         * and then compares output bytes exactly.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            ProductionInferenceCompletesByteExactlyWhileMigrationIsPending)
        {
            const FormatCase format{0, 16, false, false, "Q4_0"};
            const HostGpuExpertPackedProjection source =
                makeProjection(format, 192, 256);
            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(
                source, expected_cpu, &error)) << error;
            const auto *source_format =
                native_vnni_formats::forSourceIdentity(0, false);
            ASSERT_NE(source_format, nullptr);
            const auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format,
                source.N,
                source.K,
                801,
                4,
                13,
                ExpertTierWeightProjection::Down,
                1);
            const auto layout = manifest.deviceLayout();
            ASSERT_EQ(layout.unit_count, 24u);

            TestCUDABuffer<std::uint8_t> payload(source.payload.size());
            TestCUDABuffer<std::uint16_t> scales(source.scales.size());
            TestCUDABuffer<std::uint16_t> mins(source.mins.size());
            ASSERT_EQ(payload.upload(source.payload), cudaSuccess);
            ASSERT_EQ(scales.upload(source.scales), cudaSuccess);
            const auto source_view = makeConstView(payload, scales, mins);

            std::vector<std::int8_t> host_activation(
                static_cast<std::size_t>(source.K), 1);
            std::vector<float> host_activation_scales(
                source.blocks_per_row, 1.0f);
            TestCUDABuffer<std::int8_t> activation(host_activation.size());
            TestCUDABuffer<float> activation_scales(
                host_activation_scales.size());
            TestCUDABuffer<float> baseline_output(
                static_cast<std::size_t>(source.N));
            TestCUDABuffer<float> concurrent_output(
                static_cast<std::size_t>(source.N));
            ASSERT_EQ(activation.upload(host_activation), cudaSuccess);
            ASSERT_EQ(
                activation_scales.upload(host_activation_scales),
                cudaSuccess);

            TestCUDAStream inference_stream;
            ScopedDecodeEquivalentM1 decode_equivalent_scope;
            CUDAGemvContext *gemv_context = cudaGemvContext_create(0);
            ASSERT_NE(gemv_context, nullptr);
            TestCUDABuffer<float> gemv_partials(
                static_cast<std::size_t>(
                    NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
                static_cast<std::size_t>(source.N));
            cudaGemvContext_bindWorkspace(
                gemv_context,
                gemv_partials.data(),
                gemv_partials.bytes());
            ASSERT_TRUE(cudaNativeVNNIGemvTuned_fp32(
                activation.data(),
                payload.data(),
                scales.data(),
                nullptr,
                nullptr,
                baseline_output.data(),
                activation_scales.data(),
                source.N,
                source.K,
                1.0f,
                0.0f,
                nullptr,
                nullptr,
                source.codebook_id,
                0,
                inference_stream.opaque(),
                gemv_context,
                nullptr));
            ASSERT_EQ(inference_stream.synchronize(), cudaSuccess);

            IBackend *backend = getCUDABackend();
            ASSERT_NE(backend, nullptr);
            void *source_ready = backend->createEvent(0);
            void *inference_done = backend->createEvent(0);
            ASSERT_NE(source_ready, nullptr);
            ASSERT_NE(inference_done, nullptr);
            ASSERT_TRUE(backend->recordEvent(
                source_ready, 0, inference_stream.opaque()));

            ExpertTierWeightTransferLane lane({
                .device = DeviceId::cuda(0),
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   layout.chunkBytes(1),
                                   1u,
                                   DeviceId::cuda(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::cuda(0),
                                     "cuda_inference_overlap")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "cuda_inference_overlap",
                .perf_device = "cuda:0",
            });
            ASSERT_TRUE(lane.materialize(&error)) << error;
            std::vector<std::uint8_t> observed_cpu(
                expected_cpu.native_interleaved.size());
            ASSERT_TRUE(lane.startGpuToCpu(
                layout,
                source_view,
                observed_cpu,
                ExpertTierSourceReadiness::producerEvent(source_ready),
                &error)) << error;

            ASSERT_TRUE(cudaNativeVNNIGemvTuned_fp32(
                activation.data(),
                payload.data(),
                scales.data(),
                nullptr,
                nullptr,
                concurrent_output.data(),
                activation_scales.data(),
                source.N,
                source.K,
                1.0f,
                0.0f,
                nullptr,
                nullptr,
                source.codebook_id,
                0,
                inference_stream.opaque(),
                gemv_context,
                nullptr));
            ASSERT_TRUE(backend->recordEvent(
                inference_done, 0, inference_stream.opaque()));

            bool inference_completed_while_migration_pending = false;
            bool inference_ready = false;
            auto migration_progress = lane.progress();
            auto &gpu_context =
                GPUDeviceContextPool::instance().getContext(DeviceId::cuda(0));
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline &&
                   (migration_progress ==
                        ExpertTierWeightTransferProgress::Pending ||
                    !inference_ready))
            {
                ASSERT_TRUE(gpu_context.queryEventChecked(
                    inference_done, inference_ready));
                if (inference_ready &&
                    lane.progress() ==
                        ExpertTierWeightTransferProgress::Pending)
                {
                    inference_completed_while_migration_pending = true;
                }
                migration_progress = lane.poll(&error);
                ASSERT_NE(
                    migration_progress,
                    ExpertTierWeightTransferProgress::Failed) << error;
                std::this_thread::yield();
            }

            EXPECT_TRUE(inference_ready);
            EXPECT_TRUE(inference_completed_while_migration_pending)
                << "The production inference stream must not wait for the migration lane";
            EXPECT_EQ(
                migration_progress,
                ExpertTierWeightTransferProgress::Ready);
            EXPECT_EQ(
                observed_cpu.size(),
                expected_cpu.native_interleaved.size());
            EXPECT_TRUE(std::equal(
                observed_cpu.begin(),
                observed_cpu.end(),
                expected_cpu.native_interleaved.begin()));

            std::vector<float> baseline;
            std::vector<float> concurrent;
            ASSERT_EQ(baseline_output.download(baseline), cudaSuccess);
            ASSERT_EQ(concurrent_output.download(concurrent), cudaSuccess);
            ASSERT_EQ(baseline.size(), concurrent.size());
            EXPECT_EQ(
                std::memcmp(
                    baseline.data(),
                    concurrent.data(),
                    baseline.size() * sizeof(float)),
                0) << "Concurrent migration changed production inference bytes";

            const auto stats = lane.stats();
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
            EXPECT_EQ(stats.transfers_completed, 1u);
            EXPECT_EQ(stats.chunks_submitted, layout.unit_count);

            backend->destroyEvent(inference_done, 0);
            backend->destroyEvent(source_ready, 0);
            cudaGemvContext_destroy(gemv_context);
        }

        /**
         * @test FusedQKV's internal CUDA stream pool remains independent of migration.
         *
         * The shared harness alternates decode and prefill, reuses the Q/K/V
         * side-stream events repeatedly, and requires exact serial-branch output
         * while a long ExpertOverlay demotion is still advancing.
         */
        TEST_F(
            CUDAExpertTierWeightKernelsTest,
            FusedQKVInternalStreamPoolsRemainByteExactAndNonBlockingDuringMigration)
        {
            const FormatCase format{0, 16, false, false, "Q4_0"};
            const HostGpuExpertPackedProjection source =
                makeProjection(format, 1024, 256);
            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(
                source, expected_cpu, &error)) << error;
            const auto *source_format =
                native_vnni_formats::forSourceIdentity(0, false);
            ASSERT_NE(source_format, nullptr);
            const auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format,
                source.N,
                source.K,
                811,
                5,
                17,
                ExpertTierWeightProjection::Up,
                1);
            const auto layout = manifest.deviceLayout();
            ASSERT_GE(layout.unit_count, 64u);

            TestCUDABuffer<std::uint8_t> payload(source.payload.size());
            TestCUDABuffer<std::uint16_t> scales(source.scales.size());
            TestCUDABuffer<std::uint16_t> mins(source.mins.size());
            ASSERT_EQ(payload.upload(source.payload), cudaSuccess);
            ASSERT_EQ(scales.upload(source.scales), cudaSuccess);
            const auto source_view = makeConstView(payload, scales, mins);

            IBackend *backend = getCUDABackend();
            ASSERT_NE(backend, nullptr);
            TestCUDAStream root_stream;
            test::runFusedQKVStreamPoolMigrationStress(
                DeviceId::cuda(0),
                root_stream.opaque(),
                *backend,
                layout,
                source_view,
                std::span<const std::uint8_t>(
                    expected_cpu.native_interleaved.data(),
                    expected_cpu.native_interleaved.size()),
                "cuda_fused_qkv_migration");
        }
    } // namespace
} // namespace llaminar2
