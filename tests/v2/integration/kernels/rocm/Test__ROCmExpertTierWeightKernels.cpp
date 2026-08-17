/**
 * @file Test__ROCmExpertTierWeightKernels.cpp
 * @brief Byte-exact ROCm parity for direct ExpertOverlay tier conversion.
 *
 * Every cataloged quantized tensor is packed through the same source method
 * used by production ROCm weight preparation. The HIP demotion kernel converts
 * that projection in non-zero-origin chunks, and every resulting byte is
 * compared with the CPU NativeVNNI packer. The reverse half then streams those
 * CPU bytes back through the production promotion kernel and compares every
 * separated destination array with the device-free reference.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "execution/moe/MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "kernels/common/NativeVNNIGroupedDecodePolicy.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmWeightPacker.h"
#include "kernels/rocm/repack/ROCmExpertTierWeightKernels.h"
#include "tensors/VnniPackContext.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ExpertTierFusedQKVStreamPoolHarness.h"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

extern "C"
{
    /** Select the serial-decode arithmetic policy for the production GEMV. */
    void rocmGemv_native_vnni_set_decode_equivalent_m1_config(int enabled);

    /** Production ROCm NativeVNNI M=1 inference kernel used by overlap tests. */
    bool rocmGemv_native_vnni_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const void *d_block_scales,
        const void *d_block_mins,
        const void *d_block_emins,
        float *d_C_fp32,
        const float *d_scale_A,
        float *d_partial_fp32,
        int N,
        int K,
        std::uint8_t codebook_id,
        int device_id,
        void *stream,
        const float *d_scale_A_blockwise);

    /** Production M=1 kernel with explicit source arithmetic policy. */
    bool rocmGemv_native_vnni_fp32_with_policy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const void *d_block_scales,
        const void *d_block_mins,
        const void *d_block_emins,
        float *d_C_fp32,
        const float *d_scale_A,
        float *d_partial_fp32,
        int N,
        int K,
        std::uint8_t codebook_id,
        std::uint8_t arithmetic_policy_codebook_id,
        int device_id,
        void *stream,
        const float *d_scale_A_blockwise);

    /** Production verifier-depth NativeVNNI kernel used by promotion parity. */
    bool rocmGemv_native_vnni_small_m_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const void *d_block_scales,
        const void *d_block_mins,
        const void *d_block_emins,
        float *d_C_fp32,
        const float *d_scale_A_blockwise,
        float *d_partial_fp32,
        int M,
        int N,
        int K,
        std::uint8_t codebook_id,
        int device_id,
        void *stream);

    /** Production verifier kernel with explicit source arithmetic policy. */
    bool rocmGemv_native_vnni_small_m_fp32_with_policy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const void *d_block_scales,
        const void *d_block_mins,
        const void *d_block_emins,
        float *d_C_fp32,
        const float *d_scale_A_blockwise,
        float *d_partial_fp32,
        int M,
        int N,
        int K,
        std::uint8_t codebook_id,
        std::uint8_t arithmetic_policy_codebook_id,
        int device_id,
        void *stream);

    /** Production dense-prefill NativeVNNI kernel used by promotion parity. */
    bool rocmGemm_native_vnni_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const void *d_block_scales,
        const void *d_block_mins,
        const void *d_block_emins,
        float *d_output,
        const float *d_scales_A,
        const float *d_scales_A_blockwise,
        int M,
        int N,
        int K,
        std::uint8_t codebook_id,
        int device_id,
        void *stream);

    /** Production prefill kernel with explicit source arithmetic policy. */
    bool rocmGemm_native_vnni_fp32_with_policy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        const void *d_block_scales,
        const void *d_block_mins,
        const void *d_block_emins,
        float *d_output,
        const float *d_scales_A,
        const float *d_scales_A_blockwise,
        int M,
        int N,
        int K,
        std::uint8_t codebook_id,
        std::uint8_t arithmetic_policy_codebook_id,
        int device_id,
        void *stream);

    /** Query exact generated serial-M1 geometry for a source policy. */
    bool rocmGemv_native_vnni_query_serial_m1_config_with_policy(
        std::uint8_t arithmetic_policy_codebook_id,
        int N,
        int K,
        int *kb,
        int *target_waves_per_cu);
}

namespace llaminar2
{
    namespace
    {
        /** @brief Enable route evidence for one test without leaking env state. */
        class ScopedPerfStats final
        {
        public:
            /** @brief Save the process setting, enable collection, and reset. */
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

            /** @brief Restore the process setting and discard test records. */
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

        /** @return One PerfStats tag value, or an empty string when absent. */
        std::string perfTag(
            const PerfStatRecord &record,
            const std::string &name)
        {
            const auto found = record.tags.find(name);
            return found == record.tags.end() ? std::string{} : found->second;
        }

        /** Keep production M=1 arithmetic selected for the scoped proof. */
        class ScopedROCmDecodeEquivalentM1 final
        {
        public:
            /** @brief Enable the serial-row dispatch contract. */
            ScopedROCmDecodeEquivalentM1()
            {
                rocmGemv_native_vnni_set_decode_equivalent_m1_config(1);
            }

            /** @brief Restore the ordinary non-verifier dispatch contract. */
            ~ScopedROCmDecodeEquivalentM1()
            {
                rocmGemv_native_vnni_set_decode_equivalent_m1_config(0);
            }

            ScopedROCmDecodeEquivalentM1(
                const ScopedROCmDecodeEquivalentM1 &) = delete;
            ScopedROCmDecodeEquivalentM1 &operator=(
                const ScopedROCmDecodeEquivalentM1 &) = delete;
        };

        /**
         * @brief Own one non-default HIP stream for an integration-test scope.
         *
         * Throwing construction prevents a failed stream creation from quietly
         * turning a correctness test into a legacy-default-stream launch.
         */
        class TestHIPStream final
        {
        public:
            /** @brief Create one non-blocking HIP stream. */
            TestHIPStream()
            {
                if (hipStreamCreateWithFlags(
                        &stream_, hipStreamNonBlocking) != hipSuccess)
                {
                    throw std::runtime_error(
                        "failed to create ROCm tier-stream test stream");
                }
            }

            /** @brief Destroy the owned stream after all observed work ends. */
            ~TestHIPStream()
            {
                if (stream_ != nullptr)
                    (void)hipStreamDestroy(stream_);
            }

            TestHIPStream(const TestHIPStream &) = delete;
            TestHIPStream &operator=(const TestHIPStream &) = delete;

            /** @return Native HIP stream for explicit test copies. */
            [[nodiscard]] hipStream_t native() const noexcept
            {
                return stream_;
            }

            /** @return Opaque non-null stream required by production launchers. */
            [[nodiscard]] void *opaque() const noexcept
            {
                return reinterpret_cast<void *>(stream_);
            }

            /**
             * @brief Wait only at a test observation boundary.
             * @return HIP status for the exact owned stream.
             */
            [[nodiscard]] hipError_t synchronize() const noexcept
            {
                return hipStreamSynchronize(stream_);
            }

        private:
            hipStream_t stream_ = nullptr;
        };

        /**
         * @brief Own one typed HIP allocation, including an empty optional array.
         * @tparam T Element type exposed to the production descriptor.
         */
        template <typename T>
        class TestHIPBuffer final
        {
        public:
            /**
             * @brief Allocate an exact number of device elements.
             * @param elements Logical element count; zero deliberately stays null.
             */
            explicit TestHIPBuffer(std::size_t elements)
                : elements_(elements)
            {
                if (elements_ != 0 &&
                    hipMalloc(
                        reinterpret_cast<void **>(&pointer_), bytes()) !=
                        hipSuccess)
                {
                    throw std::runtime_error(
                        "failed to allocate ROCm tier-stream test buffer");
                }
            }

            /** @brief Release the owned device allocation, when non-empty. */
            ~TestHIPBuffer()
            {
                if (pointer_ != nullptr)
                    (void)hipFree(pointer_);
            }

            TestHIPBuffer(const TestHIPBuffer &) = delete;
            TestHIPBuffer &operator=(const TestHIPBuffer &) = delete;

            /** @return Writable device pointer, or null for an empty buffer. */
            [[nodiscard]] T *data() noexcept { return pointer_; }

            /** @return Read-only device pointer, or null for an empty buffer. */
            [[nodiscard]] const T *data() const noexcept { return pointer_; }

            /** @return Exact allocation capacity in bytes. */
            [[nodiscard]] std::size_t bytes() const noexcept
            {
                return elements_ * sizeof(T);
            }

            /**
             * @brief Upload a complete host vector before kernel execution.
             * @param source Host elements whose size must match the allocation.
             * @return HIP copy status, or invalid-value for a size mismatch.
             */
            [[nodiscard]] hipError_t upload(
                const std::vector<T> &source) noexcept
            {
                if (source.size() != elements_)
                    return hipErrorInvalidValue;
                if (elements_ == 0)
                    return hipSuccess;
                return hipMemcpy(
                    pointer_, source.data(), bytes(), hipMemcpyHostToDevice);
            }

            /**
             * @brief Download the complete allocation at a test boundary.
             * @param destination Receives exactly the allocation's elements.
             * @return HIP copy status.
             */
            [[nodiscard]] hipError_t download(
                std::vector<T> &destination) const noexcept
            {
                destination.resize(elements_);
                if (elements_ == 0)
                    return hipSuccess;
                return hipMemcpy(
                    destination.data(), pointer_, bytes(),
                    hipMemcpyDeviceToHost);
            }

        private:
            T *pointer_ = nullptr;
            std::size_t elements_ = 0;
        };

        /**
         * @brief Pack a real tensor into the common accelerator representation.
         * @param tensor Quantized source implementing `IINT8Unpackable`.
         * @return Production payload, metadata arrays, and source provenance.
         * @throws std::invalid_argument for a non-NativeVNNI tensor or geometry.
         */
        HostGpuExpertPackedProjection packProductionGpuProjection(
            const TensorBase &tensor)
        {
            const auto *unpackable =
                dynamic_cast<const IINT8Unpackable *>(&tensor);
            if (unpackable == nullptr ||
                unpackable->vnniFormatInfo() == nullptr)
            {
                throw std::invalid_argument(
                    "ROCm all-format tier test requires NativeVNNI");
            }

            const NativeVnniFormatInfo &format =
                *unpackable->vnniFormatInfo();
            const int N = static_cast<int>(tensor.rows());
            const int K = static_cast<int>(tensor.cols());
            if (N <= 0 || K <= 0 || (K % 32) != 0)
                throw std::invalid_argument("invalid ROCm tier-test geometry");

            HostGpuExpertPackedProjection projection;
            projection.N = N;
            projection.K = K;
            projection.blocks_per_row =
                static_cast<std::uint32_t>(K / 32);
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
            // This virtual is the same source-format authority used by the
            // live CUDA/ROCm weight packers; the test does not synthesize bytes.
            for (int n = 0; n < N; ++n)
            {
                for (int k_block = 0;
                     k_block < context.blocks_per_row;
                     ++k_block)
                {
                    unpackable->packVnniBlock(
                        context, n, n, k_block);
                }
            }
            return projection;
        }

        /**
         * @brief Report the first unequal byte without weakening full equality.
         * @param observed Device-produced bytes for one chunk.
         * @param valid_bytes Number of initialized bytes in `observed`.
         * @param expected Iterator to the matching CPU-oracle chunk.
         * @param first_unit Global unit origin included in a failure message.
         */
        void expectChunkBytesEqual(
            const std::vector<std::uint8_t> &observed,
            std::size_t valid_bytes,
            const std::uint8_t *expected,
            std::uint32_t first_unit)
        {
            const auto mismatch = std::mismatch(
                observed.begin(),
                observed.begin() + static_cast<std::ptrdiff_t>(valid_bytes),
                expected);
            if (mismatch.first ==
                observed.begin() + static_cast<std::ptrdiff_t>(valid_bytes))
            {
                return;
            }
            ADD_FAILURE()
                << "first_unit=" << first_unit
                << " byte="
                << std::distance(observed.begin(), mismatch.first)
                << " observed=" << static_cast<unsigned>(*mismatch.first)
                << " expected=" << static_cast<unsigned>(*mismatch.second);
        }

        /**
         * @brief Poll one production lane without introducing a host wait.
         * @param lane Materialized lane containing one active transfer.
         * @return Terminal progress, or `Pending` at the bounded deadline.
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

        /** ROCm device fixture that also publishes every production IQ table. */
        class ROCmExpertTierWeightKernelsTest : public ::testing::Test
        {
        protected:
            /** @brief Select device zero or skip when ROCm is unavailable. */
            void SetUp() override
            {
                int devices = 0;
                if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0)
                    GTEST_SKIP() << "No ROCm device available";
                ASSERT_EQ(hipSetDevice(0), hipSuccess);
                ASSERT_TRUE(rocm::ensureIQGridTablesInitialized(0))
                    << "all ROCm IQ decoder translation units must initialize";
            }
        };

        /**
         * @test All 21 real source formats are byte exact in both directions.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            EverySourceFormatMatchesProductionCpuPackerByteForByte)
        {
            constexpr int N = 70;
            constexpr int K = 256;
            constexpr std::uint32_t kUnitsPerChunk = 4;
            ASSERT_EQ(test::quantizedVerifierFormats().size(), 21u);

            TestHIPStream stream;
            bool rejection_contract_checked = false;
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
                    static_cast<std::uint32_t>(83001u + format_index));
                ASSERT_NE(tensor, nullptr);

                const HostGpuExpertPackedProjection source =
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
                const auto demotion_manifest =
                    makeGpuToCpuExpertTierWeightStreamManifest(
                        *source_format,
                        N,
                        K,
                        401,
                        3,
                        7,
                        ExpertTierWeightProjection::Down,
                        kUnitsPerChunk);
                const ExpertTierWeightDeviceLayout demotion_layout =
                    demotion_manifest.deviceLayout();
                ASSERT_TRUE(demotion_layout.valid());
                ASSERT_EQ(
                    expected_cpu.native_interleaved.size(),
                    demotion_manifest.total_stream_bytes);

                TestHIPBuffer<std::uint8_t> source_payload(
                    source.payload.size());
                TestHIPBuffer<std::uint16_t> source_scales(
                    source.scales.size());
                TestHIPBuffer<std::uint16_t> source_mins(
                    source.mins.size());
                TestHIPBuffer<std::uint32_t> source_emins(
                    source.emins.size());
                ASSERT_EQ(source_payload.upload(source.payload), hipSuccess);
                ASSERT_EQ(source_scales.upload(source.scales), hipSuccess);
                ASSERT_EQ(source_mins.upload(source.mins), hipSuccess);
                ASSERT_EQ(source_emins.upload(source.emins), hipSuccess);
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
                ASSERT_TRUE(source_view.validFor(demotion_layout));

                TestHIPBuffer<std::uint8_t> device_chunk(
                    demotion_layout.chunkBytes(kUnitsPerChunk));
                std::vector<std::uint8_t> observed_chunk(
                    demotion_layout.chunkBytes(kUnitsPerChunk));
                if (!rejection_contract_checked)
                {
                    EXPECT_FALSE(launchGpuToCpuExpertTierChunkROCm(
                        source_view,
                        demotion_layout,
                        0,
                        1,
                        device_chunk.data(),
                        device_chunk.bytes(),
                        nullptr));
                    EXPECT_FALSE(launchGpuToCpuExpertTierChunkROCm(
                        source_view,
                        demotion_layout,
                        0,
                        kUnitsPerChunk + 1,
                        device_chunk.data(),
                        device_chunk.bytes(),
                        stream.opaque()));
                }

                for (std::uint32_t first_unit = 0;
                     first_unit < demotion_layout.unit_count;
                     first_unit += kUnitsPerChunk)
                {
                    const std::uint32_t unit_count =
                        std::min<std::uint32_t>(
                            kUnitsPerChunk,
                            demotion_layout.unit_count - first_unit);
                    const std::size_t bytes =
                        demotion_layout.chunkBytes(unit_count);
                    ASSERT_TRUE(launchGpuToCpuExpertTierChunkROCm(
                        source_view,
                        demotion_layout,
                        first_unit,
                        unit_count,
                        device_chunk.data(),
                        device_chunk.bytes(),
                        stream.opaque()));
                    ASSERT_EQ(
                        hipMemcpyAsync(
                            observed_chunk.data(),
                            device_chunk.data(),
                            bytes,
                            hipMemcpyDeviceToHost,
                            stream.native()),
                        hipSuccess);
                    ASSERT_EQ(stream.synchronize(), hipSuccess);
                    expectChunkBytesEqual(
                        observed_chunk,
                        bytes,
                        expected_cpu.native_interleaved.data() +
                            demotion_layout.chunkBytes(first_unit),
                        first_unit);
                    ASSERT_FALSE(::testing::Test::HasFailure());
                }

                HostGpuExpertPackedProjection expected_gpu;
                ASSERT_TRUE(cpuToGpuExpertPackedReference(
                    expected_cpu, expected_gpu, &error)) << error;
                const auto promotion_manifest =
                    makeCpuToGpuExpertTierWeightStreamManifest(
                        expected_cpu,
                        402,
                        3,
                        7,
                        ExpertTierWeightProjection::Down,
                        kUnitsPerChunk);
                const ExpertTierWeightDeviceLayout promotion_layout =
                    promotion_manifest.deviceLayout();
                ASSERT_TRUE(promotion_layout.valid());
                ASSERT_EQ(
                    promotion_layout.direction,
                    ExpertTierWeightConversionDirection::CpuToGpu);
                ASSERT_EQ(
                    promotion_layout.gpu_codebook_id,
                    expected_gpu.codebook_id);

                TestHIPBuffer<std::uint8_t> destination_payload(
                    expected_gpu.payload.size());
                TestHIPBuffer<std::uint16_t> destination_scales(
                    expected_gpu.scales.size());
                TestHIPBuffer<std::uint16_t> destination_mins(
                    expected_gpu.mins.size());
                TestHIPBuffer<std::uint32_t> destination_emins(
                    expected_gpu.emins.size());
                const ExpertTierGpuMutableProjectionView destination_view{
                    .payload = destination_payload.data(),
                    .scales = destination_scales.data(),
                    .mins = destination_mins.data(),
                    .emins = destination_emins.data(),
                    .payload_bytes = destination_payload.bytes(),
                    .scales_bytes = destination_scales.bytes(),
                    .mins_bytes = destination_mins.bytes(),
                    .emins_bytes = destination_emins.bytes(),
                };
                ASSERT_TRUE(destination_view.validFor(promotion_layout));
                if (!rejection_contract_checked)
                {
                    EXPECT_FALSE(launchCpuToGpuExpertTierChunkROCm(
                        device_chunk.data(),
                        device_chunk.bytes(),
                        promotion_layout,
                        0,
                        1,
                        destination_view,
                        nullptr));
                    rejection_contract_checked = true;
                }

                for (std::uint32_t first_unit = 0;
                     first_unit < promotion_layout.unit_count;
                     first_unit += kUnitsPerChunk)
                {
                    const std::uint32_t unit_count =
                        std::min<std::uint32_t>(
                            kUnitsPerChunk,
                            promotion_layout.unit_count - first_unit);
                    const std::size_t bytes =
                        promotion_layout.chunkBytes(unit_count);
                    ASSERT_EQ(
                        hipMemcpyAsync(
                            device_chunk.data(),
                            expected_cpu.native_interleaved.data() +
                                promotion_layout.chunkBytes(first_unit),
                            bytes,
                            hipMemcpyHostToDevice,
                            stream.native()),
                        hipSuccess);
                    ASSERT_TRUE(launchCpuToGpuExpertTierChunkROCm(
                        device_chunk.data(),
                        bytes,
                        promotion_layout,
                        first_unit,
                        unit_count,
                        destination_view,
                        stream.opaque()));
                }
                ASSERT_EQ(stream.synchronize(), hipSuccess);

                std::vector<std::uint8_t> observed_payload;
                std::vector<std::uint16_t> observed_scales;
                std::vector<std::uint16_t> observed_mins;
                std::vector<std::uint32_t> observed_emins;
                ASSERT_EQ(
                    destination_payload.download(observed_payload), hipSuccess);
                ASSERT_EQ(
                    destination_scales.download(observed_scales), hipSuccess);
                ASSERT_EQ(
                    destination_mins.download(observed_mins), hipSuccess);
                ASSERT_EQ(
                    destination_emins.download(observed_emins), hipSuccess);
                EXPECT_EQ(observed_payload, expected_gpu.payload);
                EXPECT_EQ(observed_scales, expected_gpu.scales);
                EXPECT_EQ(observed_mins, expected_gpu.mins);
                EXPECT_EQ(observed_emins, expected_gpu.emins);
            }
        }

        /**
         * @test The persistent ROCm background lane preserves every prepared byte.
         *
         * A real Q8_0 tensor is demoted and promoted through the production
         * conversion kernels, auxiliary stream, pinned DMA chunk, and event-
         * polled pump. No synchronization or inference-stream dependency is
         * permitted inside the lane.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            PersistentBackgroundLaneRoundTripsWithoutBlockingSynchronization)
        {
            constexpr int N = 192;
            constexpr int K = 256;
            const auto &format = test::quantizedVerifierFormats().at(18);
            ASSERT_STREQ(format.label, "Q8_0");
            auto tensor = format.create({N, K}, 93001u);
            ASSERT_NE(tensor, nullptr);
            const HostGpuExpertPackedProjection source =
                packProductionGpuProjection(*tensor);

            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                tensor.get(), expected_cpu));
            const auto *source_format =
                native_vnni_formats::forSourceIdentity(
                    source.source_codebook_id,
                    source.is_superblock);
            ASSERT_NE(source_format, nullptr);
            const auto demotion_manifest =
                makeGpuToCpuExpertTierWeightStreamManifest(
                    *source_format,
                    N,
                    K,
                    901,
                    3,
                    11,
                    ExpertTierWeightProjection::Gate,
                    2);
            const auto demotion_layout = demotion_manifest.deviceLayout();

            TestHIPBuffer<std::uint8_t> source_payload(source.payload.size());
            TestHIPBuffer<std::uint16_t> source_scales(source.scales.size());
            TestHIPBuffer<std::uint16_t> source_mins(source.mins.size());
            TestHIPBuffer<std::uint32_t> source_emins(source.emins.size());
            ASSERT_EQ(source_payload.upload(source.payload), hipSuccess);
            ASSERT_EQ(source_scales.upload(source.scales), hipSuccess);
            ASSERT_EQ(source_mins.upload(source.mins), hipSuccess);
            ASSERT_EQ(source_emins.upload(source.emins), hipSuccess);
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

            IBackend *backend = getROCmBackend();
            ASSERT_NE(backend, nullptr);
            TestHIPStream producer_stream;
            void *source_ready = backend->createEvent(0);
            ASSERT_NE(source_ready, nullptr);
            ASSERT_TRUE(backend->recordEvent(
                source_ready, 0, producer_stream.opaque()));

            ExpertTierWeightTransferLane lane({
                .device = DeviceId::rocm(0),
                .staging_capacity_bytes = demotion_layout.chunkBytes(2),
                .lane_name = "rocm_tier_round_trip",
                .perf_device = "rocm:0",
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
            ASSERT_EQ(
                observed_cpu.size(),
                expected_cpu.native_interleaved.size());
            EXPECT_TRUE(std::equal(
                observed_cpu.begin(),
                observed_cpu.end(),
                expected_cpu.native_interleaved.begin()));
            backend->destroyEvent(source_ready, 0);

            HostGpuExpertPackedProjection expected_gpu;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(
                expected_cpu, expected_gpu, &error)) << error;
            const auto promotion_manifest =
                makeCpuToGpuExpertTierWeightStreamManifest(
                    expected_cpu,
                    902,
                    3,
                    11,
                    ExpertTierWeightProjection::Gate,
                    2);
            const auto promotion_layout = promotion_manifest.deviceLayout();
            TestHIPBuffer<std::uint8_t> destination_payload(
                expected_gpu.payload.size());
            TestHIPBuffer<std::uint16_t> destination_scales(
                expected_gpu.scales.size());
            TestHIPBuffer<std::uint16_t> destination_mins(
                expected_gpu.mins.size());
            TestHIPBuffer<std::uint32_t> destination_emins(
                expected_gpu.emins.size());
            const ExpertTierGpuMutableProjectionView destination_view{
                .payload = destination_payload.data(),
                .scales = destination_scales.data(),
                .mins = destination_mins.data(),
                .emins = destination_emins.data(),
                .payload_bytes = destination_payload.bytes(),
                .scales_bytes = destination_scales.bytes(),
                .mins_bytes = destination_mins.bytes(),
                .emins_bytes = destination_emins.bytes(),
            };
            ASSERT_TRUE(lane.startCpuToGpu(
                promotion_layout,
                expected_cpu.native_interleaved,
                destination_view,
                &error)) << error;
            ASSERT_EQ(
                pollLaneToCompletion(lane),
                ExpertTierWeightTransferProgress::Ready);

            std::vector<std::uint8_t> observed_payload;
            std::vector<std::uint16_t> observed_scales;
            std::vector<std::uint16_t> observed_mins;
            std::vector<std::uint32_t> observed_emins;
            ASSERT_EQ(
                destination_payload.download(observed_payload), hipSuccess);
            ASSERT_EQ(
                destination_scales.download(observed_scales), hipSuccess);
            ASSERT_EQ(destination_mins.download(observed_mins), hipSuccess);
            ASSERT_EQ(destination_emins.download(observed_emins), hipSuccess);
            EXPECT_EQ(observed_payload, expected_gpu.payload);
            EXPECT_EQ(observed_scales, expected_gpu.scales);
            EXPECT_EQ(observed_mins, expected_gpu.mins);
            EXPECT_EQ(observed_emins, expected_gpu.emins);

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_started, 2u);
            EXPECT_EQ(stats.transfers_completed, 2u);
            EXPECT_GT(stats.chunks_submitted, 2u);
            EXPECT_GT(stats.bytes_submitted, 0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
        }

        /**
         * @test Promoted asymmetric CPU weights remain executable on ROCm.
         *
         * Real Q5_1 weights are prepared into the CPU cold-tier representation,
         * streamed through the persistent promotion lane, and executed by both
         * production serial decode and verifier-depth kernels. N=512,K=2048
         * separates source-codebook arithmetic policy from the expanded
         * execution decoder, so compact and promoted weights must still
         * produce identical FP32 words rather than merely close values.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            RemoteCpuGpuEndpointsStreamQ51WithoutInferenceWaits)
        {
            ScopedPerfStats perf_stats;
            constexpr int N = 70;
            constexpr int K = 96;
            constexpr std::uint64_t promoted_epoch = 9301;
            const auto &format = test::quantizedVerifierFormat("Q5_1");
            auto tensor = format.create({N, K}, 0xa0c051u);
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
                .transaction_fingerprint = {
                    .low = 0xa0c05101u,
                    .high = 0xa0c05102u,
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
                .destination_device = DeviceId::rocm(0),
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

            TestHIPBuffer<std::uint8_t> destination_payload(
                expected_gpu.payload.size());
            TestHIPBuffer<std::uint16_t> destination_scales(
                expected_gpu.scales.size());
            TestHIPBuffer<std::uint16_t> destination_mins(
                expected_gpu.mins.size());
            TestHIPBuffer<std::uint32_t> destination_emins(
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
                    .device = DeviceId::rocm(0),
                    .staging_capacity_bytes =
                        promotion_manifest.maximum_chunk_bytes,
                    .lane_name = "rocm_remote_q51_roundtrip",
                    .perf_device = "rocm:0",
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
                                "ROCm destination factory received an unexpected physical format";
                        return false;
                    }
                    auto engine = std::make_shared<
                        rocm::ROCmQuantisedGemmKernel>(
                        N,
                        K,
                        0,
                        destination_descriptor.ptrs.d_vnni,
                        destination_descriptor.ptrs.d_scales,
                        destination_descriptor.ptrs.d_mins,
                        destination_descriptor.ptrs.d_emins,
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
                destination_payload.download(observed_payload), hipSuccess);
            ASSERT_EQ(
                destination_scales.download(observed_scales), hipSuccess);
            ASSERT_EQ(destination_mins.download(observed_mins), hipSuccess);
            ASSERT_EQ(destination_emins.download(observed_emins), hipSuccess);
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
                .transaction_fingerprint = {
                    .low = 0xa0c05201u,
                    .high = 0xa0c05202u,
                },
                .migration_index = 4,
                .layer_idx = promotion_identity.layer_idx,
                .expert_id = promotion_identity.expert_id,
                .projection = promotion_identity.projection,
                .source_participant = 1,
                .destination_participant = 0,
                .source_world_rank = 1,
                .destination_world_rank = 0,
                .source_device = DeviceId::rocm(0),
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

        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            PromotedAsymmetricWeightsExecuteDecodeAndVerifierByteExactly)
        {
            ScopedPerfStats perf_stats;
            constexpr int N = 512;
            constexpr int K = 2048;
            constexpr int verifier_rows = 4;
            constexpr int prefill_rows = 32;
            const auto &format = test::quantizedVerifierFormat("Q5_1");
            auto tensor = format.create({N, K}, 99173u);
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

            const auto manifest = makeCpuToGpuExpertTierWeightStreamManifest(
                cpu_weights,
                8301,
                2,
                19,
                ExpertTierWeightProjection::Down,
                1);
            const auto layout = manifest.deviceLayout();
            ASSERT_TRUE(layout.valid());
            ASSERT_EQ(
                layout.gpu_codebook_id,
                kNativeVnniExpandedInt8MinCodebook);
            const std::size_t blocks =
                static_cast<std::size_t>(N) * (K / 32);

            TestHIPBuffer<std::uint8_t> promoted_payload(
                blocks * layout.gpu_payload_bytes_per_block);
            TestHIPBuffer<std::uint16_t> promoted_scales(blocks);
            TestHIPBuffer<std::uint16_t> promoted_mins(blocks);
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
                .device = DeviceId::rocm(0),
                .staging_capacity_bytes = layout.chunkBytes(1),
                .lane_name = "rocm_asymmetric_promotion_execution",
                .perf_device = "rocm:0",
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
             * The promoted bank is now authoritative and contains normalized
             * asymmetric INT8 rather than the compact Q5_1 source bits.  Read
             * that exact live bank through the production demotion lane so a
             * CPU -> ROCm -> CPU migration cycle proves byte-exact multi-hop
             * behavior without retaining a host mirror of accelerator state.
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
            const auto redemotion_manifest =
                makeGpuToCpuExpertTierWeightStreamManifest(
                    *source_format,
                    promoted_descriptor,
                    8301,
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
                ExpertTierSourceReadiness::publishedResidencyBank(8301),
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

            TestHIPBuffer<std::uint8_t> compact_payload(compact.payload.size());
            TestHIPBuffer<std::uint16_t> compact_scales(compact.scales.size());
            TestHIPBuffer<std::uint16_t> compact_mins(compact.mins.size());
            ASSERT_EQ(compact_payload.upload(compact.payload), hipSuccess);
            ASSERT_EQ(compact_scales.upload(compact.scales), hipSuccess);
            ASSERT_EQ(compact_mins.upload(compact.mins), hipSuccess);

            constexpr std::size_t activation_elements =
                static_cast<std::size_t>(prefill_rows) * K;
            std::vector<std::int8_t> host_activation(activation_elements);
            for (std::size_t index = 0; index < activation_elements; ++index)
            {
                host_activation[index] = static_cast<std::int8_t>(
                    static_cast<int>((index * 11u + 3u) % 29u) - 14);
            }
            std::vector<float> host_activation_scales(prefill_rows);
            for (int row = 0; row < prefill_rows; ++row)
            {
                host_activation_scales[static_cast<std::size_t>(row)] =
                    0.03137f + static_cast<float>(row) * 0.00019f;
            }
            TestHIPBuffer<std::int8_t> activation(activation_elements);
            TestHIPBuffer<float> activation_scales(prefill_rows);
            ASSERT_EQ(activation.upload(host_activation), hipSuccess);
            ASSERT_EQ(
                activation_scales.upload(host_activation_scales),
                hipSuccess);

            TestHIPStream stream;
            ScopedROCmDecodeEquivalentM1 decode_equivalent_scope;
            TestHIPBuffer<float> partials(
                static_cast<std::size_t>(
                    NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
                verifier_rows * N);

            int source_kb = 0;
            int normalized_kb = 0;
            int ignored_waves = 0;
            ASSERT_TRUE(
                rocmGemv_native_vnni_query_serial_m1_config_with_policy(
                    compact.codebook_id,
                    N,
                    K,
                    &source_kb,
                    &ignored_waves));
            ASSERT_TRUE(
                rocmGemv_native_vnni_query_serial_m1_config_with_policy(
                    19,
                    N,
                    K,
                    &normalized_kb,
                    &ignored_waves));
            ASSERT_NE(source_kb, normalized_kb)
                << "The witness must distinguish source and normalized policies";

            for (const int rows : {1, verifier_rows, prefill_rows})
            {
                SCOPED_TRACE("rows=" + std::to_string(rows));
                TestHIPBuffer<float> compact_output(
                    static_cast<std::size_t>(rows) * N);
                TestHIPBuffer<float> promoted_output(
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
                        return rocmGemv_native_vnni_fp32_with_policy(
                            activation.data(), payload, scales, mins, nullptr,
                            output, nullptr, partials.data(), N, K, codebook,
                            arithmetic_policy_codebook, 0,
                            stream.opaque(), activation_scales.data());
                    }
                    if (rows == verifier_rows)
                    {
                        return rocmGemv_native_vnni_small_m_fp32_with_policy(
                            activation.data(), payload, scales, mins, nullptr,
                            output, activation_scales.data(), partials.data(),
                            rows, N, K, codebook,
                            arithmetic_policy_codebook, 0, stream.opaque());
                    }
                    return rocmGemm_native_vnni_fp32_with_policy(
                        activation.data(), payload, scales, mins, nullptr,
                        output, nullptr, activation_scales.data(), rows, N, K,
                        codebook, arithmetic_policy_codebook, 0,
                        stream.opaque());
                };
                ASSERT_TRUE(launch(
                    compact_payload.data(),
                    compact_scales.data(),
                    compact_mins.data(),
                    compact.codebook_id,
                    compact.codebook_id,
                    compact_output.data()));
                ASSERT_TRUE(launch(
                    promoted_payload.data(),
                    promoted_scales.data(),
                    promoted_mins.data(),
                    layout.gpu_codebook_id,
                    compact.codebook_id,
                    promoted_output.data()));
                ASSERT_EQ(stream.synchronize(), hipSuccess);

                std::vector<float> expected;
                std::vector<float> actual;
                ASSERT_EQ(compact_output.download(expected), hipSuccess);
                ASSERT_EQ(promoted_output.download(actual), hipSuccess);
                ASSERT_EQ(expected.size(), actual.size());
                ASSERT_TRUE(std::any_of(
                    expected.begin(), expected.end(),
                    [](float value) { return value != 0.0f; }));
                EXPECT_EQ(
                    std::memcmp(
                        expected.data(), actual.data(),
                        expected.size() * sizeof(float)),
                    0) << "ROCm promoted asymmetric execution changed FP32 words";
            }

            const auto small_m_records = PerfStatsCollector::snapshot(
                {"kernel.rocm_native_vnni_small_m_launch"});
            const auto prefill_records = PerfStatsCollector::snapshot(
                {"kernel.rocm_native_vnni_prefill_launch"});
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
                small_m_records, 1, compact.codebook_id, compact.codebook_id));
            EXPECT_TRUE(has_policy_record(
                small_m_records,
                1,
                layout.gpu_codebook_id,
                compact.codebook_id));
            EXPECT_TRUE(has_policy_record(
                small_m_records,
                verifier_rows,
                layout.gpu_codebook_id,
                compact.codebook_id));
            EXPECT_TRUE(has_policy_record(
                prefill_records,
                prefill_rows,
                layout.gpu_codebook_id,
                compact.codebook_id));
            EXPECT_FALSE(has_policy_record(
                small_m_records,
                1,
                layout.gpu_codebook_id,
                19));
            EXPECT_FALSE(has_policy_record(
                prefill_records,
                prefill_rows,
                layout.gpu_codebook_id,
                19));

            const auto stats = lane.stats();
            EXPECT_EQ(stats.transfers_completed, 2u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
        }

        /**
         * @test Production ROCm inference finishes unchanged during migration.
         *
         * The baseline and concurrent launches execute the real generated
         * NativeVNNI Q4_0 M=1 kernel. A 24-chunk demotion advances on the
         * independent persistent lane, and an event query must observe the
         * inference result while residency movement is still pending.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            ProductionInferenceCompletesByteExactlyWhileMigrationIsPending)
        {
            constexpr int N = 192;
            constexpr int K = 256;
            const auto &format = test::quantizedVerifierFormats().front();
            ASSERT_STREQ(format.label, "Q4_0");
            auto tensor = format.create({N, K}, 94001u);
            ASSERT_NE(tensor, nullptr);
            const HostGpuExpertPackedProjection source =
                packProductionGpuProjection(*tensor);

            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            std::string error;
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
                903,
                4,
                13,
                ExpertTierWeightProjection::Down,
                1);
            const auto layout = manifest.deviceLayout();
            ASSERT_EQ(layout.unit_count, 24u);

            TestHIPBuffer<std::uint8_t> payload(source.payload.size());
            TestHIPBuffer<std::uint16_t> scales(source.scales.size());
            TestHIPBuffer<std::uint16_t> mins(source.mins.size());
            TestHIPBuffer<std::uint32_t> emins(source.emins.size());
            ASSERT_EQ(payload.upload(source.payload), hipSuccess);
            ASSERT_EQ(scales.upload(source.scales), hipSuccess);
            ASSERT_EQ(mins.upload(source.mins), hipSuccess);
            ASSERT_EQ(emins.upload(source.emins), hipSuccess);
            const ExpertTierGpuConstProjectionView source_view{
                .payload = payload.data(),
                .scales = scales.data(),
                .mins = mins.data(),
                .emins = emins.data(),
                .payload_bytes = payload.bytes(),
                .scales_bytes = scales.bytes(),
                .mins_bytes = mins.bytes(),
                .emins_bytes = emins.bytes(),
            };

            std::vector<std::int8_t> host_activation(
                static_cast<std::size_t>(K), 1);
            std::vector<float> host_activation_scales(
                static_cast<std::size_t>(K / 32), 1.0f);
            TestHIPBuffer<std::int8_t> activation(host_activation.size());
            TestHIPBuffer<float> activation_scales(
                host_activation_scales.size());
            TestHIPBuffer<float> partials(
                static_cast<std::size_t>(
                    NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
                static_cast<std::size_t>(N));
            TestHIPBuffer<float> baseline_output(N);
            TestHIPBuffer<float> concurrent_output(N);
            ASSERT_EQ(activation.upload(host_activation), hipSuccess);
            ASSERT_EQ(
                activation_scales.upload(host_activation_scales), hipSuccess);

            TestHIPStream inference_stream;
            ScopedROCmDecodeEquivalentM1 decode_equivalent_scope;
            ASSERT_TRUE(rocmGemv_native_vnni_fp32(
                activation.data(),
                payload.data(),
                scales.data(),
                nullptr,
                nullptr,
                baseline_output.data(),
                nullptr,
                partials.data(),
                N,
                K,
                source.codebook_id,
                0,
                inference_stream.opaque(),
                activation_scales.data()));
            ASSERT_EQ(inference_stream.synchronize(), hipSuccess);

            IBackend *backend = getROCmBackend();
            ASSERT_NE(backend, nullptr);
            void *source_ready = backend->createEvent(0);
            void *inference_done = backend->createEvent(0);
            ASSERT_NE(source_ready, nullptr);
            ASSERT_NE(inference_done, nullptr);
            ASSERT_TRUE(backend->recordEvent(
                source_ready, 0, inference_stream.opaque()));

            ExpertTierWeightTransferLane lane({
                .device = DeviceId::rocm(0),
                .staging_capacity_bytes = layout.chunkBytes(1),
                .lane_name = "rocm_inference_overlap",
                .perf_device = "rocm:0",
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

            ASSERT_TRUE(rocmGemv_native_vnni_fp32(
                activation.data(),
                payload.data(),
                scales.data(),
                nullptr,
                nullptr,
                concurrent_output.data(),
                nullptr,
                partials.data(),
                N,
                K,
                source.codebook_id,
                0,
                inference_stream.opaque(),
                activation_scales.data()));
            ASSERT_TRUE(backend->recordEvent(
                inference_done, 0, inference_stream.opaque()));

            bool inference_completed_while_migration_pending = false;
            bool inference_ready = false;
            auto migration_progress = lane.progress();
            auto &gpu_context =
                GPUDeviceContextPool::instance().getContext(DeviceId::rocm(0));
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
                << "The production ROCm inference stream waited for migration";
            EXPECT_EQ(
                migration_progress,
                ExpertTierWeightTransferProgress::Ready);
            ASSERT_EQ(
                observed_cpu.size(),
                expected_cpu.native_interleaved.size());
            EXPECT_TRUE(std::equal(
                observed_cpu.begin(),
                observed_cpu.end(),
                expected_cpu.native_interleaved.begin()));

            std::vector<float> baseline;
            std::vector<float> concurrent;
            ASSERT_EQ(baseline_output.download(baseline), hipSuccess);
            ASSERT_EQ(concurrent_output.download(concurrent), hipSuccess);
            ASSERT_EQ(baseline.size(), concurrent.size());
            EXPECT_EQ(
                std::memcmp(
                    baseline.data(),
                    concurrent.data(),
                    baseline.size() * sizeof(float)),
                0) << "Concurrent migration changed ROCm inference bytes";

            const auto stats = lane.stats();
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
            EXPECT_EQ(stats.transfers_completed, 1u);
            EXPECT_EQ(stats.chunks_submitted, layout.unit_count);

            backend->destroyEvent(inference_done, 0);
            backend->destroyEvent(source_ready, 0);
        }

        /**
         * @test FusedQKV's HIP stream pool remains exact during tier movement.
         *
         * Decode and prefill both use the production Q/K/V fork/join DAG while
         * a long GPU-to-CPU conversion progresses on its independent transfer
         * stream. PerfStats proves every requested pool launch occurred.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            FusedQKVInternalStreamPoolsRemainByteExactAndNonBlockingDuringMigration)
        {
            constexpr int N = 1024;
            constexpr int K = 256;
            const auto &format = test::quantizedVerifierFormats().front();
            ASSERT_STREQ(format.label, "Q4_0");
            auto tensor = format.create({N, K}, 95001u);
            ASSERT_NE(tensor, nullptr);
            const HostGpuExpertPackedProjection source =
                packProductionGpuProjection(*tensor);
            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(
                tensor.get(), expected_cpu));
            const auto *source_format =
                native_vnni_formats::forSourceIdentity(0, false);
            ASSERT_NE(source_format, nullptr);
            const auto manifest = makeGpuToCpuExpertTierWeightStreamManifest(
                *source_format,
                N,
                K,
                904,
                5,
                17,
                ExpertTierWeightProjection::Up,
                1);
            const auto layout = manifest.deviceLayout();
            ASSERT_GE(layout.unit_count, 64u);

            TestHIPBuffer<std::uint8_t> payload(source.payload.size());
            TestHIPBuffer<std::uint16_t> scales(source.scales.size());
            TestHIPBuffer<std::uint16_t> mins(source.mins.size());
            TestHIPBuffer<std::uint32_t> emins(source.emins.size());
            ASSERT_EQ(payload.upload(source.payload), hipSuccess);
            ASSERT_EQ(scales.upload(source.scales), hipSuccess);
            ASSERT_EQ(mins.upload(source.mins), hipSuccess);
            ASSERT_EQ(emins.upload(source.emins), hipSuccess);
            const ExpertTierGpuConstProjectionView source_view{
                .payload = payload.data(),
                .scales = scales.data(),
                .mins = mins.data(),
                .emins = emins.data(),
                .payload_bytes = payload.bytes(),
                .scales_bytes = scales.bytes(),
                .mins_bytes = mins.bytes(),
                .emins_bytes = emins.bytes(),
            };

            IBackend *backend = getROCmBackend();
            ASSERT_NE(backend, nullptr);
            TestHIPStream root_stream;
            test::runFusedQKVStreamPoolMigrationStress(
                DeviceId::rocm(0),
                root_stream.opaque(),
                *backend,
                layout,
                source_view,
                std::span<const std::uint8_t>(
                    expected_cpu.native_interleaved.data(),
                    expected_cpu.native_interleaved.size()),
                "rocm_fused_qkv_migration");
        }
    } // namespace
} // namespace llaminar2
