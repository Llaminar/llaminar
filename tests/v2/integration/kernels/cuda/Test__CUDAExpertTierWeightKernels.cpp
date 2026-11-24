/**
 * @file Test__CUDAExpertTierWeightKernels.cpp
 * @brief Byte-exact CUDA parity for streamed ExpertOverlay weight conversion.
 *
 * The test uses the device-free CPU oracle to construct complete expected
 * execution bytes, then converts the same projection in arbitrary two-unit
 * chunks on a non-default CUDA stream. It covers every reversible physical CPU
 * encoding, non-64-aligned N padding, non-zero chunk origins, both directions,
 * and strict launcher rejection before the performance harness is allowed to
 * measure these kernels. Q6_K synthetic projections retain original half
 * bases plus signed source coefficients in their unchanged metadata planes.
 * IQ2_S synthetic tier metadata preserves original bases and coefficient bytes.
 * IQ4_XS uses physical24/source4 and raw encoded6 coefficient words; IQ4_NL
 * remains the independent physical4 control with no secondary plane.
 * Fixture uploads complete on an explicit stream before publishing storage to
 * independent non-blocking transfer lanes; pageable H2D staging is not completion.
 * The coded MPI proof uses real rank-local CUDA endpoints and verifies the
 * original device planes after transport publication, including untouched rows.
 */

#include "../../../fixtures/transfer/ComponentTransferExecutionPoolFixtureAccess.h"
#include "execution/moe/MoEOverlayMPIRemoteProjectionTransport.h"
#include "utils/ExpertTierWeightWireTestOracle.h"
#include "utils/GpuPreparedGemmHarness.h"

#include "execution/moe/ExpertTierWeightStream.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "tensors/IQQuantTables.h"
#include "kernels/KernelFactory.h"
#include "loaders/GPUVramPreflight.h"
#include "kernels/cuda/gemm/CUDAFloatingPointGemmKernel.h"
#include "kernels/cuda/gemm/CUDAGroupedVerifierLaunch.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "execution/moe/ExpertTierGpuBlobTransferLane.h"
#include "execution/moe/GpuExpertSlotPool.h"
#include "execution/moe/ExpertPreparedMemoryGeometry.h"
#include "execution/moe/MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/common/NativeQuantisedGroupedDecodePolicy.h"
#include "kernels/cuda/repack/CUDAExpertTierWeightKernels.h"
#include "tensors/TensorKernels.h"
#include "tensors/VnniPackContext.h"
#include "tensors/BlockStructures.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/MPIContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/MappedTransferProgressEpoch.h"

#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ExpertTierFusedQKVStreamPoolHarness.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include "kernels/common/NativeQuantisedScaleLoad.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    /** @brief Exercise the production no-peer protocol with disjoint roots on one real CUDA device. */
    struct ExpertTierGpuBlobRelayFixtureAccess
    {
        /** @return A real retained relay, allowing its logical endpoints to share the test GPU. */
        static std::unique_ptr<ExpertTierGpuBlobTransferLane> make(ExpertTierGpuBlobTransferLane::Config config)
        { return std::unique_ptr<ExpertTierGpuBlobTransferLane>(new ExpertTierGpuBlobTransferLane(std::move(config), true)); }
    };
}

extern "C"
{
    /** Initialize immutable CUDA IQ lookup tables used by production decode. */
    bool cudaNativeQuantisedInitIQGridTables_tuned();
    void cudaNativeQuantisedGemvTuned_setDecodeEquivalentM1Config(int enabled);
    int cudaNativeQuantisedGemvTuned_getDecodeEquivalentM1Config();

    /** Production M=1 NativeQuantised inference kernel used by the overlap proof. */
    bool cudaNativeQuantisedGemvTuned_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes d_meta,
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
    bool cudaNativeQuantisedGemvTuned_fp32_withPolicy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes d_meta,
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

    /** Production verifier-depth NativeQuantised kernel used by promotion parity. */
    bool cudaNativeQuantisedGemvTuned_small_m_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes d_meta,
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

    /** Production dense-prefill NativeQuantised kernel used by promotion parity. */
    bool cudaNativeQuantisedPrefill_fp32(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes d_meta,
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
    bool cudaNativeQuantisedPrefill_fp32_withPolicy(
        const std::int8_t *d_A_int8,
        const std::uint8_t *d_payload,
        ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes d_meta,
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
    using llaminar::v2::kernels::KernelFactory;
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
                      cudaNativeQuantisedGemvTuned_getDecodeEquivalentM1Config())
            {
                cudaNativeQuantisedGemvTuned_setDecodeEquivalentM1Config(1);
            }

            ~ScopedDecodeEquivalentM1()
            {
                cudaNativeQuantisedGemvTuned_setDecodeEquivalentM1Config(previous_);
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
             * @brief Complete a fixture upload before independent streams use it.
             * @param source Host elements whose size must equal the allocation.
             * @return CUDA status; empty-to-empty uploads succeed without an API call.
             * @throws std::runtime_error If the explicit copy stream cannot be created.
             */
            [[nodiscard]] cudaError_t upload(const std::vector<T> &source)
            {
                if (source.size() != elements_)
                    return cudaErrorInvalidValue;
                if (elements_ == 0)
                    return cudaSuccess;
                TestCUDAStream copy_stream;
                const auto status = cudaMemcpyAsync(
                    pointer_, source.data(), bytes(), cudaMemcpyHostToDevice,
                    copy_stream.native());
                // Pageable H2D may return after staging, before DMA finishes.
                // This cold fixture boundary must publish completed device bytes,
                // not rely on legacy-stream ordering with non-blocking consumers.
                return status == cudaSuccess ? copy_stream.synchronize() : status;
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
            projection.source_codebook_id = format.codebook == kNativeQuantisedIQ4XSSourceScaleCodebook
                ? uint8_t{4} : format.codebook;
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
                    if (format.codebook == 19)
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
                    if (format.codebook == 8)
                    {
                        // Author source metadata independently of packVnniBlock.
                        // The secondary word contains signed bytes, not a half.
                        const std::size_t source_index = block % N +
                            static_cast<std::size_t>(N) * ((block / N) / 8);
                        Q6_KBlock source{};
                        source.d = static_cast<std::uint16_t>(
                            0x2400u + (source_index % 0x0800u));
                        source.scales[0] = static_cast<std::int8_t>(
                            static_cast<int>((block * 37u + 11u) % 256u) - 128);
                        source.scales[1] = static_cast<std::int8_t>(
                            static_cast<int>((block * 19u + 71u) % 256u) - 128);
                        projection.scales[block] = source.d;
                        projection.mins[block] = static_cast<std::uint16_t>(
                            static_cast<std::uint8_t>(source.scales[0])) |
                            (static_cast<std::uint16_t>(
                                 static_cast<std::uint8_t>(source.scales[1])) << 8);
                    }
                    if (format.codebook == kNativeQuantisedIQ4XSSourceScaleCodebook)
                    {
                        // Independent original half base repeated across its
                        // eight32-value subblocks; encoded6 is raw, never half.
                        const size_t source_index = block % N + size_t(N) * ((block / N) / 8);
                        projection.scales[block] = uint16_t(0x2400u + source_index % 0x0800u);
                        projection.mins[block] = uint16_t((block * 37u + 11u) & 63u);
                    }
                    if (format.codebook == 13)
                    {
                        // Exact original source half base and coefficient byte.
                        const std::size_t source_index = block % N +
                            static_cast<std::size_t>(N) * ((block / N) / 8);
                        projection.scales[block] = static_cast<std::uint16_t>(
                            0x2400u + (source_index % 0x0800u));
                        projection.mins[block] = static_cast<std::uint8_t>(
                            (block * 37u + 11u) & 0xffu);
                    }
                }
            }
            return projection;
        }

        /**
         * @brief Pack a real quantized tensor into the common accelerator
         * representation consumed by production GEMM and MoE kernels.
         * @param tensor Source tensor implementing `IINT8Unpackable`.
         * @return Complete host projection with original source provenance.
         * @throws std::invalid_argument when the tensor is not NativeQuantised or
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
                    "all-format tier test requires a NativeQuantised tensor");
            }
            const NativeQuantisedFormatInfo &format =
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
                canonicalDeviceVnniCodebookId(format);
            projection.payload_bytes_per_block =
                static_cast<std::uint8_t>(format.payload_bytes);
            projection.is_asymmetric = canonicalDeviceVnniFormat(format).is_asymmetric;
            projection.is_superblock = format.is_superblock;
            projection.has_emins = format.has_emins;

            const std::size_t blocks =
                static_cast<std::size_t>(N) * projection.blocks_per_row;
            projection.payload.resize(blocks * format.payload_bytes);
            projection.scales.assign(nativeQuantisedScalePlaneStorageWords(projection.codebook_id, blocks), 0);
            if (canonicalDeviceVnniFormat(format).is_asymmetric)
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
            context.primary_layout = gpuPrimaryScaleLayout(projection.codebook_id);
            context.mins_array =
                canonicalDeviceVnniFormat(format).is_asymmetric ? projection.mins.data() : nullptr;
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

        /** @brief Borrow the actual on-device planes of a SourceNative loader publication. */
        ExpertTierGpuConstProjectionView sourceNativeTierView(ITensorGemm &gemm, const NativeQuantisedFormatInfo &format)
        {
            DeviceNativeQuantisedMatrixDesc descriptor;
            if (!gemm.exportNativeQuantisedMatrixDesc(descriptor))
                throw std::invalid_argument("SourceNative loader did not publish a native projection");
            const auto physical = canonicalDeviceVnniFormat(format);
            const auto packed = makeGpuExpertPackedDescriptor(descriptor, physical.payload_bytes_per_block,
                physical.is_asymmetric, physical.has_emins);
            return {packed.ptrs.d_vnni, static_cast<const uint16_t *>(packed.ptrs.d_scales),
                static_cast<const uint16_t *>(packed.ptrs.d_mins), static_cast<const uint32_t *>(packed.ptrs.d_emins),
                packed.vnni_bytes, packed.scales_bytes, packed.mins_bytes, packed.emins_bytes, packed.metadata_layout};
        }

        /**
         * @brief Build a read-only device view from owned test allocations.
         * @param payload Device payload owner.
         * @param scales Device scale owner.
         * @param mins Optional device minimum/secondary-scale owner.
         * @param scales_bytes Exact live primary-plane bytes (storage may round up).
         * @return Exact-capacity source view.
         */
        ExpertTierGpuConstProjectionView makeConstView(
            const TestCUDABuffer<std::uint8_t> &payload,
            const TestCUDABuffer<std::uint16_t> &scales,
            const TestCUDABuffer<std::uint16_t> &mins,
            std::size_t scales_bytes)
        {
            return ExpertTierGpuConstProjectionView{
                .payload = payload.data(),
                .scales = scales.data(),
                .mins = mins.data(),
                .emins = nullptr,
                .payload_bytes = payload.bytes(),
                .scales_bytes = scales_bytes,
                .mins_bytes = mins.bytes(),
                .emins_bytes = 0,
            };
        }

        /**
         * @brief Build a writable device view from owned test allocations.
         * @param payload Device payload owner.
         * @param scales Device scale owner.
         * @param mins Optional device minimum/secondary-scale owner.
         * @param scales_bytes Exact live primary-plane bytes (storage may round up).
         * @return Exact-capacity destination view.
         */
        ExpertTierGpuMutableProjectionView makeMutableView(
            TestCUDABuffer<std::uint8_t> &payload,
            TestCUDABuffer<std::uint16_t> &scales,
            TestCUDABuffer<std::uint16_t> &mins,
            std::size_t scales_bytes)
        {
            return ExpertTierGpuMutableProjectionView{
                .payload = payload.data(),
                .scales = scales.data(),
                .mins = mins.data(),
                .emins = nullptr,
                .payload_bytes = payload.bytes(),
                .scales_bytes = scales_bytes,
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
                ASSERT_TRUE(cudaNativeQuantisedInitIQGridTables_tuned())
                    << "production IQ decode tables must be initialized";
            }
        };

        /** @test Native CUDA migration preserves every coded plane through a phased, partial intersection. */
        TEST_F(CUDAExpertTierWeightKernelsTest, FileDensityPlanesRoundTripMatchesSourceNative)
        {
            using namespace cpu::native_quantised;
            namespace sb = native_quantised_superblock;
            constexpr int N = 67, K = 512;
            constexpr unsigned chunk_units = 3;
            TestCUDAStream stream;
            auto *gemv = cudaGemvContext_create(0);
            ASSERT_NE(gemv, nullptr);
            const auto destroy = std::unique_ptr<CUDAGemvContext, decltype(&cudaGemvContext_destroy)>(gemv, cudaGemvContext_destroy);
            TestCUDABuffer<float> partials(NativeQuantisedGroupedDecodePolicy::maximum_k_partitions * N);
            cudaGemvContext_bindWorkspace(gemv, partials.data(), partials.bytes());
            TestCUDABuffer<int8_t> activation(K);
            TestCUDABuffer<float> activation_scales(K / 32), source_result(N), promoted_result(N);
            ASSERT_EQ(activation.upload(std::vector<int8_t>(K, 13)), cudaSuccess);
            ASSERT_EQ(activation_scales.upload(std::vector<float>(K / 32, 0.05f)), cudaSuccess);
            unsigned qualified_coded = 0, qualified_single_scale = 0;
            for (const auto &format : test::quantizedVerifierFormats())
            {
                const auto *source = native_quantised_formats::forSourceIdentity(format.source_codebook_id, format.source_is_superblock);
                ASSERT_NE(source, nullptr);
                const bool coded = migrationUsesSuperblockCodes(*source);
                if (!coded && !expertTierPreservesNativeSingleScale(preparedEncodingForSource(*source), source->codebook_id))
                    continue;
                SCOPED_TRACE(format.label);
                const auto tensor = format.create({N, K}, 42);
                CPUNativeQuantisedPackedWeights cpu;
                ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), cpu));
                HostGpuExpertPackedProjection oracle;
                std::string error;
                ASSERT_TRUE(cpuToGpuExpertPackedReference(cpu, oracle, &error)) << error;
                const auto cpu_bytes = cpu.readView().bytes();
                const auto up = makeCpuToGpuExpertTierWeightStreamManifest(cpu, 11, 0, 4, ExpertTierWeightProjection::Down, chunk_units);
                const auto down = makeGpuToCpuExpertTierWeightStreamManifest(*source, N, K, 12, 0, 4,
                    ExpertTierWeightProjection::Down, chunk_units);
                TestCUDABuffer<uint8_t> payload(oracle.payload.size()),
                    codes(coded ? oracle.codes.size() : oracle.scaleLiveBytes()),
                    headers(coded ? oracle.superblocks.size() : oracle.mins.size() * sizeof(uint16_t));
                auto loaded = test::makeGpuPreparedGemm(tensor.get(), DeviceId::cuda(0));
                DeviceNativeQuantisedMatrixDesc source_desc;
                ASSERT_TRUE(loaded.kernel->exportNativeQuantisedMatrixDesc(source_desc));
                const auto canonical = canonicalDeviceVnniFormat(*source);
                const auto source_packed = makeGpuExpertPackedDescriptor(source_desc,
                    canonical.payload_bytes_per_block, canonical.is_asymmetric, canonical.has_emins);
                const auto live_codes = coded && !oracle.codes.empty() ? oracle.codes.size() - sb::kSuperblockCodePlanePadBytes : codes.bytes();
                const ExpertTierGpuMutableProjectionView gpu{
                    .payload = payload.data(), .scales = reinterpret_cast<uint16_t *>(live_codes ? codes.data() : payload.data()),
                    .mins = reinterpret_cast<uint16_t *>(headers.data()), .payload_bytes = payload.bytes(),
                    .scales_bytes = live_codes, .mins_bytes = headers.bytes(), .metadata_layout = oracle.metadata_layout};
                const ExpertTierGpuConstProjectionView source_view{
                    .payload = source_packed.ptrs.d_vnni,
                    .scales = static_cast<const uint16_t *>(source_packed.ptrs.d_scales),
                    .mins = static_cast<const uint16_t *>(source_packed.ptrs.d_mins),
                    .payload_bytes = source_packed.vnni_bytes, .scales_bytes = source_packed.scales_bytes,
                    .mins_bytes = source_packed.mins_bytes, .metadata_layout = source_packed.metadata_layout};
                ASSERT_TRUE(source_view.validFor(down.deviceLayout()));
                TestCUDABuffer<uint8_t> stage(up.deviceLayout().cpuStorage().maximumWireUnitBytes() * chunk_units);
                const auto run = [&](const ExpertTierWeightDeviceLayout &upload, const ExpertTierWeightDeviceLayout &download,
                    const ExpertTierGpuConstProjectionView &gpu_source, std::span<uint8_t> returned, bool capture_units = false) {
                    size_t bytes = 0;
                    EXPECT_EQ(upload.chunkBytes(0), 0u);
                    for (unsigned first = 0; first < upload.unit_count; first += chunk_units)
                    {
                        const auto units = std::min(chunk_units, upload.unit_count - first);
                        std::vector<uint8_t> wire(upload.chunkBytes(units, first));
                        ASSERT_TRUE(gatherExpertTierCPUWeightChunk(upload, cpu_bytes, first, units, wire));
                        const auto submit = [&] {
                            return launchCpuToGpuExpertTierChunkCUDA(stage.data(), stage.bytes(), upload, first, units, gpu, stream.opaque()) &&
                                   launchGpuToCpuExpertTierChunkCUDA(gpu_source, download, first, units, stage.data(), stage.bytes(), stream.opaque());
                        };
                        CUDAGraphCapture graph(stream.native(), 0);
                        if (capture_units)
                        {
                            ScopedBackendGraphCapture capture(graph, "file-density expert tier units");
                            ASSERT_TRUE(capture.begin());
                            ASSERT_TRUE(submit());
                            capture.finish();
                            ASSERT_TRUE(graph.instantiate());
                        }
                        ASSERT_EQ(cudaMemcpyAsync(stage.data(), wire.data(), wire.size(), cudaMemcpyHostToDevice, stream.native()), cudaSuccess);
                        ASSERT_TRUE(capture_units ? graph.launch() : submit());
                        std::vector<uint8_t> observed(wire.size());
                        ASSERT_EQ(cudaMemcpyAsync(observed.data(), stage.data(), observed.size(), cudaMemcpyDeviceToHost, stream.native()), cudaSuccess);
                        ASSERT_EQ(stream.synchronize(), cudaSuccess);
                        EXPECT_EQ(observed, wire);
                        ASSERT_TRUE(scatterExpertTierCPUWeightChunk(download, observed, first, units, returned));
                        bytes += observed.size();
                    }
                    EXPECT_EQ(bytes, upload.chunkBytes(upload.unit_count));
                };
                std::vector<uint8_t> received(cpu_bytes.size(), 0);
                run(up.deviceLayout(), down.deviceLayout(), gpu.asConst(), received);
                EXPECT_EQ(std::memcmp(received.data(), cpu_bytes.data(), received.size()), 0);
                std::fill(received.begin(), received.end(), 0);
                run(up.deviceLayout(), down.deviceLayout(), source_view, received);
                ASSERT_EQ(std::memcmp(received.data(), cpu_bytes.data(), received.size()), 0)
                    << "SourceNative and CPU source packing diverged before the projection";

                const auto gemv_submit = [&](const ExpertTierGpuConstProjectionView &weights, float *output) {
                    const DeviceNativeQuantisedMatrixDesc descriptor{.payload = weights.payload, .scales = weights.scales,
                        .mins = weights.mins, .metadata_layout = weights.metadata_layout};
                    return cudaNativeQuantisedGemvTuned_fp32_withPolicy(activation.data(), weights.payload,
                        native_quantised_scale::metadataPlanes(descriptor), output, activation_scales.data(),
                        N, K, 1, 0, nullptr, nullptr, oracle.codebook_id, canonicalDeviceVnniCodebookId(source->codebook_id), 0, stream.opaque(), gemv, nullptr);
                };
                ASSERT_TRUE(gemv_submit(source_view, source_result.data()));
                ASSERT_TRUE(gemv_submit(gpu.asConst(), promoted_result.data()));
                ASSERT_EQ(stream.synchronize(), cudaSuccess);
                std::vector<float> expected, actual;
                ASSERT_EQ(source_result.download(expected), cudaSuccess);
                ASSERT_EQ(promoted_result.download(actual), cudaSuccess);
                EXPECT_EQ(std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)), 0);

                // Ends inside the second source superblock, starts inside the first.
                // This second invocation reuses dirty code words; transferring only
                // the intersection must not depend on allocation initialization.
                auto upload = up.deviceLayout(), download = down.deviceLayout();
                upload.N = download.N = 65;
                upload.N_padded = download.N_padded = 128;
                upload.K = download.K = 352 - 96;
                upload.blocks_per_row = download.blocks_per_row = upload.K / 32;
                upload.cpu_first_column = download.cpu_first_column = 96;
                upload.gpu_first_column = download.gpu_first_column = 96;
                upload.cpu_first_row = download.cpu_first_row = 1;
                upload.gpu_first_row = download.gpu_first_row = 1;
                upload.unit_count = download.unit_count = 2 * upload.blocks_per_row;
                ASSERT_TRUE(upload.valid());
                ASSERT_TRUE(download.valid());
                run(upload, download, source_view, received, true);
                if (coded) ++qualified_coded;
                else ++qualified_single_scale;
            }
            EXPECT_EQ(qualified_coded, 13u);
            EXPECT_EQ(qualified_single_scale, 9u);
        }

        /** @test Original GPU K origins determine real shadow-slot capacity and both conversion directions. */
        TEST_F(CUDAExpertTierWeightKernelsTest, CodedShadowSlotRetainsOriginalRootPhase)
        {
            constexpr int n = 67, cpu_k = 768;
            constexpr unsigned chunk_units = 2;
            constexpr auto role = ExpertTierWeightProjection::Down;
            const auto device = DeviceId::cuda(0);
            const auto &source = native_quantised_formats::Q4_K;
            const auto format = ExpertWeightFormat::nativeQuantised(
                {.codebook_id = source.codebook_id, .is_superblock = true, .present = true});
            auto tensor = test::quantizedVerifierFormat("Q4_K").create({n, cpu_k}, 143);
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(), cpu));
            const auto original = cpu.readView().bytes();
            const auto upload_manifest = makeCpuToGpuExpertTierWeightStreamManifest(cpu, 1, 0, 0, role, chunk_units);
            using Projection = MoEExpertProjectionOwnership::Projection;
            const Projection whole{MoEProjectionResidence::ExpertOwner, n, cpu_k, 0, n, 0, cpu_k};
            for (const auto [origin, k, header_words] : {std::tuple{32, 256, 2}, std::tuple{64, 224, 2}, std::tuple{224, 320, 3}})
            {
                SCOPED_TRACE(origin);
                const Projection root{MoEProjectionResidence::ParticipantInputSlice, n, cpu_k, 0, n, origin, k};
                const auto upload = upload_manifest.forIntersection(ExpertTierWeightProjectionIntersection::between(whole, root)).deviceLayout();
                const auto download = makeGpuToCpuExpertTierWeightStreamManifest(source, n, k, 2, 0, 0, role, chunk_units).forIntersection(ExpertTierWeightProjectionIntersection::between(root, whole)).deviceLayout();
                ASSERT_TRUE(upload.valid());
                ASSERT_TRUE(download.valid());
                const MoEOverlayProjectionWeightManifest local{.projection = role, .N = n, .K = k,
                    .format = format, .first_column = origin};
                const auto phase = local.gpuSuperblockPhase();
                ASSERT_EQ(phase, origin / 32);
                EXPECT_EQ(upload.gpu_superblock_phase, phase);
                EXPECT_EQ(download.gpu_superblock_phase, phase);
                auto pool = GpuExpertSlotPool::createForTest(getCUDABackend(), device, 0, 0, 1,
                    {{.label = "down", .N = n, .K = k, .payload_bytes_per_block = 16,
                      .is_asymmetric = true, .codebook_id = source.codebook_id, .format = format,
                      .gpu_superblock_phase = phase}}, 0);
                auto lease = pool->acquire(0);
                ASSERT_TRUE(lease.has_value());
                const auto &slot = lease->projections.front().slot;
                const std::size_t payload = std::size_t(n) * (k / 32) * 16;
                const std::size_t codes = (std::size_t(n) * (k / 32) * 12 + 7) / 8;
                const std::size_t headers = n * header_words * sizeof(std::uint32_t);
                ASSERT_EQ(slot.mins_bytes, headers);
                ASSERT_EQ(nativeQuantisedLayoutPhase(slot.metadata_layout), phase);
                const auto bom = resolveExpertPreparedProjectionMemoryGeometry(n, k, format,
                    GPUPreparedWeightPoolLayout::MigrationReusable, phase);
                EXPECT_EQ(bom.gpu_shadow_bytes, alignGPUWeightLoadAllocation(payload) +
                    alignGPUWeightLoadAllocation(codes) + alignGPUWeightLoadAllocation(headers));
                EXPECT_GT(bom.gpu_shadow_bytes, resolveExpertPreparedProjectionMemoryGeometry(n, k, format).gpu_shadow_bytes);
                ExpertTierGpuMutableProjectionView gpu{
                    .payload = slot.d_native_quantised_payload,
                    .scales = static_cast<std::uint16_t *>(slot.d_native_quantised_scales),
                    .mins = static_cast<std::uint16_t *>(slot.d_native_quantised_mins),
                    .payload_bytes = payload, .scales_bytes = codes, .mins_bytes = headers,
                    .metadata_layout = slot.metadata_layout};
                ASSERT_TRUE(gpu.validFor(upload));
                ASSERT_TRUE(gpu.validFor(download));
                const auto capacity = chunk_units * upload.cpuStorage().maximumWireUnitBytes();
                auto &transfers = TransferEngine::instance();
                ExpertTierWeightTransferLane lane({.device = device,
                    .staging = transfers.allocatePersistentTransferStagingSlices(capacity, 1, device).front(),
                    .execution = test::ComponentTransferExecutionPoolFixtureAccess::allocate(transfers, 1, device,
                        "cuda_phased_shadow_slot").front(),
                    .progress = BackgroundTransferProgressBinding::nativeStream(),
                    .lane_name = "cuda_phased_shadow_slot", .perf_device = device.toString()});
                std::string error;
                ASSERT_TRUE(lane.materialize(&error)) << error;
                ASSERT_TRUE(lane.startCpuToGpu(upload, original, gpu, &error)) << error;
                ASSERT_EQ(pollLaneToCompletion(lane), ExpertTierWeightTransferProgress::Ready);
                std::vector<uint8_t> expected(original.size(), 0xa5), observed = expected;
                for (unsigned first = 0; first < upload.unit_count; first += chunk_units)
                {
                    const auto count = std::min(chunk_units, upload.unit_count - first);
                    std::vector<uint8_t> wire(upload.chunkBytes(count, first));
                    ASSERT_TRUE(gatherExpertTierCPUWeightChunk(upload, original, first, count, wire));
                    ASSERT_TRUE(scatterExpertTierCPUWeightChunk(download, wire, first, count, expected));
                }
                ASSERT_TRUE(lane.startGpuToCpu(download, gpu.asConst(), observed,
                    ExpertTierSourceReadiness::publishedResidencyBank(1), &error)) << error;
                ASSERT_EQ(pollLaneToCompletion(lane), ExpertTierWeightTransferProgress::Ready);
                EXPECT_EQ(observed, expected);
            }
        }

        /** @test A retained CUDA rectangle preserves neighboring code bits and untouched payload/header fields after reset. */
        TEST_F(CUDAExpertTierWeightKernelsTest, CapturedCodedPeerRectanglePreservesShiftedNeighbors)
        {
            namespace sb = native_quantised_superblock;
            constexpr unsigned N = 5, blocks = 9, stride = 12, bits = 12;
            constexpr unsigned src_shift = 4, dst_shift = 0;
            constexpr uint8_t book = 9;
            constexpr auto layout = NativeQuantisedMetadataRepresentation::SuperblockCoded;
            constexpr size_t payload_bytes = N * blocks * stride, header_bytes = N * 2 * sizeof(uint16_t);
            constexpr size_t code_bytes = (N * blocks * bits + 7) / 8 + sb::kSuperblockCodePlanePadBytes;
            const auto populate = [](size_t bytes, unsigned seed) {
                std::vector<uint8_t> value(bytes);
                for (size_t i = 0; i < bytes; ++i) value[i] = uint8_t(i * 137 + seed);
                return value;
            };
            const auto src_payload = populate(payload_bytes, 31), src_codes = populate(code_bytes, 71), src_headers = populate(header_bytes, 11);
            const auto initial_payload = populate(payload_bytes, 18), initial_codes = populate(code_bytes, 83), initial_headers = populate(header_bytes, 49);
            TestCUDABuffer<uint8_t> input(payload_bytes), input_codes(code_bytes), input_headers(header_bytes);
            TestCUDABuffer<uint8_t> output(payload_bytes), output_codes(code_bytes), output_headers(header_bytes);
            ASSERT_EQ(input.upload(src_payload), cudaSuccess);
            ASSERT_EQ(input_codes.upload(src_codes), cudaSuccess);
            ASSERT_EQ(input_headers.upload(src_headers), cudaSuccess);
            const CUDACodedPeerProjectionRectangle region{
                .rows = 3, .blocks = 7, .payload_bytes = stride, .codebook = book,
                .src_n = N, .dst_n = N, .src_row = 1, .dst_row = 2, .src_block = 2, .dst_block = 2,
                .source = {.payload = input.data(), .scales = reinterpret_cast<uint16_t *>(input_codes.data()),
                    .mins = reinterpret_cast<uint16_t *>(input_headers.data()), .metadata_layout = nativeQuantisedMetadataLayout(layout, 0, src_shift)},
                .destination = {.payload = output.data(), .scales = reinterpret_cast<uint16_t *>(output_codes.data()),
                    .mins = reinterpret_cast<uint16_t *>(output_headers.data()), .metadata_layout = nativeQuantisedMetadataLayout(layout, 0, dst_shift)}};
            TestCUDAStream stream;
            CUDAGraphCapture graph(stream.native(), 0);
            {
                ScopedBackendGraphCapture capture(graph, "coded peer rectangle");
                ASSERT_TRUE(capture.begin());
                ASSERT_TRUE(launchCUDACodedPeerProjectionRectangle(region, stream.opaque()));
                capture.finish();
            }
            ASSERT_TRUE(graph.instantiate());
            for (int repeat = 0; repeat < 3; ++repeat)
            {
                SCOPED_TRACE(repeat);
                ASSERT_EQ(output.upload(initial_payload), cudaSuccess);
                ASSERT_EQ(output_codes.upload(initial_codes), cudaSuccess);
                ASSERT_EQ(output_headers.upload(initial_headers), cudaSuccess);
                ASSERT_TRUE(graph.launch());
                ASSERT_EQ(stream.synchronize(), cudaSuccess);
                std::vector<uint8_t> actual_payload, actual_codes, actual_headers;
                ASSERT_EQ(output.download(actual_payload), cudaSuccess);
                ASSERT_EQ(output_codes.download(actual_codes), cudaSuccess);
                ASSERT_EQ(output_headers.download(actual_headers), cudaSuccess);
                for (unsigned block = 0; block < blocks; ++block)
                    for (unsigned n = 0; n < N; ++n)
                    {
                        const auto i = block * N + n;
                        const bool live = block >= 2 && n >= 2;
                        EXPECT_EQ(std::memcmp(actual_payload.data() + i * stride,
                            live ? src_payload.data() + (i - 1) * stride : initial_payload.data() + i * stride, stride), 0);
                        EXPECT_EQ(sb::loadPackedBits(actual_codes.data(), i * bits + dst_shift, bits),
                            live ? sb::loadPackedBits(src_codes.data(), (i - 1) * bits + src_shift, bits)
                                 : sb::loadPackedBits(initial_codes.data(), i * bits + dst_shift, bits));
                    }
                for (unsigned i = 0; i < 2 * N; ++i)
                    EXPECT_EQ(std::memcmp(actual_headers.data() + i * sizeof(uint16_t),
                        i % N >= 2 ? src_headers.data() + (i - 1) * sizeof(uint16_t) : initial_headers.data() + i * sizeof(uint16_t), sizeof(uint16_t)), 0);
                const auto code_end = (N * blocks * bits + 7) / 8;
                EXPECT_TRUE(std::equal(actual_codes.begin() + code_end, actual_codes.end(), initial_codes.begin() + code_end));
                EXPECT_EQ(actual_codes[code_end - 1] & 0xf0, initial_codes[code_end - 1] & 0xf0);
            }
        }

        /**
         * @brief Verify a compact CUDA relay rectangle against independent SourceNative allocations.
         * @param name Source format fixture name.
         * @param metadata Source format identity before device packing.
         * @param expected_wire_bytes Exact live bytes across all chunks, including source headers.
         * @param context Two real MPI ranks, or null for the in-process endpoint control.
         * @param sender Rank that owns the original source in the MPI case.
         */
        void proveCodedRemoteGpuPlanes(const char *name, const NativeQuantisedFormatInfo &metadata,
                                      std::size_t expected_wire_bytes,
                                      const std::shared_ptr<MPIContext> &context = {}, int sender = 0)
        {
            namespace sb = native_quantised_superblock;
            SCOPED_TRACE(name);
            const auto device = DeviceId::cuda(0);
            constexpr int src_n = 5, dst_n = 4, src_k = 512, dst_k = 256;
            const auto &format = test::quantizedVerifierFormat(name);
            const auto a = format.create({src_n, src_k}, 117);
            const auto b = format.create({dst_n, dst_k}, 193);
            const auto source_owner = std::make_shared<test::GpuPreparedGemm>(test::makeGpuPreparedGemm(a.get(), device));
            const auto destination_owner = std::make_shared<test::GpuPreparedGemm>(test::makeGpuPreparedGemm(b.get(), device));
            DeviceNativeQuantisedMatrixDesc src_desc, dst_desc;
            ASSERT_TRUE(source_owner->kernel->exportNativeQuantisedMatrixDesc(src_desc));
            ASSERT_TRUE(destination_owner->kernel->exportNativeQuantisedMatrixDesc(dst_desc));
            const auto src = makeGpuExpertPackedDescriptor(src_desc, metadata.payload_bytes, metadata.is_asymmetric, metadata.has_emins);
            const auto dst = makeGpuExpertPackedDescriptor(dst_desc, metadata.payload_bytes, metadata.is_asymmetric, metadata.has_emins);
            const auto read = [](const void *pointer, size_t bytes) {
                std::vector<uint8_t> result(bytes);
                if (bytes != 0)
                {
                    TestCUDAStream stream;
                    EXPECT_EQ(cudaMemcpyAsync(result.data(), pointer, bytes, cudaMemcpyDeviceToHost, stream.native()), cudaSuccess);
                    EXPECT_EQ(stream.synchronize(), cudaSuccess);
                }
                return result;
            };
            const bool sends = !context || context->rank() == sender;
            const bool receives = !context || context->rank() != sender;
            const auto before_payload = receives ? read(dst.ptrs.d_vnni, dst.vnni_bytes) : std::vector<uint8_t>{};
            const auto before_codes = receives ? read(dst.ptrs.d_scales, dst.scales_bytes) : std::vector<uint8_t>{};
            const auto before_headers = receives ? read(dst.ptrs.d_mins, dst.mins_bytes) : std::vector<uint8_t>{};
            const auto source_payload = receives ? read(src.ptrs.d_vnni, src.vnni_bytes) : std::vector<uint8_t>{};
            const auto source_codes = receives ? read(src.ptrs.d_scales, src.scales_bytes) : std::vector<uint8_t>{};
            const auto source_headers = receives ? read(src.ptrs.d_mins, src.mins_bytes) : std::vector<uint8_t>{};
            using P = MoEExpertProjectionOwnership::Projection;
            using R = MoEProjectionResidence;
            const auto overlap = ExpertTierWeightProjectionIntersection::between(
                P{R::ParticipantOutputSlice, 8, 512, 0, src_n, 0, src_k},
                P{R::ParticipantInputSlice, 8, 512, 3, dst_n, 256, dst_k});
            const NativeQuantisedSourceIdentity provenance{.codebook_id = metadata.codebook_id, .is_superblock = true, .present = true};
            const MoEOverlayRemoteProjectionIdentity identity{
                .expected_epoch = 811, .candidate_epoch = 812, .execution_fingerprint = {.low = 19, .high = 71},
                .layer_idx = 0, .expert_id = 1, .projection = ExpertTierWeightProjection::Down,
                .source_participant = sender, .destination_participant = 1 - sender,
                .source_world_rank = sender, .destination_world_rank = 1 - sender, .source_device = device, .destination_device = device};
            constexpr uint32_t capacity = 64;
            const auto manifest = makeMoEOverlayRemoteGpuProjectionManifest(identity, src, provenance, capacity, overlap);
            ASSERT_TRUE(manifest.carriesCodedGpuUnits());
            ASSERT_EQ(manifest.total_bytes, expected_wire_bytes);
            auto &transfers = TransferEngine::instance();
            const unsigned lane_count = context ? 1 : 2;
            const auto staging = transfers.allocatePersistentTransferStagingSlices(capacity, lane_count, device, HostTransferAccess::CrossProcessReadable);
            const auto execution = test::ComponentTransferExecutionPoolFixtureAccess::allocate(transfers, lane_count, device, "cuda_coded_remote_relay");
            const auto lane = [&](unsigned i) {
                return std::make_shared<MoEOverlayGpuRemoteProjectionLane>(MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = device, .staging = staging.at(i), .execution = execution.at(i),
                    .progress = BackgroundTransferProgressBinding::nativeStream(),
                    .lane_name = "cuda_coded_remote_relay_" + std::to_string(i), .perf_device = device.toString()});
            };
            std::string error;
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> send, receive;
            std::shared_ptr<MoEOverlayGpuRemoteProjectionSource> source;
            std::shared_ptr<MoEOverlayGpuRemoteProjectionDestination> destination;
            if (sends)
            {
                send = lane(0);
                ASSERT_TRUE(send->materialize(&error)) << error;
                source = std::make_shared<MoEOverlayGpuRemoteProjectionSource>(manifest, send, src,
                    ExpertTierSourceReadiness::publishedResidencyBank(identity.expected_epoch), source_owner);
            }
            if (receives)
            {
                receive = lane(context ? 0 : 1);
                ASSERT_TRUE(receive->materialize(&error)) << error;
                destination = std::make_shared<MoEOverlayGpuRemoteProjectionDestination>(identity, receive,
                    [&](const auto &, auto *binding, std::string *) {
                        *binding = {.descriptor = dst, .engine = std::shared_ptr<ITensorGemm>(destination_owner, destination_owner->kernel)};
                        return true;
                    }, destination_owner);
            }
            const auto wait = [&](const auto &poll) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
                auto progress = poll();
                while (progress == MoEOverlayResidencyWaveProgress::Pending && std::chrono::steady_clock::now() < deadline)
                {
                    std::this_thread::yield();
                    progress = poll();
                }
                return progress;
            };
            if (context)
            {
                // The peer owns the other endpoint; only the private MPI transport delivers chunks.
                MoEOverlayMPIRemoteProjectionTransport transport({
                    .mpi_context = context,
                    .lane_budget = MoEOverlayRemoteProjectionLaneBudget::fromUniformCycles(1, 1, 1),
                    .staging_capacity_bytes = capacity,
                    .perf_device = "mpi_remote_projection_cuda_test",
                });
                auto reservation = transport.reserve(0, identity, source, destination);
                ASSERT_EQ(reservation.status, MoEOverlayResidencyStageStartStatus::Started) << reservation.error;
                ASSERT_NE(reservation.operation, nullptr);
                ASSERT_EQ(wait([&] { return reservation.operation->poll(&error); }), MoEOverlayResidencyWaveProgress::Ready) << error;
                const auto measurement = reservation.operation->completedMeasurement();
                ASSERT_TRUE(measurement.has_value());
                EXPECT_TRUE(measurement->valid());
                EXPECT_EQ(measurement->bytes, expected_wire_bytes);
            }
            else
            {
                ASSERT_TRUE(destination->beginManifest(manifest, &error)) << error;
                size_t wire_bytes = 0;
                while (!destination->complete())
                {
                    MoEOverlayRemoteProjectionChunkView chunk;
                    ASSERT_EQ(wait([&] { return source->pollNextChunk(&chunk, &error); }), MoEOverlayResidencyWaveProgress::Ready) << error;
                    ASSERT_GT(chunk.payload.size(), 0u);
                    ASSERT_LE(chunk.payload.size(), capacity);
                    auto progress = destination->beginChunk(chunk.header, chunk.payload, &error);
                    if (progress == MoEOverlayResidencyWaveProgress::Pending)
                        progress = wait([&] { return destination->pollChunk(&error); });
                    ASSERT_EQ(progress, MoEOverlayResidencyWaveProgress::Ready) << error;
                    wire_bytes += chunk.payload.size();
                    ASSERT_TRUE(source->acknowledgeChunkSent(chunk.header, &error)) << error;
                }
                ASSERT_TRUE(destination->publishFinal(&error)) << error;
                EXPECT_EQ(wire_bytes, expected_wire_bytes);
                EXPECT_EQ(send->stats().device_to_host_submissions, receive->stats().host_to_device_submissions);
            }
            if (sends)
                EXPECT_GT(send->stats().device_to_host_submissions, 1u);
            if (!receives)
                return;
            EXPECT_GT(receive->stats().host_to_device_submissions, 1u);
            ASSERT_TRUE(destination->complete());

            // Compare original physical bytes, not decoded values that could
            // conceal a changed scale code or damage to a neighbouring row.
            const auto payload_after = read(dst.ptrs.d_vnni, dst.vnni_bytes);
            const auto codes_after = read(dst.ptrs.d_scales, dst.scales_bytes);
            const auto headers_after = read(dst.ptrs.d_mins, dst.mins_bytes);
            const auto code_bits = sb::superblockCodePlaneBits(metadata.codebook_id);
            const auto header_bytes = sb::superblockFieldBytes(metadata.codebook_id);
            const auto payload_bytes = metadata.payload_bytes;
            if (code_bits == 0)
            {
                EXPECT_TRUE(codes_after.empty());
                EXPECT_EQ(dst.ptrs.d_scales, static_cast<const void *>(dst.ptrs.d_vnni));
            }
            for (unsigned block = 0; block < dst_k / 32; ++block)
                for (unsigned row = 0; row < dst_n; ++row)
                {
                    const auto index = block * dst_n + row;
                    const auto original = (block + 8) * src_n + row + 3;
                    const bool live = row < 2;
                    EXPECT_EQ(std::memcmp(payload_after.data() + index * payload_bytes,
                        live ? source_payload.data() + original * payload_bytes : before_payload.data() + index * payload_bytes, payload_bytes), 0);
                    if (code_bits != 0)
                        EXPECT_EQ(sb::loadPackedBits(codes_after.data(), index * code_bits, code_bits), live
                            ? sb::loadPackedBits(source_codes.data(), original * code_bits, code_bits)
                            : sb::loadPackedBits(before_codes.data(), index * code_bits, code_bits));
                }
            for (unsigned row = 0; row < dst_n; ++row)
                EXPECT_EQ(std::memcmp(headers_after.data() + row * header_bytes,
                    row < 2 ? source_headers.data() + (src_n + row + 3) * header_bytes : before_headers.data() + row * header_bytes,
                    header_bytes), 0);
        }

#include "utils/ExpertTierCodedNoPeerRelayContract.inc"
        /** @test The admitted CUDA no-peer lane preserves exact coded originals and shared guard bits. */
        TEST_F(CUDAExpertTierWeightKernelsTest, CodedNoPeerHostRelayPreservesAllSourcePlanes)
        {
            proveCodedNoPeerHostRelay<TestCUDABuffer, TestCUDAStream, cudaSuccess>(
                DeviceId::cuda(0),
                [](void *output, const void *input, std::size_t bytes, cudaStream_t stream) {
                    return cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToHost, stream);
                }, launchCUDACodedPeerProjectionRectangle);
        }

        /** @test The in-process control delivers the same native CUDA endpoint contract without MPI. */
        TEST_F(CUDAExpertTierWeightKernelsTest, CodedRemoteEndpointsPreserveOriginalGpuPlanes)
        {
            proveCodedRemoteGpuPlanes("Q3_K", native_quantised_formats::Q3_K, 220);
        }

        /** @test Two real MPI ranks relay in-payload and separate source codes in opposite directions. */
        TEST_F(CUDAExpertTierWeightKernelsTest, CodedMpiRelayPreservesOriginalGpuPlanes)
        {
            int rank = -1, size = 0;
            ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
            ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
            ASSERT_EQ(size, 2);
            const auto context = std::make_shared<MPIContext>(rank, size, MPI_COMM_WORLD);
            // 16 live blocks plus two header words per transfer. Q4_K's
            // 12-bit codes occupy three bytes for each pair of live columns.
            ASSERT_NO_FATAL_FAILURE(proveCodedRemoteGpuPlanes("IQ2_XXS", native_quantised_formats::IQ2_XXS, 132, context, 0));
            ASSERT_NO_FATAL_FAILURE(proveCodedRemoteGpuPlanes("Q4_K", native_quantised_formats::Q4_K, 288, context, 1));
        }

#include "utils/ExpertTierCapturedPackedTransferContract.inc"
        /** Existing capture progression coverage; original backend completion policy is explicit. */
        TEST_F(CUDAExpertTierWeightKernelsTest,CapturedLocalPackedTransfersProgressAllFormats)
        { proveCapturedPackedTransfer<TestCUDABuffer,cudaSuccess>(DeviceId::cuda(0),CapturedPackedClient::LocalTier); }
        TEST_F(CUDAExpertTierWeightKernelsTest,CapturedRemotePackedTransfersProgressAllFormats)
        { proveCapturedPackedTransfer<TestCUDABuffer,cudaSuccess>(DeviceId::cuda(0),CapturedPackedClient::RemoteProjection); }
        /** Compact rectangle repetition uses the actual retained lane and exact completion extents. */
        TEST_F(CUDAExpertTierWeightKernelsTest,CapturedLocalRectangularTransfersAllFormats)
        { proveCapturedPackedTransfer<TestCUDABuffer,cudaSuccess>(DeviceId::cuda(0),CapturedPackedClient::LocalTier,true); }
        TEST_F(CUDAExpertTierWeightKernelsTest,CapturedRemoteRectangularTransfersAllFormats)
        { proveCapturedPackedTransfer<TestCUDABuffer,cudaSuccess>(DeviceId::cuda(0),CapturedPackedClient::RemoteProjection,true); }

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

            const auto first = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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

            const auto reused = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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
            const auto execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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
            using namespace cpu::native_quantised;
            namespace sb = native_quantised_superblock;
            const auto &param = GetParam();
            auto source = makeProjection(param);
            const auto *source_format = native_quantised_formats::forSourceIdentity(source.source_codebook_id, source.is_superblock);
            ASSERT_NE(source_format, nullptr);
            const bool coded = migrationUsesSuperblockCodes(*source_format);
            CPUNativeQuantisedPackedWeights expected_cpu;
            std::string error;
            std::unique_ptr<TensorBase> tensor;
            std::optional<test::GpuPreparedGemm> loaded;
            if (coded)
            {
                const char *name = source_format == &native_quantised_formats::IQ4_XS ? "IQ4_XS"
                    : source_format == &native_quantised_formats::Q4_K ? "Q4_K" : "Q6_K";
                tensor = test::quantizedVerifierFormat(name).create({size_t(source.N), size_t(source.K)}, 41097);
                ASSERT_TRUE(packWeightsCPUNativeQuantised(tensor.get(), expected_cpu));
                ASSERT_TRUE(cpuToGpuExpertPackedReference(expected_cpu, source, &error)) << error;
                loaded.emplace(test::makeGpuPreparedGemm(tensor.get(), DeviceId::cuda(0)));
            }
            else
                ASSERT_TRUE(gpuToCpuExpertPackedReference(source, expected_cpu, &error)) << error;
            ASSERT_TRUE(source.valid());
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
            const auto source_view = loaded ? sourceNativeTierView(*loaded->kernel, *source_format)
                : makeConstView(source_payload, source_scales, source_mins, source.scaleLiveBytes());
            ASSERT_TRUE(source_view.validFor(layout));

            const std::size_t maximum_chunk_bytes = 2 * layout.cpuStorage().maximumWireUnitBytes();
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
                const std::size_t bytes = layout.chunkBytes(units,first_unit);
                ASSERT_TRUE(launchGpuToCpuExpertTierChunkCUDA(
                    source_view, layout, first_unit, units,
                    device_cpu_chunk.data(), device_cpu_chunk.bytes(),
                    stream.opaque()));
                ASSERT_EQ(cudaMemcpyAsync(
                    observed_chunk.data(), device_cpu_chunk.data(), bytes,
                    cudaMemcpyDeviceToHost, stream.native()), cudaSuccess);
                ASSERT_EQ(stream.synchronize(), cudaSuccess);

                const auto expected_wire=test::expertTierWireOracle(layout,expected_cpu.native_interleaved,first_unit,units);
                const auto expected_begin=expected_wire.begin();
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
            const auto scales_storage = coded ? source.codes.size() : source.scaleLiveBytes();
            const auto scales_live = coded && scales_storage ? scales_storage - sb::kSuperblockCodePlanePadBytes : scales_storage;
            TestCUDABuffer<uint8_t> destination_payload(source.payload.size()), destination_scales(scales_storage),
                destination_mins(coded ? source.superblocks.size() : source.mins.size() * sizeof(uint16_t));
            const ExpertTierGpuMutableProjectionView destination_view{
                .payload = destination_payload.data(),
                .scales = reinterpret_cast<uint16_t *>(scales_storage ? destination_scales.data() : destination_payload.data()),
                .mins = reinterpret_cast<uint16_t *>(destination_mins.data()), .payload_bytes = destination_payload.bytes(),
                .scales_bytes = scales_live, .mins_bytes = destination_mins.bytes(), .metadata_layout = source.metadata_layout};
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
                const std::size_t bytes = promotion_layout.chunkBytes(units,first_unit);
                const auto expected_wire=test::expertTierWireOracle(promotion_layout,expected_cpu.native_interleaved,first_unit,units);
                const std::uint8_t *host_source=expected_wire.data();
                ASSERT_EQ(cudaMemcpyAsync(
                    device_cpu_chunk.data(), host_source, bytes,
                    cudaMemcpyHostToDevice, stream.native()), cudaSuccess);
                ASSERT_TRUE(launchCpuToGpuExpertTierChunkCUDA(
                    device_cpu_chunk.data(), bytes, promotion_layout,
                    first_unit, units, destination_view, stream.opaque()));
            }
            ASSERT_EQ(stream.synchronize(), cudaSuccess);

            std::vector<uint8_t> observed_payload, observed_primary, observed_secondary;
            ASSERT_EQ(destination_payload.download(observed_payload), cudaSuccess);
            ASSERT_EQ(destination_scales.download(observed_primary), cudaSuccess);
            ASSERT_EQ(destination_mins.download(observed_secondary), cudaSuccess);
            EXPECT_EQ(observed_payload, source.payload);
            if (coded)
            {
                const auto bits = sb::superblockCodePlaneBits(source.codebook_id);
                for (size_t index = 0; bits && index < size_t(source.N) * source.blocks_per_row; ++index)
                    EXPECT_EQ(sb::loadPackedBits(observed_primary.data(), index * bits, bits),
                        sb::loadPackedBits(source.codes.data(), index * bits, bits));
                EXPECT_EQ(observed_secondary, source.superblocks);
            }
            else
            {
                EXPECT_EQ(std::memcmp(observed_primary.data(), source.scales.data(), scales_live), 0);
                if (!observed_secondary.empty())
                    EXPECT_EQ(std::memcmp(observed_secondary.data(), source.mins.data(), observed_secondary.size()), 0);
            }
        }

        INSTANTIATE_TEST_SUITE_P(
            EveryReversiblePreparedEncoding,
            CUDAExpertTierWeightKernelsTest,
            ::testing::Values(
                FormatCase{0, 16, false, false, "Q4_0"},
                FormatCase{4, 16, false, false, "IQ4NL"},
                FormatCase{kNativeQuantisedIQ4XSSourceScaleCodebook, 16, true, true, "IQ4XS"},
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
            ASSERT_EQ(test::quantizedVerifierFormats().size(), 22u);

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

                cpu::native_quantised::CPUNativeQuantisedPackedWeights expected_cpu;
                ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(
                    tensor.get(), expected_cpu));

                const NativeQuantisedFormatInfo *source_format =
                    native_quantised_formats::forSourceIdentity(
                        format.source_codebook_id,
                        format.source_is_superblock);
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
                    manifest.deviceLayout().cpuStorageBytes());

                auto loaded = test::makeGpuPreparedGemm(tensor.get(), DeviceId::cuda(0));
                const auto source_view = sourceNativeTierView(*loaded.kernel, *source_format);
                ASSERT_TRUE(source_view.validFor(layout));

                TestCUDABuffer<std::uint8_t> device_chunk(
                    units_per_chunk * layout.cpuStorage().maximumWireUnitBytes());
                std::vector<std::uint8_t> observed(
                    units_per_chunk * layout.cpuStorage().maximumWireUnitBytes());
                for (std::uint32_t first_unit = 0;
                     first_unit < layout.unit_count;
                     first_unit += units_per_chunk)
                {
                    const std::uint32_t unit_count =
                        std::min<std::uint32_t>(
                            units_per_chunk,
                            layout.unit_count - first_unit);
                    const std::size_t bytes =
                        layout.chunkBytes(unit_count,first_unit);
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
                    const auto expected_wire=test::expertTierWireOracle(layout,expected_cpu.native_interleaved,first_unit,unit_count);
                    const auto expected_begin=expected_wire.begin();
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
            cpu::native_quantised::CPUNativeQuantisedPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(
                source, expected_cpu, &error)) << error;

            const auto *source_format =
                native_quantised_formats::forSourceIdentity(19, false);
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
                makeConstView(source_payload, source_scales, source_mins, source.scaleLiveBytes());

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
                .execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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
                destination_mins,
                source.scaleLiveBytes());
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
         * @test A shared maintenance caller preserves its device while the owning
         *       CUDA worker performs the remote projection.
         *
         * The source arrays and auxiliary stream belong to device one, while
         * the calling thread is deliberately switched back to device zero
         * immediately before submission. This is the production shape when a
         * rank progresses several GPU participants. The remote lane must submit
         * all CUDA work to its device-one worker: byte-exact output proves that
         * the real repack used the source device, the worker probe proves that
         * its CUDA context remains device one, and the caller probe proves that
         * maintenance cannot mutate unrelated thread-local CUDA state.
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
            cpu::native_quantised::CPUNativeQuantisedPackedWeights expected_cpu;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(
                tensor.get(), expected_cpu));
            const auto *source_format =
                native_quantised_formats::forSourceIdentity(
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
                .scales_bytes = source.scaleLiveBytes(),
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
                                       DeviceId::cuda(device_ordinal),
                                       HostTransferAccess::CrossProcessReadable)
                                   .front(),
                    .execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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
            /* CUDA device selection is thread-local. Device work now belongs to
             * the lane's worker, so the maintenance caller must stay on device
             * zero instead of being changed as an accidental side effect. */
            int caller_device = -1;
            ASSERT_EQ(cudaGetDevice(&caller_device), cudaSuccess);
            EXPECT_EQ(caller_device, 0);

            /* Query on the exact owner after the completed lane transaction.
             * The byte comparison above makes this an end-to-end proof rather
             * than merely checking a context constructor's initial device. */
            int worker_device = -1;
            cudaError_t worker_status = cudaSuccess;
            auto &worker = GPUDeviceContextPool::instance().getContext(
                DeviceId::cuda(device_ordinal));
            worker.submitAndWait([&]
            {
                worker_status = cudaGetDevice(&worker_device);
            });
            ASSERT_EQ(worker_status, cudaSuccess);
            EXPECT_EQ(worker_device, device_ordinal);
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
                .execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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
                // Same lane, original coordinate intersections and arbitrary
                // floating K17. No quantized-block alignment is imposed here.
                using P=MoEExpertProjectionOwnership::Projection;using R=MoEProjectionResidence;
                const P originalSource{R::ParticipantOutputSlice,32,32,2,n,1,k};
                const P originalTarget{R::ParticipantInputSlice,32,32,8,17,7,17};
                const auto rectangle=ExpertTierWeightProjectionIntersection::between(originalSource,originalTarget);
                const auto reverse=ExpertTierWeightProjectionIntersection::between(originalTarget,originalSource);
                std::vector<uint8_t> rectangleCPU(size_t(17*17)*element_bytes,0xa5);
                const auto beforeRectangleBytes=lane.stats().bytes_submitted;
                void* rectangleReady=backend->createEvent(0);ASSERT_NE(rectangleReady,nullptr);
                ASSERT_TRUE(backend->recordEvent(rectangleReady,0,producer_stream.opaque()));
                ASSERT_TRUE(lane.startGpuToCpuContiguous(source_descriptor,rectangleCPU,
                    ExpertTierSourceReadiness::producerEvent(rectangleReady),rectangle,&error)) << error;
                ASSERT_EQ(pollLaneToCompletion(lane),ExpertTierWeightTransferProgress::Ready);
                backend->destroyEvent(rectangleReady,0);
                for(size_t row=0;row<17;++row)for(size_t column=0;column<17;++column)
                    for(size_t byte=0;byte<element_bytes;++byte)
                        EXPECT_EQ(rectangleCPU[(row*17+column)*element_bytes+byte],
                            row<13?source[((row+6)*k+column+6)*element_bytes+byte]:uint8_t(0xa5));
                ASSERT_EQ(gpu_destination.upload(std::vector<uint8_t>(bytes,0xa5)),cudaSuccess);
                ASSERT_TRUE(lane.startCpuToGpuContiguous(rectangleCPU,destination_descriptor,reverse,&error)) << error;
                ASSERT_EQ(pollLaneToCompletion(lane),ExpertTierWeightTransferProgress::Ready);
                ASSERT_EQ(gpu_destination.download(observed_gpu),cudaSuccess);
                for(size_t row=0;row<size_t(n);++row)for(size_t column=0;column<size_t(k);++column)
                    for(size_t byte=0;byte<element_bytes;++byte) {
                        const size_t index=(row*k+column)*element_bytes+byte;
                        EXPECT_EQ(observed_gpu[index],row>=6 && column>=6?source[index]:uint8_t(0xa5));
                    }
                EXPECT_EQ(lane.stats().bytes_submitted-beforeRectangleBytes,size_t(2*13*17)*element_bytes);

            }

            const auto stats = lane.stats();
            // Each precision completes a whole projection and an original-coordinate
            // rectangle in both directions on this same persistent lane.
            const auto completed_per_direction = 2u * formats.size();
            EXPECT_EQ(stats.transfers_started, 2u * completed_per_direction);
            EXPECT_EQ(stats.transfers_completed, 2u * completed_per_direction);
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
                               record.value == static_cast<double>(completed_per_direction);
                    });
            };
            EXPECT_TRUE(observed_direction("gpu_to_cpu"));
            EXPECT_TRUE(observed_direction("cpu_to_gpu"));
        }

        /**
         * @test Partial-N Q5_1 remote promotion and demotion preserve every live byte.
         *
         * N70/K96 retains a 128-column CPU allocation but communicates only
         * 70 live columns. GPU payload and metadata are checked against the
         * original CPU oracle. The final CPU allocation must contain exact
         * live encoded planes and untouched poison in allocation-only columns.
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

            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu_weights;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(
                tensor.get(), cpu_weights));
            ASSERT_EQ(cpu_weights.encoding, cpu::native_quantised::CPUNativeQuantisedEncoding::Q5NativeSingleScale);
            ASSERT_TRUE(cpu_weights.is_asymmetric);
            HostGpuExpertPackedProjection expected_gpu;
            std::string error;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(
                cpu_weights, expected_gpu, &error))
                << error;
            ASSERT_EQ(
                expected_gpu.codebook_id,
                format.source_codebook_id);

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
                .scales_bytes = expected_gpu.scaleLiveBytes(),
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
                                       DeviceId::cuda(0),
                                       HostTransferAccess::CrossProcessReadable)
                                   .front(),
                    .execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
                                         1u,
                                         DeviceId::cuda(0),
                                         "cuda_remote_q51_roundtrip")
                                     .front(),
                    .progress = BackgroundTransferProgressBinding::nativeStream(),
                    .lane_name = "cuda_remote_q51_roundtrip",
                    .perf_device = "cuda:0",
                });
            ASSERT_TRUE(lane->materialize(&error)) << error;

            const NativeQuantisedSourceIdentity source_identity{
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
            ASSERT_TRUE(cursor.requiresCPUStaging());
            // The original CPU root is gathered into one exclusive bounded
            // send lane. Retain it until beginChunk/pollChunk reports the
            // exact native copy complete before reusing it for the next chunk.
            const auto send_staging = TransferEngine::instance().allocatePinnedHostBuffer(
                promotion_manifest.maximum_chunk_bytes, DeviceId::cuda(0));
            ASSERT_TRUE(send_staging->isBound());
            ASSERT_EQ(send_staging->sizeBytes(), promotion_manifest.maximum_chunk_bytes);
            ASSERT_LT(send_staging->sizeBytes(), cpu_regions[0].size());
            ASSERT_LT(send_staging->sizeBytes(), promotion_manifest.total_bytes);
            const std::span<std::uint8_t> send_bytes{
                static_cast<std::uint8_t *>(send_staging->mutableData()), send_staging->sizeBytes()};
            const auto source_hash = expertTierWeightBytesHash(cpu_regions[0]);
            EXPECT_FALSE(cursor.takeNext().has_value());
            EXPECT_FALSE(cursor.complete()); // Missing staging cannot advance source ownership.
            std::uint64_t promotion_chunks = 0;
            std::uint64_t promotion_wire_bytes = 0;
            while (auto chunk = cursor.takeNext(send_bytes))
            {
                ASSERT_EQ(chunk->payload.data(), send_bytes.data());
                ASSERT_GT(chunk->payload.size(), 0u);
                ASSERT_LE(chunk->payload.size(), send_bytes.size());
                ASSERT_EQ(chunk->header.payload_bytes, chunk->payload.size());
                promotion_wire_bytes += chunk->payload.size();
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
            EXPECT_EQ(promotion_wire_bytes, promotion_manifest.total_bytes);
            EXPECT_EQ(expertTierWeightBytesHash(cpu_regions[0]), source_hash);
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
                native_quantised_formats::forSourceIdentity(
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
            auto cpu_destination_owner = std::make_shared<std::vector<std::uint8_t>>(
                cpu_weights.native_interleaved.size(), 0xa5u);
            auto &redemoted_cpu_bytes = *cpu_destination_owner;
            // Wire units are compact transfer order, not native interleaved
            // CPU allocation offsets. Exercise the production destination's
            // authenticated scatter into the original final representation.
            MoEOverlayHostRemoteProjectionDestination cpu_destination(
                demotion_identity,
                {std::span<std::uint8_t>(redemoted_cpu_bytes), {}, {}, {}},
                cpu_destination_owner,
                demotion_manifest);
            ASSERT_TRUE(cpu_destination.beginManifest(demotion_manifest, &error)) << error;
            std::uint64_t demotion_chunks = 0;
            std::uint64_t demotion_wire_bytes = 0;
            while (!cpu_destination.complete())
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
                ASSERT_EQ(cpu_destination.beginChunk(
                              chunk.header, chunk.payload, &error),
                          MoEOverlayResidencyWaveProgress::Ready) << error;
                demotion_wire_bytes += chunk.payload.size();
                ASSERT_TRUE(source.acknowledgeChunkSent(
                    chunk.header, &error))
                    << error;
                ++demotion_chunks;
            }
            EXPECT_EQ(demotion_wire_bytes, demotion_manifest.total_bytes);
            ASSERT_TRUE(cpu_destination.publishFinal(&error)) << error;
            EXPECT_EQ(redemoted_cpu_bytes.size(),
                      cpu_weights.native_interleaved.size());
            // Original packing coordinates, not observed transfer writes,
            // determine every expected byte, including untouched padding.
            std::vector<std::uint8_t> expected_cpu_bytes(redemoted_cpu_bytes.size(), 0xa5u);
            size_t live_bytes = 0;
            const auto geometry = cpu_weights.geometry();
            for (int block = 0; block < cpu_weights.blocks_per_row; ++block)
                for (int row = 0; row < N; ++row)
                {
                    using cpu::native_quantised::Q5NativePayload;
                    const auto unit = geometry.blockOffset(row / 64, block);
                    const auto column = row % 64;
                    const auto copy = [&](size_t offset, unsigned bytes) {
                        std::memcpy(expected_cpu_bytes.data() + unit + offset, cpu_weights.native_interleaved.data() + unit + offset, bytes);
                        live_bytes += bytes;
                    };
                    for (int group = 0; group < 4; ++group)
                    {
                        copy(Q5NativePayload::nibbleOffset(group, column), 4);
                        copy(Q5NativePayload::fifthOffset(group, column), 1);
                    }
                    copy(cpu_weights.data_stride + column * 2, 2);
                    copy(cpu_weights.data_stride + 64 * 2 + column * 2, 2);
                }
            EXPECT_EQ(demotion_wire_bytes, live_bytes);
            EXPECT_LT(live_bytes, expected_cpu_bytes.size());
            EXPECT_EQ(redemoted_cpu_bytes, expected_cpu_bytes);
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
            namespace sb = native_quantised_superblock;
            const auto *source_format = native_quantised_formats::forSourceIdentity(format.source_codebook_id, format.source_is_superblock);
            ASSERT_NE(source_format, nullptr);
            auto compact = test::makeGpuPreparedGemm(tensor.get(), DeviceId::cuda(0));
            const auto compact_view = sourceNativeTierView(*compact.kernel, *source_format);
            const auto physical = canonicalDeviceVnniFormat(*source_format);
            ASSERT_EQ(physical.codebook_id, 7);
            ASSERT_TRUE(physical.is_asymmetric);
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu_weights;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(), cpu_weights));
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
                        probe_layout.cpuStorage().maximumWireUnitBytes()));
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
                physical.codebook_id);
            const std::size_t blocks =
                static_cast<std::size_t>(N) * (K / 32);

            TestCUDABuffer<std::uint8_t> promoted_payload(
                blocks * layout.gpu_payload_bytes_per_block);
            const auto pad = compact_view.metadata_layout ? sb::kSuperblockCodePlanePadBytes : 0u;
            TestCUDABuffer<uint8_t> promoted_scales(compact_view.scales_bytes + pad), promoted_mins(compact_view.mins_bytes);
            ExpertTierGpuMutableProjectionView promoted_view{
                .payload = promoted_payload.data(), .scales = reinterpret_cast<uint16_t *>(promoted_scales.data()),
                .mins = reinterpret_cast<uint16_t *>(promoted_mins.data()), .payload_bytes = promoted_payload.bytes(),
                .scales_bytes = compact_view.scales_bytes, .mins_bytes = promoted_mins.bytes(), .metadata_layout = compact_view.metadata_layout};
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
                .execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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

            // Redemote the published destination, not the original SourceNative root.
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
                .metadata_layout = compact_view.metadata_layout,
                .vnni_bytes = promoted_payload.bytes(),
                .scales_bytes = compact_view.scales_bytes,
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
                promoted_view.scales,
                promoted_view.mins,
                nullptr,
                layout.gpu_codebook_id,
                static_cast<std::uint32_t>(K / 32),
                slot_lifetime,
                NativeQuantisedSourceIdentity{
                    .codebook_id = format.source_codebook_id,
                    .is_superblock = format.source_is_superblock,
                    .present = true,
                },
                NativeQuantisedReusableDeviceAllocationFormat{
                    .payload_bytes_per_block =
                        layout.gpu_payload_bytes_per_block,
                    .has_mins = layout.gpu_is_asymmetric != 0,
                    .has_emins = !compact_view.metadata_layout && layout.gpu_has_emins != 0,
                }, compact_view.metadata_layout);
            DeviceNativeQuantisedMatrixDesc published_descriptor;
            ASSERT_TRUE(promoted_engine->exportNativeQuantisedMatrixDesc(
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
                format.source_codebook_id);
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
                physical.codebook_id);
            const ExpertTierGpuConstProjectionView promoted_source_view{
                .payload = promoted_payload.data(),
                .scales = promoted_view.scales,
                .mins = promoted_view.mins,
                .emins = nullptr,
                .payload_bytes = promoted_payload.bytes(),
                .scales_bytes = compact_view.scales_bytes,
                .mins_bytes = promoted_mins.bytes(),
                .emins_bytes = 0, .metadata_layout = compact_view.metadata_layout,
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
                    NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
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
            ASSERT_TRUE(cudaNativeQuantisedPrefill_getWorkspacePlanWithPolicy(
                physical.codebook_id,
                format.source_codebook_id,
                prefill_rows,
                N,
                K,
                0,
                &compact_prefill_workspace_bytes,
                &planned_partitions));
            ASSERT_TRUE(cudaNativeQuantisedPrefill_getWorkspacePlanWithPolicy(
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
                    native_quantised_scale::NativeQuantisedMetadataPlanes metadata,
                    std::uint8_t codebook,
                    std::uint8_t arithmetic_policy_codebook,
                    float *output)
                {
                    if (rows == 1)
                    {
                        return cudaNativeQuantisedGemvTuned_fp32_withPolicy(
                            activation.data(), payload, metadata,
                            output, activation_scales.data(), N, K,
                            1.0f, 0.0f, nullptr, nullptr, codebook,
                            arithmetic_policy_codebook, 0,
                            stream.opaque(), gemv_context, nullptr);
                    }
                    if (rows == verifier_rows)
                    {
                        return cudaNativeQuantisedGemvTuned_small_m_fp32_withPolicy(
                            activation.data(), payload, metadata,
                            output, activation_scales.data(), rows, N, K,
                            1.0f, 0.0f, nullptr, nullptr, codebook,
                            arithmetic_policy_codebook, 0,
                            stream.opaque(), gemv_context, nullptr);
                    }
                    return cudaNativeQuantisedPrefill_fp32_withPolicy(
                        activation.data(), payload, metadata,
                        output, activation_scales.data(), activation_sums.data(),
                        rows, N, K, 1.0f, 0.0f, nullptr, nullptr, codebook,
                        arithmetic_policy_codebook, 0, stream.opaque(),
                        prefill_context);
                };
                ASSERT_TRUE(launch(
                    compact_view.payload,
                    native_quantised_scale::metadataPlanes(DeviceNativeQuantisedMatrixDesc{
                        .scales = compact_view.scales, .mins = compact_view.mins, .metadata_layout = compact_view.metadata_layout}),
                    physical.codebook_id,
                    format.source_codebook_id,
                    compact_output.data()));
                ASSERT_TRUE(launch(
                    published_descriptor.payload,
                    native_quantised_scale::metadataPlanes(published_descriptor),
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
                {"kernel.cuda_native_quantised_gemv_dispatch"});
            const auto prefill_records = PerfStatsCollector::snapshot(
                {"kernel.cuda_native_quantised_prefill_calls"});
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
                decode_records, 1, physical.codebook_id, physical.codebook_id));
            EXPECT_TRUE(has_policy_record(
                decode_records,
                1,
                layout.gpu_codebook_id,
                physical.codebook_id));
            EXPECT_TRUE(has_policy_record(
                decode_records,
                verifier_rows,
                layout.gpu_codebook_id,
                physical.codebook_id));
            EXPECT_TRUE(has_policy_record(
                prefill_records,
                prefill_rows,
                layout.gpu_codebook_id,
                physical.codebook_id));
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
         * `GpuExpertSlotPool`: each NativeQuantised projection reserves the largest
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

            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu_weights;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(
                tensor.get(), cpu_weights));
            ASSERT_TRUE(cpu_weights.superblockCoded());
            ASSERT_TRUE(cpu_weights.is_asymmetric);

            HostGpuExpertPackedProjection expected;
            std::string error;
            ASSERT_TRUE(cpuToGpuExpertPackedReference(
                cpu_weights, expected, &error))
                << error;
            ASSERT_EQ(
                expected.codebook_id,
                format.source_codebook_id);

            const NativeQuantisedSourceIdentity source_identity{
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
                    .payload_bytes_per_block = expected.payload_bytes_per_block,
                    .is_asymmetric = true,
                    .has_emins = false,
                    .codebook_id = expected.codebook_id,
                    .format = ExpertWeightFormat::nativeQuantised(
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
                    staging_bytes / probe_layout.cpuStorage().maximumWireUnitBytes()));
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
                (logical_blocks * native_quantised_superblock::superblockCodePlaneBits(expected.codebook_id) + 7) / 8;
            const std::size_t logical_mins_bytes = expected.superblocks.size();
            ASSERT_GE(slot.payload_bytes, logical_payload_bytes);
            ASSERT_GE(slot.scales_bytes, logical_scales_bytes);
            ASSERT_GE(slot.mins_bytes, logical_mins_bytes);

            ExpertTierGpuMutableProjectionView destination{
                .payload = slot.d_native_quantised_payload,
                .scales = static_cast<std::uint16_t *>(
                    slot.d_native_quantised_scales),
                .mins = static_cast<std::uint16_t *>(
                    slot.d_native_quantised_mins),
                .emins = static_cast<std::uint32_t *>(
                    slot.d_native_quantised_emins),
                // The production descriptor exposes exact logical regions
                // over the deliberately larger reusable slot allocation.
                .payload_bytes = logical_payload_bytes,
                .scales_bytes = logical_scales_bytes,
                .mins_bytes = logical_mins_bytes,
                .emins_bytes = 0u, .metadata_layout = slot.metadata_layout,
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
                .execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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
            std::vector<std::uint8_t> observed_scales(logical_scales_bytes);
            std::vector<std::uint8_t> observed_mins(logical_mins_bytes);
            ASSERT_EQ(
                cudaMemcpy(
                    observed_payload.data(),
                    slot.d_native_quantised_payload,
                    observed_payload.size(),
                    cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(
                    observed_scales.data(),
                    slot.d_native_quantised_scales,
                    observed_scales.size(),
                    cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(
                    observed_mins.data(),
                    slot.d_native_quantised_mins,
                    observed_mins.size(),
                    cudaMemcpyDeviceToHost),
                cudaSuccess);
            EXPECT_EQ(observed_payload, expected.payload);
            EXPECT_TRUE(std::equal(observed_scales.begin(), observed_scales.end(), expected.codes.begin()));
            EXPECT_EQ(observed_mins, expected.superblocks);
            EXPECT_EQ(pool->slotForExpert(123, candidate_epoch), 0);
            EXPECT_EQ(lane.stats().blocking_synchronizations, 0u);
            EXPECT_EQ(lane.stats().inference_stream_waits, 0u);
        }

        /**
         * @test Real NativeQuantised inference completes unchanged during migration.
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
            cpu::native_quantised::CPUNativeQuantisedPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(
                source, expected_cpu, &error)) << error;
            const auto *source_format =
                native_quantised_formats::forSourceIdentity(0, false);
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
            const auto source_view = makeConstView(payload, scales, mins, source.scaleLiveBytes());

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
                    NativeQuantisedGroupedDecodePolicy::maximum_k_partitions) *
                static_cast<std::size_t>(source.N));
            cudaGemvContext_bindWorkspace(
                gemv_context,
                gemv_partials.data(),
                gemv_partials.bytes());
            ASSERT_TRUE(cudaNativeQuantisedGemvTuned_fp32(
                activation.data(),
                payload.data(),
                ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes{scales.data(), nullptr, nullptr, 0},
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
                .execution = ::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
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

            ASSERT_TRUE(cudaNativeQuantisedGemvTuned_fp32(
                activation.data(),
                payload.data(),
                ::llaminar2::native_quantised_scale::NativeQuantisedMetadataPlanes{scales.data(), nullptr, nullptr, 0},
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
            cpu::native_quantised::CPUNativeQuantisedPackedWeights expected_cpu;
            std::string error;
            ASSERT_TRUE(gpuToCpuExpertPackedReference(
                source, expected_cpu, &error)) << error;
            const auto *source_format =
                native_quantised_formats::forSourceIdentity(0, false);
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
            const auto source_view = makeConstView(payload, scales, mins, source.scaleLiveBytes());

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

        /**
         * @brief Service handles preserve all source/physical identities and bindings.
         *
         * Real device storage is lifetime-pinned through the service handle,
         * but its workspace and stream must never replace serving bindings.
         * This tests ownership, not arithmetic on the uninitialized test bytes.
         */
        TEST_F(CUDAExpertTierWeightKernelsTest,
               PreparedServiceViewsIsolateAllQuantizedExecutionBindings)
        {
            const auto device = DeviceId::cuda(0);
            TestCUDAStream serving_stream;
            TestCUDAStream probe_stream;
            DeviceWorkspaceManager serving_workspace(device, 4096);
            DeviceWorkspaceManager probe_workspace(device, 4096);
            constexpr int n = 64, k = 256;
            constexpr size_t blocks = n * (k / 32);
            for (const auto &format : native_quantised_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(std::string(format.quant_type));
                const auto allocation = gpuPreparedWeightPoolAllocation(*format.metadata, GPUPreparedWeightPoolLayout::MigrationReusable);
                const auto book = canonicalDeviceVnniCodebookId(*format.metadata);
                const bool coded = gpuPreparedSuperblockCoded(*format.metadata, GPUPreparedWeightPoolLayout::MigrationReusable, device);
                const auto planes = coded ? NativeQuantisedMetadataPlaneBytes::coded(book, n, k / 32, 0)
                    : NativeQuantisedMetadataPlaneBytes::of(blocks, book, allocation.has_mins, allocation.has_emins);
                const auto primary = alignGPUWeightLoadAllocation(blocks * allocation.payload_bytes_per_block);
                const auto secondary = alignGPUWeightLoadAllocation(primary + planes.scales);
                const auto extended = alignGPUWeightLoadAllocation(secondary + planes.secondary);
                auto storage = std::make_shared<TestCUDABuffer<uint8_t>>(extended + planes.emins);
                auto *payload = storage->data();
                auto *scales = reinterpret_cast<uint16_t *>(planes.scales ? payload + primary : payload);
                auto *mins = planes.secondary ? reinterpret_cast<uint16_t *>(payload + secondary) : nullptr;
                auto *emins = planes.emins ? reinterpret_cast<uint32_t *>(payload + extended) : nullptr;
                const auto layout = coded ? nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded, 0, 0) : uint8_t{0};
                const NativeQuantisedSourceIdentity identity{
                    .codebook_id = format.metadata->codebook_id,
                    .is_superblock = format.metadata->is_superblock,
                    .present = true,
                };
                auto serving = std::make_shared<cuda::CUDAQuantisedGemmKernel>(
                    n, k, 0, payload, scales, mins, emins,
                    canonicalDeviceVnniCodebookId(identity), k / 32,
                    storage, identity,
                    allocation, layout);
                serving->setGPUStream(serving_stream.opaque());
                serving->bindWorkspace(&serving_workspace);
                auto probe = KernelFactory::createExpertServiceExecutionView(serving, device);
                ASSERT_NE(probe, nullptr);
                ASSERT_NE(probe.get(), serving.get());
                auto *probe_workspace_owner = dynamic_cast<IWorkspaceConsumer *>(probe.get());
                ASSERT_NE(probe_workspace_owner, nullptr);
                probe->setGPUStream(probe_stream.opaque());
                probe_workspace_owner->bindWorkspace(&probe_workspace);
                EXPECT_EQ(serving->getGPUStream(), serving_stream.opaque());
                EXPECT_EQ(serving->getWorkspace(), &serving_workspace);

                DeviceNativeQuantisedMatrixDesc expected, observed;
                ASSERT_TRUE(serving->exportNativeQuantisedMatrixDesc(expected));
                ASSERT_TRUE(probe->exportNativeQuantisedMatrixDesc(observed));
                EXPECT_EQ(observed.payload, expected.payload);
                EXPECT_EQ(observed.scales, expected.scales);
                EXPECT_EQ(observed.mins, expected.mins);
                EXPECT_EQ(observed.emins, expected.emins);
                EXPECT_EQ(observed.n, expected.n);
                EXPECT_EQ(observed.k, expected.k);
                EXPECT_EQ(observed.codebook_id, expected.codebook_id);
                EXPECT_EQ(observed.allocation_payload_bytes_per_block, allocation.payload_bytes_per_block);
                EXPECT_EQ(observed.metadata_layout, layout);
                NativeQuantisedSourceIdentity observed_identity;
                ASSERT_TRUE(probe->exportNativeQuantisedSourceIdentity(observed_identity));
                EXPECT_EQ(observed_identity, identity);
                EXPECT_THROW((void)KernelFactory::createExpertServiceExecutionView(
                    serving, DeviceId::rocm(0)), std::invalid_argument);
                EXPECT_THROW((void)KernelFactory::createExpertServiceExecutionView(
                    serving, DeviceId::cuda(1)), std::invalid_argument);

                probe_workspace_owner->unbindWorkspace();
                probe->clearGPUStreamBinding();
                EXPECT_EQ(serving->getGPUStream(), serving_stream.opaque());
                EXPECT_EQ(serving->getWorkspace(), &serving_workspace);
                std::weak_ptr<ITensorGemm> original_lifetime = serving;
                std::weak_ptr<TestCUDABuffer<uint8_t>> weight_lifetime = storage;
                serving.reset();
                storage.reset();
                EXPECT_FALSE(original_lifetime.expired());
                EXPECT_FALSE(weight_lifetime.expired());
                probe.reset();
                EXPECT_TRUE(original_lifetime.expired());
                EXPECT_TRUE(weight_lifetime.expired());
            }
        }

        /** @brief FP16/BF16/FP32 views retain bytes but not serving library/workspace state. */
        TEST_F(CUDAExpertTierWeightKernelsTest,
               PreparedServiceViewsIsolateAllFloatingExecutionBindings)
        {
            const auto device = DeviceId::cuda(0);
            TestCUDAStream serving_stream;
            TestCUDAStream probe_stream;
            DeviceWorkspaceManager serving_workspace(device, 4096);
            DeviceWorkspaceManager probe_workspace(device, 4096);
            using Kernel = cuda::CUDAFloatingPointGemmKernel;
            constexpr int n = 64, k = 256;
            for (const auto precision : {Kernel::Precision::FP16,
                                         Kernel::Precision::BF16,
                                         Kernel::Precision::FP32})
            {
                const size_t bytes = n * k * (precision == Kernel::Precision::FP32 ? 4u : 2u);
                auto storage = std::make_shared<TestCUDABuffer<uint8_t>>(bytes);
                auto serving = std::make_shared<Kernel>(
                    storage->data(), n, k, 0, precision, storage);
                serving->setGPUStream(serving_stream.opaque());
                serving->bindWorkspace(&serving_workspace);
                auto probe = KernelFactory::createExpertServiceExecutionView(serving, device);
                ASSERT_NE(probe, nullptr);
                ASSERT_NE(probe.get(), serving.get());
                auto *probe_workspace_owner = dynamic_cast<IWorkspaceConsumer *>(probe.get());
                ASSERT_NE(probe_workspace_owner, nullptr);
                probe->setGPUStream(probe_stream.opaque());
                probe_workspace_owner->bindWorkspace(&probe_workspace);
                ContiguousFloatingPointWeightDescriptor expected, observed;
                ASSERT_TRUE(serving->exportContiguousFloatingPointWeights(expected));
                ASSERT_TRUE(probe->exportContiguousFloatingPointWeights(observed));
                EXPECT_EQ(observed.data, expected.data);
                EXPECT_EQ(observed.bytes, bytes);
                EXPECT_EQ(observed.type, expected.type);
                EXPECT_EQ(observed.n, n);
                EXPECT_EQ(observed.k, k);
                probe_workspace_owner->unbindWorkspace();
                probe->clearGPUStreamBinding();
                EXPECT_EQ(serving->getWorkspace(), &serving_workspace);
                EXPECT_THROW((void)KernelFactory::createExpertServiceExecutionView(
                    serving, DeviceId::rocm(0)), std::invalid_argument);
                std::weak_ptr<ITensorGemm> original_lifetime = serving;
                std::weak_ptr<TestCUDABuffer<uint8_t>> weight_lifetime = storage;
                serving.reset();
                storage.reset();
                EXPECT_FALSE(original_lifetime.expired());
                EXPECT_FALSE(weight_lifetime.expired());
                probe.reset();
                EXPECT_TRUE(original_lifetime.expired());
                EXPECT_TRUE(weight_lifetime.expired());
            }
        }
        /**
         * @test Actual captured IQ3_S tier movement preserves every source grid/sign.
         *
         * All512 source indices and16 sign nibbles occur at every quartet lane,
         * with odd source coefficients, signed-zero/subnormal/max-finite bases,
         * nonzero CPU-unit origins and partial N/K source tails. Both directions
         * execute the production kernels in one retained transaction. The exact
         * original payload/metadata and CPU source pack remain separate oracles.
         */
        TEST_F(CUDAExpertTierWeightKernelsTest,
               IQ3SSourceTupleAllGridsAndSignsRoundTripCaptured)
        {
            constexpr int N = 512 * 16 + 1, K = 288, source_blocks = 2;
            constexpr std::uint32_t capacity_units = ((N + 63) / 64) * (K / 32);
            constexpr std::array<std::uint16_t,10> bases{
                0x3c01,0xbc01,0x0000,0x8000,0x0001,0x8001,0x03ff,0x0401,0x7bff,0xfbff};
            std::vector<std::uint8_t> raw(static_cast<std::size_t>(N) * source_blocks * sizeof(IQ3_SBlock));
            auto *original = reinterpret_cast<IQ3_SBlock *>(raw.data());
            for (int row = 0; row < N; ++row)
                for (int sb = 0; sb < source_blocks; ++sb)
                {
                    auto &block = original[row * source_blocks + sb];
                    block.d = bases[(row / 16 + sb) % bases.size()];
                    for (unsigned sub = 0; sub < 8; ++sub)
                    {
                        const unsigned nibble = (row + sub + sb) & 15u;
                        block.scales[sub / 2] |= static_cast<std::uint8_t>(nibble << (4 * (sub % 2)));
                        for (unsigned group = 0; group < 8; ++group)
                        {
                            const unsigned index = (row / 16 + group + sb) % 512;
                            const unsigned sign = (row + group) & 15u;
                            block.qs[sub * 8 + group] = static_cast<std::uint8_t>(index);
                            block.qh[sub] |= static_cast<std::uint8_t>((index >> 8) << group);
                            block.signs[sub * 4 + group / 2] |= static_cast<std::uint8_t>(sign << (4 * (group % 2)));
                        }
                    }
                }
            auto tensor = std::make_shared<IQ3_STensor>(std::vector<std::size_t>{N,K},raw);
            const auto source = packProductionGpuProjection(*tensor);
            ASSERT_EQ(source.codebook_id,11);ASSERT_EQ(source.source_codebook_id,11);
            ASSERT_EQ(source.payload_bytes_per_block,13);ASSERT_TRUE(source.is_asymmetric);
            // Literal source fields authenticate the preparation boundary too.
            for (int kb = 0; kb < K / 32; ++kb) for (int row = 0; row < N; ++row)
            {
                const auto &block = original[row * source_blocks + kb / 8];
                const unsigned sub = kb % 8;const auto at = static_cast<std::size_t>(kb) * N + row;
                ASSERT_EQ(source.scales[at],block.d);
                ASSERT_EQ(source.mins[at],(block.scales[sub / 2] >> (4 * (sub % 2))) & 15u);
                ASSERT_EQ(std::memcmp(source.payload.data() + at * 13,block.qs + sub * 8,8),0);
                ASSERT_EQ(source.payload[at * 13 + 8],block.qh[sub]);
                ASSERT_EQ(std::memcmp(source.payload.data() + at * 13 + 9,block.signs + sub * 4,4),0);
            }
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(),cpu));
            ASSERT_EQ(cpu.encoding,cpu::native_quantised::CPUNativeQuantisedEncoding::CompactMultiScale);
            constexpr std::uint32_t first_unit = 1;
            const auto *format = native_quantised_formats::forSourceIdentity(11,true);
            ASSERT_NE(format,nullptr);
            const auto probe = makeGpuToCpuExpertTierWeightStreamManifest(*format,N,K,405,3,7,ExpertTierWeightProjection::Down,capacity_units);
            const auto demotion = probe.deviceLayout();
            const auto promotion = makeCpuToGpuExpertTierWeightStreamManifest(cpu,406,3,7,ExpertTierWeightProjection::Down,capacity_units).deviceLayout();
            ASSERT_TRUE(demotion.valid());ASSERT_TRUE(promotion.valid());
            ASSERT_EQ(demotion.cpu_block_stride,cpu.geometry().block_stride);ASSERT_EQ(promotion.gpu_codebook_id,11);
            const std::uint32_t units = demotion.unit_count - first_unit;
            TestCUDAStream stream;
            namespace sb = native_quantised_superblock;
            auto loaded = test::makeGpuPreparedGemm(tensor.get(), DeviceId::cuda(0));
            const auto in = sourceNativeTierView(*loaded.kernel, *format);
            TestCUDABuffer<uint8_t> destination_payload(source.payload.size()),
                destination_scales(in.scales_bytes + sb::kSuperblockCodePlanePadBytes), destination_coeff(in.mins_bytes);
            TestCUDABuffer<uint8_t> chunk(demotion.chunkBytes(units,first_unit));
            ExpertTierGpuMutableProjectionView out{.payload = destination_payload.data(),
                .scales = reinterpret_cast<uint16_t *>(destination_scales.data()), .mins = reinterpret_cast<uint16_t *>(destination_coeff.data()),
                .payload_bytes = destination_payload.bytes(), .scales_bytes = in.scales_bytes,
                .mins_bytes = in.mins_bytes, .metadata_layout = in.metadata_layout};
            ASSERT_TRUE(in.validFor(demotion));ASSERT_TRUE(out.validFor(promotion));
            std::vector<std::uint8_t> observed_cpu(chunk.bytes()),observed_payload(source.payload.size());
            std::vector<uint8_t> observed_scales(destination_scales.bytes()), observed_coeff(destination_coeff.bytes());
            CUDAGraphCapture graph(stream.native(),0);
            struct Pending
            {
                cudaStream_t stream;
                /** @brief Join before graph, host vectors and device owners retire. */
                ~Pending() { EXPECT_EQ(cudaStreamSynchronize(stream),cudaSuccess); }
            } pending{stream.native()};
            {
                ScopedBackendGraphCapture capture(graph,"IQ3_S exact source tuple tier roundtrip");
                ASSERT_TRUE(capture.begin());
                ASSERT_TRUE(launchGpuToCpuExpertTierChunkCUDA(in,demotion,first_unit,units,chunk.data(),chunk.bytes(),stream.opaque()));
                ASSERT_TRUE(launchCpuToGpuExpertTierChunkCUDA(chunk.data(),chunk.bytes(),promotion,first_unit,units,out,stream.opaque()));
                capture.finish();
            }
            ASSERT_TRUE(graph.instantiate());
            for (int replay = 0; replay < 3; ++replay)
            {
                ASSERT_EQ(cudaMemsetAsync(chunk.data(),0xa5,chunk.bytes(),stream.native()),cudaSuccess);
                ASSERT_EQ(cudaMemsetAsync(destination_payload.data(),0xa5,destination_payload.bytes(),stream.native()),cudaSuccess);
                ASSERT_EQ(cudaMemsetAsync(destination_scales.data(),0xa5,destination_scales.bytes(),stream.native()),cudaSuccess);
                ASSERT_EQ(cudaMemsetAsync(destination_coeff.data(),0xa5,destination_coeff.bytes(),stream.native()),cudaSuccess);
                ASSERT_TRUE(graph.launch());
                ASSERT_EQ(cudaMemcpyAsync(observed_cpu.data(),chunk.data(),chunk.bytes(),cudaMemcpyDeviceToHost,stream.native()),cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(observed_payload.data(),destination_payload.data(),destination_payload.bytes(),cudaMemcpyDeviceToHost,stream.native()),cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(observed_scales.data(),destination_scales.data(),destination_scales.bytes(),cudaMemcpyDeviceToHost,stream.native()),cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(observed_coeff.data(),destination_coeff.data(),destination_coeff.bytes(),cudaMemcpyDeviceToHost,stream.native()),cudaSuccess);
                ASSERT_EQ(stream.synchronize(),cudaSuccess);
                EXPECT_EQ(std::memcmp(observed_cpu.data(),test::expertTierWireOracle(demotion,cpu.native_interleaved,first_unit,units).data(),chunk.bytes()),0) << "first_unit=" << first_unit;
                for (int kb = 0; kb < K / 32; ++kb) for (int row = 0; row < N; ++row)
                {
                    const auto linear = static_cast<std::size_t>(kb) * N + row;
                    const bool selected = static_cast<std::uint32_t>((row / 64) * (K / 32) + kb) >= first_unit;
                    EXPECT_EQ(sb::loadPackedBits(observed_scales.data(), linear * 4, 4), selected ? source.mins[linear] : (linear % 2 ? 0xa : 5));
                    if (kb % 8 == 0)
                    {
                        uint16_t base = 0;
                        std::memcpy(&base, observed_coeff.data() + (size_t(kb / 8) * N + row) * 2, 2);
                        EXPECT_EQ(base, source.scales[linear]); // Even a partial first superblock carries its original base.
                    }
                    for (unsigned byte = 0; byte < 13; ++byte)
                        EXPECT_EQ(observed_payload[linear * 13 + byte],selected ? source.payload[linear * 13 + byte] : 0xa5u);
                }
            }
        }

        /**
         * @test First CPU-origin IQ3 promotion preserves coded bytes without grid initialization.
         *
         * This is a standalone test, not the warmed IQ-table fixture. Its focused
         * CTest starts a fresh process and performs no GPU source packing/GEMV or
         * explicit table warmup. The real lane materialize method owns cold setup; migration does not decode the grid.
         * Expected payload words come directly from the original source block.
         */
        TEST(CUDAIQ3SColdTierSetup, CPUOriginPromotionPreservesSourceWithoutWarmup)
        {
            int devices=0;
            if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)GTEST_SKIP();
            ASSERT_EQ(cudaSetDevice(0),cudaSuccess);
            constexpr int N=1,K=256;
            IQ3_SBlock original{};original.d=0x3c01u;
            for(unsigned sub=0;sub<8;++sub)
            {
                original.scales[sub/2]|=std::uint8_t(((sub*3+1)&15u)<<(4*(sub%2)));
                for(unsigned group=0;group<8;++group)
                {
                    const unsigned index=(sub*67+group*31+511)%512;
                    original.qs[sub*8+group]=static_cast<std::uint8_t>(index);
                    original.qh[sub]|=static_cast<std::uint8_t>((index>>8)<<group);
                    original.signs[sub*4+group/2]|=static_cast<std::uint8_t>(((sub+group*3)&15u)<<(4*(group%2)));
                }
            }
            std::vector<std::uint8_t> bytes(sizeof(original));std::memcpy(bytes.data(),&original,sizeof(original));
            auto tensor=std::make_shared<IQ3_STensor>(std::vector<std::size_t>{N,K},bytes);
            cpu::native_quantised::CPUNativeQuantisedPackedWeights cpu;
            ASSERT_TRUE(cpu::native_quantised::packWeightsCPUNativeQuantised(tensor.get(),cpu));
            ASSERT_EQ(cpu.encoding,cpu::native_quantised::CPUNativeQuantisedEncoding::CompactMultiScale);
            const auto layout=makeCpuToGpuExpertTierWeightStreamManifest(cpu,407,3,7,ExpertTierWeightProjection::Down,2).deviceLayout();
            ASSERT_TRUE(layout.valid());ASSERT_EQ(layout.gpu_codebook_id,11);
            TestCUDABuffer<std::uint8_t> payload(8*13);
            namespace sb = native_quantised_superblock;
            TestCUDABuffer<uint8_t> scales(4 + sb::kSuperblockCodePlanePadBytes);
            TestCUDABuffer<uint16_t> base(1);
            ExpertTierGpuMutableProjectionView destination{
                .payload=payload.data(),.scales=reinterpret_cast<uint16_t *>(scales.data()),.mins=base.data(),
                .payload_bytes=payload.bytes(),.scales_bytes=4,.mins_bytes=base.bytes(),
                .metadata_layout=nativeQuantisedMetadataLayout(NativeQuantisedMetadataRepresentation::SuperblockCoded, 0, 0)};
            ASSERT_TRUE(destination.validFor(layout));
            const auto execution=::llaminar2::test::ComponentTransferExecutionPoolFixtureAccess::allocate(TransferEngine::instance(),
                1u,DeviceId::cuda(0),"cuda_iq3_source_cold_promotion").front();
            ExpertTierWeightTransferLane lane({
                .device=DeviceId::cuda(0),
                .staging=TransferEngine::instance().allocatePersistentTransferStagingSlices(
                    2 * layout.cpuStorage().maximumWireUnitBytes(),1u,DeviceId::cuda(0)).front(),
                .execution=execution,.progress=BackgroundTransferProgressBinding::nativeStream(),
                .lane_name="cuda_iq3_source_cold_promotion",.perf_device="cuda:0"});
            std::string error;
            ASSERT_TRUE(lane.materialize(&error))<<error;
            /** @brief Drain pending test work before lane/source/destination owners retire. */
            struct Retire
            {
                ExpertTierWeightTransferLane &lane;cudaStream_t stream;
                ~Retire()
                {
                    if(!lane.quiescent())
                    {
                        EXPECT_EQ(cudaStreamSynchronize(stream),cudaSuccess);
                        (void)pollLaneToCompletion(lane);
                    }
                }
            } retire{lane,reinterpret_cast<cudaStream_t>(execution.stream())};
            ASSERT_TRUE(lane.startCpuToGpu(layout,cpu.native_interleaved,destination,&error))<<error;
            ASSERT_EQ(pollLaneToCompletion(lane),ExpertTierWeightTransferProgress::Ready);
            ASSERT_TRUE(lane.quiescent());
            std::vector<std::uint8_t> actual_payload;
            std::vector<uint8_t> actual_codes;
            std::vector<uint16_t> actual_base;
            ASSERT_EQ(payload.download(actual_payload),cudaSuccess);
            ASSERT_EQ(scales.download(actual_codes),cudaSuccess);
            ASSERT_EQ(base.download(actual_base),cudaSuccess);
            for(unsigned sub=0;sub<8;++sub)
            {
                EXPECT_EQ(actual_base[0], original.d);
                EXPECT_EQ(sb::loadPackedBits(actual_codes.data(), sub * 4, 4), (original.scales[sub/2]>>(4*(sub%2)))&15u);
                EXPECT_EQ(std::memcmp(actual_payload.data()+sub*13,original.qs+sub*8,8),0);
                EXPECT_EQ(actual_payload[sub*13+8],original.qh[sub]);
                EXPECT_EQ(std::memcmp(actual_payload.data()+sub*13+9,original.signs+sub*4,4),0);
            }
        }
    } // namespace
} // namespace llaminar2

#include "CUDAExpertTierRectangularWeightContract.inc"
