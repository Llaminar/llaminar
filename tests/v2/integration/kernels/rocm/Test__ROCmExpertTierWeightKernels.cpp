/**
 * @file Test__ROCmExpertTierWeightKernels.cpp
 * @brief Byte-exact ROCm parity for direct ExpertOverlay tier conversion.
 *
 * Every cataloged quantized tensor is packed through the same source method
 * used by production ROCm weight preparation. The HIP demotion kernel converts
 * that projection in non-zero-origin chunks, and every resulting byte is
 * compared with the CPU NativeVNNI packer. The reverse half then streams those
 * CPU bytes back through the production promotion kernel and compares every
 * separated destination array with the device-free reference. IQ4_XS uses
 * physical24 with original half bases and raw encoded6 secondary words; its
 * source identity remains4 and symmetric arithmetic is unchanged.
 */

#include "../../../utils/NativeVNNITestPartialStorage.h"
#include "../../../utils/GpuPreparedGemmHarness.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorSlice.h"
#include "backends/BackendManager.h"
#include "backends/rocm/HIPGraphCapture.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/IQQuantTables.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "kernels/KernelFactory.h"
#include "kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h"
#include "execution/moe/ExpertTierWeightTransferLane.h"
#include "execution/moe/MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/common/NativeVNNIGroupedDecodePolicy.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemv.h"
#include "kernels/cpu/primitives/GPUAlignedExpertQ8Primitives.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmWeightPacker.h"
#include "kernels/rocm/gemm/ROCmNativeM1Reduction.h"
#include "kernels/rocm/gemm/ROCmNativeVNNISerialGeometry.h"
#include "kernels/rocm/repack/ROCmExpertTierWeightKernels.h"
#include "tensors/TensorKernels.h"
#include "tensors/VnniPackContext.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include "../../../utils/QuantizedVerifierFormats.h"
#include "../../../utils/ExpertTierFusedQKVStreamPoolHarness.h"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cctype>
#include <optional>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C"
{
    /** Frozen independent pre-grouping reducer, linked from this integration target only. */
    bool testOriginalNativeM1Reducer(const float *,float *,int,int,int,int,void *);
    /** Test-only exact-bit NaN/zero/infinity partial publication. */
    bool testInjectNativeM1RNPartials(float *,int,int,void *);

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
    using llaminar::v2::kernels::KernelFactory;
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
                canonicalDeviceVnniCodebookId(format);
            projection.payload_bytes_per_block =
                static_cast<std::uint8_t>(format.payload_bytes);
            projection.is_asymmetric = canonicalDeviceVnniFormat(format).is_asymmetric;
            projection.is_superblock = format.is_superblock;
            projection.has_emins = format.has_emins;

            const std::size_t blocks =
                static_cast<std::size_t>(N) * projection.blocks_per_row;
            projection.payload.resize(blocks * format.payload_bytes);
            projection.scales.resize(blocks);
            if (projection.is_asymmetric)
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
                projection.is_asymmetric ? projection.mins.data() : nullptr;
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
         * @test Captured grouped ordinary reductions preserve every original source format.
         *
         * Real packed producers retain their source policy and immutable inputs.
         * The independent oracle captures ordinary immediate producer/reducer pairs;
         * dirty reset nodes poison all live outputs/partials on every replay.
         * A mixed pair additionally retires an inactive original-work scope before
         * submitting its ticket beside an ordinary no-work producer. The retained
         * native reducer must preserve both modes after the host scope is gone.
         */
        TEST_F(ROCmExpertTierWeightKernelsTest,CapturedOrdinaryM1DeferredReductionsAllFormats)
        {
            using namespace rocm;
            /** @brief Original source geometry and optional sealed diagnostic column ownership. */
            struct Case
            {
                int k;
                std::array<int,2> widths;
                std::optional<NativeQ8SerialGeometry> geometry;
            };
            for(const auto &format:test::quantizedVerifierFormats())
            {
                std::vector<Case> cases={{256,{192,320},{}},{256,{7680,7936},{}},{256,{192,7680},{}},{2560,{192,320},{}}};
                if(std::strcmp(format.label,"Q8_0")==0)
                    for(const auto geometry:kNativeQ8SerialGeometries)
                    {
                        // K32 publishes directly. K640 retains six empty KB16
                        // planes; K2560 exercises a completely populated tree.
                        cases.push_back({32,{1,65},geometry});
                        cases.push_back({640,{127,257},geometry});
                        cases.push_back({2560,{65,129},geometry});
                    }
                for(const auto &test_case:cases)
                {
                const auto K=test_case.k;const auto widths=test_case.widths;
                const std::size_t guard=test_case.geometry?16u:0u;
                SCOPED_TRACE(::testing::Message()<<format.label<<" K="<<K<<" N="<<widths[0]<<","<<widths[1]
                    <<" geometry="<<(test_case.geometry?int(*test_case.geometry):-1));
                TestHIPStream stream;
                std::optional<NativeQ8SerialGeometryAdmission> admission;
                if(test_case.geometry)admission.emplace(admitNativeQ8SerialGeometry(*test_case.geometry,0));
                struct Prepared
                {
                    HostGpuExpertPackedProjection packed;
                    TestHIPBuffer<std::uint8_t> payload;
                    TestHIPBuffer<std::uint16_t> scales,mins;
                    TestHIPBuffer<std::uint32_t> emins;
                    TestHIPBuffer<float> output,partials,frozen;
                    /** @brief Retain exact original prepared planes and disjoint partial ownership. */
                    Prepared(HostGpuExpertPackedProjection source,int n,std::size_t guard):packed(std::move(source)),
                        payload(packed.payload.size()),scales(packed.scales.size()),mins(packed.mins.size()),
                        emins(packed.emins.size()),output(n+guard),partials(std::size_t(NativeVNNIGroupedDecodePolicy::maximum_k_partitions)*n+guard),frozen(n+guard)
                    {
                        if(payload.upload(packed.payload)!=hipSuccess || scales.upload(packed.scales)!=hipSuccess ||
                            mins.upload(packed.mins)!=hipSuccess || emins.upload(packed.emins)!=hipSuccess)
                            throw std::runtime_error("Deferred M1 fixture weight publication failed");
                    }
                };
                std::array<std::unique_ptr<Prepared>,2> prepared;
                for(std::size_t p=0;p<2;++p)
                {
                    auto tensor=format.create({std::size_t(widths[p]),std::size_t(K)},19351u+unsigned(p));
                    ASSERT_NE(tensor,nullptr);
                    prepared[p]=std::make_unique<Prepared>(packProductionGpuProjection(*tensor),widths[p],guard);
                }
                std::vector<std::int8_t> input(K);for(int i=0;i<K;++i)input[i]=std::int8_t((i*37)%255-127);
                std::vector<float> scale(K/32);for(std::size_t i=0;i<scale.size();++i)scale[i]=(i%2)?0.03125f:0.0625f;
                TestHIPBuffer<std::int8_t> activation(K);TestHIPBuffer<float> activation_scales(K/32);
                ASSERT_EQ(activation.upload(input),hipSuccess);
                ASSERT_EQ(activation_scales.upload(scale),hipSuccess);
                std::array<std::vector<float>,2> original_partials;
                // This original publication remains alive through every captured
                // replay. It is never copied into an informal host count shadow.
                TestHIPBuffer<std::int32_t> inactive_count(1);
                ASSERT_EQ(inactive_count.upload(std::vector<std::int32_t>{0}),hipSuccess);
                const auto inactive_work=ProjectionRowWork::fromPrefix(
                    DeviceRowRange::deviceCounted(1,inactive_count.data()));
                const auto observe=[&](bool grouped,bool witness=false,bool mixed_work=false,bool active_work=false)
                {
                    if(inactive_count.upload(std::vector<std::int32_t>{active_work?1:0})!=hipSuccess)
                        throw std::runtime_error("Deferred M1 work publication failed");
                    HIPGraphCapture graph(stream.native(),0);
                    struct Join
                    {
                        TestHIPStream &stream;
                        /** @brief Retire graph and all borrowed planes before exceptional cleanup. */
                        ~Join(){(void)stream.synchronize();}
                    } retirement{stream};
                    std::array<NativeM1ReductionTicket,2> tickets;
                    std::array<NativeQ8SerialLaunch,2> receipts;
                    std::optional<ScopedNativeQ8SerialGeometry> geometry_scope;
                    // Native admission and scope construction occur before capture.
                    // Retained replay must succeed after the scope has retired.
                    if(admission)geometry_scope.emplace(*admission);
                    ScopedBackendGraphCapture capture(graph,"all-format ordinary M1 deferred reduction");
                    if(!capture.begin())throw std::runtime_error("Deferred M1 fixture capture begin failed");
                    for(std::size_t p=0;p<2;++p)
                    {
                        auto &w=*prepared[p];
                        if(hipMemsetAsync(w.output.data(),0xff,w.output.bytes(),stream.native())!=hipSuccess ||
                           hipMemsetAsync(w.partials.data(),0xa5,w.partials.bytes(),stream.native())!=hipSuccess ||
                           hipMemsetAsync(w.frozen.data(),0xff,w.frozen.bytes(),stream.native())!=hipSuccess)
                            throw std::runtime_error("Deferred M1 fixture dirty history failed");
                        std::optional<ProjectionRowWorkScope> work_scope;
                        if(mixed_work && p==0)work_scope.emplace(inactive_work);
                        const bool ok=grouped?enqueueNativeM1Producer(tickets[p],activation.data(),w.payload.data(),
                            w.scales.data(),w.mins.data(),w.emins.data(),w.output.data(),nullptr,w.partials.data(),
                            widths[p],K,w.packed.codebook_id,canonicalDeviceVnniCodebookId(w.packed.source_codebook_id),0,stream.opaque(),activation_scales.data()):
                            rocmGemv_native_vnni_fp32_with_policy(activation.data(),w.payload.data(),w.scales.data(),
                            w.mins.data(),w.emins.data(),w.output.data(),nullptr,w.partials.data(),widths[p],K,
                            w.packed.codebook_id,canonicalDeviceVnniCodebookId(w.packed.source_codebook_id),0,stream.opaque(),activation_scales.data());
                        if(!ok)throw std::runtime_error("Deferred M1 fixture original producer failed");
                        // The ticket, not this retired TLS scope, owns reduction
                        // admission. Projection one is issued with no work scope.
                        work_scope.reset();
                        if(admission)
                        {
                            const auto actual=lastNativeQ8SerialLaunch();
                            if(!actual)throw std::runtime_error("Scoped M1 producer did not publish its actual launch receipt");
                            receipts[p]=*actual;
                            const auto spec=nativeQ8SerialGeometrySpec(admission->geometry());
                            EXPECT_EQ(actual->geometry,admission->geometry());
                            EXPECT_EQ(actual->physical_n,widths[p]);
                            EXPECT_EQ(actual->k,K);
                            EXPECT_EQ(actual->stream,stream.opaque());
                            EXPECT_EQ(actual->k_partitions,MoEProjectionNumericalContract::orderedKPartitionsForWidth(K));
                            EXPECT_EQ(actual->producer.function,admission->producerResources().function);
                            EXPECT_EQ(actual->producer.grid_x,1+(widths[p]-1)/spec.tile_columns);
                            EXPECT_EQ(actual->producer.grid_y,actual->k_partitions);
                            EXPECT_EQ(actual->producer.block_x,spec.block_threads);
                            if(grouped)
                            {
                                EXPECT_EQ(actual->reducer.function,nullptr);
                                EXPECT_EQ(tickets[p].producerShape().tile_columns,spec.tile_columns);
                                EXPECT_EQ(tickets[p].producerShape().columns_per_thread,spec.columns_per_thread);
                            }
                            else EXPECT_EQ(actual->reducer.function,actual->k_partitions>1?admission->reducerResources().function:nullptr);
                        }
                    }
                    if(grouped)
                    {
                        for(std::size_t p=0;p<2;++p)
                        {
                            const auto &shape=tickets[p].producerShape();
                            if(mixed_work && !active_work && p==0)continue; // Frozen dirty output is the independent no-write oracle.
                            if(shape.partitions>1)
                            {
                                if(witness && !testInjectNativeM1RNPartials(prepared[p]->partials.data(),widths[p],shape.partitions,stream.opaque()))
                                    throw std::runtime_error("M1 RN partial witness failed");
                                if(!testOriginalNativeM1Reducer(prepared[p]->partials.data(),prepared[p]->frozen.data(),
                                    widths[p],shape.partitions,shape.tile_columns,shape.columns_per_thread,stream.opaque()))
                                    throw std::runtime_error("M1 frozen original reducer oracle failed");
                            }
                            else if(hipMemcpyAsync(prepared[p]->frozen.data(),prepared[p]->output.data(),std::size_t(widths[p])*sizeof(float),
                                hipMemcpyDeviceToDevice,stream.native())!=hipSuccess)throw std::runtime_error("M1 direct oracle copy failed");
                        }
                        EXPECT_THROW(submitNativeM1Reductions(tickets,1,stream.opaque()),std::invalid_argument);
                        if(!submitNativeM1Reductions(tickets,0,stream.opaque()))
                            throw std::runtime_error("Deferred M1 fixture grouped reducer failed");
                        EXPECT_THROW(submitNativeM1Reductions(tickets,0,stream.opaque()),std::invalid_argument);
                    }
                    capture.finish();
                    geometry_scope.reset();
                    if(admission)
                    {
                        EXPECT_FALSE(currentNativeQ8SerialGeometry().has_value());
                        std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
                        if(!graph.inspectKernelNodes(nodes,&error))throw std::runtime_error("Scoped M1 native graph inventory failed: "+error);
                        const auto spec=nativeQ8SerialGeometrySpec(admission->geometry());
                        unsigned grouped_nodes=0;
                        for(const auto &node:nodes)
                        {
                            if(node.name.find("nativeM1GroupedReduce")==std::string::npos)continue;
                            ++grouped_nodes;
                            std::string compact=node.name;
                            compact.erase(std::remove_if(compact.begin(),compact.end(),[](unsigned char c){return std::isspace(c);}),compact.end());
                            const auto mangled="nativeM1GroupedReduceILi"+std::to_string(spec.tile_columns)+"ELi"+std::to_string(spec.columns_per_thread)+"ELb0EE";
                            const auto demangled="nativeM1GroupedReduce<"+std::to_string(spec.tile_columns)+","+std::to_string(spec.columns_per_thread)+",false>";
                            EXPECT_TRUE(compact.find(mangled)!=std::string::npos || compact.find(demangled)!=std::string::npos);
                            EXPECT_EQ(node.grid_x,unsigned(1+(widths[0]-1)/spec.tile_columns+1+(widths[1]-1)/spec.tile_columns));
                            EXPECT_EQ(node.grid_y,1u);
                            EXPECT_EQ(node.grid_z,1u);
                            EXPECT_EQ(node.block_x,unsigned(spec.block_threads));
                            EXPECT_EQ(node.block_y,1u);
                            EXPECT_EQ(node.block_z,1u);
                            EXPECT_EQ(node.local_memory_bytes_per_thread,0u);
                        }
                        EXPECT_EQ(grouped_nodes,grouped && receipts[0].k_partitions>1?1u:0u);
                    }
                    if(mixed_work)
                    {
                        EXPECT_EQ(currentProjectionRowWork(1),nullptr);
                        std::vector<GPUGraphKernelNodeInfo> nodes;std::string error;
                        if(!graph.inspectKernelNodes(nodes,&error))throw std::runtime_error("Mixed-work M1 native inventory failed: "+error);
                        unsigned work_reducers=0;
                        for(const auto &node:nodes)
                        {
                            if(node.name.find("nativeM1GroupedReduce")==std::string::npos)continue;
                            auto name=node.name;
                            name.erase(std::remove_if(name.begin(),name.end(),[](unsigned char c){return std::isspace(c);}),name.end());
                            EXPECT_TRUE(name.find("ELb1EE")!=std::string::npos || name.find(",true>")!=std::string::npos);
                            ++work_reducers;
                        }
                        EXPECT_GT(work_reducers,0u);
                    }
                    if(!graph.instantiate())throw std::runtime_error("Deferred M1 fixture graph instantiate failed");
                    std::array<std::vector<float>,2> first;
                    for(int replay=0;replay<3;++replay)
                    {
                        if(!graph.launch() || stream.synchronize()!=hipSuccess)
                            throw std::runtime_error("Deferred M1 fixture retained replay failed");
                        for(std::size_t p=0;p<2;++p)
                        {
                            std::vector<float> actual;
                            if(prepared[p]->output.download(actual)!=hipSuccess)throw std::runtime_error("Deferred M1 fixture terminal observation failed");
                            if(grouped)
                            {
                                std::vector<float> frozen;
                                if(prepared[p]->frozen.download(frozen)!=hipSuccess)throw std::runtime_error("M1 frozen original terminal observation failed");
                                EXPECT_EQ(std::memcmp(frozen.data(),actual.data(),actual.size()*sizeof(float)),0);
                            }
                            if(mixed_work && !active_work && p==0)
                            {
                                for(const auto value:actual)EXPECT_EQ(std::bit_cast<std::uint32_t>(value),0xffffffffu);
                                std::vector<float> partials;
                                if(prepared[p]->partials.download(partials)!=hipSuccess)throw std::runtime_error("Inactive ticket partial observation failed");
                                for(const auto value:partials)EXPECT_EQ(std::bit_cast<std::uint32_t>(value),0xa5a5a5a5u);
                            }
                            if(admission)
                            {
                                std::vector<float> partials;
                                if(prepared[p]->partials.download(partials)!=hipSuccess)throw std::runtime_error("Scoped M1 partial observation failed");
                                if(!grouped && replay==0)original_partials[p]=partials;
                                else if(!witness)EXPECT_EQ(std::memcmp(original_partials[p].data(),partials.data(),partials.size()*sizeof(float)),0);
                                const auto live=receipts[p].k_partitions>1?std::size_t(receipts[p].k_partitions)*widths[p]:0u;
                                for(std::size_t i=live;i<partials.size();++i)
                                    EXPECT_EQ(std::bit_cast<std::uint32_t>(partials[i]),0xa5a5a5a5u)<<"unused partial/guard="<<i;
                                for(std::size_t i=std::size_t(widths[p]);i<actual.size();++i)
                                    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual[i]),0xffffffffu)<<"output guard="<<i;
                                if(!witness && K==640)
                                {
                                    EXPECT_EQ(receipts[p].k_partitions,16);
                                    EXPECT_EQ(receipts[p].nonempty_partitions,10);
                                    for(std::size_t i=std::size_t(receipts[p].nonempty_partitions)*widths[p];i<live;++i)
                                        EXPECT_EQ(std::bit_cast<std::uint32_t>(partials[i]),0u)<<"empty KB plane="<<i;
                                }
                            }
                            if(replay==0)first[p]=std::move(actual);
                            else EXPECT_EQ(std::memcmp(first[p].data(),actual.data(),actual.size()*sizeof(float)),0);
                        }
                    }
                    return first;
                };
                const auto original=observe(false);const auto grouped=observe(true);
                for(std::size_t p=0;p<2;++p)
                {
                    ASSERT_EQ(original[p].size(),grouped[p].size());
                    EXPECT_EQ(std::memcmp(original[p].data(),grouped[p].data(),original[p].size()*sizeof(float)),0);
                }
                if(!admission && K==2560 && widths[0]==192 && widths[1]==320)
                {
                    const auto mixed=observe(true,false,true);
                    EXPECT_EQ(std::memcmp(original[1].data(),mixed[1].data(),original[1].size()*sizeof(float)),0);
                    const auto active=observe(true,false,true,true);
                    for(std::size_t p=0;p<2;++p)
                        EXPECT_EQ(std::memcmp(original[p].data(),active[p].data(),original[p].size()*sizeof(float)),0);
                }
                if((std::strcmp(format.label,"Q4_0")==0 && widths[0]==192 && widths[1]==320) ||
                    (admission && K==640))(void)observe(true,true);
                std::array<NativeM1ReductionTicket,2> absent;
                EXPECT_THROW(submitNativeM1Reductions(absent,0,stream.opaque()),std::invalid_argument);
                EXPECT_THROW(submitNativeM1Reductions(absent,0,nullptr),std::invalid_argument);
                }
            }
        }

        /** @test Real public fused M1 preserves bias/slice publication across reusable slot waves. */
        TEST_F(ROCmExpertTierWeightKernelsTest,CapturedOrdinaryM1ReductionSlotWavesAndBias)
        {
            constexpr int N=192,K=256;
            const auto device=DeviceId::rocm(0);
            ScopedPerfStats perf_stats;
            test::ScopedFusedQKVConcurrencySettings concurrency_settings;
            concurrency_settings.set(device,true,false);
            TestHIPStream stream;
            auto weight=test::TestTensorFactory::createQ8_0Random({N,K},98351u);
            auto prepared=test::makeGpuPreparedGemm(weight.get(),device,"test.native_m1_reduction_waves",ModelContextId{98351});
            auto *consumer=dynamic_cast<IWorkspaceConsumer *>(prepared.kernel);
            ASSERT_NE(consumer,nullptr);
            auto requirements=consumer->getWorkspaceRequirements(1,N,K);
            const std::array<int,8> widths={N,N,N,N,N,N,N,N};
            consumer->appendFusedProjectionWorkspaceRequirements(requirements,1,widths,K);
            const auto bytes=requirements.total_bytes_with_alignment();
            PhysicalMemoryPlanBuilder plan;
            plan.add({.world_rank=0,.device=device,.total_bytes=bytes,.admission_available_bytes=bytes},PhysicalMemoryOwner::ExecutionWorkspace,bytes);
            auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
            DeviceWorkspaceManager workspace(device,bytes,memory);
            ASSERT_TRUE(workspace.allocate(requirements));
            consumer->bindWorkspace(&workspace);prepared.kernel->setGPUStream(stream.opaque());
            ASSERT_TRUE(prepared.kernel->prepareFusedProjectionGraphCapture(8));
            FP32Tensor input({1,K});
            for(std::size_t i=0;i<input.numel();++i)input.mutable_data()[i]=float(int(i%29)-14)*0.03125f;
            auto bias=std::make_shared<FP32Tensor>(std::vector<std::size_t>{N});
            for(int i=0;i<N;++i)bias->mutable_data()[i]=float(i%7-3)*0.0078125f;
            TensorSlice bias_slice(bias,SliceMetadata{.mode=SliceMode::FULL,.original_rows=N,.original_cols=1,
                .slice_start=0,.slice_end=N,.inner_is_presliced=true});
            std::array<std::unique_ptr<FP32Tensor>,8> outputs,mapped_outputs;
            for(std::size_t i=0;i<outputs.size();++i)
            {
                outputs[i]=std::make_unique<FP32Tensor>(std::vector<std::size_t>{1,N});
                mapped_outputs[i]=FP32Tensor::createMapped(std::vector<std::size_t>{1,N},device);
                ASSERT_NE(mapped_outputs[i],nullptr);
            }
            struct Retirement
            {
                TestHIPStream &stream;IWorkspaceConsumer &consumer;ITensorGemm &kernel;
                /** @brief Retire all original tensor borrows before unbinding the persistent arena. */
                ~Retirement(){(void)stream.synchronize();consumer.unbindWorkspace();kernel.clearGPUStreamBinding();}
            } retirement{stream,*consumer,*prepared.kernel};
            // Establish terminal retirement before the first asynchronous publication.
            TransferEngine::prepareDeviceInput(&input,device,stream.opaque());
            TransferEngine::prepareDeviceInput(bias.get(),device,stream.opaque());
            for(auto &output:outputs)TransferEngine::prepareDeviceOutput(output.get(),device,stream.opaque());
            for(const bool mapped:{false,true})for(const std::size_t count:{2u,4u,5u,8u})
            {
                SCOPED_TRACE(::testing::Message()<<"logical projections="<<count<<" mapped="<<mapped);
                PerfStatsCollector::reset();
                std::vector<ITensorGemm::TensorProjectionDesc> projections;
                for(std::size_t i=0;i<count;++i)
                    projections.emplace_back(prepared.kernel,mapped?mapped_outputs[i].get():outputs[i].get(),N,i%2?&bias_slice:nullptr,"slot_wave");
                const auto observe=[&](bool grouped)
                {
                    HIPGraphCapture graph(stream.native(),0);
                    struct Join
                    {
                        TestHIPStream &stream;
                        /** @brief Retire retained graph before exceptional graph destruction. */
                        ~Join(){(void)stream.synchronize();}
                    } join{stream};
                    // Placement alone does not admit an external completion event
                    // into capture. Join the actual input/bias storage owners to
                    // this exact consumer stream before either oracle or group
                    // begins recording; TensorSlice resolves to its canonical inner.
                    TransferEngine::requireDeviceInput(&input,device,stream.opaque());
                    for(const auto &p:projections)if(p.bias)
                        TransferEngine::requireDeviceInput(const_cast<TensorBase *>(p.bias),device,stream.opaque());
                    ScopedBackendGraphCapture capture(graph,"ordinary M1 original/public group slot waves");
                    if(!capture.begin())throw std::runtime_error("M1 slot wave capture failed");
                    for(const auto &p:projections)
                        if(hipMemsetAsync(p.output->gpu_data_ptr(),0xff,std::size_t(N)*sizeof(float),stream.native())!=hipSuccess)
                            throw std::runtime_error("M1 slot wave output poison failed");
                    if(grouped)
                    {
                        if(!prepared.kernel->multiply_fused_tensor(&input,projections,1,K,nullptr,&workspace))
                            throw std::runtime_error("M1 public grouped slot wave failed");
                    }
                    else for(const auto &p:projections)
                        if(!prepared.kernel->multiply_tensor(&input,p.output,1,N,K,true,1.f,0.f,p.bias,nullptr,0,&workspace))
                            throw std::runtime_error("M1 original serial bias oracle failed");
                    capture.finish();
                    if(!graph.instantiate())throw std::runtime_error("M1 slot wave graph instantiation failed");
                    std::vector<std::vector<float>> first(count);
                    for(int replay=0;replay<3;++replay)
                    {
                        if(!graph.launch() || stream.synchronize()!=hipSuccess)throw std::runtime_error("M1 slot wave retained replay failed");
                        for(std::size_t i=0;i<count;++i)
                        {
                            std::vector<float> actual(N);
                            struct HostCopyJoin
                            {
                                TestHIPStream &stream;
                                /** @brief Retire a failed terminal copy before its host destination dies. */
                                ~HostCopyJoin(){(void)stream.synchronize();}
                            } host_copy_retirement{stream};
                            if(hipMemcpyAsync(actual.data(),projections[i].output->gpu_data_ptr(),N*sizeof(float),hipMemcpyDeviceToHost,stream.native())!=hipSuccess ||
                               stream.synchronize()!=hipSuccess)throw std::runtime_error("M1 slot wave terminal output copy failed");
                            if(replay==0)first[i]=std::move(actual);
                            else EXPECT_EQ(std::memcmp(first[i].data(),actual.data(),N*sizeof(float)),0);
                        }
                    }
                    return first;
                };
                const auto original=observe(false);const auto grouped=observe(true);
                for(std::size_t i=0;i<count;++i)EXPECT_EQ(std::memcmp(original[i].data(),grouped[i].data(),N*sizeof(float)),0);
                bool grouped_route=false;
                for(const auto &record:PerfStatsCollector::snapshot({"kernel"}))
                    grouped_route|=record.name=="rocm_native_m1_grouped_reduction_calls" && record.value>0;
                EXPECT_EQ(grouped_route,!mapped);
                if(!mapped)
                {
                    auto malformed=projections;malformed[1].n=N+1;
                    EXPECT_THROW(prepared.kernel->multiply_fused_tensor(&input,malformed,1,K,nullptr,&workspace),std::invalid_argument);
                    EXPECT_THROW(prepared.kernel->multiply_fused_tensor(&input,projections,1,K+32,nullptr,&workspace),std::invalid_argument);
                    projections[1].output=projections[0].output;
                    EXPECT_THROW(prepared.kernel->multiply_fused_tensor(&input,projections,1,K,nullptr,&workspace),std::invalid_argument);
                }
            }
        }

        /**
         * @test TransferEngine exposes exactly the requested reusable HIP queues.
         *
         * The checked stream pointers are authoritative; PerfStats only makes
         * the production selection observable to a campaign artifact.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            PersistentTransferExecutionPoolIsBoundedAndIdempotent)
        {
            constexpr std::size_t lane_count = 4u;
            const DeviceId device = DeviceId::rocm(0);
            const std::string pool_name =
                "rocm_expert_tier_execution_pool_contract";

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
         * @test Independent HIP lane events remain correct on one pooled stream.
         *
         * Concurrent host submissions share one exact background queue while
         * retaining disjoint staging and completion state. Interleaved polling
         * must preserve every byte without a blocking synchronization.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            SharedPersistentExecutionLanePreservesConcurrentTransfersByteExactly)
        {
            constexpr std::size_t bytes = 1u << 20u;
            constexpr std::size_t staging_bytes = 4096u;
            constexpr int n = 1;
            constexpr int k =
                static_cast<int>(bytes / sizeof(std::uint16_t));
            const DeviceId device = DeviceId::rocm(0);

            const auto staging = TransferEngine::instance()
                                     .allocatePersistentTransferStagingSlices(
                                         staging_bytes,
                                         2u,
                                         device);
            const auto execution = TransferEngine::instance()
                                       .allocatePersistentTransferExecutionLanes(
                                           1u,
                                           device,
                                           "rocm_shared_execution_race_contract")
                                       .front();
            ExpertTierWeightTransferLane first({
                .device = device,
                .staging = staging[0],
                .execution = execution,
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "rocm_shared_execution_first",
                .perf_device = "rocm:0",
            });
            ExpertTierWeightTransferLane second({
                .device = device,
                .staging = staging[1],
                .execution = execution,
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "rocm_shared_execution_second",
                .perf_device = "rocm:0",
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
            TestHIPBuffer<std::uint8_t> first_destination(bytes);
            TestHIPBuffer<std::uint8_t> second_destination(bytes);
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
                first_destination.download(first_observed), hipSuccess);
            ASSERT_EQ(
                second_destination.download(second_observed), hipSuccess);
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
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   demotion_layout.chunkBytes(2),
                                   1u,
                                   DeviceId::rocm(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::rocm(0),
                                     "rocm_tier_round_trip")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
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
         * @test Qwen-122B Q8_0 projections are byte-exact on CPU and ROCm.
         *
         * Tier movement changes the physical executor without changing the
         * logical expert.  Exercise the production gate/up and down geometries
         * with one shared Q8 activation publication and require every FP32
         * output word to match.  Transfer-only and backend-local tests cannot
         * detect a disagreement in the cross-backend reduction tree.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            Q80ProductionProjectionCPUAndROCmAreByteExact)
        {
            const auto &format = test::quantizedVerifierFormat("Q8_0");
            constexpr std::array<std::pair<int, int>, 2> geometries{{
                {3072, 1024},
                {1024, 3072},
            }};

            ScopedROCmDecodeEquivalentM1 decode_equivalent_scope;
            for (std::size_t geometry_index = 0;
                 geometry_index < geometries.size();
                 ++geometry_index)
            {
                const auto [N, K] = geometries[geometry_index];
                SCOPED_TRACE(
                    "N=" + std::to_string(N) + " K=" +
                    std::to_string(K));
                auto tensor = format.create(
                    {static_cast<std::size_t>(N),
                     static_cast<std::size_t>(K)},
                    static_cast<std::uint32_t>(0x122b000u + geometry_index));
                ASSERT_NE(tensor, nullptr);

                cpu::native_vnni::CPUNativeVNNIGemmKernel cpu_kernel(
                    tensor.get(),
                    0,
                    -1,
                    CPUProjectionNumericalPolicy::GPUAlignedExpert);
                ASSERT_TRUE(cpu_kernel.isValid());
                const auto &cpu_packed = cpu_kernel.packedWeights();
                const HostGpuExpertPackedProjection gpu_packed =
                    packProductionGpuProjection(*tensor);
                ASSERT_EQ(gpu_packed.source_codebook_id, 19u);
                ASSERT_EQ(gpu_packed.codebook_id, 19u);

                std::vector<float> input(static_cast<std::size_t>(K));
                for (std::size_t index = 0; index < input.size(); ++index)
                {
                    const int centered = static_cast<int>(
                        (index * 73u + index / 7u +
                         geometry_index * 29u) % 509u) - 254;
                    input[index] =
                        static_cast<float>(centered) * 0.00390625f +
                        static_cast<float>(
                            static_cast<int>(index % 11u) - 5) *
                            0.000013f;
                }

                const int blocks_per_row = K / 32;
                std::vector<Q8_1Block> activation_blocks(
                    static_cast<std::size_t>(blocks_per_row));
                std::vector<std::int8_t> activation_q8(
                    static_cast<std::size_t>(K));
                std::vector<float> activation_scales(
                    static_cast<std::size_t>(blocks_per_row));
                for (int block = 0; block < blocks_per_row; ++block)
                {
                    auto &published = activation_blocks[
                        static_cast<std::size_t>(block)];
                    cpu::gpu_aligned_expert_q8::quantizeBlock(
                        input.data() + static_cast<std::size_t>(block) * 32u,
                        published);
                    std::memcpy(
                        activation_q8.data() +
                            static_cast<std::size_t>(block) * 32u,
                        published.qs,
                        32u);
                    activation_scales[static_cast<std::size_t>(block)] =
                        cpu::native_vnni::nativeVNNIFP16ScaleToFP32(
                            published.d);
                }

                std::vector<float> cpu_output(
                    static_cast<std::size_t>(N), 0.0f);
                cpu::native_vnni::gemv_native_vnni_preq(
                    cpu_packed,
                    activation_blocks.data(),
                    cpu_output.data(), llaminar2::test::NativeVNNITestPartialStorage(cpu_packed, 1).span(),
                    cpu::native_vnni::ISAPath::AUTO);

                TestHIPBuffer<std::uint8_t> payload(
                    gpu_packed.payload.size());
                TestHIPBuffer<std::uint16_t> scales(
                    gpu_packed.scales.size());
                TestHIPBuffer<std::int8_t> device_activation(
                    activation_q8.size());
                TestHIPBuffer<float> device_activation_scales(
                    activation_scales.size());
                TestHIPBuffer<float> device_output(
                    static_cast<std::size_t>(N));
                TestHIPBuffer<float> partials(
                    static_cast<std::size_t>(
                        NativeVNNIGroupedDecodePolicy::maximum_k_partitions) *
                    static_cast<std::size_t>(N));
                ASSERT_EQ(payload.upload(gpu_packed.payload), hipSuccess);
                ASSERT_EQ(scales.upload(gpu_packed.scales), hipSuccess);
                ASSERT_EQ(device_activation.upload(activation_q8), hipSuccess);
                ASSERT_EQ(
                    device_activation_scales.upload(activation_scales),
                    hipSuccess);

                TestHIPStream stream;
                ASSERT_TRUE(rocmGemv_native_vnni_fp32_with_policy(
                    device_activation.data(),
                    payload.data(),
                    scales.data(),
                    nullptr,
                    nullptr,
                    device_output.data(),
                    nullptr,
                    partials.data(),
                    N,
                    K,
                    gpu_packed.codebook_id,
                    gpu_packed.source_codebook_id,
                    0,
                    stream.opaque(),
                    device_activation_scales.data()));
                ASSERT_EQ(stream.synchronize(), hipSuccess);

                std::vector<float> rocm_output;
                ASSERT_EQ(device_output.download(rocm_output), hipSuccess);
                ASSERT_EQ(rocm_output.size(), cpu_output.size());
                if (std::memcmp(
                        rocm_output.data(),
                        cpu_output.data(),
                        cpu_output.size() * sizeof(float)) != 0)
                {
                    std::size_t first = 0;
                    while (first < cpu_output.size() &&
                           std::memcmp(
                               &rocm_output[first],
                               &cpu_output[first],
                               sizeof(float)) == 0)
                    {
                        ++first;
                    }
                    ASSERT_LT(first, cpu_output.size());
                    ADD_FAILURE()
                        << "Cross-backend Q8_0 projection differs at column "
                        << first << " CPU=" << cpu_output[first]
                        << " ROCm=" << rocm_output[first];
                }
            }
        }

        /**
         * @test A shared maintenance caller preserves its device while the owning
         *       ROCm worker performs the remote projection.
         *
         * The source arrays and auxiliary stream belong to device one, while
         * the calling thread is deliberately switched back to device zero
         * immediately before submission. This is the production shape when a
         * rank progresses several GPU participants. The remote lane must submit
         * all HIP work to its device-one worker: byte-exact output proves that
         * the real repack used the source device, the worker probe proves that
         * its HIP context remains device one, and the caller probe proves that
         * maintenance cannot mutate unrelated thread-local HIP state.
         */
        TEST_F(
            ROCmExpertTierWeightKernelsTest,
            SecondaryDeviceRemoteProjectionRestoresOwningDeviceBeforeRepack)
        {
            int device_count = 0;
            ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
            if (device_count < 2)
                GTEST_SKIP() << "Secondary-device projection requires two ROCm devices";

            constexpr int device_ordinal = 1;
            constexpr int N = 70;
            constexpr int K = 96;
            constexpr std::uint32_t units_per_chunk = 2;
            ASSERT_EQ(hipSetDevice(device_ordinal), hipSuccess);

            const auto &format = test::quantizedVerifierFormats().at(18);
            ASSERT_STREQ(format.label, "Q8_0");
            auto tensor = format.create({N, K}, 0x51ec0d1u);
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
            ASSERT_TRUE(source_view.validFor(layout));

            auto lane = std::make_shared<MoEOverlayGpuRemoteProjectionLane>(
                MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = DeviceId::rocm(device_ordinal),
                    .staging = TransferEngine::instance()
                                   .allocatePersistentTransferStagingSlices(
                                       layout.chunkBytes(units_per_chunk),
                                       1u,
                                       DeviceId::rocm(device_ordinal))
                                   .front(),
                    .execution = TransferEngine::instance()
                                     .allocatePersistentTransferExecutionLanes(
                                         1u,
                                         DeviceId::rocm(device_ordinal),
                                         "rocm_secondary_device_remote_projection")
                                     .front(),
                    .progress = BackgroundTransferProgressBinding::nativeStream(),
                    .lane_name = "rocm_secondary_device_remote_projection",
                    .perf_device = "rocm:1",
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
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
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
            /* HIP device selection is thread-local. Device work now belongs to
             * the lane's worker, so the maintenance caller must stay on device
             * zero instead of being changed as an accidental side effect. */
            int caller_device = -1;
            ASSERT_EQ(hipGetDevice(&caller_device), hipSuccess);
            EXPECT_EQ(caller_device, 0);

            /* Query on the exact owner after the completed lane transaction.
             * The byte comparison above makes this an end-to-end proof rather
             * than merely checking a context constructor's initial device. */
            int worker_device = -1;
            hipError_t worker_status = hipSuccess;
            auto &worker = GPUDeviceContextPool::instance().getContext(
                DeviceId::rocm(device_ordinal));
            worker.submitAndWait([&]
            {
                worker_status = hipGetDevice(&worker_device);
            });
            ASSERT_EQ(worker_status, hipSuccess);
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
            ROCmExpertTierWeightKernelsTest,
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

            IBackend *backend = getROCmBackend();
            ASSERT_NE(backend, nullptr);
            ExpertTierWeightTransferLane lane({
                .device = DeviceId::rocm(0),
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   staging_bytes,
                                   1u,
                                   DeviceId::rocm(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::rocm(0),
                                     "rocm_floating_tier_round_trip")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
                .lane_name = "rocm_floating_tier_round_trip",
                .perf_device = "rocm:0",
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

                TestHIPBuffer<std::uint8_t> gpu_source(bytes);
                TestHIPStream producer_stream;
                ASSERT_EQ(
                    hipMemcpyAsync(
                        gpu_source.data(),
                        source.data(),
                        bytes,
                        hipMemcpyHostToDevice,
                        producer_stream.native()),
                    hipSuccess);
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

                TestHIPBuffer<std::uint8_t> gpu_destination(bytes);
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
                    gpu_destination.download(observed_gpu), hipSuccess);
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
                .execution_fingerprint = {
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
                    .staging = TransferEngine::instance()
                                   .allocatePersistentTransferStagingSlices(
                                       promotion_manifest.maximum_chunk_bytes,
                                       1u,
                                       DeviceId::rocm(0))
                                   .front(),
                    .execution = TransferEngine::instance()
                                     .allocatePersistentTransferExecutionLanes(
                                         1u,
                                         DeviceId::rocm(0),
                                         "rocm_remote_q51_roundtrip")
                                     .front(),
                    .progress = BackgroundTransferProgressBinding::nativeStream(),
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
                .execution_fingerprint = {
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
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   layout.chunkBytes(1),
                                   1u,
                                   DeviceId::rocm(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::rocm(0),
                                     "rocm_asymmetric_promotion_execution")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
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
            constexpr std::size_t activation_blocks_per_row =
                static_cast<std::size_t>(K) / 32u;
            constexpr std::size_t activation_block_elements =
                static_cast<std::size_t>(prefill_rows) *
                activation_blocks_per_row;
            std::vector<std::int8_t> host_activation(activation_elements);
            for (std::size_t index = 0; index < activation_elements; ++index)
            {
                host_activation[index] = static_cast<std::int8_t>(
                    static_cast<int>((index * 11u + 3u) % 29u) - 14);
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
            TestHIPBuffer<std::int8_t> activation(activation_elements);
            TestHIPBuffer<float> activation_scales(
                activation_block_elements);
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
            constexpr std::uint8_t normalized_default_policy =
                native_vnni_formats::Q8_0.codebook_id;
            ASSERT_NE(
                compact.source_codebook_id,
                normalized_default_policy)
                << "The witness must distinguish source-policy provenance";
            ASSERT_TRUE(
                rocmGemv_native_vnni_query_serial_m1_config_with_policy(
                    compact.source_codebook_id,
                    N,
                    K,
                    &source_kb,
                    &ignored_waves));
            ASSERT_TRUE(
                rocmGemv_native_vnni_query_serial_m1_config_with_policy(
                    normalized_default_policy,
                    N,
                    K,
                    &normalized_kb,
                    &ignored_waves));
            const int contract_kb =
                MoEProjectionNumericalContract::orderedKPartitionsForWidth(K);
            ASSERT_GT(contract_kb, 0);
            EXPECT_EQ(source_kb, contract_kb)
                << "The compact source must use the cross-backend reduction tree";
            EXPECT_EQ(normalized_kb, contract_kb)
                << "Expanded storage must not invent a placement-specific tree";

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
                        // Include migration-only physical23 with its original source
                        // policy in the deferred ordinary publication coverage.
                        rocm::NativeM1ReductionTicket ticket;
                        if(!rocm::enqueueNativeM1Producer(ticket,activation.data(),payload,scales,mins,nullptr,
                            output,nullptr,partials.data(),N,K,codebook,arithmetic_policy_codebook,0,
                            stream.opaque(),activation_scales.data()))return false;
                        return rocm::submitNativeM1Reductions(std::span(&ticket,1),0,stream.opaque());
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
                    compact.source_codebook_id,
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
                small_m_records,
                1,
                compact.codebook_id,
                compact.source_codebook_id));
            EXPECT_TRUE(has_policy_record(
                small_m_records,
                1,
                layout.gpu_codebook_id,
                compact.source_codebook_id));
            EXPECT_TRUE(has_policy_record(
                small_m_records,
                verifier_rows,
                layout.gpu_codebook_id,
                compact.source_codebook_id));
            EXPECT_TRUE(has_policy_record(
                prefill_records,
                prefill_rows,
                layout.gpu_codebook_id,
                compact.source_codebook_id));
            EXPECT_FALSE(has_policy_record(
                small_m_records,
                1,
                layout.gpu_codebook_id,
                normalized_default_policy));
            EXPECT_FALSE(has_policy_record(
                prefill_records,
                prefill_rows,
                layout.gpu_codebook_id,
                normalized_default_policy));

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
                .staging = TransferEngine::instance()
                               .allocatePersistentTransferStagingSlices(
                                   layout.chunkBytes(1),
                                   1u,
                                   DeviceId::rocm(0))
                               .front(),
                .execution = TransferEngine::instance()
                                 .allocatePersistentTransferExecutionLanes(
                                     1u,
                                     DeviceId::rocm(0),
                                     "rocm_inference_overlap")
                                 .front(),
                .progress = BackgroundTransferProgressBinding::nativeStream(),
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

        /**
         * @brief Service handles preserve all source/physical identities and bindings.
         *
         * Real device storage is lifetime-pinned through the service handle,
         * but its workspace and stream must never replace serving bindings.
         * This tests ownership, not arithmetic on the uninitialized test bytes.
         */
        TEST_F(ROCmExpertTierWeightKernelsTest,
               PreparedServiceViewsIsolateAllQuantizedExecutionBindings)
        {
            const auto device = DeviceId::rocm(0);
            TestHIPStream serving_stream;
            TestHIPStream probe_stream;
            DeviceWorkspaceManager serving_workspace(device, 4096);
            DeviceWorkspaceManager probe_workspace(device, 4096);
            constexpr int n = 64, k = 256;
            constexpr size_t blocks = n * (k / 32);
            for (const auto &format : native_vnni_formats::kAllSourceFormats)
            {
                SCOPED_TRACE(std::string(format.quant_type));
                auto storage = std::make_shared<TestHIPBuffer<uint8_t>>(blocks * 40);
                auto *payload = storage->data();
                auto *scales = reinterpret_cast<uint16_t *>(payload + blocks * 32);
                auto *mins = scales + blocks;
                auto *emins = reinterpret_cast<uint32_t *>(mins + blocks);
                const NativeVnniSourceIdentity identity{
                    .codebook_id = format.metadata->codebook_id,
                    .is_superblock = format.metadata->is_superblock,
                    .present = true,
                };
                auto serving = std::make_shared<rocm::ROCmQuantisedGemmKernel>(
                    n, k, 0, payload, scales, mins, emins,
                    canonicalDeviceVnniCodebookId(identity), k / 32,
                    storage, identity,
                    NativeVnniReusableDeviceAllocationFormat{32, true, true});
                serving->setGPUStream(serving_stream.opaque());
                serving->bindWorkspace(&serving_workspace);
                auto probe = KernelFactory::createExpertServiceExecutionView(serving, device);
                ASSERT_NE(probe, nullptr);
                ASSERT_NE(probe.get(), serving.get());
                auto *probe_workspace_owner = dynamic_cast<IWorkspaceConsumer *>(probe.get());
                ASSERT_NE(probe_workspace_owner, nullptr);
                probe->setGPUStream(probe_stream.opaque());
                probe_workspace_owner->bindWorkspace(&probe_workspace);
                EXPECT_EQ(serving->requireGPUStream(), serving_stream.opaque());
                EXPECT_EQ(serving->getWorkspace(), &serving_workspace);

                DeviceNativeVNNIMatrixDesc expected, observed;
                ASSERT_TRUE(serving->exportNativeVNNIMatrixDesc(expected));
                ASSERT_TRUE(probe->exportNativeVNNIMatrixDesc(observed));
                EXPECT_EQ(observed.payload, expected.payload);
                EXPECT_EQ(observed.scales, expected.scales);
                EXPECT_EQ(observed.mins, expected.mins);
                EXPECT_EQ(observed.emins, expected.emins);
                EXPECT_EQ(observed.n, expected.n);
                EXPECT_EQ(observed.k, expected.k);
                EXPECT_EQ(observed.codebook_id, expected.codebook_id);
                EXPECT_EQ(observed.allocation_payload_bytes_per_block, 32);
                NativeVnniSourceIdentity observed_identity;
                ASSERT_TRUE(probe->exportNativeVNNISourceIdentity(observed_identity));
                EXPECT_EQ(observed_identity, identity);
                EXPECT_THROW((void)KernelFactory::createExpertServiceExecutionView(
                    serving, DeviceId::cuda(0)), std::invalid_argument);
                EXPECT_THROW((void)KernelFactory::createExpertServiceExecutionView(
                    serving, DeviceId::rocm(1)), std::invalid_argument);

                probe_workspace_owner->unbindWorkspace();
                probe->clearGPUStreamBinding();
                EXPECT_EQ(serving->requireGPUStream(), serving_stream.opaque());
                EXPECT_EQ(serving->getWorkspace(), &serving_workspace);
                std::weak_ptr<ITensorGemm> original_lifetime = serving;
                std::weak_ptr<TestHIPBuffer<uint8_t>> weight_lifetime = storage;
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
        TEST_F(ROCmExpertTierWeightKernelsTest,
               PreparedServiceViewsIsolateAllFloatingExecutionBindings)
        {
            const auto device = DeviceId::rocm(0);
            TestHIPStream serving_stream;
            TestHIPStream probe_stream;
            DeviceWorkspaceManager serving_workspace(device, 4096);
            DeviceWorkspaceManager probe_workspace(device, 4096);
            using Kernel = rocm::ROCmFloatingPointGemmKernel;
            constexpr int n = 64, k = 256;
            for (const auto precision : {Kernel::Precision::FP16,
                                         Kernel::Precision::BF16,
                                         Kernel::Precision::FP32})
            {
                const size_t bytes = n * k * (precision == Kernel::Precision::FP32 ? 4u : 2u);
                auto storage = std::make_shared<TestHIPBuffer<uint8_t>>(bytes);
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
                    serving, DeviceId::cuda(0)), std::invalid_argument);
                std::weak_ptr<ITensorGemm> original_lifetime = serving;
                std::weak_ptr<TestHIPBuffer<uint8_t>> weight_lifetime = storage;
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
        TEST_F(ROCmExpertTierWeightKernelsTest,
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
            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(tensor.get(),cpu));
            ASSERT_EQ(cpu.encoding,cpu::native_vnni::CPUNativeVNNIEncoding::IQ3SExpandedSourceScale);
            constexpr std::uint32_t first_unit = 1;
            const auto *format = native_vnni_formats::forSourceIdentity(11,true);
            ASSERT_NE(format,nullptr);
            const auto probe = makeGpuToCpuExpertTierWeightStreamManifest(*format,N,K,405,3,7,ExpertTierWeightProjection::Down,capacity_units);
            const auto demotion = probe.deviceLayout();
            const auto promotion = makeCpuToGpuExpertTierWeightStreamManifest(cpu,406,3,7,ExpertTierWeightProjection::Down,capacity_units).deviceLayout();
            ASSERT_TRUE(demotion.valid());ASSERT_TRUE(promotion.valid());
            ASSERT_EQ(demotion.cpu_block_stride,2432u);ASSERT_EQ(promotion.gpu_codebook_id,11);
            const std::uint32_t units = demotion.unit_count - first_unit;
            TestHIPStream stream;
            TestHIPBuffer<std::uint8_t> input_payload(source.payload.size()),destination_payload(source.payload.size());
            TestHIPBuffer<std::uint16_t> input_scales(source.scales.size()),input_coeff(source.mins.size());
            TestHIPBuffer<std::uint16_t> destination_scales(source.scales.size()),destination_coeff(source.mins.size());
            TestHIPBuffer<std::uint8_t> chunk(demotion.chunkBytes(units));
            ASSERT_EQ(input_payload.upload(source.payload),hipSuccess);
            ASSERT_EQ(input_scales.upload(source.scales),hipSuccess);ASSERT_EQ(input_coeff.upload(source.mins),hipSuccess);
            ExpertTierGpuConstProjectionView in{.payload=input_payload.data(),.scales=input_scales.data(),.mins=input_coeff.data(),
                .payload_bytes=input_payload.bytes(),.scales_bytes=input_scales.bytes(),.mins_bytes=input_coeff.bytes()};
            ExpertTierGpuMutableProjectionView out{.payload=destination_payload.data(),.scales=destination_scales.data(),.mins=destination_coeff.data(),
                .payload_bytes=destination_payload.bytes(),.scales_bytes=destination_scales.bytes(),.mins_bytes=destination_coeff.bytes()};
            ASSERT_TRUE(in.validFor(demotion));ASSERT_TRUE(out.validFor(promotion));
            std::vector<std::uint8_t> observed_cpu(chunk.bytes()),observed_payload(source.payload.size());
            std::vector<std::uint16_t> observed_scales(source.scales.size()),observed_coeff(source.mins.size());
            HIPGraphCapture graph(stream.native(),0);
            struct Pending
            {
                hipStream_t stream;
                /** @brief Join before graph, host vectors and device owners retire. */
                ~Pending() { EXPECT_EQ(hipStreamSynchronize(stream),hipSuccess); }
            } pending{stream.native()};
            {
                ScopedBackendGraphCapture capture(graph,"IQ3_S exact source tuple tier roundtrip");
                ASSERT_TRUE(capture.begin());
                ASSERT_TRUE(launchGpuToCpuExpertTierChunkROCm(in,demotion,first_unit,units,chunk.data(),chunk.bytes(),stream.opaque()));
                ASSERT_TRUE(launchCpuToGpuExpertTierChunkROCm(chunk.data(),chunk.bytes(),promotion,first_unit,units,out,stream.opaque()));
                capture.finish();
            }
            ASSERT_TRUE(graph.instantiate());
            for (int replay = 0; replay < 3; ++replay)
            {
                ASSERT_EQ(hipMemsetAsync(chunk.data(),0xa5,chunk.bytes(),stream.native()),hipSuccess);
                ASSERT_EQ(hipMemsetAsync(destination_payload.data(),0xa5,destination_payload.bytes(),stream.native()),hipSuccess);
                ASSERT_EQ(hipMemsetAsync(destination_scales.data(),0xa5,destination_scales.bytes(),stream.native()),hipSuccess);
                ASSERT_EQ(hipMemsetAsync(destination_coeff.data(),0xa5,destination_coeff.bytes(),stream.native()),hipSuccess);
                ASSERT_TRUE(graph.launch());
                ASSERT_EQ(hipMemcpyAsync(observed_cpu.data(),chunk.data(),chunk.bytes(),hipMemcpyDeviceToHost,stream.native()),hipSuccess);
                ASSERT_EQ(hipMemcpyAsync(observed_payload.data(),destination_payload.data(),destination_payload.bytes(),hipMemcpyDeviceToHost,stream.native()),hipSuccess);
                ASSERT_EQ(hipMemcpyAsync(observed_scales.data(),destination_scales.data(),destination_scales.bytes(),hipMemcpyDeviceToHost,stream.native()),hipSuccess);
                ASSERT_EQ(hipMemcpyAsync(observed_coeff.data(),destination_coeff.data(),destination_coeff.bytes(),hipMemcpyDeviceToHost,stream.native()),hipSuccess);
                ASSERT_EQ(stream.synchronize(),hipSuccess);
                expectChunkBytesEqual(observed_cpu,chunk.bytes(),cpu.native_interleaved.data() + demotion.chunkBytes(first_unit),first_unit);
                for (int kb = 0; kb < K / 32; ++kb) for (int row = 0; row < N; ++row)
                {
                    const auto linear = static_cast<std::size_t>(kb) * N + row;
                    const bool selected = static_cast<std::uint32_t>((row / 64) * (K / 32) + kb) >= first_unit;
                    EXPECT_EQ(observed_scales[linear],selected ? source.scales[linear] : 0xa5a5u);
                    EXPECT_EQ(observed_coeff[linear],selected ? source.mins[linear] : 0xa5a5u);
                    for (unsigned byte = 0; byte < 13; ++byte)
                        EXPECT_EQ(observed_payload[linear * 13 + byte],selected ? source.payload[linear * 13 + byte] : 0xa5u);
                }
            }
        }

        /**
         * @test First CPU-origin IQ3 promotion initializes the existing resident grid.
         *
         * This is a standalone test, not the warmed IQ-table fixture. Its focused
         * CTest starts a fresh process and performs no GPU source packing/GEMV or
         * explicit table warmup. The real lane materialize method owns cold setup.
         * Expected payload words come directly from the original source block.
         */
        TEST(ROCmIQ3SColdTierSetup, CPUOriginPromotionPreservesSourceWithoutWarmup)
        {
            int devices=0;
            if(hipGetDeviceCount(&devices)!=hipSuccess||devices==0)GTEST_SKIP();
            ASSERT_EQ(hipSetDevice(0),hipSuccess);
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
            cpu::native_vnni::CPUNativeVNNIPackedWeights cpu;
            ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(tensor.get(),cpu));
            ASSERT_EQ(cpu.encoding,cpu::native_vnni::CPUNativeVNNIEncoding::IQ3SExpandedSourceScale);
            const auto layout=makeCpuToGpuExpertTierWeightStreamManifest(cpu,407,3,7,ExpertTierWeightProjection::Down,2).deviceLayout();
            ASSERT_TRUE(layout.valid());ASSERT_EQ(layout.gpu_codebook_id,11);
            TestHIPBuffer<std::uint8_t> payload(8*13);
            TestHIPBuffer<std::uint16_t> scales(8),coefficients(8);
            ExpertTierGpuMutableProjectionView destination{
                .payload=payload.data(),.scales=scales.data(),.mins=coefficients.data(),
                .payload_bytes=payload.bytes(),.scales_bytes=scales.bytes(),.mins_bytes=coefficients.bytes()};
            ASSERT_TRUE(destination.validFor(layout));
            const auto execution=TransferEngine::instance().allocatePersistentTransferExecutionLanes(
                1u,DeviceId::rocm(0),"rocm_iq3_source_cold_promotion").front();
            ExpertTierWeightTransferLane lane({
                .device=DeviceId::rocm(0),
                .staging=TransferEngine::instance().allocatePersistentTransferStagingSlices(
                    layout.chunkBytes(2),1u,DeviceId::rocm(0)).front(),
                .execution=execution,.progress=BackgroundTransferProgressBinding::nativeStream(),
                .lane_name="rocm_iq3_source_cold_promotion",.perf_device="rocm:0"});
            std::string error;
            ASSERT_TRUE(lane.materialize(&error))<<error;
            /** @brief Drain pending test work before lane/source/destination owners retire. */
            struct Retire
            {
                ExpertTierWeightTransferLane &lane;hipStream_t stream;
                ~Retire()
                {
                    if(!lane.quiescent())
                    {
                        EXPECT_EQ(hipStreamSynchronize(stream),hipSuccess);
                        (void)pollLaneToCompletion(lane);
                    }
                }
            } retire{lane,reinterpret_cast<hipStream_t>(execution.stream())};
            ASSERT_TRUE(lane.startCpuToGpu(layout,cpu.native_interleaved,destination,&error))<<error;
            ASSERT_EQ(pollLaneToCompletion(lane),ExpertTierWeightTransferProgress::Ready);
            ASSERT_TRUE(lane.quiescent());
            std::vector<std::uint8_t> actual_payload;
            std::vector<std::uint16_t> actual_scales,actual_coefficients;
            ASSERT_EQ(payload.download(actual_payload),hipSuccess);
            ASSERT_EQ(scales.download(actual_scales),hipSuccess);
            ASSERT_EQ(coefficients.download(actual_coefficients),hipSuccess);
            for(unsigned sub=0;sub<8;++sub)
            {
                EXPECT_EQ(actual_scales[sub],original.d);
                EXPECT_EQ(actual_coefficients[sub],(original.scales[sub/2]>>(4*(sub%2)))&15u);
                EXPECT_EQ(std::memcmp(actual_payload.data()+sub*13,original.qs+sub*8,8),0);
                EXPECT_EQ(actual_payload[sub*13+8],original.qh[sub]);
                EXPECT_EQ(std::memcmp(actual_payload.data()+sub*13+9,original.signs+sub*4,4),0);
            }
        }
    } // namespace
} // namespace llaminar2
