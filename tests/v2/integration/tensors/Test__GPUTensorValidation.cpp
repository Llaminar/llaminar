/**
 * @file Test__GPUTensorValidation.cpp
 * @brief CUDA/ROCm regressions for stream-ordered device tensor validation.
 *
 * Each test enqueues source publication and immediately invokes the validator
 * on the same non-default stream. The validator must observe that publication,
 * classify exact all-zero tensors correctly, and report nonzero/NaN/Inf values
 * for every supported floating representation. The validator's compact
 * event-published result is the observation boundary. Fixture destruction also
 * retires its exact stream so rejected declarations cannot leave pending setup
 * uploads referring to released original host/native storage.
 */

#include <gtest/gtest.h>

#include "backends/IBackend.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/StageVerifier.h"
#include "tensors/GpuTensorView.h"
#include "tensors/GPUTensorVerification.h"
#include "tensors/Tensors.h"
#include "tensors/TensorVerification.h"
#include <array>

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif
#ifdef HAVE_ROCM
#include "backends/rocm/ROCmBackend.h"
#endif

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace llaminar2;

namespace
{
    constexpr size_t kElementCount = 4096;

    /**
     * @brief Minimal stage exposing one device-only tensor as an input.
     *
     * KV-cache adapters use GpuTensorView rather than TensorBase because the
     * cache owns allocation and lifetime. This stage reproduces that production
     * contract so entry validation cannot regress to treating a null host
     * pointer as missing storage.
     */
    class DeviceViewInputStage final : public IComputeStage
    {
    public:
        DeviceViewInputStage(DeviceId device, ITensor &input)
            : IComputeStage(device), input_(input) {}

        bool execute(IDeviceContext *) override { return true; }
        ComputeStageType type() const override { return ComputeStageType::ATTENTION; }
        bool supportsBackend(ComputeBackendType backend) const override
        {
            return (device().is_cuda() && backend == ComputeBackendType::GPU_CUDA) ||
                   (device().is_rocm() && backend == ComputeBackendType::GPU_ROCM);
        }

    protected:
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo info;
            info.addInput("cache_view", &input_, input_.rows(), input_.cols());
            return info;
        }

    private:
        ITensor &input_;
    };

    /**
     * @brief Own persistent test resources for one explicit GPU stream.
     *
     * The host upload buffer is pinned so hostToDeviceOnStream() remains a real
     * asynchronous publication. Destruction occurs only after validate() has
     * observed its completion event, so no extra synchronization is necessary.
     */
    class ValidationResources
    {
    public:
        ValidationResources(IBackend &backend, DeviceType type)
            : backend_(backend), type_(type)
        {
            if (!backend_.setDevice(0))
                throw std::runtime_error("failed to select GPU 0 for tensor validation test");
            stream_ = backend_.createStream(0);
            if (!stream_)
                throw std::runtime_error("failed to create explicit validation stream");
            device_data_ = backend_.allocate(kElementCount * sizeof(float), 0);
            host_data_ = backend_.allocatePinned(kElementCount * sizeof(float), 0);
            if (!device_data_ || !host_data_)
                throw std::runtime_error("failed to allocate validation test storage");
            validator_ = getTensorValidator(type_, 0);
            if (!validator_)
                throw std::runtime_error("backend tensor validator is unavailable");
        }

        ~ValidationResources()
        {
            // Static rejection tests can exit after a test upload but before
            // validate() publishes its terminal result. Retire this fixture's
            // exact stream before releasing pinned/native original storage.
            if (stream_ && !backend_.synchronizeStream(stream_, 0))
                std::terminate();
            if (host_data_)
                backend_.freePinned(host_data_, 0);
            if (device_data_)
                backend_.free(device_data_, 0);
            if (stream_)
                backend_.destroyStream(stream_, 0);
        }

        ValidationResources(const ValidationResources &) = delete;
        ValidationResources &operator=(const ValidationResources &) = delete;

        /** Enqueue bytes and validate them without an intervening host wait. */
        [[nodiscard]] TensorValidationResult validate(
            const void *source,
            size_t bytes,
            size_t num_elements,
            TensorValidationDataType data_type)
        {
            std::memcpy(host_data_, source, bytes);
            if (!backend_.hostToDeviceOnStream(device_data_, host_data_, bytes, 0, stream_))
                throw std::runtime_error("failed to enqueue validation test input");
            return validator_->validate(
                device_data_,
                num_elements,
                data_type,
                ExplicitGPUStream{stream_});
        }

        /** Enqueue bytes that a subsequent stage-stream validation must observe. */
        void upload(const void *source, size_t bytes)
        {
            std::memcpy(host_data_, source, bytes);
            if (!backend_.hostToDeviceOnStream(device_data_, host_data_, bytes, 0, stream_))
                throw std::runtime_error("failed to enqueue device-view validation input");
        }

        /** @brief Submit an acquired-extent scan after the original input publication. */
        [[nodiscard]] TensorValidationResult observe(
            TensorValidationExtent extent, TensorValidationDataType type)
        {
            return validator_->validateObserved(device_data_, extent, type, ExplicitGPUStream{stream_});
        }

        [[nodiscard]] void *deviceData() const noexcept { return device_data_; }
        [[nodiscard]] void *stream() const noexcept { return stream_; }

    private:
        IBackend &backend_;
        DeviceType type_;
        void *stream_ = nullptr;
        void *device_data_ = nullptr;
        void *host_data_ = nullptr;
        ITensorValidator *validator_ = nullptr;
    };

    /** Exercise exact zero and exceptional-value detection for all data types. */
    void verifyAllFloatingRepresentations(IBackend &backend, DeviceType type)
    {
        ValidationResources resources(backend, type);

        std::vector<float> fp32(kElementCount, 0.0f);
        auto result = resources.validate(
            fp32.data(), fp32.size() * sizeof(float), fp32.size(),
            TensorValidationDataType::FP32);
        EXPECT_TRUE(result.appears_zero);
        EXPECT_EQ(result.zero_count, kElementCount);
        EXPECT_FALSE(result.has_nan);
        EXPECT_FALSE(result.has_inf);

        fp32[17] = 1.0f;
        fp32[23] = std::numeric_limits<float>::quiet_NaN();
        fp32[29] = std::numeric_limits<float>::infinity();
        result = resources.validate(
            fp32.data(), fp32.size() * sizeof(float), fp32.size(),
            TensorValidationDataType::FP32);
        EXPECT_FALSE(result.appears_zero);
        EXPECT_EQ(result.nan_count, 1u);
        EXPECT_EQ(result.inf_count, 1u);

        std::vector<uint16_t> bf16(kElementCount, 0u);
        result = resources.validate(
            bf16.data(), bf16.size() * sizeof(uint16_t), bf16.size(),
            TensorValidationDataType::BF16);
        EXPECT_TRUE(result.appears_zero);
        EXPECT_EQ(result.zero_count, kElementCount);

        bf16[17] = 0x3F80u; // 1.0
        bf16[23] = 0x7FC1u; // quiet NaN
        bf16[29] = 0x7F80u; // +Inf
        result = resources.validate(
            bf16.data(), bf16.size() * sizeof(uint16_t), bf16.size(),
            TensorValidationDataType::BF16);
        EXPECT_FALSE(result.appears_zero);
        EXPECT_EQ(result.nan_count, 1u);
        EXPECT_EQ(result.inf_count, 1u);

        std::vector<uint16_t> fp16(kElementCount, 0u);
        result = resources.validate(
            fp16.data(), fp16.size() * sizeof(uint16_t), fp16.size(),
            TensorValidationDataType::FP16);
        EXPECT_TRUE(result.appears_zero);
        EXPECT_EQ(result.zero_count, kElementCount);

        fp16[17] = 0x3C00u; // 1.0
        fp16[23] = 0x7E01u; // quiet NaN
        fp16[29] = 0x7C00u; // +Inf
        result = resources.validate(
            fp16.data(), fp16.size() * sizeof(uint16_t), fp16.size(),
            TensorValidationDataType::FP16);
        EXPECT_FALSE(result.appears_zero);
        EXPECT_EQ(result.nan_count, 1u);
        EXPECT_EQ(result.inf_count, 1u);
    }

    /**
     * @brief Prove stage validation accepts an authoritative hostless GPU view.
     *
     * The upload and scan share one explicit non-default stream. Entry
     * verification must use ITensor's residency contract and must never ask the
     * cache view for raw host data.
     */
    void verifyDeviceOnlyViewAtStageEntry(IBackend &backend, DeviceId device)
    {
#if LLAMINAR_ASSERTIONS_ACTIVE
        ValidationResources resources(backend, device.type);
        std::vector<float> values(kElementCount, 1.0f);
        resources.upload(values.data(), values.size() * sizeof(float));

        GpuTensorView view(
            resources.deviceData(),
            /*rows=*/64,
            /*cols=*/kElementCount / 64,
            TensorType::FP32,
            device);
        auto stage = std::make_unique<DeviceViewInputStage>(device, view);
        stage->setGPUStream(resources.stream());
        ComputeNode node("device_view_consumer", std::move(stage), device);

        EXPECT_NO_THROW(verifyStageEntry(node, 0));
#else
        (void)backend;
        (void)device;
#endif
    }
    /** @brief Borrow exact original native output/count views for verifier negatives. */
    class ObservedOutputStage final : public IComputeStage
    {
    public:
        /** @brief Keep the declaration immutable while its external original roots remain alive. */
        ObservedOutputStage(DeviceId device,StageDumpInfo::OutputBuffer output)
            : IComputeStage(device),output_(std::move(output)) {}
        /** @brief No inference work is required for a diagnostic declaration test. */
        bool execute(IDeviceContext *) override { return true; }
        /** @brief Use the ordinary sparse indexer output-validation policy. */
        ComputeStageType type() const override { return ComputeStageType::QWEN4EXP_QSA_INDEXER; }
        /** @brief This fixture submits only through the original validator, not execute(). */
        bool supportsBackend(ComputeBackendType) const override { return false; }
    protected:
        /** @brief Preserve original subview/count owner identity in every snapshot. */
        StageDumpInfo buildDumpInfoImpl() const override
        {
            StageDumpInfo result;
            result.outputs.push_back(output_);
            return result;
        }
    private:
        StageDumpInfo::OutputBuffer output_;
    };

    /**
     * @brief Empty native observations still reject foreign/missing/unbounded owners.
     * @param backend Actual backend owning the original allocation/stream.
     * @param device Exact payload/count authority.
     */
    void verifyObservedStageBindings(IBackend &backend,DeviceId device)
    {
#if LLAMINAR_ASSERTIONS_ACTIVE
        ValidationResources resources(backend,device.type);
        std::vector<std::uint32_t> bank(kElementCount,0x7fc00001u);
        bank[kElementCount-2]=0;bank[kElementCount-1]=0;
        resources.upload(bank.data(),bank.size()*4);
        GpuTensorView payload(resources.deviceData(),kElementCount,1,TensorType::FP32,device);
        GpuTensorView frontier(static_cast<std::int32_t *>(resources.deviceData())+kElementCount-2,
                               2,1,TensorType::INT32,device);
        INT32Tensor hostFrontier({2});
        StageDumpInfo::OutputBuffer output{};
        output.name="observed_native_subview";
        output.rows=4;output.cols=128;output.dtype="FP32";
        output.element_size=4;output.byte_size=4*128*4;
        output.tensor=&payload;output.tensor_byte_offset=16;
        output.semantic_extent=SnapshotStatePrefixRows{{&frontier,0},16,4};
        auto observe=[&](StageDumpInfo::OutputBuffer declaration,DeviceId owner,void *stream) {
            auto stage=std::make_unique<ObservedOutputStage>(owner,declaration);
            stage->setGPUStream(stream);
            ComputeNode node("observed_native_binding",std::move(stage),owner);
            verifyStageExit(node,-1);
        };
        EXPECT_NO_THROW(observe(output,device,resources.stream()));
        auto bad=output;
        std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.tensor=&hostFrontier;
        EXPECT_THROW(observe(bad,device,resources.stream()),verification::VerificationFailure);
        bad=output;std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.tensor=&payload;
        EXPECT_THROW(observe(bad,device,resources.stream()),verification::VerificationFailure);
        bad=output;std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.byteOffset=4;
        EXPECT_THROW(observe(bad,device,resources.stream()),verification::VerificationFailure);
        bad=output;std::get<SnapshotStatePrefixRows>(bad.semantic_extent).frontier.tensor=nullptr;
        EXPECT_THROW(observe(bad,device,resources.stream()),verification::VerificationFailure);
        bad=output;bad.tensor_byte_offset=payload.size_bytes();
        EXPECT_THROW(observe(bad,device,resources.stream()),verification::VerificationFailure);
        const auto foreign=device.is_cuda() ? DeviceId::rocm(0) : DeviceId::cuda(0);
        EXPECT_THROW(observe(output,foreign,resources.stream()),verification::VerificationFailure);
        // Integer/no-numerical-scan policy must still authenticate the native
        // original owner before returning its device-only no-op result.
        StageDumpInfo::OutputBuffer integer{};
        integer.name="integer_native_owner";
        integer.rows=2;integer.cols=1;integer.dtype="INT32";
        integer.element_size=4;integer.byte_size=8;integer.tensor=&frontier;
        EXPECT_NO_THROW(observe(integer,device,resources.stream()));
        EXPECT_THROW(observe(integer,foreign,resources.stream()),verification::VerificationFailure);
#else
        (void)backend;(void)device;
#endif
    }

    /**
     * @brief Prove all three scanners use the acquired native extent, not capacity.
     * @param backend Actual selected native backend.
     * @param device Original allocation/stream owner.
     *
     * Payload and two frontier words share this real fixture allocation with
     * disjoint checked subviews. Acquired metadata never crosses to the host;
     * only the validator's existing compact result carries checked-elements.
     */
    void verifyObservedRepresentations(IBackend &backend, DeviceId device)
    {
        ValidationResources resources(backend, device.type);
        constexpr std::size_t rows=4,columns=128,frontierWord=kElementCount-2;
        auto *nativeWords=static_cast<std::int32_t *>(resources.deviceData());
        TensorValidationExtent extent{{SnapshotContiguousRowKind::StatePrefixRows,rows,16,4},
                                      columns,nativeWords+frontierWord};
        for (auto type : {TensorValidationDataType::FP32,TensorValidationDataType::BF16,
                          TensorValidationDataType::FP16})
        {
            SCOPED_TRACE(static_cast<int>(type));
            const auto width=type==TensorValidationDataType::FP32 ? 4u : 2u;
            const std::uint32_t one=type==TensorValidationDataType::FP32 ? 0x3f800000u :
                type==TensorValidationDataType::BF16 ? 0x3f80u : 0x3c00u;
            const std::uint32_t nan=type==TensorValidationDataType::FP32 ? 0x7fc00001u :
                type==TensorValidationDataType::BF16 ? 0x7fc1u : 0x7e01u;
            const std::uint32_t inf=type==TensorValidationDataType::FP32 ? 0x7f800000u :
                type==TensorValidationDataType::BF16 ? 0x7f80u : 0x7c00u;
            std::vector<std::uint32_t> bank(kElementCount,0);
            auto set=[&](std::size_t index,std::uint32_t bits) {
                auto *at=reinterpret_cast<std::byte *>(bank.data())+index*width;
                if (width==4) std::memcpy(at,&bits,4);
                else { const auto narrow=static_cast<std::uint16_t>(bits);std::memcpy(at,&narrow,2); }
            };
            auto publish=[&](int prior,int live) {
                const std::array<std::int32_t,2> words{prior,live};
                std::memcpy(bank.data()+frontierWord,words.data(),sizeof(words));
                resources.upload(bank.data(),bank.size()*4);
                return resources.observe(extent,type);
            };
            for (auto [prior,live] : std::array<std::array<int,2>,8>{{
                {8,8},{0,1},{0,2},{0,3},{3,1},{5,2},{7,1},{0,0}}})
            {
                const auto observedRows=(prior+live)/4;
                for (std::size_t i=0;i<rows*columns;++i) set(i,i<observedRows*columns ? one : nan);
                const auto result=publish(prior,live);
                EXPECT_TRUE(result.valid);
                EXPECT_FALSE(result.invalid_extent);
                EXPECT_FALSE(result.has_nan);
                EXPECT_EQ(result.total_checked,observedRows*columns);
            }
            for (std::size_t i=0;i<rows*columns;++i) set(i,i<columns ? 0u : nan);
            auto result=publish(3,1);
            EXPECT_TRUE(result.valid);
            EXPECT_TRUE(result.appears_zero);
            EXPECT_EQ(result.zero_count,columns);
            EXPECT_EQ(result.total_checked,columns);
            set(0,nan);set(1,inf);
            result=publish(3,1);
            EXPECT_FALSE(result.valid);
            EXPECT_EQ(result.nan_count,1u);
            EXPECT_EQ(result.inf_count,1u);
            for (auto [prior,live] : std::array<std::array<int,2>,3>{{{-1,0},{0,-1},{16,1}}})
            {
                result=publish(prior,live);
                EXPECT_FALSE(result.valid);
                EXPECT_TRUE(result.invalid_extent);
                EXPECT_EQ(result.total_checked,0u);
                EXPECT_EQ(result.nan_count,0u);
            }
            // Static invalid declarations reject before the initializer enqueue.
            auto malformed=extent;
            malformed.frontier=nullptr;
            EXPECT_THROW(resources.observe(malformed,type),std::invalid_argument);
            malformed=extent;malformed.columns=std::numeric_limits<std::size_t>::max();
            EXPECT_THROW(resources.observe(malformed,type),std::invalid_argument);
            malformed=extent;malformed.geometry.kind=static_cast<SnapshotContiguousRowKind>(255);
            EXPECT_THROW(resources.observe(malformed,type),std::invalid_argument);
        }
    }
} // namespace

#ifdef HAVE_CUDA
TEST(Test__GPUTensorValidation, CUDAExactProducerStreamCoversAllFloatingTypes)
{
    CUDABackend backend;
    verifyAllFloatingRepresentations(backend, DeviceType::CUDA);
    verifyDeviceOnlyViewAtStageEntry(backend, DeviceId::cuda(0));
}
#endif

#ifdef HAVE_ROCM
TEST(Test__GPUTensorValidation, ROCmExactProducerStreamCoversAllFloatingTypes)
{
    ROCmBackend backend;
    verifyAllFloatingRepresentations(backend, DeviceType::ROCm);
    verifyDeviceOnlyViewAtStageEntry(backend, DeviceId::rocm(0));
}
#endif

#ifdef HAVE_CUDA
/** @test All CUDA floating scanners consume actual acquired rows on the original producer stream. */
TEST(Test__GPUTensorValidation, CUDAObservedNativeRows)
{
    CUDABackend backend;
    verifyObservedRepresentations(backend,DeviceId::cuda(0));
    verifyObservedStageBindings(backend,DeviceId::cuda(0));
}
#endif
#ifdef HAVE_ROCM
/** @test All HIP floating scanners consume actual acquired rows on the original producer stream. */
TEST(Test__GPUTensorValidation, ROCmObservedNativeRows)
{
    ROCmBackend backend;
    verifyObservedRepresentations(backend,DeviceId::rocm(0));
    verifyObservedStageBindings(backend,DeviceId::rocm(0));
}
#endif
