/**
 * @file Test__GPUTensorValidation.cpp
 * @brief CUDA/ROCm regressions for stream-ordered device tensor validation.
 *
 * Each test enqueues source publication and immediately invokes the validator
 * on the same non-default stream. The validator must observe that publication,
 * classify exact all-zero tensors correctly, and report nonzero/NaN/Inf values
 * for every supported floating representation. No test-side stream or device
 * synchronization is used; the validator's compact event-published result is
 * the only host observation boundary.
 */

#include <gtest/gtest.h>

#include "backends/IBackend.h"
#include "execution/compute_stages/IComputeStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/StageVerifier.h"
#include "tensors/GpuTensorView.h"
#include "tensors/GPUTensorVerification.h"

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
