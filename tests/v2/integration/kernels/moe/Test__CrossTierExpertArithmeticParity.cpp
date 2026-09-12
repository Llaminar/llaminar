/**
 * @file Test__CrossTierExpertArithmeticParity.cpp
 * @brief CPU/CUDA/ROCm GPU-aligned expert arithmetic parity integration tests.
 *
 * ExpertOverlay may execute one logical expert on different physical backends
 * in adjacent movement epochs. These focused tests keep the expensive common
 * all-codebook proof in a small translation unit so arithmetic regressions can
 * be iterated without rebuilding either backend's monolithic MoE test binary.
 */

#include "NativeVNNIExpertTransferParityTest.h"
#include "execution/moe/DeviceMoEExpertDescriptorBuilder.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{
    /** @brief RAII owner for one explicit non-default backend test stream. */
    class ScopedBackendStream final
    {
    public:
        /**
         * @brief Create a stream on the exact backend endpoint.
         * @param device CUDA or ROCm device whose backend owns the stream.
         */
        explicit ScopedBackendStream(llaminar2::DeviceId device)
            : backend_(llaminar2::getBackendFor(device)),
              ordinal_(device.ordinal)
        {
            if (!backend_ || !device.is_gpu() || ordinal_ < 0 ||
                ordinal_ >= backend_->deviceCount())
                throw std::runtime_error(
                    "Cross-tier arithmetic parity received an unavailable GPU endpoint");
            stream_ = backend_->createStream(ordinal_);
            if (!stream_)
                throw std::runtime_error(
                    "Cross-tier arithmetic parity could not create an explicit stream");
        }

        /** @brief Release the stream after the helper's terminal event drain. */
        ~ScopedBackendStream()
        {
            if (stream_)
                backend_->destroyStream(stream_, ordinal_);
        }

        ScopedBackendStream(const ScopedBackendStream &) = delete;
        ScopedBackendStream &operator=(const ScopedBackendStream &) = delete;

        /** @return The exact non-default stream used by every GPU operation. */
        [[nodiscard]] void *get() const noexcept { return stream_; }

    private:
        llaminar2::IBackend *backend_ = nullptr;
        int ordinal_ = -1;
        void *stream_ = nullptr;
    };

    /**
     * @brief Prepare native floating bytes shared by the two backend oracles.
     * @param type FP16, BF16, or FP32 weight storage; activations stay FP32.
     * @param shape Row-major projection geometry.
     * @param seed Distinct, deterministic gate/up/down operands.
     * @return Owning tensor whose rounding happens once before either loader.
     */
    std::unique_ptr<llaminar2::TensorBase> makeFloatingExpertWeight(
        llaminar2::TensorType type, const std::vector<size_t> &shape, size_t seed)
    {
        using namespace llaminar2;
        using test::TestTensorFactory;
        std::vector<float> values(shape[0] * shape[1]);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = 0.00317f * static_cast<float>(
                static_cast<int>((i * (37u + seed * 2u) + seed * 11u) % 103u) - 51);
        switch (type)
        {
        case TensorType::FP16:
        {
            auto tensor = TestTensorFactory::createFP16(shape);
            tensor->from_fp32(values.data(), values.size());
            return tensor;
        }
        case TensorType::BF16:
        {
            auto tensor = TestTensorFactory::createBF16(shape);
            tensor->from_fp32(values.data(), values.size());
            return tensor;
        }
        case TensorType::FP32:
        {
            auto tensor = TestTensorFactory::createFP32(shape);
            std::copy(values.begin(), values.end(), tensor->mutable_data());
            return tensor;
        }
        default:
            throw std::invalid_argument("Floating expert proof requires FP16, BF16 or FP32");
        }
    }

    /**
     * @brief Compare real CPU expert execution with captured GPU runtime decode.
     *
     * A quantized GGUF can contain floating routed-expert layers. The same
     * expert must publish identical words after a tier change in those layers
     * too. In particular, a 32-wide synthetic dot misses the full 256-lane
     * reduction tree used by a production 3072x1024 expert.
     *
     * @param device CUDA or ROCm participant paired with a CPU expert engine.
     * @param stream Exact, non-default stream owning setup and captured work.
     * @param create_weight Shared source factory; registered gates use the
     *        model-free operands and diagnostics may supply native GGUF views.
     */
    void runFloatingCrossTierDecodeParity(
        llaminar2::DeviceId device, void *stream,
        const std::function<std::unique_ptr<llaminar2::TensorBase>(
            llaminar2::TensorType, const std::vector<size_t> &, uint32_t)> &
            create_weight = makeFloatingExpertWeight)
    {
        using namespace llaminar2;
        using namespace llaminar2::test;
        using namespace native_vnni_transfer_parity_detail;
        using CpuKernel = gemm::FloatingPointGemmKernel;
        constexpr int d_model = 3072;
        constexpr int intermediate = 1024;
        auto *backend = getBackendFor(device);
        auto kernel = llaminar::v2::kernels::KernelFactory::createMoEKernel(device);
        ASSERT_NE(kernel, nullptr);
        kernel->setGPUStream(stream);
        auto requirements = device.is_cuda()
            ? MoEWorkspaceBuffers::cudaMoE(1, d_model, intermediate, 1, 1)
            : MoEWorkspaceBuffers::rocmMoE(1, d_model, intermediate, 1, 1);
        DeviceWorkspaceManager workspace(
            device, requirements.total_bytes_with_alignment() + 4u * 1024u * 1024u);
        ASSERT_TRUE(workspace.allocate(requirements));
        auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
        ASSERT_NE(consumer, nullptr);
        consumer->bindWorkspace(&workspace);

        for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
        {
            SCOPED_TRACE(device.to_string() + " floating format=" +
                         std::to_string(static_cast<int>(type)));
            auto gate_weight = create_weight(type, {intermediate, d_model}, 1u);
            auto up_weight = create_weight(type, {intermediate, d_model}, 2u);
            auto down_weight = create_weight(type, {d_model, intermediate}, 3u);
            const auto model = ModelContextId{4200000u + static_cast<uint64_t>(type)};
            auto gate_gpu = makeGpuPreparedFloatingPointGemm(
                gate_weight.get(), device, "test.cross_tier_float.gate", model);
            auto up_gpu = makeGpuPreparedFloatingPointGemm(
                up_weight.get(), device, "test.cross_tier_float.up", model);
            auto down_gpu = makeGpuPreparedFloatingPointGemm(
                down_weight.get(), device, "test.cross_tier_float.down", model);
            CpuKernel gate_cpu(gate_weight.get(), CpuKernel::NumericalPolicy::GPUAlignedExpert);
            CpuKernel up_cpu(up_weight.get(), CpuKernel::NumericalPolicy::GPUAlignedExpert);
            CpuKernel down_cpu(down_weight.get(), CpuKernel::NumericalPolicy::GPUAlignedExpert);
            DeviceMoEExpertDescriptor expert{};
            expert.logical_expert_id = 0;
            expert.owner_participant = 0;
            ASSERT_TRUE(exportDeviceMoEExpertWeightDescriptors(
                gate_gpu.kernel, up_gpu.kernel, down_gpu.kernel,
                d_model, intermediate, expert));
            const int gate_table = kernel->uploadGroupedExpertFloatingGateUpDescriptorTables(
                &expert.floating_gate, &expert.floating_up, expert.weight_format,
                1, d_model, intermediate);
            const int down_table = kernel->uploadGroupedExpertFloatingDownDescriptorTable(
                &expert.floating_down, expert.weight_format, 1, d_model, intermediate);
            ASSERT_GE(gate_table, 0);
            ASSERT_GE(down_table, 0);
            auto runtime = makeRuntimeTable(device, stream, {expert}, 0, {1}, {1}, 1, 1, 1);

            for (const auto pattern : {ExpertInputPattern::Smooth, ExpertInputPattern::Scrambled})
            for (const float amplitude : {1.0f, 4.0f, 32.0f})
            {
                SCOPED_TRACE("amplitude=" + std::to_string(amplitude) +
                             " pattern=" + std::to_string(static_cast<int>(pattern)));
                auto hidden = makeHidden(1, d_model, 0u, pattern);
                for (size_t i = 0; i < hidden->numel(); ++i)
                    hidden->mutable_data()[i] *= amplitude;
                auto cpu_gate = TestTensorFactory::createFP32({1u, intermediate});
                auto cpu_up = TestTensorFactory::createFP32({1u, intermediate});
                auto cpu_output = TestTensorFactory::createFP32({1u, d_model});
                std::vector<ITensorGemm::TensorProjectionDesc> projections = {
                    {&gate_cpu, cpu_gate.get(), intermediate, nullptr, "gate"},
                    {&up_cpu, cpu_up.get(), intermediate, nullptr, "up"}};
                ASSERT_TRUE(gate_cpu.multiply_fused_tensor(hidden.get(), projections, 1, d_model));
                ASSERT_TRUE(down_cpu.multiply_tensor_with_fused_swiglu(
                    cpu_gate.get(), cpu_up.get(), cpu_output.get(), 1, d_model, intermediate));
                expectCPUExpertPacketParity(
                    {&gate_cpu, &up_cpu, &down_cpu}, hidden->data(), cpu_output->data(),
                    1, d_model, intermediate);
                auto router = TestTensorFactory::createFP32({1u, d_model});
                std::fill_n(router->mutable_data(), router->numel(), 0.0f);
                auto output = TestTensorFactory::createFP32({1u, d_model});
                auto routes = TestTensorFactory::createFP32({1u, 1u, d_model});
                for (auto *tensor : std::array<TensorBase *, 4>{
                         hidden.get(), router.get(), output.get(), routes.get()})
                {
                    ASSERT_TRUE(tensor->ensureOnDevice(device, stream));
                    TransferEngine::requireDeviceInput(tensor, device, stream);
                }
                ASSERT_TRUE(kernel->prepareRouteLaunch(router.get(), {
                    .kind = MoERouteLaunchKind::RuntimeDecode, .physical_rows = 1,
                    .d_model = d_model, .num_experts = 1, .top_k = 1}));
                ASSERT_TRUE(kernel->prepareGroupedRuntimeDecodeLaunchState(
                    gate_table, down_table, 1, d_model, intermediate,
                    MoEDecodeDescriptorSource::RuntimePlacementTable));
                auto &context = GPUDeviceContextPool::instance().getContext(device);
                auto graph = context.createGraphCapture(stream);
                ASSERT_NE(graph, nullptr);
                ScopedBackendGraphCapture capture(context, *graph, "floating cross-tier runtime decode");
                ASSERT_TRUE(capture.begin());
                const bool routed = kernel->decodeRouteSelect(
                    runtime->deviceLayerState(0), hidden.get(), router.get(),
                    d_model, 1, 1, true, nullptr, nullptr, false, false, nullptr,
                    RoutedExpertRowExecutionPolicy::ParticipantAssigned);
                const bool computed = kernel->groupedExpertDecodeFromRuntime(
                    runtime->deviceLayerState(0), hidden.get(), gate_table, down_table,
                    1, output.get(), d_model, intermediate,
                    MoEDecodeDescriptorSource::RuntimePlacementTable, routes.get());
                const bool reduced = kernel->reduceCanonicalRouteContributions(
                    routes.get(), output.get(), 1, 1, d_model);
                capture.finish();
                ASSERT_TRUE(routed);
                ASSERT_TRUE(computed);
                ASSERT_TRUE(reduced);
                ASSERT_TRUE(graph->instantiate());
                ASSERT_TRUE(graph->launch());
                TransferEngine::publishDeviceWrite(output.get(), device, stream);
                ASSERT_TRUE(output->ensureOnHost(stream));
                ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
                expectByteEqual("CPU vs captured GPU floating expert",
                    std::vector<float>(output->data(), output->data() + d_model),
                    std::vector<float>(cpu_output->data(), cpu_output->data() + d_model), d_model);
            }
        }
        ASSERT_TRUE(backend->synchronizeStream(stream, device.ordinal));
        consumer->unbindWorkspace();
    }
} // namespace


#ifdef HAVE_CUDA
/** @test Prove native FP16/BF16/FP32 experts remain exact across CPU/CUDA. */
TEST(Test__CrossTierExpertArithmeticParity, CPUAndCUDAFloatingExpertDecodeIsByteExact)
{
    const auto device = llaminar2::DeviceId::cuda(0);
    auto *backend = llaminar2::getBackendFor(device);
    if (!backend || backend->deviceCount() <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ScopedBackendStream stream(device);
    runFloatingCrossTierDecodeParity(device, stream.get());
}

/** @test Prove all quantized GPU-aligned experts are CPU/CUDA byte-identical. */
TEST(
    Test__CrossTierExpertArithmeticParity,
    CPUAndCUDAAllQuantizedExpertFormatsAreByteExactAcrossM)
{
    const auto device = llaminar2::DeviceId::cuda(0);
    auto *backend = llaminar2::getBackendFor(device);
    if (!backend || backend->deviceCount() <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ScopedBackendStream stream(device);
    llaminar2::test::runCPUToGPUAllFormatExpertArithmeticParity(
        "CUDA",
        device,
        stream.get());
}
#endif

#ifdef HAVE_ROCM
/** @test Prove native FP16/BF16/FP32 experts remain exact across CPU/ROCm. */
TEST(Test__CrossTierExpertArithmeticParity, CPUAndROCmFloatingExpertDecodeIsByteExact)
{
    const auto device = llaminar2::DeviceId::rocm(0);
    auto *backend = llaminar2::getBackendFor(device);
    if (!backend || backend->deviceCount() <= 0)
        GTEST_SKIP() << "No ROCm device available";
    ScopedBackendStream stream(device);
    runFloatingCrossTierDecodeParity(device, stream.get());
}

/** @test Prove all quantized GPU-aligned experts are CPU/ROCm byte-identical. */
TEST(
    Test__CrossTierExpertArithmeticParity,
    CPUAndROCmAllQuantizedExpertFormatsAreByteExactAcrossM)
{
    const auto device = llaminar2::DeviceId::rocm(0);
    auto *backend = llaminar2::getBackendFor(device);
    if (!backend || backend->deviceCount() <= 0)
        GTEST_SKIP() << "No ROCm device available";
    ScopedBackendStream stream(device);
    llaminar2::test::runCPUToGPUAllFormatExpertArithmeticParity(
        "ROCm",
        device,
        stream.get());
}
#endif
