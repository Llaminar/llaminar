/**
 * @file Test__FloatingOutputPartitions.cpp
 * @brief Native floating mirrored heads match independent serial TP shards.
 *
 * A real prepared kernel emits full-width outputs through captured ordinary
 * and grouped-verifier APIs. The oracle uses distinct prepared shard kernels
 * and ordinary M=1 calls; neither backend BLAS nor the mirrored scope can
 * silently certify itself. Every floating format and TP degree 1..8 is covered,
 * including odd column/K tails, M=1..16 and deeper grouped rows.
 */
#include "utils/GpuPreparedGemmHarness.h"
#include "utils/ScopedGPUStream.h"
#include "utils/TestTensorFactory.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "transfer/TransferEngine.h"
#include <gtest/gtest.h>
#include <cstring>
#include <future>

namespace llaminar2::test
{
namespace
{
/** @brief Own a real prepared projection, its weight lease and declared scratch. */
struct Projection
{
    std::unique_ptr<TensorBase> weight;
    GpuPreparedGemm gpu;
    std::unique_ptr<gemm::FloatingPointGemmKernel> cpu;
    std::unique_ptr<DeviceWorkspaceManager> workspace;
    ITensorGemm *kernel = nullptr;

    /** @brief Clear borrowed bindings before their storage is released. */
    ~Projection()
    {
        if (auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel))
            consumer->unbindWorkspace();
        if (kernel)
            kernel->clearGPUStreamBinding();
    }

    /** @brief Prepare exact native bytes through the production kernel factory. */
    void prepare(DeviceId device, void *stream, int max_rows)
    {
        if (device.is_gpu())
        {
            gpu = makeGpuPreparedFloatingPointGemm(weight.get(), device,
                "test.floating_output_partition", ModelContextId{840091});
            kernel = gpu.kernel;
            kernel->setGPUStream(stream);
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(kernel);
            ASSERT_NE(consumer, nullptr);
            auto requirements = consumer->getWorkspaceRequirements(
                max_rows, static_cast<int>(weight->rows()), static_cast<int>(weight->cols()));
            const std::array<int, 1> widths{static_cast<int>(weight->rows())};
            consumer->appendFusedProjectionWorkspaceRequirements(
                requirements, max_rows, widths, static_cast<int>(weight->cols()));
            workspace = std::make_unique<DeviceWorkspaceManager>(
                device, requirements.total_bytes_with_alignment());
            ASSERT_TRUE(workspace->allocate(requirements));
            consumer->bindWorkspace(workspace.get());
            ASSERT_TRUE(kernel->prepareFusedProjectionGraphCapture(1));
        }
        else
        {
            cpu = std::make_unique<gemm::FloatingPointGemmKernel>(weight.get());
            kernel = cpu.get();
        }
    }
};

/** @brief Preserve the source dtype and copy a native row interval without conversion. */
std::unique_ptr<TensorBase> weights(TensorType type, int n, int k,
                                    const TensorBase *source = nullptr, int first = 0)
{
    const std::vector<size_t> shape{size_t(n), size_t(k)};
    std::unique_ptr<TensorBase> result;
    if (type == TensorType::FP32)
        result = TestTensorFactory::createFP32Random(shape, -.5f, .5f, 840091);
    else if (type == TensorType::FP16)
        result = TestTensorFactory::createFP16Random(shape, -.5f, .5f, 840091);
    else
        result = TestTensorFactory::createBF16Random(shape, -.5f, .5f, 840091);
    if (source)
        std::memcpy(result->raw_mutable_data(),
            static_cast<const uint8_t *>(source->raw_data()) +
                size_t(first) * k * (type == TensorType::FP32 ? 4u : 2u),
            result->size_bytes());
    return result;
}

/** @brief Compare captured full-width rows with independently prepared serial shards. */
void provePartitions(DeviceId device)
{
    constexpr int N = 131, K = 257, max_m = 64;
    std::unique_ptr<ScopedGPUStream> stream_owner;
    if (device.is_gpu())
        stream_owner = std::make_unique<ScopedGPUStream>(device);
    void *stream = stream_owner ? stream_owner->get() : nullptr;
    auto *backend = device.is_gpu() ? getBackendFor(device) : nullptr;
    auto input = TestTensorFactory::createFP32Random({max_m, K}, -.7f, .7f, 840093);
    auto output = TestTensorFactory::createFP32({max_m, N});
    if (device.is_gpu())
    {
        ASSERT_TRUE(input->ensureOnDevice(device, stream));
        ASSERT_TRUE(output->allocateOnDevice(device, stream));
    }
    std::vector<float> expected(max_m * N), actual(max_m * N);
    for (auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
    {
        SCOPED_TRACE(static_cast<int>(type));
        Projection full;
        full.weight = weights(type, N, K);
        ASSERT_NO_FATAL_FAILURE(full.prepare(device, stream, max_m));
        ASSERT_EQ(full.kernel->get_n(), N);
        ASSERT_EQ(full.kernel->get_k(), K);
        EXPECT_THROW(full.kernel->beginOutputPartitionEquivalenceScope(N - 1, 16), std::invalid_argument);
        EXPECT_THROW(full.kernel->beginOutputPartitionEquivalenceScope(N, 0), std::invalid_argument);
        EXPECT_THROW(full.kernel->beginOutputPartitionEquivalenceScope(N, N + 1), std::invalid_argument);

        for (int degree = 1; degree <= 8; ++degree)
        {
            SCOPED_TRACE(degree);
            const int shard_n = (N + degree - 1) / degree;
            for (int first = 0; first < N; first += shard_n)
            {
                const int width = std::min(shard_n, N - first);
                Projection shard;
                shard.weight = weights(type, width, K, full.weight.get(), first);
                ASSERT_NO_FATAL_FAILURE(shard.prepare(device, stream, 1));
                auto row_output = TestTensorFactory::createFP32({1u, size_t(width)});
                if (device.is_gpu())
                    ASSERT_TRUE(row_output->allocateOnDevice(device, stream));
                std::vector<float> row_values(width);
                for (int row = 0; row < max_m; ++row)
                {
                    ASSERT_TRUE(shard.kernel->multiply_tensor(input.get(), row_output.get(),
                        1, width, K, true, 1.f, 0.f, nullptr, nullptr, -1,
                        shard.workspace.get(), row));
                    if (device.is_gpu())
                    {
                        ASSERT_TRUE(backend->deviceToHost(row_values.data(), row_output->gpu_data_ptr(),
                            width * sizeof(float), device.ordinal, stream));
                    }
                    else
                        std::memcpy(row_values.data(), row_output->data(), width * sizeof(float));
                    std::copy(row_values.begin(), row_values.end(), expected.begin() + row * N + first);
                }
            }
            for (int m = 1; m <= max_m; ++m)
            {
                if (m > 16 && m != 31 && m != max_m)
                    continue;
                SCOPED_TRACE(m);
                for (int lane : {0, 1, 2})
                {
                    SCOPED_TRACE(lane);
                    // Scope lifetime is recording-only. Replay after destruction
                    // must retain the complete arithmetic choice in its graph.
                    auto launch = [&]()
                    {
                        if (lane == 2)
                            return full.kernel->multiply_fused_verifier_rows_decode_equivalent(
                                input.get(), {{full.kernel, output.get(), N, nullptr, "head"}},
                                m, K, nullptr, full.workspace.get());
                        if (lane == 1)
                            return full.kernel->multiply_fused_tensor(
                                input.get(), {{full.kernel, output.get(), N, nullptr, "head"}},
                                m, K, nullptr, full.workspace.get());
                        return full.kernel->multiply_tensor(input.get(), output.get(), m, N, K,
                            true, 1.f, 0.f, nullptr, nullptr, -1, full.workspace.get());
                    };
                    if (device.is_gpu())
                    {
                        auto &context = GPUDeviceContextPool::instance().getContext(device);
                        auto graph = context.createGraphCapture(stream);
                        {
                            auto scope = full.kernel->beginOutputPartitionEquivalenceScope(N, shard_n);
                            ScopedBackendGraphCapture recording(context, *graph, "floating_output_partition");
                            ASSERT_TRUE(recording.begin());
                            const bool ok = launch();
                            recording.finish();
                            ASSERT_TRUE(ok);
                            ASSERT_TRUE(graph->instantiate());
                        }
                        ASSERT_TRUE(graph->launch());
                        ASSERT_TRUE(backend->deviceToHost(actual.data(), output->gpu_data_ptr(),
                            size_t(m) * N * sizeof(float), device.ordinal, stream));
                    }
                    else
                    {
                        auto scope = full.kernel->beginOutputPartitionEquivalenceScope(N, shard_n);
                        ASSERT_TRUE(launch());
                        std::memcpy(actual.data(), output->data(), size_t(m) * N * sizeof(float));
                    }
                    ASSERT_EQ(std::memcmp(actual.data(), expected.data(), size_t(m) * N * sizeof(float)), 0);
                }
            }
        }
    }
}

/** @test CPU uses the same native-format and serial-partition contract. */
TEST(FloatingOutputPartitions, CPU) { provePartitions(DeviceId::cpu()); }
#ifdef HAVE_CUDA
/** @test CUDA retained ordinary/verifier heads are byte-identical to serial TP. */
TEST(FloatingOutputPartitions, CUDA) { provePartitions(DeviceId::cuda(0)); }
#endif
#ifdef HAVE_ROCM
/** @test ROCm retained ordinary/verifier heads include wide FP32, not just BF16. */
TEST(FloatingOutputPartitions, ROCm) { provePartitions(DeviceId::rocm(0)); }
#endif

/** @test Recording scopes nest and cannot affect another thread or kernel. */
TEST(FloatingOutputPartitions, CPUScopeIsolation)
{
    auto w = weights(TensorType::FP32, 131, 257);
    gemm::FloatingPointGemmKernel first(w.get()), second(w.get());
    auto active = [&](const ITensorGemm &kernel) {
        return FloatingOutputPartitionScope::requiresFixedColumns(kernel, 131);
    };
    EXPECT_FALSE(active(first));
    {
        auto outer = first.beginOutputPartitionEquivalenceScope(131, 33);
        EXPECT_TRUE(active(first));
        EXPECT_FALSE(active(second));
        EXPECT_THROW(FloatingOutputPartitionScope::requiresFixedColumns(first, 130), std::logic_error);
        EXPECT_FALSE(std::async(std::launch::async, [&] { return active(first); }).get());
        {
            auto inner = second.beginOutputPartitionEquivalenceScope(131, 17);
            EXPECT_TRUE(active(first));
            EXPECT_TRUE(active(second));
        }
        EXPECT_TRUE(active(first));
        EXPECT_FALSE(active(second));
    }
    EXPECT_FALSE(active(first));
}
}
}
