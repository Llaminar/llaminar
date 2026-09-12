/**
 * @file Test__TransferEngine_CopyActivation.cpp
 * @brief Integration tests for TransferEngine::copyActivation() on real GPUs.
 *
 * copyActivation() performs a tensor→tensor copy INTO the destination tensor's
 * buffer on a target device, auto-selecting the optimal transport:
 *   - same physical GPU            → device-to-device copy (intra-VRAM)
 *   - same-vendor, different GPU   → peer copy (NCCL/RCCL or peer DMA)
 *   - cross-vendor GPU (CUDA↔ROCm) → host-staged bounce (no direct path)
 *   - source on host                → direct H2D upload
 *
 * These paths require real device backends, so they live in the integration
 * suite (the pure host/guard paths are covered by the unit suite in
 * unit/transfer/Test__TransferEngine). Every GPU-dependent test guards on
 * hardware availability with GTEST_SKIP so the suite passes on CPU-only hosts.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include "v2/tensors/TensorClasses.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/GPUDeviceContextPool.h"
#include "v2/backends/IGPUGraphCapture.h"
#include "v2/backends/IWorkerGPUContext.h"
#include "v2/execution/local_execution/graph/GraphCaptureGuard.h"
#include "v2/collective/BackendRouter.h"
#include "v2/transfer/TransferEngine.h"
#include "v2/transfer/TransferMethod.h"
#include "../../utils/ScopedGPUStream.h"

using namespace llaminar2;

/**
 * @brief Fixture detecting available CUDA/ROCm devices for copyActivation tests.
 *
 * Mirrors the detection strategy of Test__TensorBase_TransferTo: probes the
 * registered backends and device counts so individual tests can skip cleanly
 * when the required hardware (single GPU, two same-vendor GPUs, or both
 * vendors) is not present.
 */
class Test__TransferEngine_CopyActivation : public ::testing::Test
{
protected:
    /** @brief Distinguish tensor-to-tensor copies from PP ownership handoffs. */
    enum class ActivationOperation { Copy, Handoff };
    /** @brief Discover real devices and create explicit producer streams. */
    void SetUp() override
    {
#ifdef HAVE_CUDA
        if (auto *cuda = getCUDABackend(); cuda != nullptr)
        {
            cuda_available_ = true;
            cuda_device_ = DeviceId::cuda(0);
            cuda_stream_ =
                std::make_unique<llaminar2::test::ScopedGPUStream>(cuda_device_);
            // Query the backend directly (hipGetDeviceCount/cudaGetDeviceCount) rather
            // than DeviceManager, which would require an explicit initialize() call.
            if (cuda->deviceCount() >= 2)
            {
                cuda_device_1_ = DeviceId::cuda(1);
                multi_cuda_ = true;
            }
        }
#endif
#ifdef HAVE_ROCM
        if (auto *rocm = getROCmBackend(); rocm != nullptr)
        {
            rocm_available_ = true;
            rocm_device_ = DeviceId::rocm(0);
            rocm_stream_ =
                std::make_unique<llaminar2::test::ScopedGPUStream>(rocm_device_);
            if (rocm->deviceCount() >= 2)
            {
                rocm_device_1_ = DeviceId::rocm(1);
                multi_rocm_ = true;
            }
        }
#endif

        // Same-vendor peer copies route through the collective backend router
        // (NCCL/RCCL or peer DMA). Initialize the global router so those paths
        // can be exercised when 2+ same-vendor GPUs are present.
        if (multi_cuda_ || multi_rocm_)
        {
            GlobalBackendRouter::initForTests();
        }
    }

    /** @brief Retire communicator owners before process-local registries exit. */
    void TearDown() override
    {
        GlobalBackendRouter::shutdown();
    }

    /** @return Exact producer stream owned by this fixture for the device. */
    void *streamFor(DeviceId device) const
    {
        if (device.type == DeviceType::CUDA && cuda_stream_)
            return cuda_stream_->get();
        if (device.type == DeviceType::ROCm && rocm_stream_)
            return rocm_stream_->get();
        throw std::runtime_error(
            "No explicit copyActivation test stream for requested GPU");
    }

    /// @brief Create an FP32 tensor filled with a deterministic ramp pattern.
    std::unique_ptr<FP32Tensor> makePatternTensor()
    {
        auto t = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 128});
        float *data = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
        {
            data[i] = static_cast<float>(i) * 0.001f;
        }
        return t;
    }

    /// @brief Create an FP32 tensor whose host buffer is filled with a poison
    ///        sentinel (-1.0f) that is neither zero nor any ramp value.
    ///
    /// Using a poison value (rather than zeros) makes verifyPattern() impossible
    /// to pass trivially: the test can only succeed if a real device→host
    /// download returns the data that was genuinely transferred into the
    /// destination's *device* buffer. If no transfer occurred, the host buffer
    /// would still hold the sentinel and verification would fail.
    std::unique_ptr<FP32Tensor> makePoisonTensor()
    {
        auto t = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 128});
        float *data = t->mutable_data();
        for (size_t i = 0; i < t->numel(); ++i)
        {
            data[i] = -1.0f;
        }
        return t;
    }

    /**
     * @brief Make a tensor authoritative on a GPU with its ramp pattern resident.
     *
     * Uploads the host pattern to the device and marks the device copy as the
     * authoritative source, so copyActivation will read from device memory.
     */
    bool makeGpuResident(FP32Tensor *t, DeviceId device)
    {
        void *stream = streamFor(device);
        if (!t->ensureOnDevice(device, stream))
        {
            return false;
        }
        TransferEngine::publishCurrentDeviceWrite(t, stream);
        return t->isDeviceAuthoritative(device);
    }

    /// @brief Sync a tensor to host and verify it matches the ramp pattern.
    bool verifyPattern(FP32Tensor *t)
    {
        if (!t->ensureOnHost())
        {
            return false;
        }
        const float *data = t->data();
        for (size_t i = 0; i < t->numel(); ++i)
        {
            if (std::abs(data[i] - static_cast<float>(i) * 0.001f) > 1e-6f)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Prove that an activation copy consumes its published generation.
     *
     * A mapped timeline holds the producer before its final write. The copy
     * must import that producer event; waiting for an unrelated copy stream
     * alone can finish successfully while copying the previous generation.
     * All allocations and initial uploads precede the gate so an allocator
     * synchronization cannot accidentally make the test pass.
     *
     * @param source_device Device holding the blocked activation producer.
     * @param destination_device Device receiving the activation.
     * @param operation Copy to another tensor or hand off this tensor's owner.
     */
    void provePendingProducerCopy(DeviceId source_device, DeviceId destination_device,
                                  ActivationOperation operation = ActivationOperation::Copy)
    {
        auto *backend = getBackendFor(source_device);
        ASSERT_NE(backend, nullptr);
        auto &transfers = TransferEngine::instance();
        const DeviceId devices[] = {source_device};
        auto control = transfers.allocateMappedHostRegion(64u, devices);
        auto *signal = static_cast<std::uint64_t *>(control->mutableHostData());
        std::atomic_ref<std::uint64_t>(*signal).store(0u, std::memory_order_release);
        auto src = makePatternTensor();
        auto dst = makePoisonTensor();
        // PP handoff storage and any collective communicator must exist before
        // holding the producer. Lazy setup otherwise hides a missing event.
        TransferEngine::allocateDeviceStorage(src.get(), destination_device);
        llaminar2::test::ScopedGPUStream destination_stream(destination_device);
        ASSERT_TRUE(makeGpuResident(src.get(), source_device));
        ASSERT_TRUE(dst->ensureOnDevice(destination_device, destination_stream.get()));
        ASSERT_TRUE(src->ensureOnHost());
        ASSERT_TRUE(dst->ensureOnHost());
        const auto bytes = src->numel() * sizeof(float);
        void *producer = streamFor(source_device);
        // Host verification above changes authority. Re-publish the resident
        // source so warmup actually exercises the intended GPU transport.
        TransferEngine::publishDeviceWrite(src.get(), source_device, producer);
        const auto warm = transfers.copyActivation(src.get(), dst.get(), destination_device, bytes);
        ASSERT_TRUE(warm.success) << warm.error;

        auto &context = GPUDeviceContextPool::instance().getContext(source_device);
        std::unique_ptr<IGPUGraphCapture> graph;
        context.submitAndWait([&] {
            graph = context.createGraphCapture(producer);
            if (!graph) throw std::runtime_error("Missing captured producer");
            ScopedBackendGraphCapture capture(context, *graph, "activation producer proof");
            if (!capture.begin() || !backend->streamWaitTimelineSignal64(
                    producer, control->deviceAlias(source_device), 1u, source_device.ordinal) ||
                !backend->memset(src->gpu_data_ptr(), 0x3f, bytes,
                                  source_device.ordinal, producer))
                throw std::runtime_error("Cannot capture the activation producer");
            capture.finish();
            if (!graph->instantiate()) throw std::runtime_error("Cannot instantiate producer");
        });

        // Release is test-owned and bounded even if an assertion or a copy
        // throws. The normal path releases only AFTER submission returned,
        // proving nonblocking progress without an inference-speed threshold.
        std::jthread release([signal](std::stop_token stop) {
            std::mutex mutex;
            std::unique_lock lock(mutex);
            std::condition_variable_any wake;
            wake.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
            std::atomic_ref<std::uint64_t>(*signal).store(1u, std::memory_order_release);
        });
        context.submitAndWait([&] {
            if (!graph->launch()) throw std::runtime_error("Cannot launch captured producer");
        });
        TransferEngine::publishDeviceWrite(src.get(), source_device, producer);
        const auto result = operation == ActivationOperation::Copy
            ? transfers.copyActivation(src.get(), dst.get(), destination_device, bytes)
            : transfers.transferActivation(src.get(), destination_device, bytes);
        ASSERT_TRUE(result.success) << result.error;
        EXPECT_EQ(std::atomic_ref<std::uint64_t>(*signal).load(std::memory_order_acquire), 0u)
            << "Transfer blocked waiting for the held producer";
        release.request_stop();
        release.join();
        auto *received = operation == ActivationOperation::Copy ? dst.get() : src.get();
        ASSERT_TRUE(received->ensureOnHost());
        const auto *actual = reinterpret_cast<const unsigned char *>(received->data());
        for (std::size_t byte = 0u; byte < bytes; ++byte)
        {
            ASSERT_EQ(actual[byte], 0x3f)
                << "Copied stale activation before its producer event at byte " << byte;
        }
    }

    bool cuda_available_ = false;
    bool rocm_available_ = false;
    bool multi_cuda_ = false;
    bool multi_rocm_ = false;

    DeviceId cuda_device_ = DeviceId::cpu();
    DeviceId cuda_device_1_ = DeviceId::cpu();
    DeviceId rocm_device_ = DeviceId::cpu();
    DeviceId rocm_device_1_ = DeviceId::cpu();
    std::unique_ptr<llaminar2::test::ScopedGPUStream> cuda_stream_;
    std::unique_ptr<llaminar2::test::ScopedGPUStream> rocm_stream_;
};

/** @brief CUDA copies must join a producer on a different explicit stream. */
TEST_F(Test__TransferEngine_CopyActivation, PendingProducer_CUDA)
{
    if (!cuda_available_) GTEST_SKIP() << "Need CUDA";
    provePendingProducerCopy(cuda_device_, cuda_device_);
}

/** @brief ROCm obeys the same generation/event contract as CUDA. */
TEST_F(Test__TransferEngine_CopyActivation, PendingProducer_ROCm)
{
    if (!rocm_available_) GTEST_SKIP() << "Need ROCm";
    provePendingProducerCopy(rocm_device_, rocm_device_);
}

/** @brief Both CUDA tensor copy and PP handoff import the cross-device producer. */
TEST_F(Test__TransferEngine_CopyActivation, PendingProducer_CUDA_Peer)
{
    if (!multi_cuda_) GTEST_SKIP() << "Need two CUDA devices";
    for (auto operation : {ActivationOperation::Copy, ActivationOperation::Handoff})
        provePendingProducerCopy(cuda_device_, cuda_device_1_, operation);
}

/** @brief Both ROCm tensor copy and PP handoff import the cross-device producer. */
TEST_F(Test__TransferEngine_CopyActivation, PendingProducer_ROCm_Peer)
{
    if (!multi_rocm_) GTEST_SKIP() << "Need two ROCm devices";
    for (auto operation : {ActivationOperation::Copy, ActivationOperation::Handoff})
        provePendingProducerCopy(rocm_device_, rocm_device_1_, operation);
}

// =============================================================================
// Host source → GPU (direct H2D)
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, HostSource_to_CUDA_H2D)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Source stays host-authoritative; destination is fresh.
    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, HostSource_to_ROCm_H2D)
{
    if (!rocm_available_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_TO_DEVICE);
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

// =============================================================================
// Same physical GPU → device-to-device copy
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, SamePhysicalGPU_CUDA_D2D)
{
    if (!cuda_available_)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Source resident & authoritative on cuda:0; destination targets cuda:0 too.
    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), cuda_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_));
    // Intra-GPU copy: both src and dst live on the same physical card.
    EXPECT_TRUE(src->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, SamePhysicalGPU_ROCm_D2D)
{
    if (!rocm_available_)
    {
        GTEST_SKIP() << "No ROCm device available";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), rocm_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_));
    // Intra-GPU copy: both src and dst live on the same physical card.
    EXPECT_TRUE(src->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

// =============================================================================
// Same-vendor, different GPU → peer copy (NCCL/RCCL or peer DMA)
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, CUDA_to_CUDA_Peer)
{
    if (!multi_cuda_)
    {
        GTEST_SKIP() << "Need 2+ CUDA devices";
    }
    if (GlobalBackendRouter::get() == nullptr)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Distinct physical cards: src on cuda:0, dst on cuda:1.
    ASSERT_NE(cuda_device_.gpu_ordinal(), cuda_device_1_.gpu_ordinal());

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), cuda_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_1_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    // dst landed on the *other* physical card; src stayed on its own card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_1_));
    EXPECT_FALSE(dst->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, ROCm_to_ROCm_Peer)
{
    if (!multi_rocm_)
    {
        GTEST_SKIP() << "Need 2+ ROCm devices";
    }
    if (GlobalBackendRouter::get() == nullptr)
    {
        GTEST_SKIP() << "GlobalBackendRouter not initialized";
    }

    // Distinct physical cards: src on rocm:0, dst on rocm:1.
    ASSERT_NE(rocm_device_.gpu_ordinal(), rocm_device_1_.gpu_ordinal());

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), rocm_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_1_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND);
    // dst landed on the *other* physical card; src stayed on its own card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_1_));
    EXPECT_FALSE(dst->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

// =============================================================================
// Cross-vendor (CUDA↔ROCm) → host-staged bounce
// =============================================================================

TEST_F(Test__TransferEngine_CopyActivation, CrossVendor_CUDA_to_ROCm_HostStaged)
{
    if (!cuda_available_ || !rocm_available_)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm devices";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), cuda_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), rocm_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_STAGED);
    // Source was on a CUDA card; destination is now authoritative on a ROCm card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}

TEST_F(Test__TransferEngine_CopyActivation, CrossVendor_ROCm_to_CUDA_HostStaged)
{
    if (!cuda_available_ || !rocm_available_)
    {
        GTEST_SKIP() << "Need both CUDA and ROCm devices";
    }

    auto src = makePatternTensor();
    auto dst = makePoisonTensor();
    ASSERT_TRUE(makeGpuResident(src.get(), rocm_device_));
    const size_t bytes = src->numel() * sizeof(float);

    auto result =
        TransferEngine::instance().copyActivation(src.get(), dst.get(), cuda_device_, bytes);

    ASSERT_TRUE(result.success) << result.error;
    EXPECT_EQ(result.method_used, TransferMethod::HOST_STAGED);
    // Source was on a ROCm card; destination is now authoritative on a CUDA card.
    EXPECT_TRUE(dst->isDeviceAuthoritative(cuda_device_));
    EXPECT_TRUE(src->isDeviceAuthoritative(rocm_device_));
    EXPECT_TRUE(verifyPattern(dst.get()));
}
