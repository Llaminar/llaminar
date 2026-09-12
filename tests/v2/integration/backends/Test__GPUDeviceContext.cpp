/**
 * @file Test__GPUDeviceContext.cpp
 * @brief Integration tests for GPU Device Context infrastructure
 *
 * Tests:
 * - IWorkerGPUContext interface via concrete implementations
 * - GPUDeviceContextPool singleton
 * - NvidiaDeviceContext (if CUDA available)
 * - AMDDeviceContext (if ROCm available)
 *
 * **Thread Safety Model**:
 * The device context follows a strict ownership model where all GPU state
 * is owned by a dedicated worker thread. Tests verify both thread-safe
 * methods (submitAndWait, submitAsync) and worker-thread-only methods
 * (stream/event creation, BLAS handle access).
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/moe/MoEOverlayLocalCapacityPlanner.h"
#include "planning/CapturedGraphMemoryEstimator.h"
#include "planning/MemoryPlanner.h"
#include "transfer/TransferEngine.h"

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#include <hip/hip_runtime.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace llaminar2;

// ===========================================================================
// Skip macros for hardware availability
// ===========================================================================

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define SKIP_IF_NO_CUDA()                                      \
    do                                                         \
    {                                                          \
        ensureNvidiaFactoryRegistered();                       \
        if (!GPUDeviceContextPool::instance().hasNvidiaSupport()) \
            GTEST_SKIP() << "CUDA not available";              \
    } while (false)
#else
#define SKIP_IF_NO_CUDA() GTEST_SKIP() << "CUDA backend not linked in this test binary"
#endif

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define SKIP_IF_NO_ROCM()                                   \
    do                                                      \
    {                                                       \
        ensureAMDFactoryRegistered();                       \
        if (!GPUDeviceContextPool::instance().hasAMDSupport()) \
            GTEST_SKIP() << "ROCm not available";           \
    } while (false)
#else
#define SKIP_IF_NO_ROCM() GTEST_SKIP() << "ROCm backend not linked in this test binary"
#endif

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define SKIP_IF_NO_GPU() SKIP_IF_NO_ROCM()
#elif defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define SKIP_IF_NO_GPU() SKIP_IF_NO_CUDA()
#else
#define SKIP_IF_NO_GPU() GTEST_SKIP() << "No GPU backend linked in this test binary"
#endif

namespace
{
    constexpr size_t kRuntimeRetirementAllocationBytes = 64u * 1024u * 1024u;

    void expectAuxiliaryStreamsReuseByName(IWorkerGPUContext &ctx)
    {
        ctx.resetAuxiliaryStreams();

        bool created = false;
        void *transfer_stream = ctx.getOrCreateAuxiliaryStream("moe_gpu_direct_transfer_test", &created);
        ASSERT_NE(transfer_stream, nullptr);
        EXPECT_TRUE(created);

        created = true;
        void *reused_transfer_stream = ctx.getOrCreateAuxiliaryStream("moe_gpu_direct_transfer_test", &created);
        EXPECT_EQ(reused_transfer_stream, transfer_stream);
        EXPECT_FALSE(created);

        created = false;
        void *other_stream = ctx.getOrCreateAuxiliaryStream("moe_gpu_direct_transfer_other", &created);
        ASSERT_NE(other_stream, nullptr);
        EXPECT_NE(other_stream, transfer_stream);
        EXPECT_TRUE(created);

        ctx.resetAuxiliaryStreams();

        created = false;
        void *after_reset = ctx.getOrCreateAuxiliaryStream("moe_gpu_direct_transfer_test", &created);
        ASSERT_NE(after_reset, nullptr);
        EXPECT_TRUE(created);

        ctx.resetAuxiliaryStreams();
    }

    /**
     * @brief Prove the complete public model-retirement transaction on a GPU.
     *
     * The live allocation is included in the ticket's exact BOM, then released
     * before completion just as the final model owner would be. Completion must
     * destroy the worker generation, reset the native runtime generation, and
     * leave both the canonical backend allocator and worker-context pool usable.
     */
    void expectExclusiveRuntimeGenerationRetirement(DeviceId device)
    {
        IBackend *const backend = getBackendFor(device);
        ASSERT_NE(backend, nullptr);

        auto &pool = GPUDeviceContextPool::instance();
        ASSERT_TRUE(pool.getContext(device).isInitialized());

        const std::uint64_t runtime_generation_before =
            backend->deviceRuntimeGeneration(device.gpu_ordinal());
        ASSERT_NE(runtime_generation_before, 0u);

        void *const allocation = backend->allocate(
            kRuntimeRetirementAllocationBytes,
            device.gpu_ordinal());
        ASSERT_NE(allocation, nullptr);

        TransferEngine engine;
        auto ticket = engine.beginExclusiveModelRetirement(
            ModelDeviceMemoryRetention{
                .device = device,
                .prepared_weight_bytes =
                    kRuntimeRetirementAllocationBytes,
                .reusable_workspace_bytes = 0u,
            });

        // Model owners release their canonical allocations between the two
        // typed retirement phases; the backend tracker must therefore be empty
        // before TransferEngine is permitted to reset the runtime generation.
        backend->free(allocation, device.gpu_ordinal());
        const DeviceMemoryReclamationReceipt receipt =
            engine.completeExclusiveModelRetirement(std::move(ticket));

        EXPECT_TRUE(receipt.runtime_reset_invoked);
        EXPECT_EQ(
            receipt.retired_runtime_generation,
            runtime_generation_before);
        EXPECT_EQ(
            receipt.successor_runtime_generation,
            runtime_generation_before + 1u);
        EXPECT_EQ(
            receipt.runtime_post_reset_state,
            DeviceRuntimePostResetState::Quiescent);
        EXPECT_EQ(
            backend->deviceRuntimeGeneration(device.gpu_ordinal()),
            runtime_generation_before + 1u);
        EXPECT_GE(
            receipt.releasedCanonicalBytes(),
            kRuntimeRetirementAllocationBytes);

        // Neither a stale worker nor a stale allocator handle may survive the
        // primary-context reset. Both authorities must lazily bind to the new
        // runtime generation and remain usable by the next model admission.
        ASSERT_TRUE(pool.getContext(device).isInitialized());
        void *const fresh_allocation =
            backend->allocate(4096u, device.gpu_ordinal());
        ASSERT_NE(fresh_allocation, nullptr);
        backend->free(fresh_allocation, device.gpu_ordinal());

        /*
         * A process campaign keeps its ClusterInventory while retiring one
         * model generation. Prove that the same typed mutation used by
         * production admission replaces that deliberately stale discovery
         * value with the newly recreated backend generation's live capacity.
         */
        RankInventory runtime_inventory;
        runtime_inventory.rank = 0;
        runtime_inventory.gpus.push_back(DeviceInfo{
            .type = device.type,
            .local_device_id = device.ordinal,
            .memory_bytes = 1u,
            .free_memory_bytes = 1u,
        });
        const std::size_t runtime_total_bytes =
            backend->deviceMemoryTotal(device.gpu_ordinal());
        const std::size_t runtime_free_bytes =
            backend->deviceMemoryFree(device.gpu_ordinal());
        ASSERT_GT(runtime_total_bytes, 0u);
        ASSERT_GT(runtime_free_bytes, 0u);
        installMoEOverlayRuntimeGPUCapacityObservation(
            runtime_inventory,
            device,
            runtime_total_bytes,
            runtime_free_bytes);
        ASSERT_EQ(runtime_inventory.gpus.size(), 1u);
        EXPECT_EQ(
            runtime_inventory.gpus.front().memory_bytes,
            runtime_total_bytes);
        EXPECT_EQ(
            runtime_inventory.gpus.front().free_memory_bytes,
            runtime_free_bytes);
    }

    /**
     * @brief Prove one model retires a same-vendor two-GPU set as one edge.
     *
     * Both workers and both canonical allocations are materialized before any
     * ticket is captured. Completion must destroy/exclude both worker contexts
     * before the first native reset, preserve ticket order in its receipts,
     * advance each device generation exactly once, and leave both endpoints
     * immediately usable by the next model admission.
     *
     * @param first First exact physical participant.
     * @param second Second exact physical participant of the same backend.
     */
    void expectExclusiveMultiDeviceRuntimeGenerationRetirement(
        DeviceId first,
        DeviceId second)
    {
        ASSERT_TRUE(first.is_gpu());
        ASSERT_TRUE(second.is_gpu());
        ASSERT_EQ(first.type, second.type);
        ASSERT_NE(first, second);

        std::array<DeviceId, 2> devices{first, second};
        std::array<IBackend *, 2> backends{};
        std::array<std::uint64_t, 2> generations{};
        std::array<void *, 2> allocations{};
        auto &pool = GPUDeviceContextPool::instance();
        for (size_t index = 0u; index < devices.size(); ++index)
        {
            backends[index] = getBackendFor(devices[index]);
            ASSERT_NE(backends[index], nullptr);
            ASSERT_TRUE(pool.getContext(devices[index]).isInitialized());
            generations[index] = backends[index]->deviceRuntimeGeneration(
                devices[index].gpu_ordinal());
            ASSERT_NE(generations[index], 0u);
            allocations[index] = backends[index]->allocate(
                kRuntimeRetirementAllocationBytes,
                devices[index].gpu_ordinal());
            ASSERT_NE(allocations[index], nullptr);
        }

        TransferEngine engine;
        std::vector<ExclusiveModelRetirementTicket> tickets;
        tickets.reserve(devices.size());
        for (const DeviceId device : devices)
        {
            tickets.push_back(engine.beginExclusiveModelRetirement(
                ModelDeviceMemoryRetention{
                    .device = device,
                    .prepared_weight_bytes =
                        kRuntimeRetirementAllocationBytes,
                    .reusable_workspace_bytes = 0u,
                }));
        }

        /* Release every final model allocation as one ownership boundary. No
         * native runtime may reset until the complete participant set is free. */
        for (size_t index = 0u; index < devices.size(); ++index)
        {
            backends[index]->free(
                allocations[index],
                devices[index].gpu_ordinal());
        }

        const auto receipts = engine.completeExclusiveModelRetirements(
            std::move(tickets));
        ASSERT_EQ(receipts.size(), devices.size());
        for (size_t index = 0u; index < devices.size(); ++index)
        {
            EXPECT_EQ(receipts[index].device, devices[index]);
            EXPECT_TRUE(receipts[index].runtime_reset_invoked);
            EXPECT_EQ(
                receipts[index].retired_runtime_generation,
                generations[index]);
            EXPECT_EQ(
                receipts[index].successor_runtime_generation,
                generations[index] + 1u);
            EXPECT_EQ(
                receipts[index].runtime_post_reset_state,
                DeviceRuntimePostResetState::Quiescent);
            EXPECT_EQ(
                backends[index]->deviceRuntimeGeneration(
                    devices[index].gpu_ordinal()),
                generations[index] + 1u);
            EXPECT_GE(
                receipts[index].releasedCanonicalBytes(),
                kRuntimeRetirementAllocationBytes);

            ASSERT_TRUE(pool.getContext(devices[index]).isInitialized());
            void *const fresh_allocation = backends[index]->allocate(
                4096u, devices[index].gpu_ordinal());
            ASSERT_NE(fresh_allocation, nullptr);
            backends[index]->free(
                fresh_allocation, devices[index].gpu_ordinal());
        }
    }

    /**
     * @brief Prove acquisition remains excluded for the whole retirement scope.
     */
    void expectContextAcquisitionExcludedDuringRetirement(DeviceId device)
    {
        auto &pool = GPUDeviceContextPool::instance();
        ASSERT_TRUE(pool.getContext(device).isInitialized());

        {
            auto retirement =
                pool.beginExclusiveGenerationRetirement(device);
            EXPECT_TRUE(retirement.receipt().retiredLiveContext());
            EXPECT_THROW((void)pool.getContext(device), std::logic_error);
        }

        ASSERT_TRUE(pool.getContext(device).isInitialized());
    }
} // namespace

// ===========================================================================
// GPUDeviceContextPool Tests
// ===========================================================================

/** @test Retained MTP capacity owns every general/helper slot before execution. */
TEST(Test__GPUDeviceContextPool,
     RetainedMTPGraphCapacityIsPricedBeforeExecution)
{
    MTPRuntimeConfig retained_mtp;
    retained_mtp.enabled = false;
    retained_mtp.graph_capacity_draft_tokens = 15;

    const CapturedServingGraphMemoryInventory inventory =
        resolveCapturedServingGraphMemoryInventory(
            {32, 64, 128},
            retained_mtp);
    const MTPGraphOwnerPlan owner_plan(retained_mtp);
    ASSERT_EQ(inventory.fixed_executable_count, 6u);
    ASSERT_EQ(
        inventory.mtp_graph_owners.auxiliaryExecutableSlotCount(),
        107u);
    ASSERT_EQ(owner_plan.generalAuxiliaryExecutableSlotCount(), 25u);
    ASSERT_EQ(owner_plan.boundedHelperExecutableSlotCount(), 82u);

    ModelMemoryProfile profile;
    // Workspace planning consults the production architecture's typed
    // sharding policy even when this fixture carries no weight inventory.
    // Keep the synthetic geometry attached to a registered dense schema so
    // this integration test exercises the same planner boundary as admission.
    profile.architecture = "qwen3";
    profile.n_layers = 64;
    profile.d_model = 1024;
    profile.d_ff = 4096;
    profile.n_heads = 16;
    profile.n_kv_heads = 4;
    profile.head_dim = 64;
    profile.vocab_size = 32000;
    profile.max_seq_len = 4096;

    DevicePlanConfig cfg;
#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
    cfg.device = DeviceId::rocm(0);
#else
    cfg.device = DeviceId::cuda(0);
#endif
    cfg.device_compute_units = 1;
    cfg.device_total_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    cfg.device_free_bytes = cfg.device_total_bytes;
    cfg.batch_size = 1;
    cfg.max_seq_len = 4096;
    cfg.activation_seq_len = 64;
    cfg.kv_precision = "fp16";
    cfg.captured_serving_graphs = inventory;

    const MemoryPlan plan = MemoryPlanner::plan(profile, {cfg});
    ASSERT_EQ(plan.devices.size(), 1u);
    EXPECT_EQ(
        plan.devices.front().captured_graph_bytes(),
        estimateCapturedGraphExecutableBytes(
            cfg.device,
            CapturedGraphExecutableInventory{
                .model_graph_identity_count =
                    /*two prefill + decode + prefix bridge + four MTP forwards=*/8u,
                .model_graph_topology_variant_count = 1u,
                // Match the physical classes declared by runtime graph owners;
                // the total slot count alone overprices bounded CUDA helpers.
                .auxiliary_executable_count =
                    owner_plan.generalAuxiliaryExecutableSlotCount(),
                .bounded_helper_executable_count =
                    owner_plan.boundedHelperExecutableSlotCount(),
            }));
    EXPECT_GT(
        plan.devices.front().captured_graph_bytes(),
        estimateCapturedGraphExecutableBytes(
            cfg.device,
            /*two prefill + decode + prefix bridge + four MTP forwards=*/8u));
}

TEST(Test__GPUDeviceContextPool, SingletonInstance)
{
    auto &pool1 = GPUDeviceContextPool::instance();
    auto &pool2 = GPUDeviceContextPool::instance();
    EXPECT_EQ(&pool1, &pool2) << "Pool should be a singleton";
}

TEST(Test__GPUDeviceContextPool, HasSupportQueries)
{
    auto &pool = GPUDeviceContextPool::instance();

    // These should not throw, just return bool
    bool has_nvidia = pool.hasNvidiaSupport();
    bool has_amd = pool.hasAMDSupport();

    // Log what's available
    std::cout << "NVIDIA (CUDA) support: " << (has_nvidia ? "yes" : "no") << std::endl;
    std::cout << "AMD (ROCm) support: " << (has_amd ? "yes" : "no") << std::endl;

    // Not asserting either is true - could be CPU-only build
    (void)has_nvidia;
    (void)has_amd;
}

TEST(Test__GPUDeviceContextPool, DeviceCountQueries)
{
    auto &pool = GPUDeviceContextPool::instance();

    int nvidia_count = pool.nvidiaDeviceCount();
    int amd_count = pool.amdDeviceCount();

    std::cout << "NVIDIA device count: " << nvidia_count << std::endl;
    std::cout << "AMD device count: " << amd_count << std::endl;

    // Device count should be non-negative
    EXPECT_GE(nvidia_count, 0);
    EXPECT_GE(amd_count, 0);

    // If support is available, should have at least one device
    if (pool.hasNvidiaSupport())
    {
        EXPECT_GE(nvidia_count, 1) << "hasNvidiaSupport() true but no devices";
    }
    if (pool.hasAMDSupport())
    {
        EXPECT_GE(amd_count, 1) << "hasAMDSupport() true but no devices";
    }
}

TEST(Test__GPUDeviceContextPool, GetContextByType_CUDA)
{
    SKIP_IF_NO_CUDA();

    auto &pool = GPUDeviceContextPool::instance();

    // Test various CUDA type strings
    auto &ctx1 = pool.getContext("cuda", 0);
    EXPECT_TRUE(ctx1.isInitialized());

    auto &ctx2 = pool.getContext("CUDA", 0);
    EXPECT_TRUE(ctx2.isInitialized());

    // Both should return the same context (same device ordinal)
    EXPECT_EQ(&ctx1, &ctx2);
}

TEST(Test__GPUDeviceContextPool, GetContextByType_ROCm)
{
    SKIP_IF_NO_ROCM();

    auto &pool = GPUDeviceContextPool::instance();

    // Test various ROCm type strings
    auto &ctx1 = pool.getContext("rocm", 0);
    EXPECT_TRUE(ctx1.isInitialized());

    auto &ctx2 = pool.getContext("ROCm", 0);
    EXPECT_TRUE(ctx2.isInitialized());

    auto &ctx3 = pool.getContext("hip", 0);
    EXPECT_TRUE(ctx3.isInitialized());

    auto &ctx4 = pool.getContext("HIP", 0);
    EXPECT_TRUE(ctx4.isInitialized());

    // All should return the same context (same device ordinal)
    EXPECT_EQ(&ctx1, &ctx2);
    EXPECT_EQ(&ctx2, &ctx3);
    EXPECT_EQ(&ctx3, &ctx4);
}

TEST(Test__GPUDeviceContextPool, GetContextInvalidType)
{
    auto &pool = GPUDeviceContextPool::instance();

    // Invalid device type should throw
    EXPECT_THROW(pool.getContext("invalid_type", 0), std::invalid_argument);
    EXPECT_THROW(pool.getContext("vulkan", 0), std::invalid_argument);
    EXPECT_THROW(pool.getContext("", 0), std::invalid_argument);
}

TEST(Test__GPUDeviceContextPool, GetContextInvalidOrdinal)
{
    auto &pool = GPUDeviceContextPool::instance();

    // Negative ordinal should throw
    if (pool.hasNvidiaSupport())
    {
        EXPECT_THROW(pool.getNvidiaContext(-1), std::runtime_error);
    }
    if (pool.hasAMDSupport())
    {
        EXPECT_THROW(pool.getAMDContext(-1), std::runtime_error);
    }

    // Ordinal beyond device count should throw
    if (pool.hasNvidiaSupport())
    {
        int count = pool.nvidiaDeviceCount();
        EXPECT_THROW(pool.getNvidiaContext(count + 100), std::runtime_error);
    }
    if (pool.hasAMDSupport())
    {
        int count = pool.amdDeviceCount();
        EXPECT_THROW(pool.getAMDContext(count + 100), std::runtime_error);
    }
}

TEST(Test__GPUDeviceContextPool, ExclusiveRetirementRejectsCPU)
{
    auto &pool = GPUDeviceContextPool::instance();
    EXPECT_THROW(
        (void)pool.retireExclusiveGeneration(DeviceId::cpu()),
        std::invalid_argument);
}

TEST(Test__GPUDeviceContextPool, CUDAExclusiveRetirementCreatesFreshGeneration)
{
    SKIP_IF_NO_CUDA();

    auto &pool = GPUDeviceContextPool::instance();
    ASSERT_TRUE(pool.getNvidiaContext(0).isInitialized());

    const auto first =
        pool.retireExclusiveGeneration(DeviceId::cuda(0));
    ASSERT_TRUE(first.retiredLiveContext());
    EXPECT_EQ(first.device, DeviceId::cuda(0));

    // Lazy acquisition after the terminal edge must materialize a distinct
    // lifecycle generation even if the host allocator reuses an object address.
    ASSERT_TRUE(pool.getNvidiaContext(0).isInitialized());
    const auto second =
        pool.retireExclusiveGeneration(DeviceId::cuda(0));
    EXPECT_GT(second.retired_generation, first.retired_generation);

    // Leave a live context for the remaining backend integration cases.
    ASSERT_TRUE(pool.getNvidiaContext(0).isInitialized());
}

TEST(Test__GPUDeviceContextPool, ROCmExclusiveRetirementCreatesFreshGeneration)
{
    SKIP_IF_NO_ROCM();

    auto &pool = GPUDeviceContextPool::instance();
    ASSERT_TRUE(pool.getAMDContext(0).isInitialized());

    const auto first =
        pool.retireExclusiveGeneration(DeviceId::rocm(0));
    ASSERT_TRUE(first.retiredLiveContext());
    EXPECT_EQ(first.device, DeviceId::rocm(0));

    ASSERT_TRUE(pool.getAMDContext(0).isInitialized());
    const auto second =
        pool.retireExclusiveGeneration(DeviceId::rocm(0));
    EXPECT_GT(second.retired_generation, first.retired_generation);

    // Later tests consume the ordinary live generation, not a retired handle.
    ASSERT_TRUE(pool.getAMDContext(0).isInitialized());
}

TEST(Test__GPUDeviceContextPool,
     CUDAAcquisitionIsExcludedAcrossRuntimeRetirementScope)
{
    SKIP_IF_NO_CUDA();
    expectContextAcquisitionExcludedDuringRetirement(DeviceId::cuda(0));
}

TEST(Test__GPUDeviceContextPool,
     ROCmAcquisitionIsExcludedAcrossRuntimeRetirementScope)
{
    SKIP_IF_NO_ROCM();
    expectContextAcquisitionExcludedDuringRetirement(DeviceId::rocm(0));
}

TEST(Test__GPUDeviceContextPool,
     CUDAExclusiveModelRetirementResetsRuntimeGeneration)
{
    SKIP_IF_NO_CUDA();
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SCOPED_TRACE("CUDA retirement/admission iteration " +
                     std::to_string(iteration));
        expectExclusiveRuntimeGenerationRetirement(DeviceId::cuda(0));
    }
}

TEST(Test__GPUDeviceContextPool,
     ROCmExclusiveModelRetirementResetsRuntimeGeneration)
{
    SKIP_IF_NO_ROCM();
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SCOPED_TRACE("ROCm retirement/admission iteration " +
                     std::to_string(iteration));
        expectExclusiveRuntimeGenerationRetirement(DeviceId::rocm(0));
    }
}

TEST(Test__GPUDeviceContextPool,
     CUDAExclusiveMultiDeviceModelRetirementIsStableForTwentyCycles)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    if (pool.nvidiaDeviceCount() < 2)
        GTEST_SKIP() << "Two CUDA devices are required";

    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SCOPED_TRACE("CUDA batch retirement/admission iteration " +
                     std::to_string(iteration));
        expectExclusiveMultiDeviceRuntimeGenerationRetirement(
            DeviceId::cuda(0), DeviceId::cuda(1));
    }
}

TEST(Test__GPUDeviceContextPool,
     ROCmExclusiveMultiDeviceModelRetirementIsStableForTwentyCycles)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    if (pool.amdDeviceCount() < 2)
        GTEST_SKIP() << "Two ROCm devices are required";

    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SCOPED_TRACE("ROCm batch retirement/admission iteration " +
                     std::to_string(iteration));
        expectExclusiveMultiDeviceRuntimeGenerationRetirement(
            DeviceId::rocm(0), DeviceId::rocm(1));
    }
}

TEST(Test__GPUDeviceContextPool, ConcurrentAccess)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();

    // Determine which backend to test
    bool use_cuda = pool.hasNvidiaSupport();
    const std::string device_type = use_cuda ? "cuda" : "rocm";

    constexpr int NUM_THREADS = 8;
    std::vector<IWorkerGPUContext *> contexts(NUM_THREADS, nullptr);
    std::vector<std::thread> threads;

    // Multiple threads requesting the same context
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&, i]()
                             { contexts[i] = &pool.getContext(device_type, 0); });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    // All threads should get the same context
    for (int i = 1; i < NUM_THREADS; ++i)
    {
        EXPECT_EQ(contexts[0], contexts[i])
            << "Thread " << i << " got different context than thread 0";
    }
}

// ===========================================================================
// NvidiaDeviceContext Tests
// ===========================================================================

TEST(Test__NvidiaDeviceContext, CreationAndInitialization)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    EXPECT_TRUE(ctx.isInitialized());
    EXPECT_EQ(ctx.deviceOrdinal(), 0);
    EXPECT_FALSE(ctx.deviceName().empty());

    std::cout << "NVIDIA device 0: " << ctx.deviceName() << std::endl;
}

TEST(Test__NvidiaDeviceContext, SubmitAndWait)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    std::atomic<bool> work_executed{false};
    ctx.submitAndWait([&]()
                      { work_executed.store(true); });

    EXPECT_TRUE(work_executed.load()) << "Work should have executed";
}

TEST(Test__NvidiaDeviceContext, SubmitAndWaitReturnValue)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Verify we can capture return values via lambda capture
    int result = 0;
    ctx.submitAndWait([&]()
                      { result = 42; });

    EXPECT_EQ(result, 42);
}

TEST(Test__NvidiaDeviceContext, SubmitAsync)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    std::atomic<int> counter{0};
    auto future = ctx.submitAsync([&]()
                                  { counter.fetch_add(1); });

    future.wait();
    EXPECT_EQ(counter.load(), 1);
}

TEST(Test__NvidiaDeviceContext, MultipleAsyncSubmissions)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    constexpr int NUM_TASKS = 100;
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&]()
                                          { counter.fetch_add(1); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    EXPECT_EQ(counter.load(), NUM_TASKS);
}

TEST(Test__NvidiaDeviceContext, SubmitAsyncPreserveOrder)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Verify FIFO ordering - tasks should execute in submission order
    std::vector<int> execution_order;
    std::mutex mutex;

    constexpr int NUM_TASKS = 50;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&, i]()
                                          {
            std::lock_guard<std::mutex> lock(mutex);
            execution_order.push_back(i); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    ASSERT_EQ(execution_order.size(), static_cast<size_t>(NUM_TASKS));
    for (int i = 0; i < NUM_TASKS; ++i)
    {
        EXPECT_EQ(execution_order[i], i) << "Task " << i << " executed out of order";
    }
}

TEST(Test__NvidiaDeviceContext, StreamCreationAndDestruction)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *stream = nullptr;
    ctx.submitAndWait([&]()
                      { stream = ctx.createStream(); });

    EXPECT_NE(stream, nullptr) << "createStream() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyStream(stream); });

    // No assertion after destroy - just verify it doesn't crash
}

TEST(Test__NvidiaDeviceContext, MultipleStreams)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    constexpr int NUM_STREAMS = 4;
    std::vector<void *> streams(NUM_STREAMS, nullptr);

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            streams[i] = ctx.createStream();
            EXPECT_NE(streams[i], nullptr) << "Stream " << i << " is null";
        } });

    // All streams should be unique
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        for (int j = i + 1; j < NUM_STREAMS; ++j)
        {
            EXPECT_NE(streams[i], streams[j])
                << "Streams " << i << " and " << j << " are the same";
        }
    }

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            ctx.destroyStream(streams[i]);
        } });
}

TEST(Test__NvidiaDeviceContext, AuxiliaryStreamReusesByNameAndResets)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);
    expectAuxiliaryStreamsReuseByName(ctx);
}

TEST(Test__NvidiaDeviceContext, DefaultStream)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *default_stream = nullptr;
    ctx.submitAndWait([&]()
                      { default_stream = ctx.defaultStream(); });

    EXPECT_NE(default_stream, nullptr)
        << "The context-owned execution stream must never be CUDA's implicit default stream";
}

TEST(Test__NvidiaDeviceContext, EventCreationAndDestruction)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *event = nullptr;
    ctx.submitAndWait([&]()
                      { event = ctx.createEvent(); });

    EXPECT_NE(event, nullptr) << "createEvent() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyEvent(event); });
}

TEST(Test__NvidiaDeviceContext, EventRecordAndSynchronize)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    ctx.submitAndWait([&]()
                      {
        void *stream = ctx.createStream();
        void *event = ctx.createEvent();
        ASSERT_NE(stream, nullptr);
        ASSERT_NE(event, nullptr);
        EXPECT_THROW(ctx.recordEvent(event, nullptr), std::invalid_argument);
        ctx.recordEvent(event, stream);
        ctx.synchronizeEvent(event);
        ctx.destroyEvent(event);
        ctx.destroyStream(stream); });
}

TEST(Test__NvidiaDeviceContext, EventQueryCheckedReportsCompletion)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    bool query_before_sync_ok = false;
    bool ready_before_sync = false;
    bool query_after_sync_ok = false;
    bool ready_after_sync = false;

    ctx.submitAndWait([&]()
                      {
        void* stream = ctx.createStream();
        void* event = ctx.createEvent();
        ASSERT_NE(stream, nullptr);
        ASSERT_NE(event, nullptr);

        ASSERT_TRUE(ctx.recordEventChecked(event, stream));
        query_before_sync_ok = ctx.queryEventChecked(event, ready_before_sync);
        ctx.synchronizeEvent(event);
        query_after_sync_ok = ctx.queryEventChecked(event, ready_after_sync);

        ctx.destroyEvent(event);
        ctx.destroyStream(stream); });

    EXPECT_TRUE(query_before_sync_ok);
    EXPECT_TRUE(query_after_sync_ok);
    EXPECT_TRUE(ready_after_sync);
}

TEST(Test__NvidiaDeviceContext, ExactStreamQueryIsTypedAndNonBlocking)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);
    ctx.submitAndWait([&]()
                      {
        void *stream = ctx.createStream();
        ASSERT_NE(stream, nullptr);

        const GPUStreamExecutionState before_fence =
            ctx.queryStreamExecutionState(stream, "cuda stream query integration test");
        EXPECT_TRUE(before_fence == GPUStreamExecutionState::Pending ||
                    before_fence == GPUStreamExecutionState::Complete);

        ASSERT_TRUE(ctx.synchronizeStreamChecked(stream));
        EXPECT_EQ(
            ctx.queryStreamExecutionState(stream, "cuda stream query completed fence"),
            GPUStreamExecutionState::Complete);
        EXPECT_THROW(
            ctx.queryStreamExecutionState(nullptr, "cuda null stream"),
            std::invalid_argument);
        EXPECT_THROW(
            ctx.queryStreamExecutionState(stream, {}),
            std::invalid_argument);

        ctx.destroyStream(stream); });
}

TEST(Test__NvidiaDeviceContext, EventWait)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    ctx.submitAndWait([&]()
                      {
        void* stream1 = ctx.createStream();
        void* stream2 = ctx.createStream();
        void* event = ctx.createEvent();

        // Record event on stream1
        ctx.recordEvent(event, stream1);

        // Make stream2 wait for the event
        ctx.waitEvent(event, stream2);

        // Cleanup
        ctx.destroyEvent(event);
        ctx.destroyStream(stream1);
        ctx.destroyStream(stream2); });
}

TEST(Test__NvidiaDeviceContext, BlasHandle)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    void *handle = nullptr;
    ctx.submitAndWait([&]()
                      { handle = ctx.blasHandle(); });

    EXPECT_NE(handle, nullptr) << "BLAS handle should be available";
}

TEST(Test__NvidiaDeviceContext, Synchronize)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Synchronize should not throw
    ctx.synchronize();

    // Synchronize after some work
    ctx.submitAsync([&]()
                    {
        // Simulate some GPU work
        void* stream = ctx.createStream();
        ctx.destroyStream(stream); });

    ctx.synchronize();
}

TEST(Test__NvidiaDeviceContext, CollectiveCommInitiallyNull)
{
    SKIP_IF_NO_CUDA();

    auto &ctx = GPUDeviceContextPool::instance().getNvidiaContext(0);

    // Initially should be null (not set until collective backend initializes)
    void *comm = ctx.collectiveComm();
    // Don't assert null - it may have been set by previous tests
    (void)comm;
}

// ===========================================================================
// AMDDeviceContext Tests (mirror NVIDIA tests)
// ===========================================================================

TEST(Test__AMDDeviceContext, CreationAndInitialization)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    EXPECT_TRUE(ctx.isInitialized());
    EXPECT_EQ(ctx.deviceOrdinal(), 0);
    EXPECT_FALSE(ctx.deviceName().empty());

    std::cout << "AMD device 0: " << ctx.deviceName() << std::endl;
}

TEST(Test__AMDDeviceContext, SubmitAndWait)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    std::atomic<bool> work_executed{false};
    ctx.submitAndWait([&]()
                      { work_executed.store(true); });

    EXPECT_TRUE(work_executed.load()) << "Work should have executed";
}

TEST(Test__AMDDeviceContext, SubmitAndWaitReturnValue)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    int result = 0;
    ctx.submitAndWait([&]()
                      { result = 42; });

    EXPECT_EQ(result, 42);
}

TEST(Test__AMDDeviceContext, SubmitAsync)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    std::atomic<int> counter{0};
    auto future = ctx.submitAsync([&]()
                                  { counter.fetch_add(1); });

    future.wait();
    EXPECT_EQ(counter.load(), 1);
}

TEST(Test__AMDDeviceContext, MultipleAsyncSubmissions)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    constexpr int NUM_TASKS = 100;
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&]()
                                          { counter.fetch_add(1); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    EXPECT_EQ(counter.load(), NUM_TASKS);
}

TEST(Test__AMDDeviceContext, SubmitAsyncPreserveOrder)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    std::vector<int> execution_order;
    std::mutex mutex;

    constexpr int NUM_TASKS = 50;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        futures.push_back(ctx.submitAsync([&, i]()
                                          {
            std::lock_guard<std::mutex> lock(mutex);
            execution_order.push_back(i); }));
    }

    for (auto &f : futures)
    {
        f.wait();
    }

    ASSERT_EQ(execution_order.size(), static_cast<size_t>(NUM_TASKS));
    for (int i = 0; i < NUM_TASKS; ++i)
    {
        EXPECT_EQ(execution_order[i], i) << "Task " << i << " executed out of order";
    }
}

TEST(Test__AMDDeviceContext, StreamCreationAndDestruction)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *stream = nullptr;
    ctx.submitAndWait([&]()
                      { stream = ctx.createStream(); });

    EXPECT_NE(stream, nullptr) << "createStream() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyStream(stream); });
}

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
TEST(Test__AMDDeviceContext, StreamCreationUsesContextDeviceWhenCallerDeviceDiffers)
{
    SKIP_IF_NO_ROCM();

    auto &pool = GPUDeviceContextPool::instance();
    if (pool.amdDeviceCount() < 2)
    {
        GTEST_SKIP() << "requires at least two ROCm devices";
    }

    int original_device = 0;
    (void)hipGetDevice(&original_device);

    auto &ctx = pool.getAMDContext(0);

    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    void *stream = ctx.createStream();
    ASSERT_NE(stream, nullptr) << "createStream() should return non-null";

    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    int *device_word = nullptr;
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&device_word), sizeof(int)), hipSuccess);

    hipError_t memset_err = hipMemsetAsync(
        device_word,
        0,
        sizeof(int),
        static_cast<hipStream_t>(stream));
    EXPECT_EQ(memset_err, hipSuccess) << hipGetErrorString(memset_err);
    if (memset_err == hipSuccess)
    {
        EXPECT_EQ(hipStreamSynchronize(static_cast<hipStream_t>(stream)), hipSuccess);
    }

    ASSERT_EQ(hipFree(device_word), hipSuccess);
    ctx.destroyStream(stream);

    if (original_device >= 0)
    {
        (void)hipSetDevice(original_device);
    }
}
#endif

TEST(Test__AMDDeviceContext, MultipleStreams)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    constexpr int NUM_STREAMS = 4;
    std::vector<void *> streams(NUM_STREAMS, nullptr);

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            streams[i] = ctx.createStream();
            EXPECT_NE(streams[i], nullptr) << "Stream " << i << " is null";
        } });

    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        for (int j = i + 1; j < NUM_STREAMS; ++j)
        {
            EXPECT_NE(streams[i], streams[j])
                << "Streams " << i << " and " << j << " are the same";
        }
    }

    ctx.submitAndWait([&]()
                      {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            ctx.destroyStream(streams[i]);
        } });
}

TEST(Test__AMDDeviceContext, AuxiliaryStreamReusesByNameAndResets)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);
    expectAuxiliaryStreamsReuseByName(ctx);
}

TEST(Test__AMDDeviceContext, DefaultStream)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *default_stream = nullptr;
    ctx.submitAndWait([&]()
                      { default_stream = ctx.defaultStream(); });

    EXPECT_NE(default_stream, nullptr)
        << "The context-owned execution stream must never be HIP's implicit default stream";
}

TEST(Test__AMDDeviceContext, EventCreationAndDestruction)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *event = nullptr;
    ctx.submitAndWait([&]()
                      { event = ctx.createEvent(); });

    EXPECT_NE(event, nullptr) << "createEvent() should return non-null";

    ctx.submitAndWait([&]()
                      { ctx.destroyEvent(event); });
}

TEST(Test__AMDDeviceContext, EventRecordAndSynchronize)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    ctx.submitAndWait([&]()
                      {
        void *stream = ctx.createStream();
        void *event = ctx.createEvent();
        ASSERT_NE(stream, nullptr);
        ASSERT_NE(event, nullptr);
        EXPECT_THROW(ctx.recordEvent(event, nullptr), std::invalid_argument);
        ctx.recordEvent(event, stream);
        ctx.synchronizeEvent(event);
        ctx.destroyEvent(event);
        ctx.destroyStream(stream); });
}

TEST(Test__AMDDeviceContext, EventQueryCheckedReportsCompletion)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    bool query_before_sync_ok = false;
    bool ready_before_sync = false;
    bool query_after_sync_ok = false;
    bool ready_after_sync = false;

    ctx.submitAndWait([&]()
                      {
        void* stream = ctx.createStream();
        void* event = ctx.createEvent();
        ASSERT_NE(stream, nullptr);
        ASSERT_NE(event, nullptr);

        ASSERT_TRUE(ctx.recordEventChecked(event, stream));
        query_before_sync_ok = ctx.queryEventChecked(event, ready_before_sync);
        ctx.synchronizeEvent(event);
        query_after_sync_ok = ctx.queryEventChecked(event, ready_after_sync);

        ctx.destroyEvent(event);
        ctx.destroyStream(stream); });

    EXPECT_TRUE(query_before_sync_ok);
    EXPECT_TRUE(query_after_sync_ok);
    EXPECT_TRUE(ready_after_sync);
}

TEST(Test__AMDDeviceContext, ExactStreamQueryIsTypedAndNonBlocking)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);
    ctx.submitAndWait([&]()
                      {
        void *stream = ctx.createStream();
        ASSERT_NE(stream, nullptr);

        const GPUStreamExecutionState before_fence =
            ctx.queryStreamExecutionState(stream, "rocm stream query integration test");
        EXPECT_TRUE(before_fence == GPUStreamExecutionState::Pending ||
                    before_fence == GPUStreamExecutionState::Complete);

        ASSERT_TRUE(ctx.synchronizeStreamChecked(stream));
        EXPECT_EQ(
            ctx.queryStreamExecutionState(stream, "rocm stream query completed fence"),
            GPUStreamExecutionState::Complete);
        EXPECT_THROW(
            ctx.queryStreamExecutionState(nullptr, "rocm null stream"),
            std::invalid_argument);
        EXPECT_THROW(
            ctx.queryStreamExecutionState(stream, {}),
            std::invalid_argument);

        ctx.destroyStream(stream); });
}

TEST(Test__AMDDeviceContext, EventWait)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    ctx.submitAndWait([&]()
                      {
        void* stream1 = ctx.createStream();
        void* stream2 = ctx.createStream();
        void* event = ctx.createEvent();

        ctx.recordEvent(event, stream1);
        ctx.waitEvent(event, stream2);

        ctx.destroyEvent(event);
        ctx.destroyStream(stream1);
        ctx.destroyStream(stream2); });
}

TEST(Test__AMDDeviceContext, BlasHandle)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *handle = nullptr;
    ctx.submitAndWait([&]()
                      { handle = ctx.blasHandle(); });

    EXPECT_NE(handle, nullptr) << "BLAS handle should be available";
}

TEST(Test__AMDDeviceContext, Synchronize)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    ctx.synchronize();

    ctx.submitAsync([&]()
                    {
        void* stream = ctx.createStream();
        ctx.destroyStream(stream); });

    ctx.synchronize();
}

TEST(Test__AMDDeviceContext, CollectiveCommInitiallyNull)
{
    SKIP_IF_NO_ROCM();

    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);

    void *comm = ctx.collectiveComm();
    (void)comm;
}

// ===========================================================================
// Cross-Platform Interface Tests
// ===========================================================================

TEST(Test__IWorkerGPUContext, InterfaceConsistency)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();

    // Get any available context
    IWorkerGPUContext *ctx = nullptr;
    if (pool.hasNvidiaSupport())
    {
        ctx = &pool.getNvidiaContext(0);
    }
    else if (pool.hasAMDSupport())
    {
        ctx = &pool.getAMDContext(0);
    }

    ASSERT_NE(ctx, nullptr);

    // Verify interface methods are callable
    EXPECT_TRUE(ctx->isInitialized());
    EXPECT_GE(ctx->deviceOrdinal(), 0);
    EXPECT_FALSE(ctx->deviceName().empty());

    // Test work submission
    std::atomic<int> counter{0};
    ctx->submitAndWait([&]()
                       { counter.fetch_add(1); });
    EXPECT_EQ(counter.load(), 1);

    // Test async submission
    auto future = ctx->submitAsync([&]()
                                   { counter.fetch_add(1); });
    future.wait();
    EXPECT_EQ(counter.load(), 2);

    // Test synchronize
    ctx->synchronize();
}

TEST(Test__IWorkerGPUContext, MultiDeviceSameVendor)
{
    auto &pool = GPUDeviceContextPool::instance();

    if (pool.hasNvidiaSupport() && pool.nvidiaDeviceCount() >= 2)
    {
        auto &ctx0 = pool.getNvidiaContext(0);
        auto &ctx1 = pool.getNvidiaContext(1);

        EXPECT_NE(&ctx0, &ctx1) << "Different devices should have different contexts";
        EXPECT_EQ(ctx0.deviceOrdinal(), 0);
        EXPECT_EQ(ctx1.deviceOrdinal(), 1);
    }

    if (pool.hasAMDSupport() && pool.amdDeviceCount() >= 2)
    {
        auto &ctx0 = pool.getAMDContext(0);
        auto &ctx1 = pool.getAMDContext(1);

        EXPECT_NE(&ctx0, &ctx1) << "Different devices should have different contexts";
        EXPECT_EQ(ctx0.deviceOrdinal(), 0);
        EXPECT_EQ(ctx1.deviceOrdinal(), 1);
    }
}

// ===========================================================================
// Stress Tests
// ===========================================================================

TEST(Test__GPUDeviceContext_Stress, RapidSubmitAndWait)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = pool.hasNvidiaSupport()
                                 ? &pool.getNvidiaContext(0)
                                 : &pool.getAMDContext(0);

    constexpr int NUM_ITERATIONS = 1000;
    std::atomic<int> counter{0};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        ctx->submitAndWait([&]()
                           { counter.fetch_add(1); });
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(counter.load(), NUM_ITERATIONS);

    std::cout << "Rapid submitAndWait: " << NUM_ITERATIONS << " iterations in "
              << duration_ms << " ms (" << (NUM_ITERATIONS * 1000.0 / duration_ms)
              << " ops/sec)" << std::endl;
}

TEST(Test__GPUDeviceContext_Stress, ConcurrentSubmitFromMultipleThreads)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = pool.hasNvidiaSupport()
                                 ? &pool.getNvidiaContext(0)
                                 : &pool.getAMDContext(0);

    constexpr int NUM_THREADS = 4;
    constexpr int SUBMISSIONS_PER_THREAD = 100;
    std::atomic<int> counter{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        threads.emplace_back([&]()
                             {
            for (int i = 0; i < SUBMISSIONS_PER_THREAD; ++i) {
                ctx->submitAndWait([&]() {
                    counter.fetch_add(1);
                });
            } });
    }

    for (auto &t : threads)
    {
        t.join();
    }

    EXPECT_EQ(counter.load(), NUM_THREADS * SUBMISSIONS_PER_THREAD);
}

TEST(Test__GPUDeviceContext_Stress, StreamEventChurn)
{
    SKIP_IF_NO_GPU();

    auto &pool = GPUDeviceContextPool::instance();
    IWorkerGPUContext *ctx = pool.hasNvidiaSupport()
                                 ? &pool.getNvidiaContext(0)
                                 : &pool.getAMDContext(0);

    constexpr int NUM_ITERATIONS = 100;

    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        ctx->submitAndWait([&]()
                           {
            void* stream = ctx->createStream();
            void* event = ctx->createEvent();

            ctx->recordEvent(event, stream);
            ctx->synchronizeEvent(event);

            ctx->destroyEvent(event);
            ctx->destroyStream(stream); });
    }
}
