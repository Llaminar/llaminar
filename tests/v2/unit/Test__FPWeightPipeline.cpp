/**
 * @file Test__FPWeightPipeline.cpp
 * @brief Unit tests for floating-point weight pipeline support
 *
 * Tests the RAW_FP passthrough path through the GPU weight loading pipeline:
 * - WeightVRAMPool::planRawWeight() allocation
 * - DeviceLoadPipeline passthrough (no GPU repack, D2D copy from staging)
 * - LoadOrchestrator::planRawWeight() delegation
 * - Byte-for-byte parity of uploaded FP data
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <numeric>
#include <string>

#include "loaders/gpu_pipeline/DeviceLoadPipeline.h"
#include "loaders/gpu_pipeline/WeightVRAMPool.h"
#include "loaders/gpu_pipeline/PinnedRingBuffer.h"
#include "loaders/gpu_pipeline/LoadOrchestrator.h"
#include "loaders/gpu_pipeline/RepackFormat.h"
#include "backends/BackendManager.h"

#ifdef HAVE_ROCM
#include "kernels/rocm/repack/VnniRepackKernels.h"
#include <hip/hip_runtime.h>
#endif

#ifdef HAVE_CUDA
#include "kernels/cuda/repack/CUDAVnniRepackKernels.h"
#endif

namespace llaminar2
{

// ============================================================================
// WeightVRAMPool unit tests for planRawWeight
// ============================================================================

TEST(Test__FPWeightPipeline, PlanRawWeight_BasicAllocation)
{
    WeightVRAMPool pool;
    pool.planRawWeight("fp32_weight", /*N=*/1024, /*K=*/2048, /*raw_bytes=*/1024 * 2048 * 4);

    EXPECT_EQ(pool.numPlannedWeights(), 1u);
    EXPECT_GE(pool.totalPlannedBytes(), 1024u * 2048 * 4);
}

TEST(Test__FPWeightPipeline, PlanRawWeight_MultipleWeights)
{
    WeightVRAMPool pool;
    const size_t fp32_bytes = 512 * 1024 * 4;
    const size_t fp16_bytes = 1024 * 2048 * 2;
    const size_t bf16_bytes = 256 * 512 * 2;

    pool.planRawWeight("w_fp32", 512, 1024, fp32_bytes);
    pool.planRawWeight("w_fp16", 1024, 2048, fp16_bytes);
    pool.planRawWeight("w_bf16", 256, 512, bf16_bytes);

    EXPECT_EQ(pool.numPlannedWeights(), 3u);
    // Total should be at least the sum of all raw bytes (plus alignment)
    EXPECT_GE(pool.totalPlannedBytes(), fp32_bytes + fp16_bytes + bf16_bytes);
}

TEST(Test__FPWeightPipeline, PlanRawWeight_DuplicateNameThrows)
{
    WeightVRAMPool pool;
    pool.planRawWeight("w1", 64, 64, 64 * 64 * 4);

    EXPECT_THROW(
        pool.planRawWeight("w1", 32, 32, 32 * 32 * 4),
        std::runtime_error);
}

TEST(Test__FPWeightPipeline, PlanRawWeight_AfterAllocateThrows)
{
    WeightVRAMPool pool;
    pool.planRawWeight("w1", 64, 64, 64 * 64 * 4);
    ASSERT_TRUE(pool.allocate(nullptr, 0, 0));

    EXPECT_THROW(
        pool.planRawWeight("w2", 32, 32, 32 * 32 * 4),
        std::runtime_error);
}

TEST(Test__FPWeightPipeline, PlanRawWeight_SlotHasNoScales)
{
    WeightVRAMPool pool;
    pool.planRawWeight("w1", 64, 128, 64 * 128 * 4);
    ASSERT_TRUE(pool.allocate(nullptr, 0, 0));

    auto slot = pool.getSlot("w1");
    ASSERT_TRUE(slot.has_value());
    EXPECT_GT(slot->payload_bytes, 0u);
    EXPECT_EQ(slot->staging_bytes, 64u * 128 * 4);
    // Raw FP weights have no scales/mins/emins
    EXPECT_EQ(slot->d_native_vnni_scales, nullptr);
    EXPECT_EQ(slot->d_native_vnni_mins, nullptr);
    EXPECT_EQ(slot->d_native_vnni_emins, nullptr);
}

TEST(Test__FPWeightPipeline, PlanRawWeight_MixedWithQuantized)
{
    WeightVRAMPool pool;
    // Plan a quantized weight
    pool.planWeight("q4_weight", 512, 1024, 16, /*is_asymmetric=*/false,
                    /*has_emins=*/false, /*raw_gguf_bytes=*/262144);
    // Plan a raw FP weight
    pool.planRawWeight("fp16_weight", 512, 1024, 512 * 1024 * 2);

    EXPECT_EQ(pool.numPlannedWeights(), 2u);
    ASSERT_TRUE(pool.allocate(nullptr, 0, 0));

    auto q4_slot = pool.getSlot("q4_weight");
    auto fp_slot = pool.getSlot("fp16_weight");
    ASSERT_TRUE(q4_slot.has_value());
    ASSERT_TRUE(fp_slot.has_value());

    // Quantized slot should have scales, FP slot should not
    // (with nullptr backend, both d_base_ is nullptr, but scales pointer logic is tested)
    EXPECT_EQ(fp_slot->d_native_vnni_scales, nullptr);
    EXPECT_EQ(fp_slot->d_native_vnni_mins, nullptr);
}

// ============================================================================
// LoadOrchestrator unit tests for planRawWeight
// ============================================================================

TEST(Test__FPWeightPipeline, Orchestrator_PlanRawWeight)
{
    LoadOrchestrator orch(nullptr);
    orch.addDevice(0);

    orch.planRawWeight(0, "fp32_attn_q", 1024, 2048, 1024 * 2048 * 4);

    auto *pool = orch.getPool(0);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->numPlannedWeights(), 1u);
}

TEST(Test__FPWeightPipeline, Orchestrator_PlanRawWeight_UnknownDeviceThrows)
{
    LoadOrchestrator orch(nullptr);
    orch.addDevice(0);

    EXPECT_THROW(
        orch.planRawWeight(99, "w1", 64, 64, 64 * 64 * 4),
        std::runtime_error);
}

TEST(Test__FPWeightPipeline, Orchestrator_MixedPlanQuantizedAndRaw)
{
    LoadOrchestrator orch(nullptr);
    orch.addDevice(0);

    orch.planWeight(0, "q4_w", 512, 1024, 16, false, false, 262144);
    orch.planRawWeight(0, "fp16_w", 512, 1024, 512 * 1024 * 2);

    auto *pool = orch.getPool(0);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->numPlannedWeights(), 2u);
}

// ============================================================================
// DeviceLoadPipeline tests for RAW_FP passthrough
// ============================================================================

class Test__FPDeviceLoadPipeline : public ::testing::Test
{
  protected:
    IBackend* backend_ = nullptr;
    RepackKernels kernels_{};

    void SetUp() override
    {
#ifdef HAVE_ROCM
        backend_ = getROCmBackend();
        if (backend_)
        {
            kernels_.vnniRepack = launchVnniRepack;
        }
#endif
#ifdef HAVE_CUDA
        if (!backend_)
        {
            backend_ = getCUDABackend();
            if (backend_)
            {
                kernels_.vnniRepack = launchVnniRepackCUDA;
            }
        }
#endif
        if (!backend_)
        {
            GTEST_SKIP() << "No GPU backend available (need HAVE_ROCM or HAVE_CUDA)";
        }
        backend_->setDevice(0);
    }
};

TEST_F(Test__FPDeviceLoadPipeline, RAW_FP_SingleWeight_ByteForByteParity)
{
    // Simulate a small FP32 weight (64x128 = 32KB)
    const int N = 64;
    const int K = 128;
    const size_t raw_bytes = static_cast<size_t>(N) * K * sizeof(float);
    const int num_streams = 3;

    // Create known data pattern
    std::vector<float> host_data(N * K);
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // Setup pool with raw plan
    WeightVRAMPool pool;
    pool.planRawWeight("fp32_test", N, K, raw_bytes);
    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    // Setup pinned ring
    PinnedRingBuffer pinned(raw_bytes, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    // Create pipeline
    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    // Create RAW_FP job
    WeightJob job;
    job.name = "fp32_test";
    job.host_raw_data = host_data.data();
    job.raw_bytes = raw_bytes;
    job.format = RepackFormat::RAW_FP;
    job.N = N;
    job.K = K;
    job.is_asymmetric = false;

    ASSERT_TRUE(pipeline.processJobs({job}));
    EXPECT_EQ(pipeline.numProcessed(), 1u);

    // Download and verify byte-for-byte parity
    auto slot = pool.getSlot("fp32_test");
    ASSERT_TRUE(slot.has_value());
    EXPECT_EQ(slot->payload_bytes, raw_bytes);
    EXPECT_EQ(slot->d_native_vnni_scales, nullptr);

    std::vector<float> gpu_data(N * K);
    backend_->deviceToHost(gpu_data.data(), slot->d_native_vnni_payload, raw_bytes, 0);
    EXPECT_EQ(gpu_data, host_data) << "FP32 byte-for-byte parity failed";

    pinned.release();
    pool.release();
}

TEST_F(Test__FPDeviceLoadPipeline, RAW_FP_FP16Weight_ByteForByteParity)
{
    // Simulate a small FP16 weight (128x256 = 64KB)
    const int N = 128;
    const int K = 256;
    const size_t raw_bytes = static_cast<size_t>(N) * K * sizeof(uint16_t);
    const int num_streams = 3;

    // Create known FP16 data pattern
    std::vector<uint16_t> host_data(N * K);
    for (size_t i = 0; i < host_data.size(); ++i)
    {
        // Incrementing FP16 values (0x3C00 = 1.0, 0x4000 = 2.0, etc.)
        host_data[i] = static_cast<uint16_t>(0x3000 + (i & 0x0FFF));
    }

    // Setup pool with raw plan
    WeightVRAMPool pool;
    pool.planRawWeight("fp16_test", N, K, raw_bytes);
    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    // Setup pinned ring
    PinnedRingBuffer pinned(raw_bytes, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    // Create pipeline
    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    // Create RAW_FP job
    WeightJob job;
    job.name = "fp16_test";
    job.host_raw_data = host_data.data();
    job.raw_bytes = raw_bytes;
    job.format = RepackFormat::RAW_FP;
    job.N = N;
    job.K = K;
    job.is_asymmetric = false;

    ASSERT_TRUE(pipeline.processJobs({job}));
    EXPECT_EQ(pipeline.numProcessed(), 1u);

    // Download and verify
    auto slot = pool.getSlot("fp16_test");
    ASSERT_TRUE(slot.has_value());

    std::vector<uint16_t> gpu_data(N * K);
    backend_->deviceToHost(gpu_data.data(), slot->d_native_vnni_payload, raw_bytes, 0);
    EXPECT_EQ(gpu_data, host_data) << "FP16 byte-for-byte parity failed";

    pinned.release();
    pool.release();
}

TEST_F(Test__FPDeviceLoadPipeline, RAW_FP_MixedWithQuantized)
{
    // Test that RAW_FP and quantized weights can coexist in the same pipeline run
    const int N = 32;
    const int K = 64; // 2 blocks per row for Q4_0
    const int blocks_per_row = K / 32;
    const int total_blocks = N * blocks_per_row;
    const size_t q4_raw_bytes = total_blocks * 18; // Q4_0: 18 bytes/block
    const size_t fp_raw_bytes = static_cast<size_t>(N) * K * sizeof(float);
    const int num_streams = 3;

    // Create FP32 data
    std::vector<float> fp_data(N * K, 3.14f);

    // Setup pool with mixed plan
    WeightVRAMPool pool;
    pool.planWeight("q4_weight", N, K, 16, false, false, q4_raw_bytes);
    pool.planRawWeight("fp_weight", N, K, fp_raw_bytes);
    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    // Verify both slots exist with correct properties
    auto q4_slot = pool.getSlot("q4_weight");
    auto fp_slot = pool.getSlot("fp_weight");
    ASSERT_TRUE(q4_slot.has_value());
    ASSERT_TRUE(fp_slot.has_value());

    // Q4 slot has scales; FP slot does not
    EXPECT_NE(q4_slot->d_native_vnni_scales, nullptr);
    EXPECT_EQ(fp_slot->d_native_vnni_scales, nullptr);

    // Both have payload pointers
    EXPECT_NE(q4_slot->d_native_vnni_payload, nullptr);
    EXPECT_NE(fp_slot->d_native_vnni_payload, nullptr);

    // Payload pointers should be different (non-overlapping regions)
    EXPECT_NE(q4_slot->d_native_vnni_payload, fp_slot->d_native_vnni_payload);

    // Setup pipeline and run just the FP job (to verify passthrough path alone)
    PinnedRingBuffer pinned(fp_raw_bytes, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    WeightJob fp_job;
    fp_job.name = "fp_weight";
    fp_job.host_raw_data = fp_data.data();
    fp_job.raw_bytes = fp_raw_bytes;
    fp_job.format = RepackFormat::RAW_FP;
    fp_job.N = N;
    fp_job.K = K;
    fp_job.is_asymmetric = false;

    ASSERT_TRUE(pipeline.processJobs({fp_job}));

    // Verify FP data
    std::vector<float> gpu_data(N * K);
    backend_->deviceToHost(gpu_data.data(), fp_slot->d_native_vnni_payload, fp_raw_bytes, 0);
    EXPECT_EQ(gpu_data, fp_data);

    pinned.release();
    pool.release();
}

TEST_F(Test__FPDeviceLoadPipeline, RAW_FP_MultipleWeights_StreamReuse)
{
    // Test multiple FP weights flowing through the pipeline with stream cycling
    const int num_streams = 2; // fewer streams than weights forces reuse
    const int N = 32;
    const int K = 64;
    const size_t raw_bytes = static_cast<size_t>(N) * K * sizeof(float);

    // Create 5 different FP weights with distinct data
    std::vector<std::vector<float>> all_data(5);
    for (int w = 0; w < 5; ++w)
    {
        all_data[w].resize(N * K);
        std::iota(all_data[w].begin(), all_data[w].end(),
                  static_cast<float>(w * 1000));
    }

    // Plan all weights
    WeightVRAMPool pool;
    for (int w = 0; w < 5; ++w)
    {
        pool.planRawWeight("fp_w" + std::to_string(w), N, K, raw_bytes);
    }
    ASSERT_TRUE(pool.allocate(backend_, 0, num_streams));

    size_t max_bytes = raw_bytes;
    PinnedRingBuffer pinned(max_bytes, num_streams);
    ASSERT_TRUE(pinned.allocate(backend_, 0));

    DeviceLoadPipeline pipeline(*backend_, 0, pool, pinned, kernels_, num_streams);
    ASSERT_TRUE(pipeline.initialize());

    // Create all jobs
    std::vector<WeightJob> jobs;
    for (int w = 0; w < 5; ++w)
    {
        WeightJob job;
        job.name = "fp_w" + std::to_string(w);
        job.host_raw_data = all_data[w].data();
        job.raw_bytes = raw_bytes;
        job.format = RepackFormat::RAW_FP;
        job.N = N;
        job.K = K;
        job.is_asymmetric = false;
        jobs.push_back(job);
    }

    ASSERT_TRUE(pipeline.processJobs(jobs));
    EXPECT_EQ(pipeline.numProcessed(), 5u);

    // Verify each weight independently
    for (int w = 0; w < 5; ++w)
    {
        auto slot = pool.getSlot("fp_w" + std::to_string(w));
        ASSERT_TRUE(slot.has_value()) << "Missing slot for fp_w" << w;

        std::vector<float> gpu_data(N * K);
        backend_->deviceToHost(gpu_data.data(), slot->d_native_vnni_payload, raw_bytes, 0);
        EXPECT_EQ(gpu_data, all_data[w]) << "Data mismatch for fp_w" << w;
    }

    pinned.release();
    pool.release();
}

TEST_F(Test__FPDeviceLoadPipeline, LoadOrchestrator_EndToEnd_RawFP)
{
    // Test the full LoadOrchestrator path with RAW_FP weights
    LoadOrchestrator orch(backend_);
    orch.addDevice(0);

    const int N = 64;
    const int K = 128;
    const size_t raw_bytes = static_cast<size_t>(N) * K * sizeof(float);

    // Plan two FP weights
    orch.planRawWeight(0, "attn_q", N, K, raw_bytes);
    orch.planRawWeight(0, "attn_v", N, K, raw_bytes);

    // Allocate
    orch.allocate(raw_bytes, 3);

    // Create data
    std::vector<float> data_q(N * K, 1.5f);
    std::vector<float> data_v(N * K, -2.5f);

    // Submit jobs
    WeightJob job_q;
    job_q.name = "attn_q";
    job_q.host_raw_data = data_q.data();
    job_q.raw_bytes = raw_bytes;
    job_q.format = RepackFormat::RAW_FP;
    job_q.N = N;
    job_q.K = K;
    job_q.is_asymmetric = false;
    orch.addWeightJob(0, job_q);

    WeightJob job_v;
    job_v.name = "attn_v";
    job_v.host_raw_data = data_v.data();
    job_v.raw_bytes = raw_bytes;
    job_v.format = RepackFormat::RAW_FP;
    job_v.N = N;
    job_v.K = K;
    job_v.is_asymmetric = false;
    orch.addWeightJob(0, job_v);

    // Execute pipeline
    ASSERT_NO_THROW(orch.load());

    // Verify results
    auto *pool = orch.getPool(0);
    ASSERT_NE(pool, nullptr);

    auto slot_q = pool->getSlot("attn_q");
    auto slot_v = pool->getSlot("attn_v");
    ASSERT_TRUE(slot_q.has_value());
    ASSERT_TRUE(slot_v.has_value());

    std::vector<float> gpu_q(N * K), gpu_v(N * K);
    backend_->deviceToHost(gpu_q.data(), slot_q->d_native_vnni_payload, raw_bytes, 0);
    backend_->deviceToHost(gpu_v.data(), slot_v->d_native_vnni_payload, raw_bytes, 0);

    EXPECT_EQ(gpu_q, data_q);
    EXPECT_EQ(gpu_v, data_v);

    orch.release();
}

} // namespace llaminar2
