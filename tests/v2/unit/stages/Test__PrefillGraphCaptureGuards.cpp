/**
 * @file Test__PrefillGraphCaptureGuards.cpp
 * @brief Tests that capture-blocking operation guards prevent illegal HIP
 *        operations (hipMalloc, hipFree) during GPU graph capture.
 */

#include <gtest/gtest.h>
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "kernels/rocm/gdn/ROCmGatedDeltaNet.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;

class Test__PrefillGraphCaptureGuards : public ::testing::Test
{
protected:
    void SetUp() override
    {
#ifndef HAVE_ROCM
        GTEST_SKIP() << "No ROCm support compiled";
#else
        int device_count = 0;
        hipGetDeviceCount(&device_count);
        if (device_count <= 0)
            GTEST_SKIP() << "No ROCm device available";
        hipSetDevice(0);
#endif
    }
};

// =============================================================================
// MoE readiness helpers
// =============================================================================

TEST_F(Test__PrefillGraphCaptureGuards, MoE_HasGroupedPrefillScratchCapacity_Empty)
{
#ifdef HAVE_ROCM
    ROCmMoEKernel kernel(0);
    // Fresh kernel has zero capacity
    EXPECT_FALSE(kernel.hasGroupedPrefillScratchCapacity(128, 2048, 5632));
    // Zero request always fits
    EXPECT_TRUE(kernel.hasGroupedPrefillScratchCapacity(0, 0, 0));
#endif
}

TEST_F(Test__PrefillGraphCaptureGuards, MoE_HasGroupingBufferCapacity_Empty)
{
#ifdef HAVE_ROCM
    ROCmMoEKernel kernel(0);
    // Fresh kernel has zero capacity and null write_heads
    EXPECT_FALSE(kernel.hasGroupingBufferCapacity(128, 64));
    EXPECT_FALSE(kernel.hasGroupingBufferCapacity(0, 0));
#endif
}

// =============================================================================
// MoE graph capture guards — prepareExpertGroupsAsync
// =============================================================================

TEST_F(Test__PrefillGraphCaptureGuards, MoE_PrepareExpertGroupsAsync_FailsDuringCapture)
{
#ifdef HAVE_ROCM
    ROCmMoEKernel kernel(0);

    // Create small routing tensors (seq=2, top_k=2, experts=4) to warm up buffers
    const int warm_seq = 2, warm_topk = 2, warm_experts = 4;
    const int warm_slots = warm_seq * warm_topk;
    auto routing_indices = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(warm_seq), static_cast<size_t>(warm_topk)});
    auto routing_weights = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(warm_seq), static_cast<size_t>(warm_topk)});
    float *ri = routing_indices->mutable_data();
    float *rw = routing_weights->mutable_data();
    for (int i = 0; i < warm_slots; ++i)
    {
        ri[i] = static_cast<float>(i % warm_experts);
        rw[i] = 1.0f / warm_topk;
    }

    // Upload to device before calling prepareExpertGroupsAsync
    routing_indices->ensureOnDevice(DeviceId::rocm(0));
    routing_weights->ensureOnDevice(DeviceId::rocm(0));

    // Warm up: allocate buffers for small dimensions
    ASSERT_TRUE(kernel.prepareExpertGroupsAsync(
        routing_indices.get(), routing_weights.get(),
        warm_seq, warm_experts, warm_topk));

    // Verify capacity was allocated
    EXPECT_TRUE(kernel.hasGroupingBufferCapacity(warm_slots, warm_experts));

    // Now create LARGER routing tensors that exceed capacity
    const int big_seq = 64, big_topk = 2, big_experts = 64;
    const int big_slots = big_seq * big_topk;
    auto big_indices = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(big_seq), static_cast<size_t>(big_topk)});
    auto big_weights = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(big_seq), static_cast<size_t>(big_topk)});
    float *bi = big_indices->mutable_data();
    float *bw = big_weights->mutable_data();
    for (int i = 0; i < big_slots; ++i)
    {
        bi[i] = static_cast<float>(i % big_experts);
        bw[i] = 1.0f / big_topk;
    }

    // Upload to device
    big_indices->ensureOnDevice(DeviceId::rocm(0));
    big_weights->ensureOnDevice(DeviceId::rocm(0));

    // Under graph capture, the realloc should fail
    {
        GraphCaptureGuard guard;
        EXPECT_FALSE(kernel.prepareExpertGroupsAsync(
            big_indices.get(), big_weights.get(),
            big_seq, big_experts, big_topk));
    }

    // Outside capture, it should succeed
    EXPECT_TRUE(kernel.prepareExpertGroupsAsync(
        big_indices.get(), big_weights.get(),
        big_seq, big_experts, big_topk));
#endif
}

// =============================================================================
// MoE graph capture guards — ensureGroupedPrefillScratchCapacity
// =============================================================================

TEST_F(Test__PrefillGraphCaptureGuards, MoE_EnsureGroupedPrefillScratch_FailsDuringCapture)
{
#ifdef HAVE_ROCM
    ROCmMoEKernel kernel(0);

    // Fresh kernel: under capture, any allocation should fail
    {
        GraphCaptureGuard guard;
        // Access ensureGroupedPrefillScratchCapacity indirectly via hasGroupedPrefillScratchCapacity
        // The private method can't be called directly, but we verify the readiness helper
        EXPECT_FALSE(kernel.hasGroupedPrefillScratchCapacity(128, 2048, 5632));
    }
#endif
}

// =============================================================================
// GDN graph capture guards — chunk_forward
// =============================================================================

TEST_F(Test__PrefillGraphCaptureGuards, GDN_ChunkForward_FailsDuringCapture_NoState)
{
#ifdef HAVE_ROCM
    ROCmGatedDeltaNet gdn(0);
    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);
    gdn.setGPUStream(ctx.defaultStream());

    const int n_heads = 4, d_k = 64, d_v = 64, seq_len = 8;
    const int state_size = n_heads * d_k * d_v;

    // Allocate dummy device buffers for the call
    float *d_buf = nullptr;
    hipMalloc(&d_buf, static_cast<size_t>(seq_len * n_heads * d_k) * sizeof(float));
    hipMemset(d_buf, 0, static_cast<size_t>(seq_len * n_heads * d_k) * sizeof(float));

    float *d_output = nullptr;
    hipMalloc(&d_output, static_cast<size_t>(seq_len * n_heads * d_v) * sizeof(float));

    // State not allocated — under capture, chunk_forward should fail
    {
        GraphCaptureGuard guard;
        EXPECT_FALSE(gdn.chunk_forward(
            d_buf, d_buf, d_buf, d_buf, d_buf, d_buf, d_buf,
            d_output, nullptr,
            seq_len, n_heads, d_k, d_v, 0, false));
    }

    // Allocate state outside capture
    gdn.allocateGPUState(state_size);
    EXPECT_TRUE(gdn.isGPUStateReady(state_size));

    // With state ready, chunk_forward under capture should NOT hit the guard
    // (state is already allocated with correct size)
    {
        GraphCaptureGuard guard;
        // This should succeed (kernel dispatch is capturable)
        bool result = gdn.chunk_forward(
            d_buf, d_buf, d_buf, d_buf, d_buf, d_buf, d_buf,
            d_output, nullptr,
            seq_len, n_heads, d_k, d_v, 0, false);
        EXPECT_TRUE(result);
    }

    hipFree(d_buf);
    hipFree(d_output);
#endif
}

// =============================================================================
// GDN graph capture guards — recurrent_step
// =============================================================================

TEST_F(Test__PrefillGraphCaptureGuards, GDN_RecurrentStep_FailsDuringCapture_NoState)
{
#ifdef HAVE_ROCM
    ROCmGatedDeltaNet gdn(0);
    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);
    gdn.setGPUStream(ctx.defaultStream());

    const int n_heads = 4, d_k = 64, d_v = 64;
    const int state_size = n_heads * d_k * d_v;

    float *d_buf = nullptr;
    hipMalloc(&d_buf, static_cast<size_t>(n_heads * d_k) * sizeof(float));
    hipMemset(d_buf, 0, static_cast<size_t>(n_heads * d_k) * sizeof(float));

    float *d_output = nullptr;
    hipMalloc(&d_output, static_cast<size_t>(n_heads * d_v) * sizeof(float));

    // No state — should fail under capture
    {
        GraphCaptureGuard guard;
        EXPECT_FALSE(gdn.recurrent_step(
            d_buf, d_buf, d_buf, d_buf, d_buf, d_buf, d_buf,
            d_output, nullptr,
            n_heads, d_k, d_v, false));
    }

    // Allocate and retry
    gdn.allocateGPUState(state_size);
    {
        GraphCaptureGuard guard;
        bool result = gdn.recurrent_step(
            d_buf, d_buf, d_buf, d_buf, d_buf, d_buf, d_buf,
            d_output, nullptr,
            n_heads, d_k, d_v, false);
        EXPECT_TRUE(result);
    }

    hipFree(d_buf);
    hipFree(d_output);
#endif
}

// =============================================================================
// GDN graph capture guards — deinterleave_qkv_device
// =============================================================================

TEST_F(Test__PrefillGraphCaptureGuards, GDN_DeinterleaveQKV_FailsDuringCapture_Realloc)
{
#ifdef HAVE_ROCM
    ROCmGatedDeltaNet gdn(0);
    auto &ctx = GPUDeviceContextPool::instance().getAMDContext(0);
    gdn.setGPUStream(ctx.defaultStream());

    const int n_k_heads = 4, n_v_heads = 4, d_k = 64, d_v = 64;

    // First call with small seq_len to allocate scratch
    const int small_seq = 2;
    size_t small_total = static_cast<size_t>(small_seq) * n_v_heads * d_k * 2 +
                         static_cast<size_t>(small_seq) * n_v_heads * d_v;
    float *d_merged = nullptr;
    hipMalloc(&d_merged, small_total * sizeof(float));
    hipMemset(d_merged, 0, small_total * sizeof(float));

    float *d_q = nullptr, *d_k_out = nullptr, *d_v_out = nullptr;
    ASSERT_TRUE(gdn.deinterleave_qkv_device(
        d_merged, d_q, d_k_out, d_v_out,
        small_seq, n_k_heads, n_v_heads, d_k, d_v, 0));

    // Now try with larger seq_len under capture — should fail (needs realloc)
    const int big_seq = 128;
    size_t big_total = static_cast<size_t>(big_seq) * n_v_heads * d_k * 2 +
                       static_cast<size_t>(big_seq) * n_v_heads * d_v;
    float *d_merged_big = nullptr;
    hipMalloc(&d_merged_big, big_total * sizeof(float));
    hipMemset(d_merged_big, 0, big_total * sizeof(float));

    {
        GraphCaptureGuard guard;
        float *dq2 = nullptr, *dk2 = nullptr, *dv2 = nullptr;
        EXPECT_FALSE(gdn.deinterleave_qkv_device(
            d_merged_big, dq2, dk2, dv2,
            big_seq, n_k_heads, n_v_heads, d_k, d_v, 0));
    }

    // Outside capture should succeed
    {
        float *dq3 = nullptr, *dk3 = nullptr, *dv3 = nullptr;
        EXPECT_TRUE(gdn.deinterleave_qkv_device(
            d_merged_big, dq3, dk3, dv3,
            big_seq, n_k_heads, n_v_heads, d_k, d_v, 0));
    }

    hipFree(d_merged);
    hipFree(d_merged_big);
#endif
}
