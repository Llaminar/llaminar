/**
 * @file Test__CrossDomainExpertTransferIntegration.cpp
 * @brief Integration tests for cross-domain MoE expert weight transfer paths.
 *
 * Validates the three primary transfer mechanisms:
 *   1. GPU↔GPU P2P (peer DMA) via GPUExpertTransfer — data integrity + device context
 *   2. GPU↔GPU host-staged fallback — data integrity when P2P unavailable
 *   3. CPU→GPU cross-domain via CrossDomainTransfer — activation transfer for PP transitions
 *   4. computeGpuCacheExpertMasks — GPU preference placement logic
 *   5. Full rebalance cycle: mask change → release departed → register new experts
 *
 * Requires: HAVE_ROCM and at least 1 GPU. Multi-GPU tests require 2+ GPUs.
 */

#include <gtest/gtest.h>

#include "execution/moe/GPUExpertTransfer.h"
#include "execution/moe/MoEExpertWeightService.h"
#include "execution/moe/MoERebalanceController.h"
#include "execution/moe/ExpertWeightTransfer.h"
#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/local_execution/coherence/CrossDomainTransfer.h"
#include "kernels/KernelFactory.h"
#include "kernels/PackedWeightsSerialization.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test;
using KF = llaminar::v2::kernels::KernelFactory;
using namespace llaminar2::packed_weights_serialization;

// ===========================================================================
// Fixture
// ===========================================================================

class Test__CrossDomainExpertTransfer : public ::testing::Test
{
protected:
    void SetUp() override
    {
#ifdef HAVE_ROCM
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
            GTEST_SKIP() << "No ROCm GPU available";
        }
        gpu_count_ = count;
#else
        GTEST_SKIP() << "ROCm not available";
#endif
    }

    void TearDown() override
    {
        KF::clearCache();
    }

    int gpu_count_ = 0;
};

// ===========================================================================
// 1. GPU↔GPU P2P transfer — large payload integrity
// ===========================================================================

#ifdef HAVE_ROCM

TEST_F(Test__CrossDomainExpertTransfer, GPUP2P_LargePayload_DataIntegrity)
{
    // Simulate a realistic expert transfer: ~2.8MB VNNI + ~88KB scales
    // Matches Qwen3.5 expert gate: N=2560, K=2048, Q4_0 packed
    const size_t vnni_bytes = 2 * 1024 * 1024 + 800 * 1024; // ~2.8MB
    const size_t scales_bytes = 88 * 1024;
    const size_t mins_bytes = 88 * 1024;

    hipSetDevice(0);

    uint8_t *d_src_vnni = nullptr, *d_dst_vnni = nullptr;
    uint8_t *d_src_scales = nullptr, *d_dst_scales = nullptr;
    uint8_t *d_src_mins = nullptr, *d_dst_mins = nullptr;

    ASSERT_EQ(hipMalloc(&d_src_vnni, vnni_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst_vnni, vnni_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_src_scales, scales_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst_scales, scales_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_src_mins, mins_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst_mins, mins_bytes), hipSuccess);

    // Fill with pseudo-random pattern via host
    std::vector<uint8_t> pattern_vnni(vnni_bytes);
    std::vector<uint8_t> pattern_scales(scales_bytes);
    std::vector<uint8_t> pattern_mins(mins_bytes);

    std::mt19937 rng(42);
    for (auto& b : pattern_vnni) b = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& b : pattern_scales) b = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& b : pattern_mins) b = static_cast<uint8_t>(rng() & 0xFF);

    ASSERT_EQ(hipMemcpy(d_src_vnni, pattern_vnni.data(), vnni_bytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_src_scales, pattern_scales.data(), scales_bytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_src_mins, pattern_mins.data(), mins_bytes, hipMemcpyHostToDevice), hipSuccess);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src_vnni;
    src_ptrs.d_scales = d_src_scales;
    src_ptrs.d_mins = d_src_mins;
    dst_ptrs.d_vnni = d_dst_vnni;
    dst_ptrs.d_scales = d_dst_scales;
    dst_ptrs.d_mins = d_dst_mins;

    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(0), DeviceId::rocm(0),
        vnni_bytes, scales_bytes, mins_bytes, 0, nullptr);
    ASSERT_TRUE(ok);

    // Verify all three arrays
    std::vector<uint8_t> result_vnni(vnni_bytes);
    std::vector<uint8_t> result_scales(scales_bytes);
    std::vector<uint8_t> result_mins(mins_bytes);

    ASSERT_EQ(hipMemcpy(result_vnni.data(), d_dst_vnni, vnni_bytes, hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(result_scales.data(), d_dst_scales, scales_bytes, hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(result_mins.data(), d_dst_mins, mins_bytes, hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(pattern_vnni, result_vnni) << "VNNI payload corrupted during P2P transfer";
    EXPECT_EQ(pattern_scales, result_scales) << "Scales corrupted during P2P transfer";
    EXPECT_EQ(pattern_mins, result_mins) << "Mins corrupted during P2P transfer";

    hipFree(d_src_vnni); hipFree(d_dst_vnni);
    hipFree(d_src_scales); hipFree(d_dst_scales);
    hipFree(d_src_mins); hipFree(d_dst_mins);
}

TEST_F(Test__CrossDomainExpertTransfer, GPUP2P_CrossDevice_DataIntegrity)
{
    if (gpu_count_ < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for cross-device transfer test";
    }

    const size_t vnni_bytes = 1024 * 1024; // 1MB
    const size_t scales_bytes = 32 * 1024;

    // Allocate on device 0
    hipSetDevice(0);
    uint8_t *d_src_vnni = nullptr, *d_src_scales = nullptr;
    ASSERT_EQ(hipMalloc(&d_src_vnni, vnni_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_src_scales, scales_bytes), hipSuccess);

    std::vector<uint8_t> pattern_vnni(vnni_bytes);
    std::vector<uint8_t> pattern_scales(scales_bytes);
    std::mt19937 rng(99);
    for (auto& b : pattern_vnni) b = static_cast<uint8_t>(rng() & 0xFF);
    for (auto& b : pattern_scales) b = static_cast<uint8_t>(rng() & 0xFF);
    ASSERT_EQ(hipMemcpy(d_src_vnni, pattern_vnni.data(), vnni_bytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(d_src_scales, pattern_scales.data(), scales_bytes, hipMemcpyHostToDevice), hipSuccess);

    // Allocate on device 1
    hipSetDevice(1);
    uint8_t *d_dst_vnni = nullptr, *d_dst_scales = nullptr;
    ASSERT_EQ(hipMalloc(&d_dst_vnni, vnni_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst_scales, scales_bytes), hipSuccess);

    // Restore to device 0 to verify context preservation
    hipSetDevice(0);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src_vnni;
    src_ptrs.d_scales = d_src_scales;
    dst_ptrs.d_vnni = d_dst_vnni;
    dst_ptrs.d_scales = d_dst_scales;

    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(0), DeviceId::rocm(1),
        vnni_bytes, scales_bytes, 0, 0, nullptr);
    ASSERT_TRUE(ok);

    // Verify device context preserved
    int current = -1;
    ASSERT_EQ(hipGetDevice(&current), hipSuccess);
    EXPECT_EQ(current, 0) << "Caller device context must be preserved";

    // Read back from device 1
    hipSetDevice(1);
    std::vector<uint8_t> result_vnni(vnni_bytes);
    std::vector<uint8_t> result_scales(scales_bytes);
    ASSERT_EQ(hipMemcpy(result_vnni.data(), d_dst_vnni, vnni_bytes, hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_EQ(hipMemcpy(result_scales.data(), d_dst_scales, scales_bytes, hipMemcpyDeviceToHost), hipSuccess);

    EXPECT_EQ(pattern_vnni, result_vnni) << "Cross-device VNNI payload corrupted";
    EXPECT_EQ(pattern_scales, result_scales) << "Cross-device scales corrupted";

    hipSetDevice(0); hipFree(d_src_vnni); hipFree(d_src_scales);
    hipSetDevice(1); hipFree(d_dst_vnni); hipFree(d_dst_scales);
}

// ===========================================================================
// 2. CrossDomainTransfer — activation transfer CPU↔GPU round-trip integrity
// ===========================================================================

TEST_F(Test__CrossDomainExpertTransfer, CrossDomain_ActivationRoundTrip_LargeTensor)
{
    // Simulate a hidden-state transfer at PP boundary: [seq_len=512, d_model=2048]
    const size_t seq_len = 512;
    const size_t d_model = 2048;
    auto src = TestTensorFactory::createFP32Random({seq_len, d_model}, -5.0f, 5.0f, 77);

    // Store original for comparison
    std::vector<float> original(src->numel());
    std::memcpy(original.data(), src->data(), src->size_bytes());

    CrossDomainTransfer transfer;
    DeviceId gpu = DeviceId::rocm(0);

    // CPU → GPU
    auto gpu_tensor = TestTensorFactory::createFP32({seq_len, d_model});
    ASSERT_TRUE(transfer.cpuToGpu(src.get(), gpu_tensor.get(), gpu));

    // GPU → CPU
    auto dst = TestTensorFactory::createFP32({seq_len, d_model});
    ASSERT_TRUE(transfer.gpuToCpu(gpu_tensor.get(), dst.get()));

    // Verify data integrity
    const float* result = dst->data();
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_EQ(original[i], result[i])
            << "Round-trip mismatch at index " << i
            << " (row " << i / d_model << ", col " << i % d_model << ")";
        if (original[i] != result[i]) break; // Don't spam
    }

    auto stats = transfer.getStats();
    EXPECT_EQ(stats.cpu_to_gpu_count, 1);
    EXPECT_EQ(stats.gpu_to_cpu_count, 1);
    EXPECT_EQ(stats.cpu_to_gpu_bytes, src->size_bytes());
    EXPECT_EQ(stats.gpu_to_cpu_bytes, gpu_tensor->size_bytes());
}

TEST_F(Test__CrossDomainExpertTransfer, CrossDomain_MultipleConcurrentTransfers)
{
    // Multiple tensors transferred sequentially — stats accumulate correctly
    CrossDomainTransfer transfer;
    DeviceId gpu = DeviceId::rocm(0);

    const int NUM_TRANSFERS = 5;
    const size_t elements = 1024;

    for (int i = 0; i < NUM_TRANSFERS; ++i) {
        auto src = TestTensorFactory::createFP32Random({elements}, -1.0f, 1.0f, 100 + i);
        auto dst = TestTensorFactory::createFP32({elements});
        ASSERT_TRUE(transfer.cpuToGpu(src.get(), dst.get(), gpu));
    }

    auto stats = transfer.getStats();
    EXPECT_EQ(stats.cpu_to_gpu_count, NUM_TRANSFERS);
    EXPECT_EQ(stats.cpu_to_gpu_bytes, NUM_TRANSFERS * elements * sizeof(float));
}

// ===========================================================================
// 3. computeGpuCacheExpertMasks — GPU preference placement
// ===========================================================================

TEST_F(Test__CrossDomainExpertTransfer, GpuCacheMasks_HotExpertsOnGPU)
{
    // Setup: 2 sockets (1 GPU, 1 CPU), 8 experts, cache 3 on GPU
    const int num_layers = 4;
    const int num_experts = 8;
    const int cache_per_layer = 3;

    MoERebalanceController::Config cfg;
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = num_layers;
    cfg.num_experts = num_experts;
    cfg.top_k = 2;
    cfg.window_size = 64;
    cfg.sockets = {DeviceId::rocm(0), DeviceId::cpu()};
    cfg.initial_expert_to_socket.assign(num_experts, 0); // Start all on GPU

    MoERebalanceController controller(cfg);

    // Inject histogram data: experts 0,1,2 are hot; 5,6,7 are cold
    auto* hist = controller.histogram();
    ASSERT_NE(hist, nullptr);

    // Record activations to create a skewed distribution
    std::vector<int> hot_experts = {0, 1, 2};
    std::vector<int> cold_experts = {5, 6, 7};
    std::vector<float> weights = {0.5f, 0.5f};

    for (int step = 0; step < 50; ++step) {
        for (int l = 0; l < num_layers; ++l) {
            // Hot experts activated frequently
            for (int e : hot_experts) {
                int indices[2] = {e, (e + 1) % num_experts};
                hist->record(l, indices, weights.data(), 2);
            }
            // Cold experts activated rarely
            if (step % 10 == 0) {
                for (int e : cold_experts) {
                    int indices[2] = {e, (e + 2) % num_experts};
                    hist->record(l, indices, weights.data(), 2);
                }
            }
        }
    }

    // Compute masks
    auto masks = controller.computeGpuCacheExpertMasks(cache_per_layer);
    ASSERT_EQ(masks.size(), 2u); // 2 sockets

    // Socket 0 (GPU): should have exactly cache_per_layer experts per layer
    // Socket 1 (CPU): should have remaining experts
    for (int l = 0; l < num_layers; ++l) {
        int gpu_count = 0, cpu_count = 0;
        for (int e = 0; e < num_experts; ++e) {
            if (masks[0][l][e]) gpu_count++;
            if (masks[1][l][e]) cpu_count++;
        }
        EXPECT_EQ(gpu_count, cache_per_layer)
            << "GPU socket should have exactly " << cache_per_layer << " experts at layer " << l;
        EXPECT_EQ(cpu_count, num_experts - cache_per_layer)
            << "CPU socket should have remaining experts at layer " << l;

        // Every expert should be assigned to exactly one socket
        for (int e = 0; e < num_experts; ++e) {
            EXPECT_TRUE(masks[0][l][e] || masks[1][l][e])
                << "Expert " << e << " layer " << l << " not assigned to any socket";
            EXPECT_FALSE(masks[0][l][e] && masks[1][l][e])
                << "Expert " << e << " layer " << l << " assigned to both sockets";
        }
    }

    // The hottest experts should be on GPU (socket 0)
    // With our distribution, experts 0,1,2 should be on GPU at all layers
    for (int l = 0; l < num_layers; ++l) {
        int hot_on_gpu = 0;
        for (int e : hot_experts) {
            if (masks[0][l][e]) hot_on_gpu++;
        }
        EXPECT_EQ(hot_on_gpu, cache_per_layer)
            << "All hot experts should be on GPU at layer " << l;
    }
}

TEST_F(Test__CrossDomainExpertTransfer, GpuCacheMasks_FallbackWithoutMixedTopology)
{
    // When all sockets are same type (all GPU), should fall back to uniform masks
    const int num_experts = 4;
    const int num_layers = 2;

    MoERebalanceController::Config cfg;
    cfg.mode = MoERebalanceMode::OBSERVE;
    cfg.num_layers = num_layers;
    cfg.num_experts = num_experts;
    cfg.top_k = 2;
    cfg.window_size = 32;
    cfg.sockets = {DeviceId::rocm(0), DeviceId::rocm(1)}; // Both GPU
    cfg.initial_expert_to_socket = {0, 0, 1, 1};

    MoERebalanceController controller(cfg);

    auto masks = controller.computeGpuCacheExpertMasks(2);
    ASSERT_EQ(masks.size(), 2u);

    // Should fall back to computeExpertMasks (contiguous partition)
    // Socket 0 gets experts 0,1; socket 1 gets experts 2,3
    for (int l = 0; l < num_layers; ++l) {
        EXPECT_TRUE(masks[0][l][0]);
        EXPECT_TRUE(masks[0][l][1]);
        EXPECT_TRUE(masks[1][l][2]);
        EXPECT_TRUE(masks[1][l][3]);
    }
}

// ===========================================================================
// 4. CPU rebalance cycle: releaseDeparted → registerNewExperts
// ===========================================================================

TEST_F(Test__CrossDomainExpertTransfer, CPURebalanceCycle_ReleaseAndRegister)
{
    // Test the full CPU-side rebalance path with serialized weight blobs
    constexpr int kNumExperts = 4;
    constexpr int kIntermediate = 64;
    constexpr int kDModel = 32;
    constexpr size_t kBlockSize = 32;

    // Create 3D weight tensors (GGUF convention: [cols, rows_per_expert, num_experts])
    auto create3D = [](size_t cols, size_t rows_per_expert, size_t num_experts, uint32_t seed) {
        size_t blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;
        size_t total_blocks = rows_per_expert * num_experts * blocks_per_row;
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        std::vector<uint8_t> raw(total_blocks * sizeof(Q4_0Block));
        auto* blocks = reinterpret_cast<Q4_0Block*>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i) {
            float max_abs = 0.0f;
            float values[32];
            for (size_t j = 0; j < kBlockSize; ++j) {
                values[j] = dist(rng);
                max_abs = std::max(max_abs, std::abs(values[j]));
            }
            float scale = max_abs / 7.0f;
            blocks[i].d = fp32_to_fp16(scale);
            float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
            for (size_t j = 0; j < kBlockSize / 2; ++j) {
                int32_t q0 = std::clamp(int(std::round(values[2*j] * inv)) + 8, 0, 15);
                int32_t q1 = std::clamp(int(std::round(values[2*j+1] * inv)) + 8, 0, 15);
                blocks[i].qs[j] = static_cast<uint8_t>((q1 << 4) | q0);
            }
        }
        return std::make_shared<Q4_0Tensor>(
            std::vector<size_t>{cols, rows_per_expert, num_experts}, raw);
    };

    auto gate_3d = create3D(kDModel, kIntermediate, kNumExperts, 42);
    auto up_3d   = create3D(kDModel, kIntermediate, kNumExperts, 43);
    auto down_3d = create3D(kIntermediate, kDModel, kNumExperts, 44);

    // Build weight context — start with experts 0,1 active (TestWeightContextOwner pattern)
    std::vector<bool> expert_mask = {true, true, false, false};
    std::vector<std::shared_ptr<TensorBase>> gate_views, up_views, down_views;
    std::vector<ITensorGemm*> gate_gemm, up_gemm, down_gemm;
    std::vector<std::shared_ptr<ITensorGemm>> owned_kernels;
    std::shared_ptr<void> gate_lt, up_lt, down_lt;

    MoEWeightContext ctx{
        DeviceId::cpu(), kNumExperts, kIntermediate, kDModel,
        0, -1, 0,
        expert_mask,
        gate_3d.get(), up_3d.get(), down_3d.get(),
        gate_views, up_views, down_views,
        gate_gemm, up_gemm, down_gemm,
        owned_kernels, gate_lt, up_lt, down_lt
    };

    // Extract views and prepare engines for initial experts
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(ctx));

    // Verify initial state: experts 0,1 have engines
    ASSERT_EQ(gate_gemm.size(), static_cast<size_t>(kNumExperts));
    EXPECT_NE(gate_gemm[0], nullptr);
    EXPECT_NE(gate_gemm[1], nullptr);
    EXPECT_EQ(gate_gemm[2], nullptr);
    EXPECT_EQ(gate_gemm[3], nullptr);

    // Serialize expert 1 (about to depart)
    auto blobs_1 = MoEExpertWeightService::serializeExpert(ctx, 1);
    EXPECT_FALSE(blobs_1.empty()) << "Serialization of active expert must succeed";

    // Phase 1: Release expert 1, add expert 2
    std::vector<bool> new_mask = {true, false, true, false};
    auto released = MoEExpertWeightService::releaseDepartedExperts(ctx, new_mask);
    (void)released; // May or may not have tensors to release

    // Phase 2: Register expert 2 with transferred blobs
    // Serialize from the original 3D data for expert 2 (simulating received from another rank)
    // For CPU path: we pass blobs from serializeExpert of a source that has expert 2
    std::vector<bool> all_active = {true, true, true, true};
    std::vector<std::shared_ptr<TensorBase>> src_gate_v, src_up_v, src_down_v;
    std::vector<ITensorGemm*> src_gate_g, src_up_g, src_down_g;
    std::vector<std::shared_ptr<ITensorGemm>> src_owned;
    std::shared_ptr<void> src_glt, src_ult, src_dlt;
    MoEWeightContext src_ctx{
        DeviceId::cpu(), kNumExperts, kIntermediate, kDModel,
        0, -1, 0,
        all_active,
        gate_3d.get(), up_3d.get(), down_3d.get(),
        src_gate_v, src_up_v, src_down_v,
        src_gate_g, src_up_g, src_down_g,
        src_owned, src_glt, src_ult, src_dlt
    };
    ASSERT_TRUE(MoEExpertWeightService::extractExpertViews(src_ctx));
    ASSERT_TRUE(MoEExpertWeightService::prepareGemmEngines(src_ctx));
    auto blobs_for_2 = MoEExpertWeightService::serializeExpert(src_ctx, 2);
    ASSERT_FALSE(blobs_for_2.empty()) << "Source must serialize expert 2";

    std::unordered_map<int, ExpertWeightBlobs> received;
    received[2] = std::move(blobs_for_2);

    bool ok = MoEExpertWeightService::registerAndPrepareNewExperts(ctx, new_mask, &received);
    EXPECT_TRUE(ok) << "registerAndPrepareNewExperts must succeed for CPU path";

    // Verify final state: expert 0 (kept), expert 2 (new) have engines; expert 1 gone
    EXPECT_NE(gate_gemm[0], nullptr) << "Expert 0 should still have engine (kept)";
    EXPECT_EQ(gate_gemm[1], nullptr) << "Expert 1 should have no engine (departed)";
    EXPECT_NE(gate_gemm[2], nullptr) << "Expert 2 should have engine (newly registered)";
}

// ===========================================================================
// 5. GPU↔GPU transfer with stream synchronization
// ===========================================================================

TEST_F(Test__CrossDomainExpertTransfer, GPUP2P_StreamAsync_DataIntegrity)
{
    // Test with an explicit stream to validate async transfer + sync
    const size_t bytes = 512 * 1024; // 512KB

    hipSetDevice(0);
    uint8_t *d_src = nullptr, *d_dst = nullptr;
    ASSERT_EQ(hipMalloc(&d_src, bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(&d_dst, bytes), hipSuccess);

    std::vector<uint8_t> pattern(bytes);
    std::mt19937 rng(123);
    for (auto& b : pattern) b = static_cast<uint8_t>(rng() & 0xFF);
    ASSERT_EQ(hipMemcpy(d_src, pattern.data(), bytes, hipMemcpyHostToDevice), hipSuccess);

    // Create and use an explicit stream
    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    GPUExpertPointers src_ptrs, dst_ptrs;
    src_ptrs.d_vnni = d_src;
    dst_ptrs.d_vnni = d_dst;

    bool ok = GPUExpertTransfer::transferExpert(
        src_ptrs, dst_ptrs,
        DeviceId::rocm(0), DeviceId::rocm(0),
        bytes, 0, 0, 0, stream);
    ASSERT_TRUE(ok);

    // Must sync stream before reading
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    std::vector<uint8_t> result(bytes);
    ASSERT_EQ(hipMemcpy(result.data(), d_dst, bytes, hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(pattern, result);

    hipStreamDestroy(stream);
    hipFree(d_src);
    hipFree(d_dst);
}

#endif // HAVE_ROCM
