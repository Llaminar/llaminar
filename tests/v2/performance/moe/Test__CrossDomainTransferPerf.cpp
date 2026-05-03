/**
 * @file Test__CrossDomainTransferPerf.cpp
 * @brief Performance benchmarks for cross-domain MoE expert weight transfer paths.
 *
 * Measures:
 *   1. GPU↔GPU P2P DMA bandwidth (same-device memcpy baseline + cross-device)
 *   2. Host-staged fallback bandwidth comparison
 *   3. CPU↔GPU activation transfer (CrossDomainTransfer) bandwidth
 *   4. Expert serialize → transfer → deserialize round-trip latency
 *   5. computeGpuCacheExpertMasks computation time for large expert counts
 *
 * Requires: HAVE_ROCM and at least 1 GPU for GPU benchmarks.
 */

#include <gtest/gtest.h>

#include "execution/moe/GPUExpertTransfer.h"
#include "execution/moe/MoEExpertWeightService.h"
#include "execution/moe/MoERebalanceController.h"
#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/local_execution/coherence/CrossDomainTransfer.h"
#include "kernels/KernelFactory.h"
#include "kernels/PackedWeightsSerialization.h"
#include "backends/DeviceId.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "utils/TestTensorFactory.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test;
using KF = llaminar::v2::kernels::KernelFactory;
using namespace llaminar2::packed_weights_serialization;

// ===========================================================================
// Utilities
// ===========================================================================

namespace {

struct BandwidthResult {
    double mean_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
    double bandwidth_gbps = 0; // GB/s based on min time
    size_t bytes = 0;
    int iterations = 0;
};

template<typename Fn>
BandwidthResult benchmarkBandwidth(Fn&& fn, size_t bytes, int warmup = 2, int iterations = 10)
{
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    BandwidthResult r;
    r.bytes = bytes;
    r.iterations = iterations;
    r.min_ms = *std::min_element(times.begin(), times.end());
    r.max_ms = *std::max_element(times.begin(), times.end());
    r.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    // BW based on min (best case)
    r.bandwidth_gbps = (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (r.min_ms / 1000.0);
    return r;
}

void printBW(const char* label, const BandwidthResult& r)
{
    double mb = static_cast<double>(r.bytes) / (1024.0 * 1024.0);
    std::cout << "  " << std::left << std::setw(45) << label
              << " mean=" << std::fixed << std::setprecision(3) << r.mean_ms << " ms"
              << "  min=" << std::setprecision(3) << r.min_ms << " ms"
              << "  BW=" << std::setprecision(2) << r.bandwidth_gbps << " GB/s"
              << "  (" << std::setprecision(1) << mb << " MB)\n";
}

void printLatency(const char* label, double mean_ms, double min_ms, double max_ms)
{
    std::cout << "  " << std::left << std::setw(45) << label
              << " mean=" << std::fixed << std::setprecision(3) << mean_ms << " ms"
              << "  min=" << std::setprecision(3) << min_ms << " ms"
              << "  max=" << std::setprecision(3) << max_ms << " ms\n";
}

} // anonymous namespace

// ===========================================================================
// Fixture
// ===========================================================================

class Test__CrossDomainTransferPerf : public ::testing::Test
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
// 1. GPU D2D bandwidth — baseline same-device memcpy
// ===========================================================================

#ifdef HAVE_ROCM

TEST_F(Test__CrossDomainTransferPerf, GPUD2D_SameDevice_Bandwidth)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n"
              << "║  GPU D2D Transfer Bandwidth — Same Device                            ║\n"
              << "╠══════════════════════════════════════════════════════════════════════╣\n";

    hipSetDevice(0);

    // Test multiple sizes to show bandwidth scaling
    std::vector<size_t> sizes = {
        256 * 1024,         // 256KB — small expert component
        1024 * 1024,        // 1MB
        3 * 1024 * 1024,    // 3MB — typical expert gate/up weight
        8 * 1024 * 1024,    // 8MB — full expert (gate+up+down)
    };

    for (size_t sz : sizes) {
        uint8_t *d_src = nullptr, *d_dst = nullptr;
        ASSERT_EQ(hipMalloc(&d_src, sz), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_dst, sz), hipSuccess);

        // Initialize source
        std::vector<uint8_t> fill(sz, 0xAB);
        hipMemcpy(d_src, fill.data(), sz, hipMemcpyHostToDevice);

        auto result = benchmarkBandwidth([&]() {
            GPUExpertPointers src_ptrs, dst_ptrs;
            src_ptrs.d_vnni = d_src;
            dst_ptrs.d_vnni = d_dst;
            GPUExpertTransfer::transferExpert(
                src_ptrs, dst_ptrs,
                DeviceId::rocm(0), DeviceId::rocm(0),
                sz, 0, 0, 0, nullptr);
            hipDeviceSynchronize();
        }, sz, /*warmup=*/3, /*iterations=*/10);

        char label[64];
        snprintf(label, sizeof(label), "D2D same-dev %zuKB", sz / 1024);
        printBW(label, result);

        hipFree(d_src);
        hipFree(d_dst);
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
}

TEST_F(Test__CrossDomainTransferPerf, GPUD2D_CrossDevice_Bandwidth)
{
    if (gpu_count_ < 2) {
        GTEST_SKIP() << "Need 2+ GPUs for cross-device bandwidth test";
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n"
              << "║  GPU D2D Transfer Bandwidth — Cross Device (0 → 1)                   ║\n"
              << "╠══════════════════════════════════════════════════════════════════════╣\n";

    const size_t expert_size = 3 * 1024 * 1024; // 3MB typical expert component

    hipSetDevice(0);
    uint8_t *d_src = nullptr;
    ASSERT_EQ(hipMalloc(&d_src, expert_size), hipSuccess);
    std::vector<uint8_t> fill(expert_size, 0xCD);
    hipMemcpy(d_src, fill.data(), expert_size, hipMemcpyHostToDevice);

    hipSetDevice(1);
    uint8_t *d_dst = nullptr;
    ASSERT_EQ(hipMalloc(&d_dst, expert_size), hipSuccess);

    hipSetDevice(0);

    auto result = benchmarkBandwidth([&]() {
        GPUExpertPointers src_ptrs, dst_ptrs;
        src_ptrs.d_vnni = d_src;
        dst_ptrs.d_vnni = d_dst;
        GPUExpertTransfer::transferExpert(
            src_ptrs, dst_ptrs,
            DeviceId::rocm(0), DeviceId::rocm(1),
            expert_size, 0, 0, 0, nullptr);
        hipDeviceSynchronize();
    }, expert_size, /*warmup=*/3, /*iterations=*/10);

    printBW("D2D cross-dev (0→1) 3MB", result);

    // Also measure raw hipMemcpyPeer as baseline
    auto baseline = benchmarkBandwidth([&]() {
        hipMemcpyPeer(d_dst, 1, d_src, 0, expert_size);
        hipDeviceSynchronize();
    }, expert_size, /*warmup=*/3, /*iterations=*/10);

    printBW("hipMemcpyPeer baseline 3MB", baseline);

    // GPUExpertTransfer should be within 20% of raw hipMemcpyPeer
    double overhead = (result.min_ms - baseline.min_ms) / baseline.min_ms * 100.0;
    std::cout << "  Overhead vs raw hipMemcpyPeer: "
              << std::setprecision(1) << overhead << "%\n";
    EXPECT_LT(overhead, 20.0)
        << "GPUExpertTransfer should have <20% overhead vs raw hipMemcpyPeer";

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    hipSetDevice(0); hipFree(d_src);
    hipSetDevice(1); hipFree(d_dst);
}

// ===========================================================================
// 2. CPU↔GPU activation transfer bandwidth (CrossDomainTransfer)
// ===========================================================================

TEST_F(Test__CrossDomainTransferPerf, CrossDomain_ActivationTransfer_Bandwidth)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n"
              << "║  CrossDomainTransfer Activation Bandwidth (CPU↔GPU)                   ║\n"
              << "╠══════════════════════════════════════════════════════════════════════╣\n";

    DeviceId gpu = DeviceId::rocm(0);

    // Test sizes: [hidden_state at PP boundary]
    // seq_len * d_model * sizeof(float) bytes
    struct Case { size_t seq_len; size_t d_model; const char* label; };
    std::vector<Case> cases = {
        {1, 2048, "Decode (1×2048)"},
        {64, 2048, "Short prefill (64×2048)"},
        {512, 2048, "Medium prefill (512×2048)"},
        {2048, 2048, "Long prefill (2048×2048)"},
    };

    for (auto& c : cases) {
        auto src = TestTensorFactory::createFP32Random({c.seq_len, c.d_model}, -3.0f, 3.0f, 55);
        auto dst = TestTensorFactory::createFP32({c.seq_len, c.d_model});
        size_t bytes = c.seq_len * c.d_model * sizeof(float);

        // CPU → GPU
        CrossDomainTransfer transfer;
        auto h2d = benchmarkBandwidth([&]() {
            transfer.cpuToGpu(src.get(), dst.get(), gpu);
        }, bytes, /*warmup=*/2, /*iterations=*/8);

        char lbl[80];
        snprintf(lbl, sizeof(lbl), "CPU→GPU %s", c.label);
        printBW(lbl, h2d);

        // GPU → CPU
        auto d2h = benchmarkBandwidth([&]() {
            transfer.gpuToCpu(dst.get(), src.get());
        }, bytes, /*warmup=*/2, /*iterations=*/8);

        snprintf(lbl, sizeof(lbl), "GPU→CPU %s", c.label);
        printBW(lbl, d2h);
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
}

// ===========================================================================
// 3. Expert serialize → transfer → deserialize round-trip latency
// ===========================================================================

TEST_F(Test__CrossDomainTransferPerf, ExpertSerializeDeserialize_RoundTripLatency)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n"
              << "║  Expert Serialize → Deserialize Round-Trip Latency                    ║\n"
              << "╠══════════════════════════════════════════════════════════════════════╣\n";

    // Simulate Qwen3.5-35B expert dimensions
    struct ExpertConfig { int N; int K; const char* name; };
    std::vector<ExpertConfig> configs = {
        {2560, 2048, "gate/up (2560×2048)"},
        {2048, 2560, "down (2048×2560)"},
    };

    for (auto& cfg : configs) {
        auto tensor = TestTensorFactory::createQ4_0Random(
            {static_cast<size_t>(cfg.N), static_cast<size_t>(cfg.K)}, /*seed=*/42);
        ASSERT_NE(tensor, nullptr);

        // Prepare engine once
        auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
        ASSERT_NE(engine, nullptr);
        auto packed = engine->cloneWeights();
        ASSERT_NE(packed, nullptr);
        auto blob = serialize(*packed);
        ASSERT_FALSE(blob.empty());
        size_t blob_bytes = blob.size();

        // Benchmark: serialize only
        std::vector<double> ser_times;
        for (int i = 0; i < 10; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto p = engine->cloneWeights();
            auto b = serialize(*p);
            auto t1 = std::chrono::high_resolution_clock::now();
            ser_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        double ser_mean = std::accumulate(ser_times.begin(), ser_times.end(), 0.0) / ser_times.size();
        double ser_min = *std::min_element(ser_times.begin(), ser_times.end());
        double ser_max = *std::max_element(ser_times.begin(), ser_times.end());

        char lbl[80];
        snprintf(lbl, sizeof(lbl), "Serialize %s", cfg.name);
        printLatency(lbl, ser_mean, ser_min, ser_max);

        // Benchmark: deserialize only
        std::vector<double> des_times;
        for (int i = 0; i < 10; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto e = KF::createExpertGemmFromTransferBlob(blob);
            auto t1 = std::chrono::high_resolution_clock::now();
            des_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            KF::clearCache();
        }
        double des_mean = std::accumulate(des_times.begin(), des_times.end(), 0.0) / des_times.size();
        double des_min = *std::min_element(des_times.begin(), des_times.end());
        double des_max = *std::max_element(des_times.begin(), des_times.end());

        snprintf(lbl, sizeof(lbl), "Deserialize %s", cfg.name);
        printLatency(lbl, des_mean, des_min, des_max);

        // Compute effective "transfer" BW (memcpy speed for the blob)
        double effective_bw_gbps = (static_cast<double>(blob_bytes) / (1024.0 * 1024.0 * 1024.0))
                                   / (des_min / 1000.0);
        std::cout << "    Blob size: " << std::setprecision(1)
                  << static_cast<double>(blob_bytes) / (1024.0 * 1024.0)
                  << " MB, effective BW: " << std::setprecision(2)
                  << effective_bw_gbps << " GB/s (deserialize)\n";
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
}

// ===========================================================================
// 4. computeGpuCacheExpertMasks — scalability benchmark
// ===========================================================================

TEST_F(Test__CrossDomainTransferPerf, GpuCacheMasks_Scalability)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n"
              << "║  computeGpuCacheExpertMasks Latency                                   ║\n"
              << "╠══════════════════════════════════════════════════════════════════════╣\n";

    // Test with varying expert/layer counts
    struct Config { int experts; int layers; int cache_per_layer; const char* name; };
    std::vector<Config> configs = {
        {8, 24, 3, "Small (8E, 24L, cache=3)"},
        {64, 28, 16, "Medium (64E, 28L, cache=16)"},
        {128, 64, 32, "Large (128E, 64L, cache=32)"},
    };

    for (auto& cfg : configs) {
        MoERebalanceController::Config rcfg;
        rcfg.mode = MoERebalanceMode::OBSERVE;
        rcfg.num_layers = cfg.layers;
        rcfg.num_experts = cfg.experts;
        rcfg.top_k = 2;
        rcfg.window_size = 128;
        rcfg.sockets = {DeviceId::rocm(0), DeviceId::cpu()};
        rcfg.initial_expert_to_socket.assign(cfg.experts, 0);

        MoERebalanceController controller(rcfg);

        // Populate histogram with realistic distribution (Zipf-like)
        auto* hist = controller.histogram();
        std::mt19937 rng(12345);
        for (int step = 0; step < 100; ++step) {
            for (int l = 0; l < cfg.layers; ++l) {
                // Zipf: expert i has probability proportional to 1/(i+1)
                std::vector<float> probs(cfg.experts);
                float sum = 0;
                for (int e = 0; e < cfg.experts; ++e) {
                    probs[e] = 1.0f / (e + 1);
                    sum += probs[e];
                }
                // Pick top-2 by weighted random
                std::discrete_distribution<int> d(probs.begin(), probs.end());
                int e1 = d(rng);
                int e2 = d(rng);
                while (e2 == e1) e2 = d(rng);
                int indices[2] = {e1, e2};
                float weights[2] = {0.6f, 0.4f};
                hist->record(l, indices, weights, 2);
            }
        }

        // Benchmark mask computation
        std::vector<double> times;
        for (int i = 0; i < 50; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto masks = controller.computeGpuCacheExpertMasks(cfg.cache_per_layer);
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            (void)masks;
        }

        double mean_us = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_us = *std::min_element(times.begin(), times.end());
        double max_us = *std::max_element(times.begin(), times.end());

        std::cout << "  " << std::left << std::setw(45) << cfg.name
                  << " mean=" << std::fixed << std::setprecision(1) << mean_us << " µs"
                  << "  min=" << std::setprecision(1) << min_us << " µs"
                  << "  max=" << std::setprecision(1) << max_us << " µs\n";

        // Mask computation should be under 1ms even for 128 experts × 64 layers
        EXPECT_LT(mean_us, 1000.0)
            << cfg.name << ": mask computation should complete under 1ms";
    }

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";
}

// ===========================================================================
// 5. Full expert GPU transfer — 3 weight components (gate+up+down)
// ===========================================================================

TEST_F(Test__CrossDomainTransferPerf, FullExpertGPUTransfer_ThreeComponents)
{
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n"
              << "║  Full Expert GPU Transfer (gate + up + down) — Same Device             ║\n"
              << "╠══════════════════════════════════════════════════════════════════════╣\n";

    hipSetDevice(0);

    // Qwen3.5 expert sizes (Q4_0 packed, VNNI format):
    // gate: 2560×2048/32 blocks × 18 bytes ≈ 2.9MB
    // up:   2560×2048/32 blocks × 18 bytes ≈ 2.9MB
    // down: 2048×2560/32 blocks × 18 bytes ≈ 2.9MB
    const size_t gate_vnni = 3 * 1024 * 1024;
    const size_t up_vnni = 3 * 1024 * 1024;
    const size_t down_vnni = 3 * 1024 * 1024;
    const size_t gate_scales = 80 * 1024;
    const size_t up_scales = 80 * 1024;
    const size_t down_scales = 80 * 1024;
    const size_t total_bytes = gate_vnni + up_vnni + down_vnni + gate_scales + up_scales + down_scales;

    // Allocate all buffers
    struct Bufs { uint8_t *vnni, *scales; };
    auto allocBuf = [](size_t vnni_sz, size_t scales_sz) -> std::pair<Bufs, Bufs> {
        Bufs src, dst;
        hipMalloc(&src.vnni, vnni_sz);
        hipMalloc(&src.scales, scales_sz);
        hipMalloc(&dst.vnni, vnni_sz);
        hipMalloc(&dst.scales, scales_sz);
        return {src, dst};
    };

    auto [gate_src, gate_dst] = allocBuf(gate_vnni, gate_scales);
    auto [up_src, up_dst] = allocBuf(up_vnni, up_scales);
    auto [down_src, down_dst] = allocBuf(down_vnni, down_scales);

    auto result = benchmarkBandwidth([&]() {
        GPUExpertPointers g_src{gate_src.vnni, gate_src.scales, nullptr, nullptr};
        GPUExpertPointers g_dst{gate_dst.vnni, gate_dst.scales, nullptr, nullptr};
        GPUExpertTransfer::transferExpert(g_src, g_dst,
            DeviceId::rocm(0), DeviceId::rocm(0), gate_vnni, gate_scales, 0, 0, nullptr);

        GPUExpertPointers u_src{up_src.vnni, up_src.scales, nullptr, nullptr};
        GPUExpertPointers u_dst{up_dst.vnni, up_dst.scales, nullptr, nullptr};
        GPUExpertTransfer::transferExpert(u_src, u_dst,
            DeviceId::rocm(0), DeviceId::rocm(0), up_vnni, up_scales, 0, 0, nullptr);

        GPUExpertPointers d_src{down_src.vnni, down_src.scales, nullptr, nullptr};
        GPUExpertPointers d_dst{down_dst.vnni, down_dst.scales, nullptr, nullptr};
        GPUExpertTransfer::transferExpert(d_src, d_dst,
            DeviceId::rocm(0), DeviceId::rocm(0), down_vnni, down_scales, 0, 0, nullptr);

        hipDeviceSynchronize();
    }, total_bytes, /*warmup=*/3, /*iterations=*/10);

    printBW("Full expert (gate+up+down) 3×3MB", result);

    double mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << "  Total payload: " << std::setprecision(1) << mb << " MB\n";
    std::cout << "  Per-expert latency: " << std::setprecision(3) << result.min_ms << " ms (best)\n";

    // Full expert transfer should complete in under 5ms on modern GPUs
    EXPECT_LT(result.min_ms, 5.0)
        << "Full expert transfer should complete in <5ms on modern GPUs";

    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n\n";

    // Cleanup
    hipFree(gate_src.vnni); hipFree(gate_src.scales);
    hipFree(gate_dst.vnni); hipFree(gate_dst.scales);
    hipFree(up_src.vnni); hipFree(up_src.scales);
    hipFree(up_dst.vnni); hipFree(up_dst.scales);
    hipFree(down_src.vnni); hipFree(down_src.scales);
    hipFree(down_dst.vnni); hipFree(down_dst.scales);
}

#endif // HAVE_ROCM
