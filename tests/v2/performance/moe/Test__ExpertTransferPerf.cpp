/**
 * @file Test__ExpertTransferPerf.cpp
 * @brief Performance test comparing expert weight transfer paths:
 *
 *   1. Full VNNI repack from raw tensor data (prepareExpertGemmLocal)
 *   2. Transfer blob deserialization (createExpertGemmFromTransferBlob)
 *   3. Serialization overhead (cloneWeights + serialize)
 *
 * Validates that the transfer blob path achieves meaningful speedup
 * over full repack, justifying the cross-rank transfer optimization.
 *
 * Test dimensions simulate real Qwen3.5-35B expert sizes:
 *   gate/up: N=2560 (intermediate_size), K=2048 (d_model)
 *   down:    N=2048 (d_model), K=2560 (intermediate_size)
 */

#include <gtest/gtest.h>

#include "kernels/KernelFactory.h"
#include "kernels/PackedWeightsSerialization.h"
#include "kernels/cpu/native_vnni/CPUPackedWeights.h"
#include "backends/DeviceId.h"
#include "utils/TestTensorFactory.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace llaminar2;
using KF = llaminar::v2::kernels::KernelFactory;
using namespace llaminar2::test;
using namespace llaminar2::packed_weights_serialization;

namespace {

struct TimingResult {
    double mean_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
    int iterations = 0;
};

/// Run a function multiple times and collect timing stats.
template<typename Fn>
TimingResult benchmark(Fn&& fn, int warmup = 1, int iterations = 5)
{
    // Warmup
    for (int i = 0; i < warmup; ++i)
        fn();

    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    TimingResult r;
    r.iterations = iterations;
    r.min_ms = *std::min_element(times.begin(), times.end());
    r.max_ms = *std::max_element(times.begin(), times.end());
    r.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    return r;
}

void printResult(const char* label, const TimingResult& r, size_t blob_bytes = 0)
{
    std::cout << "  " << std::left << std::setw(40) << label
              << " mean=" << std::fixed << std::setprecision(2) << r.mean_ms << " ms"
              << "  min=" << r.min_ms << " ms"
              << "  max=" << r.max_ms << " ms";
    if (blob_bytes > 0) {
        double mb = static_cast<double>(blob_bytes) / (1024.0 * 1024.0);
        std::cout << "  (" << std::setprecision(1) << mb << " MB)";
    }
    std::cout << "\n";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class Test__ExpertTransferPerf : public ::testing::Test
{
protected:
    void TearDown() override
    {
        KF::clearCache();
    }
};

// ---------------------------------------------------------------------------
// Single-expert benchmark: full repack vs transfer blob
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferPerf, SingleExpert_RepackVsTransferBlob)
{
    // Simulate Qwen3.5-35B gate/up expert: N=2560, K=2048
    const int N = 2560;
    const int K = 2048;

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n"
              << "║  Expert Transfer Performance: Single Expert (N=" << N << " K=" << K << ")  ║\n"
              << "╠══════════════════════════════════════════════════════════════════╣\n";

    auto tensor = TestTensorFactory::createQ4_0Random({N, K}, /*seed=*/42);
    ASSERT_NE(tensor, nullptr);

    // --- Path A: Full VNNI repack (prepareExpertGemmLocal) ---
    auto repack_timing = benchmark([&]() {
        KF::clearCache();
        auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
        ASSERT_NE(engine, nullptr);
    }, /*warmup=*/1, /*iterations=*/5);
    printResult("Full VNNI repack (prepareExpertGemmLocal)", repack_timing);

    // --- Prepare the blob (done once by sender) ---
    auto engine_for_serialize = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(engine_for_serialize, nullptr);
    auto packed = engine_for_serialize->cloneWeights();
    ASSERT_NE(packed, nullptr);
    auto blob = serialize(*packed);
    ASSERT_FALSE(blob.empty());

    // --- Path B: Serialize (sender overhead) ---
    auto serialize_timing = benchmark([&]() {
        auto p = engine_for_serialize->cloneWeights();
        auto b = serialize(*p);
        ASSERT_FALSE(b.empty());
    }, /*warmup=*/1, /*iterations=*/5);
    printResult("Serialization (sender side)", serialize_timing, blob.size());

    // --- Path C: Create from transfer blob (receiver) ---
    auto transfer_timing = benchmark([&]() {
        auto engine = KF::createExpertGemmFromTransferBlob(blob);
        ASSERT_NE(engine, nullptr);
    }, /*warmup=*/1, /*iterations=*/5);
    printResult("Transfer blob deserialize (receiver)", transfer_timing, blob.size());

    // --- Summary ---
    double speedup = repack_timing.mean_ms / transfer_timing.mean_ms;
    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n"
              << "║  Speedup (repack vs transfer blob): "
              << std::fixed << std::setprecision(1) << speedup << "x"
              << "                              ║\n"
              << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // The transfer blob path should be significantly faster than full repack
    // (deserialization is just memcpy + metadata parse vs full VNNI interleaving)
    EXPECT_GT(speedup, 2.0)
        << "Transfer blob path should be at least 2x faster than full VNNI repack";
}

// ---------------------------------------------------------------------------
// Batch expert benchmark: simulate 8-expert rebalance
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferPerf, BatchExperts_RebalanceSimulation)
{
    // Simulate: 8 experts transferred during rebalance (gate only, for timing)
    const int NUM_EXPERTS = 8;
    const int N = 2560;
    const int K = 2048;

    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n"
              << "║  Batch Expert Transfer: " << NUM_EXPERTS << " experts (N=" << N << " K=" << K << ")      ║\n"
              << "╠══════════════════════════════════════════════════════════════════╣\n";

    // Create expert tensors and pre-serialize blobs
    std::vector<std::unique_ptr<Q4_0Tensor>> tensors;
    std::vector<std::vector<uint8_t>> blobs;

    for (int i = 0; i < NUM_EXPERTS; ++i) {
        tensors.push_back(TestTensorFactory::createQ4_0Random({N, K}, /*seed=*/100 + i));
        auto engine = KF::prepareExpertGemmLocal(tensors.back().get(), DeviceId::cpu());
        ASSERT_NE(engine, nullptr);
        auto packed = engine->cloneWeights();
        auto blob = serialize(*packed);
        blobs.push_back(std::move(blob));
    }
    KF::clearCache();

    // Batch repack from raw data
    auto batch_repack_timing = benchmark([&]() {
        KF::clearCache();
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            auto engine = KF::prepareExpertGemmLocal(tensors[i].get(), DeviceId::cpu());
            ASSERT_NE(engine, nullptr);
        }
    }, /*warmup=*/1, /*iterations=*/3);
    printResult("Batch repack (8 experts)", batch_repack_timing);

    // Batch transfer blob
    auto batch_transfer_timing = benchmark([&]() {
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            auto engine = KF::createExpertGemmFromTransferBlob(blobs[i]);
            ASSERT_NE(engine, nullptr);
        }
    }, /*warmup=*/1, /*iterations=*/3);

    size_t total_blob_bytes = 0;
    for (const auto& b : blobs) total_blob_bytes += b.size();
    printResult("Batch transfer blob (8 experts)", batch_transfer_timing, total_blob_bytes);

    double speedup = batch_repack_timing.mean_ms / batch_transfer_timing.mean_ms;
    double saved_ms = batch_repack_timing.mean_ms - batch_transfer_timing.mean_ms;

    std::cout << "╠══════════════════════════════════════════════════════════════════╣\n"
              << "║  Batch speedup: " << std::fixed << std::setprecision(1) << speedup << "x"
              << "    Time saved: " << std::setprecision(0) << saved_ms << " ms"
              << "                       ║\n"
              << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    EXPECT_GT(speedup, 2.0)
        << "Batch transfer should be at least 2x faster than batch repack";
}

// ---------------------------------------------------------------------------
// Serialization bandwidth measurement
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferPerf, SerializationBandwidth)
{
    // Measure serialization throughput (relevant for MPI sender overhead)
    const int N = 2560;
    const int K = 2048;

    auto tensor = TestTensorFactory::createQ4_0Random({N, K}, /*seed=*/55);
    auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(engine, nullptr);

    std::vector<uint8_t> blob;
    auto timing = benchmark([&]() {
        auto packed = engine->cloneWeights();
        blob = serialize(*packed);
    }, /*warmup=*/2, /*iterations=*/10);

    double mb = static_cast<double>(blob.size()) / (1024.0 * 1024.0);
    double bw_gbps = (mb / 1024.0) / (timing.mean_ms / 1000.0);

    std::cout << "\n  Serialization bandwidth: "
              << std::fixed << std::setprecision(2) << bw_gbps << " GB/s"
              << " (" << std::setprecision(1) << mb << " MB in "
              << std::setprecision(2) << timing.mean_ms << " ms)\n\n";

    // Serialization should be memory-bandwidth-bound (>1 GB/s on modern hardware)
    EXPECT_GT(bw_gbps, 0.5) << "Serialization should achieve >0.5 GB/s";
}

// ---------------------------------------------------------------------------
// Deserialization bandwidth measurement
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferPerf, DeserializationBandwidth)
{
    // Measure deserialization throughput (relevant for MPI receiver overhead)
    const int N = 2560;
    const int K = 2048;

    auto tensor = TestTensorFactory::createQ4_0Random({N, K}, /*seed=*/66);
    auto engine = KF::prepareExpertGemmLocal(tensor.get(), DeviceId::cpu());
    ASSERT_NE(engine, nullptr);
    auto packed = engine->cloneWeights();
    auto blob = serialize(*packed);
    ASSERT_FALSE(blob.empty());

    auto timing = benchmark([&]() {
        auto result = KF::createExpertGemmFromTransferBlob(blob);
        ASSERT_NE(result, nullptr);
    }, /*warmup=*/2, /*iterations=*/10);

    double mb = static_cast<double>(blob.size()) / (1024.0 * 1024.0);
    double bw_gbps = (mb / 1024.0) / (timing.mean_ms / 1000.0);

    std::cout << "\n  Deserialization bandwidth: "
              << std::fixed << std::setprecision(2) << bw_gbps << " GB/s"
              << " (" << std::setprecision(1) << mb << " MB in "
              << std::setprecision(2) << timing.mean_ms << " ms)\n\n";

    // Deserialization should be very fast (mostly memcpy of pre-packed data)
    EXPECT_GT(bw_gbps, 1.0) << "Deserialization should achieve >1 GB/s";
}

// ---------------------------------------------------------------------------
// End-to-end: 3-projection expert transfer time budget
// ---------------------------------------------------------------------------

TEST_F(Test__ExpertTransferPerf, ThreeProjection_TransferTimeBudget)
{
    // Full expert = gate + up + down (3 projections)
    // Target: <20ms total for one expert's 3 projections via transfer path
    const int GATE_N = 2560, GATE_K = 2048;
    const int UP_N = 2560, UP_K = 2048;
    const int DOWN_N = 2048, DOWN_K = 2560;

    auto gate = TestTensorFactory::createQ4_0Random({GATE_N, GATE_K}, /*seed=*/1);
    auto up   = TestTensorFactory::createQ4_0Random({UP_N, UP_K}, /*seed=*/2);
    auto down = TestTensorFactory::createQ4_0Random({DOWN_N, DOWN_K}, /*seed=*/3);

    // Prepare blobs (sender side)
    auto gate_engine = KF::prepareExpertGemmLocal(gate.get(), DeviceId::cpu());
    auto up_engine   = KF::prepareExpertGemmLocal(up.get(), DeviceId::cpu());
    auto down_engine = KF::prepareExpertGemmLocal(down.get(), DeviceId::cpu());
    ASSERT_NE(gate_engine, nullptr);
    ASSERT_NE(up_engine, nullptr);
    ASSERT_NE(down_engine, nullptr);

    auto gate_blob = serialize(*gate_engine->cloneWeights());
    auto up_blob   = serialize(*up_engine->cloneWeights());
    auto down_blob = serialize(*down_engine->cloneWeights());

    size_t total_bytes = gate_blob.size() + up_blob.size() + down_blob.size();

    // Measure full 3-projection transfer (receiver side)
    auto timing = benchmark([&]() {
        auto g = KF::createExpertGemmFromTransferBlob(gate_blob);
        auto u = KF::createExpertGemmFromTransferBlob(up_blob);
        auto d = KF::createExpertGemmFromTransferBlob(down_blob);
        ASSERT_NE(g, nullptr);
        ASSERT_NE(u, nullptr);
        ASSERT_NE(d, nullptr);
    }, /*warmup=*/2, /*iterations=*/5);

    double mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    std::cout << "\n  3-projection transfer: "
              << std::fixed << std::setprecision(2) << timing.mean_ms << " ms"
              << " (" << std::setprecision(1) << mb << " MB total)\n\n";

    // Time budget: <20ms for a full expert transfer (3 projections)
    // This is the receiver-side overhead that determines rebalance latency
    EXPECT_LT(timing.mean_ms, 20.0)
        << "Full 3-projection transfer must complete in <20ms (receiver side)";
}
