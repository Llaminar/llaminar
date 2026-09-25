/**
 * @file Test__CPUNativeVNNIVerifierWorkers.cpp
 * @brief Prove real worker ownership and serial-row bytes in small verifier grids.
 *
 * A historical M=2 shortcut advertised a multi-worker grouped schedule but
 * executed every N tile on the calling thread. These model-free regressions
 * observe tasks completed by the production launcher, rather than trusting
 * its requested worker count or enforcing a noisy performance threshold.
 * Both ISA registrations use the canonical quantized-format inventory.
 * Complete FFN transactions preserve every intermediate output byte. A second
 * process registration disables kernel diagnostics and audits C++ allocations
 * only during warmed execution, independently of weight/workspace preparation.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "utils/NativeVNNITestPartialStorage.h"
#include "utils/QuantizedVerifierFormats.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <new>
#include <set>
#include <string>
#include <vector>

namespace
{
/** @brief Process-local C++ allocation witness, armed only around a test invocation. */
std::atomic<bool> audit_cpp_allocations{false};
/** @brief Includes allocations by all members of the tested OpenMP team. */
std::atomic<size_t> audited_cpp_allocations{0};

/** @brief Count one successful C++ allocation without allocating diagnostic storage. */
void observeCppAllocation()
{
    if (audit_cpp_allocations.load(std::memory_order_relaxed))
        audited_cpp_allocations.fetch_add(1, std::memory_order_relaxed);
}

/** @brief Enable allocation observation for one warmed, synchronous invocation. */
class CppAllocationAudit final
{
public:
    /** @brief Start only after fixture creation and runtime warmup have completed. */
    CppAllocationAudit()
    {
        audited_cpp_allocations.store(0, std::memory_order_relaxed);
        audit_cpp_allocations.store(true, std::memory_order_relaxed);
    }
    /** @brief Never include GTest formatting or fixture teardown in the witness. */
    ~CppAllocationAudit() { audit_cpp_allocations.store(false, std::memory_order_relaxed); }
    CppAllocationAudit(const CppAllocationAudit &) = delete;
    CppAllocationAudit &operator=(const CppAllocationAudit &) = delete;
    /** @brief Stop after the invocation has joined its workers and return its count. */
    size_t finish()
    {
        audit_cpp_allocations.store(false, std::memory_order_relaxed);
        return audited_cpp_allocations.load(std::memory_order_relaxed);
    }
};
} // namespace

// These replacements belong only to this dedicated integration executable.
// They observe strings/maps in the real code, not a source-text proxy. C heap
// allocations inside libgomp are deliberately outside this C++ allocation proof.
/** @brief Allocate ordinary test-process C++ storage and publish its witness. */
void *operator new(std::size_t size)
{
    if (void *result = std::malloc(std::max<std::size_t>(size, 1)))
    {
        observeCppAllocation();
        return result;
    }
    throw std::bad_alloc();
}
/** @brief Count array storage through the same scalar allocation witness. */
void *operator new[](std::size_t size) { return ::operator new(size); }
/** @brief Allocate over-aligned storage without losing its requested alignment. */
void *operator new(std::size_t size, std::align_val_t alignment)
{
    void *result = nullptr;
    if (posix_memalign(&result, static_cast<std::size_t>(alignment), std::max<std::size_t>(size, 1)) != 0)
        throw std::bad_alloc();
    observeCppAllocation();
    return result;
}
/** @brief Count aligned arrays through the same aligned allocation witness. */
void *operator new[](std::size_t size, std::align_val_t alignment) { return ::operator new(size, alignment); }
/** @brief Release ordinary C++ storage using its matching C allocator. */
void operator delete(void *pointer) noexcept { std::free(pointer); }
/** @brief Release ordinary array storage. */
void operator delete[](void *pointer) noexcept { std::free(pointer); }
/** @brief Release sized ordinary storage; the allocator owns its actual extent. */
void operator delete(void *pointer, std::size_t) noexcept { std::free(pointer); }
/** @brief Release sized ordinary array storage. */
void operator delete[](void *pointer, std::size_t) noexcept { std::free(pointer); }
/** @brief Release aligned storage allocated with posix_memalign. */
void operator delete(void *pointer, std::align_val_t) noexcept { std::free(pointer); }
/** @brief Release aligned array storage. */
void operator delete[](void *pointer, std::align_val_t) noexcept { std::free(pointer); }
/** @brief Release sized aligned storage. */
void operator delete(void *pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }
/** @brief Release sized aligned array storage. */
void operator delete[](void *pointer, std::size_t, std::align_val_t) noexcept { std::free(pointer); }

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;

/** @brief Restore OpenMP caller policy even when an assertion aborts a case. */
class VerifierWorkerScope final
{
public:
    /** @brief Retain the caller's team width and disable dynamic team shrinking. */
    VerifierWorkerScope()
        : workers_(omp_get_max_threads()), dynamic_(omp_get_dynamic())
    {
        omp_set_dynamic(0);
    }

    /** @brief Restore both independent OpenMP policy values. */
    ~VerifierWorkerScope()
    {
        omp_set_num_threads(workers_);
        omp_set_dynamic(dynamic_);
    }

private:
    int workers_; ///< Caller-owned worker budget.
    int dynamic_; ///< Caller-owned permission to shrink a team.
};

/**
 * @brief Compare complete grouped output and exact task ownership with serial decode.
 * @param format Canonical source-format creator and label.
 * @param n Logical output width, including a possible masked final tile.
 * @param k Logical input width.
 *
 * Explicit full-K pair-grid execution isolates the affected production
 * launcher independently of changing learned Auto choices. A 64-column task
 * owns all of its K arithmetic, so scheduling must not alter any output byte.
 */
void proveVerifierWorkers(const QuantizedVerifierFormatCase &format, int n, int k)
{
    constexpr int max_rows = 15;
    VerifierWorkerScope restore;
    auto weights = format.create({size_t(n), size_t(k)}, 9421);
    ASSERT_NE(weights, nullptr);
    CPUNativeVNNIGemmKernel kernel(weights.get());
    ASSERT_TRUE(kernel.isValid());
    const auto &packed = kernel.packedWeights();
    auto inputs = TestTensorFactory::createFP32Random(
        {size_t(max_rows), size_t(k)}, -0.75f, 0.75f, 9427);
    std::vector<Q8_1Block> quantized(size_t(max_rows) * packed.blocks_per_row);
    quantize_activations_to_q8_1(inputs->data(), quantized.data(),
        max_rows, k, packed.blocks_per_row);
    std::vector<float> serial(size_t(max_rows) * n);
    // A poisoned, padded output detects missing tiles and tail overwrites.
    const int stride = n + 7;
    std::vector<float> grouped(size_t(max_rows) * stride, -12345.0f);

    for (const int workers : {1, 2, 3, 7, 28, 31})
    {
        omp_set_num_threads(workers);
        NativeVNNITestPartialStorage partials(packed, max_rows);
        for (int row = 0; row < max_rows; ++row)
        {
            gemv_native_vnni_preq(packed,
                quantized.data() + size_t(row) * packed.blocks_per_row,
                serial.data() + size_t(row) * n, partials.span());
        }
        for (const int rows : {2, 3, 4, 15})
        {
            SCOPED_TRACE(std::string(format.label) + " N=" + std::to_string(n) +
                " K=" + std::to_string(k) + " M=" + std::to_string(rows) +
                " workers=" + std::to_string(workers));
            std::fill(grouped.begin(), grouped.end(), -12345.0f);
            PerfStatsCollector::reset();
            gemm_native_vnni_preq_decode_equivalent_rows(
                packed, quantized.data(), grouped.data(), partials.span(),
                rows, stride, ISAPath::AUTO,
                VerifierRowsPolicy::FullKTwoRowPairGridNbc1);
            for (int row = 0; row < rows; ++row)
            {
                ASSERT_EQ(std::memcmp(grouped.data() + size_t(row) * stride,
                    serial.data() + size_t(row) * n, size_t(n) * sizeof(float)), 0);
                for (int column = n; column < stride; ++column)
                    ASSERT_EQ(grouped[size_t(row) * stride + column], -12345.0f);
            }
            ASSERT_TRUE(std::all_of(grouped.begin() + size_t(rows) * stride,
                grouped.end(), [](float value) { return value == -12345.0f; }));

            const int tasks = ((rows + 1) / 2) * ((n + 63) / 64);
            std::set<int> owners;
            double completed = 0.0;
            for (const auto &record : PerfStatsCollector::snapshot(
                     {"kernel.cpu_native_vnni_verifier_worker_tasks"}))
            {
                EXPECT_EQ(record.tags.at("workers"), std::to_string(workers));
                EXPECT_EQ(record.tags.at("route"), "grouped_full_k_pair_grid");
                owners.insert(std::stoi(record.tags.at("worker")));
                completed += record.value;
            }
            EXPECT_EQ(completed, tasks);
            EXPECT_EQ(owners.size(), size_t(std::min(workers, tasks)));
        }
    }
}

/** @test Every quantized codebook keeps the promised small-grid worker ownership. */
TEST(CPUNativeVNNIVerifierWorkers, AllFormatsUseResolvedTeamAndPreserveRowBytes)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    for (const auto &format : quantizedVerifierFormats())
        proveVerifierWorkers(format, 257, 256);
}

/** @test The real Qwen projection geometry cannot silently execute on one worker. */
TEST(CPUNativeVNNIVerifierWorkers, DenseProjectionUsesResolvedTeamAndPreservesRowBytes)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    for (const auto &format : quantizedVerifierFormats())
        if (format.tensor_type == TensorType::Q6_K)
            proveVerifierWorkers(format, 2048, 2048);
}

/** @brief Prove real FFN and router-Q8 bytes, auditing allocations when collection is disabled. */
void proveExpertTransaction()
{
    VerifierWorkerScope restore;
    constexpr int width = 256, max_rows = 31;
    constexpr int blocks = width / Q8_1Block::BLOCK_SIZE;
    constexpr auto policy = CPUProjectionNumericalPolicy::GPUAlignedExpert;
    using Desc = CPUNativeVNNIGemmKernel::BatchedPrequantizedProjectionDesc;
    auto input = TestTensorFactory::createFP32Random({max_rows, width}, -0.75f, 0.75f, 7103);
    std::vector<Q8_1Block> q8(max_rows * blocks), actual_q8(q8.size()), expected_q8(q8.size());
    std::vector<float> gate(max_rows * width), up(gate.size()), down(gate.size());
    std::vector<float> expected_gate(gate.size()), expected_up(gate.size()), expected_down(gate.size());
    std::array<float, width> activated{};

    for (const auto &format : quantizedVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        omp_set_num_threads(1);
        auto gate_weight = format.create({width, width}, 7109);
        auto up_weight = format.create({width, width}, 7117);
        auto down_weight = format.create({width, width}, 7121);
        CPUNativeVNNIGemmKernel gate_kernel(gate_weight.get(), 0, -1, policy);
        CPUNativeVNNIGemmKernel up_kernel(up_weight.get(), 0, -1, policy);
        CPUNativeVNNIGemmKernel down_kernel(down_weight.get(), 0, -1, policy);
        ASSERT_TRUE(gate_kernel.isValid() && up_kernel.isValid() && down_kernel.isValid());
        FP32Tensor router_output({max_rows, width});
        const std::vector<ITensorGemm::TensorProjectionDesc> router_projection{
            {&gate_kernel, &router_output, width, nullptr, "router_gate"}};
        quantize_activations_to_q8_1(input->data(), q8.data(), max_rows, width, blocks, policy);
        NativeVNNITestPartialStorage serial_partials(gate_kernel.packedWeights(), 1);
        for (int row = 0; row < max_rows; ++row)
        {
            gemv_native_vnni_preq(gate_kernel.packedWeights(), q8.data() + row * blocks,
                expected_gate.data() + row * width, serial_partials.span());
            gemv_native_vnni_preq(up_kernel.packedWeights(), q8.data() + row * blocks,
                expected_up.data() + row * width, serial_partials.span());
            primitives::compute_swiglu_gpu_aligned_expert_serial(expected_gate.data() + row * width,
                expected_up.data() + row * width, activated.data(), width);
            quantize_activations_to_q8_1(activated.data(), expected_q8.data() + row * blocks,
                1, width, blocks, policy);
            gemv_native_vnni_preq(down_kernel.packedWeights(), expected_q8.data() + row * blocks,
                expected_down.data() + row * width, serial_partials.span());
        }
        for (int workers : {1, 3, 7})
        for (int rows : {2, 3, 15, max_rows})
        {
            SCOPED_TRACE("workers=" + std::to_string(workers) + " rows=" + std::to_string(rows));
            omp_set_num_threads(workers);
            // One single-row member and one grouped member exercise unequal
            // schedules. Their immutable weights may be shared, not their rows.
            const std::array<Desc, 4> gate_up{{
                {.kernel=&gate_kernel, .input_q8=q8.data(), .output=gate.data(), .rows=1, .n=width, .ldc=width},
                {.kernel=&up_kernel, .input_q8=q8.data(), .output=up.data(), .rows=1, .n=width, .ldc=width},
                {.kernel=&gate_kernel, .input_q8=q8.data()+blocks, .output=gate.data()+width, .rows=rows-1, .n=width, .ldc=width},
                {.kernel=&up_kernel, .input_q8=q8.data()+blocks, .output=up.data()+width, .rows=rows-1, .n=width, .ldc=width},
            }};
            std::array<Desc, 2> down_desc{{
                {.kernel=&down_kernel, .input_q8=actual_q8.data(), .output=down.data(), .rows=1, .n=width, .ldc=width},
                {.kernel=&down_kernel, .input_q8=actual_q8.data()+blocks, .output=down.data()+width, .rows=rows-1, .n=width, .ldc=width},
            }};
            NativeVNNITestPartialStorage partials(std::max(
                NativeVNNITestPartialStorage::bundleFloats(gate_up.data(), gate_up.size(), 1),
                NativeVNNITestPartialStorage::bundleFloats(down_desc.data(), down_desc.size(), 1)));
            const auto run = [&] {
                return CPUNativeVNNIGemmKernel::execute_moe_grouped_ffn_transaction_preq_decode_equivalent(
                    gate_up.data(), gate_up.size(), width, gate.data(), up.data(), actual_q8.data(),
                    rows, width, blocks, down_desc.data(), down_desc.size(), partials.span());
            };
            const auto verify = [&] {
                EXPECT_EQ(std::memcmp(gate.data(), expected_gate.data(), rows * width * sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(up.data(), expected_up.data(), rows * width * sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(down.data(), expected_down.data(), rows * width * sizeof(float)), 0);
                EXPECT_EQ(std::memcmp(actual_q8.data(), expected_q8.data(), rows * blocks * sizeof(Q8_1Block)), 0);
            };
            std::fill(gate.begin(), gate.end(), -12345.0f);
            // Invalid later phases must be rejected before the first gate write.
            down_desc.back().ldc = width - 1;
            EXPECT_FALSE(run());
            EXPECT_TRUE(std::all_of(gate.begin(), gate.end(), [](float x) { return x == -12345.0f; }));
            down_desc.back().ldc = width;
            PerfStatsCollector::reset();
            ASSERT_TRUE(run());
            verify();
            if (!PerfStatsCollector::isDomainEnabled("kernel"))
            {
                // Warmup above pays for OpenMP and immutable policy setup. The
                // real FFN may not allocate C++ diagnostics when disabled.
                CppAllocationAudit audit;
                const bool complete = run();
                const size_t allocations = audit.finish();
                ASSERT_TRUE(complete);
                EXPECT_EQ(allocations, 0u);
                verify();
            }

            // An already-active team remains supported. Shared output
            // workshares and their barriers retain the same byte contract.
            std::atomic<bool> completed{true};
#pragma omp parallel num_threads(workers)
            {
                if (!run()) completed.store(false, std::memory_order_relaxed);
            }
            ASSERT_TRUE(completed.load(std::memory_order_relaxed));
            verify();

            // The router-published Q8 wrapper is another production entry to
            // the same math. Check both its serial and grouped diagnostic sites.
            NativeVNNITestPartialStorage router_partials(gate_kernel.packedWeights(), rows);
            for (const int router_rows : {1, rows})
            {
                const auto project = [&] {
                    return gate_kernel.multiply_fused_router_q8_hidden_grouped_decode_equivalent(
                        q8.data(), router_projection, router_rows, width, router_partials.span());
                };
                ASSERT_TRUE(project());
                if (!PerfStatsCollector::isDomainEnabled("kernel"))
                {
                    CppAllocationAudit audit;
                    const bool projected = project();
                    const size_t allocations = audit.finish();
                    ASSERT_TRUE(projected);
                    EXPECT_EQ(allocations, 0u) << "router rows=" << router_rows;
                }
                EXPECT_EQ(std::memcmp(router_output.data(), expected_gate.data(),
                    router_rows * width * sizeof(float)), 0);
            }
        }
    }
}

/** @test Enabled production diagnostics preserve every intermediate output byte. */
TEST(CPUNativeVNNIVerifierWorkers, ExpertTransactionPreservesAllFormats)
{
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("kernel"));
    proveExpertTransaction();
}

/** @test Disabled collection must not allocate before the collector rejects a record. */
TEST(CPUNativeVNNIVerifierWorkers, DisabledDiagnosticsDoNotAllocate)
{
    ASSERT_FALSE(PerfStatsCollector::isDomainEnabled("kernel"));
    // Prove the interposer observes real allocation before trusting a zero.
    CppAllocationAudit audit;
    void *probe = ::operator new(19);
    const size_t witnessed = audit.finish();
    ::operator delete(probe);
    ASSERT_EQ(witnessed, 1u);
    proveExpertTransaction();
}
} // namespace
} // namespace llaminar2::test
