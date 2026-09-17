/**
 * @file Test__CPUProjectionWorkspace.cpp
 * @brief All-format proof of participant-owned CPU transform and Q8 scratch.
 *
 * The real prepared kernels execute serial decode and grouped verification.
 * Shared weights see separate, reused arenas; missing or undersized storage
 * fails before any transform write. This is a model-free preflight regression,
 * not a replacement for captured model parity or a performance certificate.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "execution/compute_stages/stages/LMHeadStage.h"
#include "utils/CPUProjectionTestWorkspace.h"
#include "utils/NativeVNNITestPartialStorage.h"
#include "utils/PreparedWeightTestHarness.h"
#include "utils/QuantizedVerifierFormats.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <future>

namespace llaminar2::test
{
namespace
{
/** @brief Restore the caller's worker budget after the focused thread sweep. */
struct WorkerScope final
{
    int previous = omp_get_max_threads(); ///< Initial worker count restored on retirement.
    ~WorkerScope() { omp_set_num_threads(previous); }
};

/** @test Remote metadata declares every real prepared engine's exact named buffers. */
TEST(CPUProjectionWorkspace, MetadataMatchesPreparedRequirementsAllFormatsAndTPWidths)
{
    using namespace cpu::native_vnni;
    WorkerScope restore;
    ASSERT_TRUE(CPUExecutionGeometry::local().isValid());
    for (const auto &format : quantizedMoEVerifierFormats())
    for (const int k : {256, 4096})
    {
        auto weight = format.create({513u, size_t(k)}, 773);
        for (int shards = 1; shards <= 8; ++shards)
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative,
                                 CPUProjectionNumericalPolicy::GPUAlignedExpert})
        {
            const int n = (513 + shards - 1) / shards;
            CPUNativeVNNIGemmKernel kernel(weight.get(), 0, n, policy);
            ASSERT_TRUE(kernel.isValid());
            for (const int workers : {1, 2, 3, 8, 28})
            for (const int rows : {1, 2, 3, 15, 31})
            {
                SCOPED_TRACE(std::string(format.label) + " K=" + std::to_string(k) +
                    " TP=" + std::to_string(shards) + " workers=" + std::to_string(workers) +
                    " rows=" + std::to_string(rows));
                omp_set_num_threads(workers);
                const auto actual = kernel.getWorkspaceRequirements(rows);
                const auto metadata = CPUProjectionWorkspaceContract::sourceNative(format.label, {
                    .rows = rows, .n = n, .k = k, .workers = workers,
                    .execution = CPUExecutionGeometry::local(), .numerical_policy = policy});
                ASSERT_EQ(actual.buffers.size(), metadata.buffers.size());
                EXPECT_EQ(actual.total_bytes_with_alignment(), metadata.total_bytes_with_alignment());
                for (size_t i = 0; i < actual.buffers.size(); ++i)
                {
                    EXPECT_EQ(actual.buffers[i].name, metadata.buffers[i].name);
                    EXPECT_EQ(actual.buffers[i].size_bytes, metadata.buffers[i].size_bytes);
                }
            }
        }
    }
}

/** @brief Ordinary and fused projections must borrow the declared Q8 bank without changing output bytes. */
void proveQ8Workspace(cpu::native_vnni::CPUNativeVNNIGemmKernel &kernel, int n, int k)
{
    constexpr int max_rows = 31;
    const size_t blocks = (size_t(k) + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;
    const auto requirements = kernel.getWorkspaceRequirements(max_rows);
    CPUProjectionTestWorkspace first(max_rows, k, requirements), second(max_rows, k, requirements), short_bank(1, k);
    auto input = TestTensorFactory::createFP32Random({max_rows, size_t(k)}, -.7f, .7f, 647);
    auto output = TestTensorFactory::createFP32({max_rows, size_t(n)});
    auto serial = TestTensorFactory::createFP32({max_rows, size_t(n)});
    auto row_output = TestTensorFactory::createFP32({1u, size_t(n)});
    const auto q8 = std::find_if(requirements.buffers.begin(), requirements.buffers.end(),
        [](const auto &buffer) { return buffer.name == kCPUProjectionQ8; });
    ASSERT_NE(q8, requirements.buffers.end());
    EXPECT_EQ(q8->size_bytes, max_rows * blocks * sizeof(Q8_1Block));
    EXPECT_THROW(kernel.multiply_tensor(input.get(), output.get(), 2, n, k), std::runtime_error);
    EXPECT_THROW(kernel.multiply_tensor(input.get(), output.get(), 2, n, k,
        true, 1.f, 0.f, nullptr, nullptr, -1, short_bank.get()), std::runtime_error);

    WorkerScope workers_scope;
    for (int workers = 1; workers <= 8; ++workers)
    {
        omp_set_num_threads(workers);
        for (int row = 0; row < max_rows; ++row)
        {
            ASSERT_TRUE(kernel.multiply_tensor(input.get(), row_output.get(), 1, n, k,
                true, 1.f, 0.f, nullptr, nullptr, -1, first.get(), row));
            std::memcpy(serial->mutable_data() + size_t(row) * n, row_output->data(), size_t(n) * sizeof(float));
        }
        for (int rows : {1, 2, 3, 15, 31})
        {
            SCOPED_TRACE("Q8 workers=" + std::to_string(workers) + " rows=" + std::to_string(rows));
            auto *workspace = rows % 2 ? first.get() : second.get();
            auto bank = CPUInvocationWorkspace::require<Q8_1Block>(workspace, kCPUProjectionQ8, max_rows * blocks);
            auto bytes = std::as_writable_bytes(bank);
            std::fill(bytes.begin(), bytes.end(), std::byte{0x5a});
            ASSERT_TRUE(kernel.multiply_tensor(input.get(), output.get(), rows, n, k,
                true, 1.f, 0.f, nullptr, nullptr, -1, workspace));
            EXPECT_EQ(std::memcmp(output->data(), serial->data(), size_t(rows) * n * sizeof(float)), 0);
            EXPECT_TRUE(std::all_of(bytes.begin() + rows * blocks * sizeof(Q8_1Block), bytes.end(),
                [](std::byte byte) { return byte == std::byte{0x5a}; }));
            // One shared input feeds both projections. The bank is not doubled
            // for output cardinality and remains distinct from output storage.
            auto twin = TestTensorFactory::createFP32({size_t(rows), size_t(n)});
            std::vector<ITensorGemm::TensorProjectionDesc> projections{
                {&kernel, output.get(), n, nullptr, "first"},
                {&kernel, twin.get(), n, nullptr, "second"}};
            ASSERT_TRUE(kernel.multiply_fused_tensor(input.get(), projections, rows, k, nullptr, workspace));
            EXPECT_EQ(std::memcmp(twin->data(), serial->data(), size_t(rows) * n * sizeof(float)), 0);
        }
    }
}

/** @brief Prove exact serial/grouped output and arena isolation on one immutable engine. */
void proveWorkspace(ITensorGemm &kernel, int n, int k)
{
    // Stages declare scratch through the public prepared-engine geometry, not
    // the test's known shape or a format-specific weight descriptor.
    ASSERT_EQ(kernel.get_n(), n);
    ASSERT_EQ(kernel.get_k(), k);
    constexpr int max_rows = 31;
    auto *consumer = dynamic_cast<IWorkspaceConsumer *>(&kernel);
    ASSERT_NE(consumer, nullptr);
    const auto requirements = consumer->getWorkspaceRequirements(max_rows, n, k);
    CPUProjectionTestWorkspace first(max_rows, k, requirements), second(max_rows, k, requirements), short_tile(1, k);
    auto gate = TestTensorFactory::createFP32Random({max_rows, size_t(k)}, -0.7f, 0.7f, 617);
    auto up = TestTensorFactory::createFP32Random({max_rows, size_t(k)}, -0.9f, 0.9f, 619);
    auto output = TestTensorFactory::createFP32({max_rows, size_t(n)});
    auto serial = TestTensorFactory::createFP32({max_rows, size_t(n)});
    auto row_gate = TestTensorFactory::createFP32({1u, size_t(k)});
    auto row_up = TestTensorFactory::createFP32({1u, size_t(k)});
    auto row_output = TestTensorFactory::createFP32({1u, size_t(n)});
    EXPECT_EQ(consumer->workspaceBindingPolicy(), WorkspaceBindingPolicy::Invocation);
    EXPECT_THROW(kernel.multiply_tensor_with_fused_swiglu(
        gate.get(), up.get(), output.get(), 2, n, k), std::runtime_error);
    EXPECT_THROW(kernel.multiply_tensor_with_fused_swiglu(
        gate.get(), up.get(), output.get(), 2, n, k, 1.f, 0.f, short_tile.get()), std::runtime_error);

    WorkerScope worker_scope;
    for (int workers = 1; workers <= 8; ++workers)
    {
        omp_set_num_threads(workers);
        // One serial reference per worker geometry, independent of grouped rows.
        for (int row = 0; row < max_rows; ++row)
        {
            std::memcpy(row_gate->mutable_data(), gate->data() + size_t(row) * k, size_t(k) * sizeof(float));
            std::memcpy(row_up->mutable_data(), up->data() + size_t(row) * k, size_t(k) * sizeof(float));
            ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu(
                row_gate.get(), row_up.get(), row_output.get(), 1, n, k, 1.f, 0.f, first.get()));
            std::memcpy(serial->mutable_data() + size_t(row) * n, row_output->data(), size_t(n) * sizeof(float));
        }
        for (int rows = 2; rows <= max_rows; ++rows)
        {
            SCOPED_TRACE("workers=" + std::to_string(workers) + " rows=" + std::to_string(rows));
            auto *workspace = rows % 2 ? first.get() : second.get();
            auto scratch = CPUInvocationWorkspace::require<float>(workspace, kCPUSwiGLUInput, max_rows * size_t(k));
            std::fill(scratch.begin(), scratch.end(), 12345.f);
            ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                gate.get(), up.get(), output.get(), rows, n, k, 1.f, 0.f, workspace));
            EXPECT_EQ(std::memcmp(output->data(), serial->data(), size_t(rows) * n * sizeof(float)), 0);
            EXPECT_TRUE(std::all_of(scratch.begin() + size_t(rows) * k, scratch.end(),
                [](float value) { return value == 12345.f; }));
        }
    }

    // Two actual participants reuse one immutable engine concurrently. No bind
    // operation on the engine may select or steal either participant's arena.
    omp_set_num_threads(1);
    auto other_gate = TestTensorFactory::createFP32Random({max_rows, size_t(k)}, -0.4f, 0.4f, 631);
    auto expected_left = TestTensorFactory::createFP32({max_rows, size_t(n)});
    auto expected_right = TestTensorFactory::createFP32({max_rows, size_t(n)});
    ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
        gate.get(), up.get(), expected_left.get(), max_rows, n, k, 1.f, 0.f, first.get()));
    ASSERT_TRUE(kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
        other_gate.get(), up.get(), expected_right.get(), max_rows, n, k, 1.f, 0.f, second.get()));
    auto concurrent = [&](DeviceWorkspaceManager *workspace, const TensorBase *input_gate) {
        omp_set_num_threads(1);
        auto result = TestTensorFactory::createFP32({max_rows, size_t(n)});
        for (int repeat = 0; repeat < 20; ++repeat)
            if (!kernel.multiply_tensor_with_fused_swiglu_verifier_rows_decode_equivalent(
                    input_gate, up.get(), result.get(), max_rows, n, k, 1.f, 0.f, workspace))
                throw std::runtime_error("Concurrent CPU projection failed");
        return result;
    };
    auto left = std::async(std::launch::async, concurrent, first.get(), gate.get());
    auto right = std::async(std::launch::async, concurrent, second.get(), other_gate.get());
    const auto a = left.get(), b = right.get();
    EXPECT_EQ(std::memcmp(a->data(), expected_left->data(), a->size_bytes()), 0);
    EXPECT_EQ(std::memcmp(b->data(), expected_right->data(), b->size_bytes()), 0);
    EXPECT_FALSE(consumer->hasWorkspace());
    EXPECT_EQ(consumer->getWorkspace(), nullptr);
}

/** @brief Alpha/beta and optional rotation preserve bytes under a retained workshare. */
void proveOutputAccumulation(cpu::native_vnni::CPUNativeVNNIGemmKernel &kernel, int n, int k)
{
    constexpr int max_rows = 31;
    auto requirements = kernel.getWorkspaceRequirements(max_rows, n, k);
    kernel.appendOutputAccumulationWorkspaceRequirements(requirements, max_rows, n);
    CPUProjectionTestWorkspace workspace(max_rows, k, requirements), missing_product(max_rows, k);
    auto input = TestTensorFactory::createFP32Random({max_rows, size_t(k)}, -.6f, .6f, 661);
    auto initial = TestTensorFactory::createFP32Random({max_rows, size_t(n)}, -.8f, .8f, 673);
    auto direct = TestTensorFactory::createFP32({max_rows, size_t(n)});
    auto team = TestTensorFactory::createFP32({max_rows, size_t(n)});
    std::memcpy(team->mutable_data(), initial->data(), initial->size_bytes());
    EXPECT_THROW(kernel.multiply_tensor(input.get(), team.get(), 2, n, k,
        true, -.75f, .25f, nullptr, nullptr, -1, missing_product.get()), std::runtime_error);
    EXPECT_EQ(std::memcmp(team->data(), initial->data(), team->size_bytes()), 0);
    const auto partial_requirement = std::find_if(requirements.buffers.begin(), requirements.buffers.end(),
        [](const auto &buffer) { return buffer.name == kCPUProjectionPartials; });
    if (partial_requirement != requirements.buffers.end())
    {
        auto without_partials = requirements;
        std::erase_if(without_partials.buffers,
            [](const auto &buffer) { return buffer.name == kCPUProjectionPartials; });
        CPUProjectionTestWorkspace missing_partials(max_rows, k, without_partials);
        EXPECT_THROW(kernel.multiply_tensor(input.get(), team.get(), 2, n, k,
            true, 1.f, 0.f, nullptr, nullptr, -1, missing_partials.get()), std::runtime_error);
        EXPECT_EQ(std::memcmp(team->data(), initial->data(), team->size_bytes()), 0);
    }

    WorkerScope restore;
    for (int workers = 1; workers <= 8; ++workers)
    for (const int rows : {1, 2, 3, 15, max_rows})
    for (const float beta : {0.f, .25f})
    {
        SCOPED_TRACE("epilogue workers=" + std::to_string(workers) + " rows=" + std::to_string(rows)
            + " beta=" + std::to_string(beta));
        omp_set_num_threads(workers);
        std::memcpy(direct->mutable_data(), initial->data(), initial->size_bytes());
        std::memcpy(team->mutable_data(), initial->data(), initial->size_bytes());
        ASSERT_TRUE(kernel.multiply_tensor(input.get(), direct.get(), rows, n, k,
            true, -.75f, beta, nullptr, nullptr, -1, workspace.get()));
        auto product = CPUInvocationWorkspace::require<float>(workspace.get(), kCPUProjectionProduct, max_rows * size_t(n));
        std::fill(product.begin(), product.end(), 12345.f);
        const size_t declared_partials = workspace.get()->getBufferSize(kCPUProjectionPartials) / sizeof(float);
        auto partials = CPUInvocationWorkspace::require<float>(workspace.get(), kCPUProjectionPartials, declared_partials);
        std::fill(partials.begin(), partials.end(), 12345.f);
        std::atomic<bool> completed{true};
#pragma omp parallel
        {
            if (!kernel.multiply_tensor(input.get(), team.get(), rows, n, k,
                    true, -.75f, beta, nullptr, nullptr, -1, workspace.get()))
                completed.store(false, std::memory_order_relaxed);
        }
        ASSERT_TRUE(completed.load());
        EXPECT_EQ(std::memcmp(team->data(), direct->data(), team->size_bytes()), 0);
        const size_t touched = beta == 0.f ? 0u : size_t(rows) * n;
        EXPECT_TRUE(std::all_of(product.begin() + touched, product.end(), [](float x) { return x == 12345.f; }));
        const size_t used_partials = cpu::native_vnni::nativeVNNIProjectionPartialFloats(kernel.packedWeights(), rows);
        EXPECT_TRUE(std::all_of(partials.begin() + used_partials, partials.end(), [](float x) { return x == 12345.f; }));
    }
}

/** @test Mixed-format bundle admission preserves full-K, private and shared ownership. */
TEST(CPUProjectionWorkspace, FusedPlanPricesEveryMemberAndPreservesPrivateTrees)
{
    using namespace cpu::native_vnni;
    for (const auto &format : quantizedMoEVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto first_weight = format.create({65u, 256u}, 719);
        auto second_weight = TestTensorFactory::createQ4_0Random({193u, 256u}, 727);
        CPUNativeVNNIGemmKernel first(first_weight.get());
        CPUNativeVNNIGemmKernel second(second_weight.get(), 0, -1,
            CPUProjectionNumericalPolicy::GPUAlignedExpert);
        ASSERT_TRUE(first.isValid());
        ASSERT_TRUE(second.isValid());
        // Payload pointers are deliberately absent. Admission needs metadata,
        // not execution, to resolve disjoint partial intervals and tail padding.
        std::array<FusedVerifierRowsDesc, 2> descriptors{{
            {.packed = &first.packedWeights(), .N = 65, .ldc = 65},
            {.packed = &second.packedWeights(), .N = 193, .ldc = 193, .rows = 3}}};
        for (int rows : {1, 2, 3, 15, 31})
        for (int workers : {1, 2, 3, 7, 16, 128})
        {
            const auto plan = planNativeVNNIFusedRows(descriptors, rows, workers);
            EXPECT_EQ(plan.partial_route, NativeVNNIFusedPartialRoute::SharedBank);
            EXPECT_EQ(plan.projections[0].rows, rows);
            EXPECT_EQ(plan.projections[0].partial_sums_size, 0u);
            EXPECT_EQ(plan.projections[1].rows, 3);
            EXPECT_EQ(plan.projections[1].partial_sums_offset, 0u);
            EXPECT_EQ(plan.sharedPartialFloats(), cpuProjectionPartialFloats(3, 193,
                MoEProjectionNumericalContract::orderedKPartitionsForWidth(256)));
        }
        descriptors[0].packed = &second.packedWeights();
        descriptors[0].N = descriptors[0].ldc = 193;
        const auto private_plan = planNativeVNNIFusedRows(descriptors, 1, 1);
        EXPECT_EQ(private_plan.partial_route, NativeVNNIFusedPartialRoute::OutputTileLocal);
        EXPECT_EQ(private_plan.sharedPartialFloats(), 0u);
        const auto shared_plan = planNativeVNNIFusedRows(descriptors, 1, 128);
        EXPECT_EQ(shared_plan.partial_route, NativeVNNIFusedPartialRoute::SharedBank);
        EXPECT_EQ(shared_plan.projections[1].partial_sums_offset,
            shared_plan.projections[0].partial_sums_size);
        EXPECT_EQ(shared_plan.sharedPartialFloats(), cpuProjectionPartialFloats(4, 193,
            MoEProjectionNumericalContract::orderedKPartitionsForWidth(256)));
        descriptors[0].packed = descriptors[1].packed = &first.packedWeights();
        const auto full_k_plan = planNativeVNNIFusedRows(descriptors, 1, 128);
        EXPECT_EQ(full_k_plan.partial_route, NativeVNNIFusedPartialRoute::FullK);
        EXPECT_EQ(full_k_plan.sharedPartialFloats(), 0u);
        EXPECT_THROW(planNativeVNNIFusedRows({}, 1, 1), std::invalid_argument);
        EXPECT_THROW(planNativeVNNIFusedRows(descriptors, 0, 1), std::invalid_argument);
        EXPECT_THROW(planNativeVNNIFusedRows(descriptors, 1, 0), std::invalid_argument);
        descriptors[1].rows = -1;
        EXPECT_THROW(planNativeVNNIFusedRows(descriptors, 1, 1), std::invalid_argument);
        descriptors[1].rows = 1;
        descriptors[1].verifier_schedule = VerifierRowsPolicy::WideRows;
        EXPECT_THROW(planNativeVNNIFusedRows(descriptors, 1, 1), std::invalid_argument);
    }
}

/** @test Two mixed-mode partial readers reuse one admitted bank without new ordering edges. */
TEST(CPUProjectionWorkspace, FusedMixedMembersReuseTheLargestBank)
{
    using namespace cpu::native_vnni;
    WorkerScope restore;
    omp_set_num_threads(4);
    constexpr int k = 256, rows = 5;
    auto input = TestTensorFactory::createFP32Random({rows, k}, -.7f, .7f, 743);
    AlignedVector<Q8_1Block> q8(size_t(rows) * (k / 32));
    quantize_activations_to_q8_1(input->data(), q8.data(), rows, k, k / 32);
    for (const auto &format : quantizedMoEVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto weight = format.create({193u, k}, 751);
        CPUNativeVNNIGemmKernel native(weight.get());
        CPUNativeVNNIGemmKernel partitioned(weight.get(), 0, -1,
            CPUProjectionNumericalPolicy::GPUAlignedExpert);
        std::array<std::vector<float>, 3> output, expected;
        std::array<FusedVerifierRowsDesc, 3> descriptors;
        for (int member = 0; member < 3; ++member)
        {
            output[member].resize(rows * 193, 12345.f);
            expected[member].resize(rows * 193);
            const auto &packed = member == 1 ? native.packedWeights() : partitioned.packedWeights();
            descriptors[member] = {.packed = &packed, .output = output[member].data(), .N = 193, .ldc = 193};
            NativeVNNITestPartialStorage scratch(packed, 1);
            for (int row = 0; row < rows; ++row)
                gemv_native_vnni_preq(packed, q8.data() + row * (k / 32),
                    expected[member].data() + row * 193, scratch.span());
        }
        const auto plan = planNativeVNNIFusedRows(descriptors, rows, 4);
        ASSERT_EQ(plan.partial_route, NativeVNNIFusedPartialRoute::SharedBank);
        ASSERT_EQ(plan.projections[1].partial_sums_size, 0u);
        ASSERT_EQ(plan.projections[0].partial_sums_offset, 0u);
        ASSERT_EQ(plan.projections[2].partial_sums_offset, 0u);
        ASSERT_EQ(plan.sharedPartialFloats(), plan.projections[0].partial_sums_size);
        NativeVNNITestPartialStorage storage(plan.sharedPartialFloats() + 64);
        std::fill(storage.span().begin(), storage.span().end(), 12345.f);
        const auto exact = storage.span().first(plan.sharedPartialFloats());
        EXPECT_THROW(gemm_native_vnni_fused_verifier_rows_preq(q8.data(), descriptors.data(),
            exact.first(exact.size() - 1), 3, rows, k / 32), std::runtime_error);
        for (const auto &member : output)
            EXPECT_TRUE(std::all_of(member.begin(), member.end(), [](float x) { return x == 12345.f; }));
        ASSERT_TRUE(gemm_native_vnni_fused_verifier_rows_preq(q8.data(), descriptors.data(),
            exact, 3, rows, k / 32));
        for (int member = 0; member < 3; ++member)
            EXPECT_EQ(std::memcmp(output[member].data(), expected[member].data(), output[member].size() * sizeof(float)), 0);
        EXPECT_TRUE(std::all_of(storage.span().begin() + exact.size(), storage.span().end(),
            [](float x) { return x == 12345.f; }));
    }
}

/** @test Large native trees use admitted worker tiles, preserving tails and independent serial rows. */
TEST(CPUProjectionWorkspace, FusedLargeTreesUseExclusiveWorkerTiles)
{
    using namespace cpu::native_vnni;
    WorkerScope restore;
    constexpr int n = 513, k = 16384, max_rows = 5, workers = 32;
    omp_set_num_threads(workers);
    auto input = TestTensorFactory::createFP32Random({max_rows, k}, -.6f, .6f, 757);
    AlignedVector<Q8_1Block> q8(size_t(max_rows) * (k / 32));
    quantize_activations_to_q8_1(input->data(), q8.data(), max_rows, k, k / 32);
    for (const auto &format : quantizedMoEVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto weight = format.create({n, k}, 761);
        CPUNativeVNNIGemmKernel kernel(weight.get());
        ASSERT_TRUE(kernel.isValid());
        const auto &packed = kernel.packedWeights();
        std::vector<float> expected(max_rows * n);
        NativeVNNITestPartialStorage serial_storage(packed, 1);
        for (int row = 0; row < max_rows; ++row)
            gemv_native_vnni_preq(packed, q8.data() + row * (k / 32),
                expected.data() + row * n, serial_storage.span());
        for (int rows : {1, max_rows})
        {
            SCOPED_TRACE(rows);
            std::array<std::vector<float>, 2> output{
                std::vector<float>(size_t(rows) * n, 12345.f),
                std::vector<float>(size_t(rows) * n, 12345.f)};
            std::array<FusedVerifierRowsDesc, 2> descriptors{{
                {.packed = &packed, .output = output[0].data(), .N = n, .ldc = n},
                {.packed = &packed, .output = output[1].data(), .N = n, .ldc = n}}};
            const auto plan = planNativeVNNIFusedRows(descriptors, rows, workers);
            ASSERT_GT(plan.projections[0].k_tiles, MoEProjectionNumericalContract::ordered_k_partitions);
            ASSERT_EQ(plan.partial_route, rows == 1 ? NativeVNNIFusedPartialRoute::SharedBank
                : NativeVNNIFusedPartialRoute::OutputTileWorkspace);
            const auto requirements = kernel.getWorkspaceRequirements(rows);
            CPUProjectionTestWorkspace workspace(rows, k, requirements);
            auto bank = CPUInvocationWorkspace::available<float>(workspace.get(), kCPUProjectionPartials);
            ASSERT_GE(bank.size(), plan.sharedPartialFloats());
            std::fill(bank.begin(), bank.end(), 12345.f);
            const auto exact = bank.first(plan.sharedPartialFloats());
            EXPECT_THROW(gemm_native_vnni_fused_verifier_rows_preq(q8.data(), descriptors.data(),
                exact.first(exact.size() - 1), 2, rows, k / 32), std::runtime_error);
            for (const auto &member : output)
                EXPECT_TRUE(std::all_of(member.begin(), member.end(), [](float x) { return x == 12345.f; }));
            // A retained team sees the same bank, but each tile owner receives
            // a disjoint stride. Poisoned unused capacity exposes over-writes.
            std::atomic<bool> complete{true};
#pragma omp parallel num_threads(workers)
            {
                if (!gemm_native_vnni_fused_verifier_rows_preq(q8.data(), descriptors.data(),
                        exact, 2, rows, k / 32)) complete.store(false);
            }
            EXPECT_TRUE(complete.load());
            for (const auto &member : output)
                EXPECT_EQ(std::memcmp(member.data(), expected.data(), member.size() * sizeof(float)), 0);
            EXPECT_TRUE(std::all_of(bank.begin() + exact.size(), bank.end(), [](float x) { return x == 12345.f; }));
        }
    }
}

/** @test A retained team's actual worker count, not a future nested team, owns arithmetic. */
TEST(CPUProjectionWorkspace, FusedRetainedTeamUsesItsActualWorkerGeometry)
{
    using namespace cpu::native_vnni;
    WorkerScope restore;
    constexpr int n = 512, k = 16384, rows = 3;
    auto input = TestTensorFactory::createFP32Random({rows, k}, -.7f, .7f, 733);
    AlignedVector<Q8_1Block> q8(size_t(rows) * (k / 32));
    auto expected = TestTensorFactory::createFP32({rows, n});
    auto actual = TestTensorFactory::createFP32({rows, n});
    for (const auto &format : quantizedMoEVerifierFormats())
    for (const int workers : {2, 8})
    {
        SCOPED_TRACE(format.label);
        SCOPED_TRACE("workers=" + std::to_string(workers));
        omp_set_num_threads(workers);
        auto weight = format.create({n, k}, 739);
        CPUNativeVNNIGemmKernel kernel(weight.get());
        const auto &packed = kernel.packedWeights();
        ASSERT_TRUE(kernel.isValid());
        quantize_activations_to_q8_1(input->data(), q8.data(), rows, k, k / 32);
        NativeVNNITestPartialStorage partials(packed, rows);
        for (int row = 0; row < rows; ++row)
            gemv_native_vnni_preq(packed, q8.data() + size_t(row) * (k / 32),
                expected->mutable_data() + size_t(row) * n, partials.span());
        FusedVerifierRowsDesc descriptor{.packed = &packed,
            .output = actual->mutable_data(), .N = n, .ldc = n};
        const auto current = planNativeVNNIFusedRows({&descriptor, 1}, rows, workers);
        const int future_workers = workers == 2 ? 8 : 2;
        const auto nested = planNativeVNNIFusedRows({&descriptor, 1}, rows, future_workers);
        ASSERT_NE(current.projections[0].k_tiles, nested.projections[0].k_tiles);
        // The explicit team can be smaller OR larger than nthreads-var. Its
        // actual participants own both the arithmetic and shared bank extent.
        omp_set_num_threads(future_workers);
        for (int lane = 0; lane < 3; ++lane)
        {
            SCOPED_TRACE("lane=" + std::to_string(lane));
            std::atomic<bool> completed{true};
#pragma omp parallel num_threads(workers)
            {
                try
                {
                    if (lane == 0)
                    {
                        for (int row = 0; row < rows; ++row)
                            gemv_native_vnni_preq(packed, q8.data() + size_t(row) * (k / 32),
                                actual->mutable_data() + size_t(row) * n, partials.span());
                    }
                    else if (lane == 1)
                        gemm_native_vnni_preq_decode_equivalent_rows(packed, q8.data(),
                            actual->mutable_data(), partials.span(), rows, n);
                    else if (!gemm_native_vnni_fused_verifier_rows_preq(q8.data(), &descriptor,
                                 partials.span(), 1, rows, k / 32))
                        completed.store(false, std::memory_order_relaxed);
                }
                catch (const std::exception &)
                {
                    // Contract checks precede worksharing. Preserve a readable
                    // failing test instead of unwinding through an OpenMP team.
                    completed.store(false, std::memory_order_relaxed);
                }
            }
            EXPECT_TRUE(completed.load());
            if (completed.load())
                EXPECT_EQ(std::memcmp(actual->data(), expected->data(), expected->size_bytes()), 0);
        }
    }
}

/** @test Mirrored terminal admission uses serial policy N but allocates physical N. */
TEST(CPUProjectionWorkspace, MirroredHeadDeclaresSerialTreeForPhysicalOutput)
{
    using namespace cpu::native_vnni;
    WorkerScope restore;
    omp_set_num_threads(16);
    constexpr int n = 2048, serial_n = 512, k = 16384, rows = 3;
    auto weight = TestTensorFactory::createQ4_0Random({n, k}, 701);
    auto prepared = makePreparedGemmFixture(weight.get(), DeviceId::cpu(), "output.weight");
    auto *kernel = dynamic_cast<CPUNativeVNNIGemmKernel *>(prepared.store->gemmKernel(prepared.ref));
    ASSERT_NE(kernel, nullptr);
    ASSERT_EQ(nativeVNNIProjectionPartialFloats(kernel->packedWeights(), rows), 0u);
    auto input = TestTensorFactory::createFP32Random({rows, k}, -.5f, .5f, 709);
    auto output = TestTensorFactory::createFP32({rows, n});
    LMHeadStage::Params params;
    params.hidden_states = input.get();
    params.lm_head_weight = weight.get();
    params.logits = output.get();
    params.seq_len = rows;
    params.d_model = k;
    params.vocab_size = n;
    params.device_id = DeviceId::cpu();
    params.serial_equivalent_partition_width = serial_n;
    params.compute_all_positions = true;
    params.prepared_store = prepared.store.get();
    params.prepared_ref = prepared.ref;
    LMHeadStage stage(params);
    const auto requirements = stage.getWorkspaceRequirements(rows, n, k);
    ASSERT_EQ(cpuNativeVNNISerialOutputPartitionN(), 0);
    const auto partials = std::find_if(requirements.buffers.begin(), requirements.buffers.end(),
        [](const auto &buffer) { return buffer.name == kCPUProjectionPartials; });
    ASSERT_NE(partials, requirements.buffers.end());
    const auto scope = kernel->beginOutputPartitionEquivalenceScope(n, serial_n);
    const auto expected = nativeVNNIProjectionPartialFloats(kernel->packedWeights(), rows);
    ASSERT_GT(expected, 0u);
    EXPECT_EQ(partials->size_bytes, expected * sizeof(float));
    CPUProjectionTestWorkspace workspace(rows, k, requirements);
    ASSERT_TRUE(kernel->multiply_tensor(input.get(), output.get(), rows, n, k,
        true, 1.f, 0.f, nullptr, nullptr, -1, workspace.get()));
}

/** @test Every NativeVNNI source codebook uses the same admitted transform contract. */
TEST(CPUProjectionWorkspace, AllQuantizedFormats)
{
    using namespace cpu::native_vnni;
    for (const auto &format : quantizedMoEVerifierFormats())
    for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative, CPUProjectionNumericalPolicy::GPUAlignedExpert})
    {
        SCOPED_TRACE(std::string(format.label) + " policy=" + std::to_string(static_cast<int>(policy)));
        auto weight = format.create({128u, 256u}, 313);
        CPUNativeVNNIGemmKernel kernel(weight.get(), 0, -1, policy);
        ASSERT_TRUE(kernel.isValid());
        proveQ8Workspace(kernel, 128, 256);
        proveWorkspace(kernel, 128, 256);
        proveOutputAccumulation(kernel, 128, 256);
    }
}

/** @test Every source format's optional rotated representation retains participant isolation. */
TEST(CPUProjectionWorkspace, RotatedQuantizedFormats)
{
    using namespace cpu::native_vnni;
    ActivationRotation rotation(256, 128);
    for (const auto &format : quantizedMoEVerifierFormats())
    {
        SCOPED_TRACE(format.label);
        auto weight = format.create({128u, 256u}, 683);
        weight->setActivationRotation(&rotation);
        CPUNativeVNNIGemmKernel kernel(weight.get());
        ASSERT_TRUE(kernel.isValid());
        const auto requirements = kernel.getWorkspaceRequirements(31);
        const auto found = std::find_if(requirements.buffers.begin(), requirements.buffers.end(),
            [](const auto &buffer) { return buffer.name == kCPUProjectionRotation; });
        ASSERT_NE(found, requirements.buffers.end());
        EXPECT_EQ(found->size_bytes, 31u * 256u * sizeof(float));
        proveWorkspace(kernel, 128, 256);
        proveOutputAccumulation(kernel, 128, 256);
    }
}

/** @test FP16, BF16 and FP32 retain identical ownership and serial-row semantics. */
TEST(CPUProjectionWorkspace, AllFloatingFormats)
{
    using Kernel = gemm::FloatingPointGemmKernel;
    for (const auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
    for (const auto policy : {Kernel::NumericalPolicy::BackendNative, Kernel::NumericalPolicy::GPUAlignedExpert})
    {
        SCOPED_TRACE(static_cast<int>(type));
        std::unique_ptr<TensorBase> weight;
        if (type == TensorType::FP16) weight = TestTensorFactory::createFP16Random({128u, 256u}, -0.1f, 0.1f, 317);
        else if (type == TensorType::BF16) weight = TestTensorFactory::createBF16Random({128u, 256u}, -0.1f, 0.1f, 317);
        else weight = TestTensorFactory::createFP32Random({128u, 256u}, -0.1f, 0.1f, 317);
        Kernel kernel(weight.get(), policy);
        proveWorkspace(kernel, 128, 256);

        // A private service view pins the prepared tensor and its geometry even
        // after the original owner is released. It must not allocate a weight
        // copy or inherit ITensorGemm's unbound (zero-width) geometry.
        auto source = std::make_shared<Kernel>(
            std::shared_ptr<const TensorBase>(std::move(weight)), policy);
        Kernel view{std::shared_ptr<const Kernel>(source)};
        source.reset();
        EXPECT_EQ(view.packedWeightBytes(), 0u);
        proveWorkspace(view, 128, 256);
    }
}
}
}
