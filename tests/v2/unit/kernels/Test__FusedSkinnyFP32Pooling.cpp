/**
 * @file Test__FusedSkinnyFP32Pooling.cpp
 * @brief Fork-count and byte-identity contracts of pooled skinny FP32 fused projections.
 *
 * A backend-native fused FP32 transaction whose members all route to the
 * decode-equivalent skinny primitive (one-row decode, or small M-by-N shapes
 * such as the GDN alpha/beta projections) runs every member's tasks in one
 * OpenMP workshare. These device-free checks drive the public engine:
 *
 * - every pooled output is byte-identical to that member's own public
 *   multiply_tensor result across member counts, M including row-tile tails,
 *   N tails, bias, and worker counts from one upward;
 * - a pooled fused call forks exactly one team, where the per-member calls
 *   fork one per member;
 * - the pooled primitive keeps the beta epilogue identical to the
 *   single-member primitive and rejects overlapping storage.
 *
 * Timing and economy are not asserted here; they are Release A/B evidence.
 */
#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "tensors/TensorKernels.h"
#include "tensors/Tensors.h"
#include "utils/OpenMPUtils.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using llaminar2::test::TestTensorFactory;

namespace
{
    using Engine = gemm::FloatingPointGemmKernel;

    /** @brief One fused member: weights, its engine, bias and both outputs. */
    struct Member
    {
        std::unique_ptr<FP32Tensor> weights;
        std::unique_ptr<Engine> engine;
        std::unique_ptr<FP32Tensor> bias;
        std::unique_ptr<FP32Tensor> fused_output;
        std::unique_ptr<FP32Tensor> direct_output;
    };

    /** @brief Build one member with deterministic N-by-K weights and an optional bias. */
    Member makeMember(int n, int k, bool with_bias, uint32_t seed)
    {
        Member member;
        member.weights = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(n), static_cast<size_t>(k)}, -0.5f, 0.5f, seed);
        member.engine = std::make_unique<Engine>(member.weights.get());
        if (with_bias)
            member.bias = TestTensorFactory::createFP32Random({static_cast<size_t>(n)}, -0.25f, 0.25f, seed + 7u);
        return member;
    }

    /** @brief Byte-compare two FP32 buffers of @p count words. */
    bool sameBytes(const float *left, const float *right, size_t count)
    {
        return std::memcmp(left, right, count * sizeof(float)) == 0;
    }

    /** @brief Worker counts from one through the available processors, plus an oversubscribed team. */
    std::vector<int> workerCounts()
    {
        const int processors = std::max(1, omp_get_num_procs());
        std::vector<int> counts;
        for (int workers = 1; workers <= std::min(processors, 8); ++workers)
            counts.push_back(workers);
        if (processors > 8)
            counts.push_back(processors);
        counts.push_back(processors + 3);
        return counts;
    }
} // namespace

/**
 * @test Pooled fused outputs equal each member's own public multiply bytes.
 * @details M=1 admits any N (decode); M>1 admits N<=64. Each case runs the
 *          pooled multiply_fused_tensor and the member's own multiply_tensor
 *          at the same worker count and compares bytes exactly.
 */
TEST(FusedSkinnyFP32Pooling, PooledOutputsMatchPerMemberBytes)
{
    const int saved_threads = omp_get_max_threads();
    struct Shape { int m; std::vector<int> n; };
    const std::vector<Shape> shapes{
        {1, {48, 48}},
        {1, {300, 17, 1}},
        {1, {7, 129, 64, 3}},
        {2, {48, 48}},
        {3, {1, 64}},
        {4, {33, 48, 5}},
        {5, {48, 48}},
        {11, {48, 48, 2, 64}},
        {12, {48, 48}},
        {64, {16, 9}},
    };
    constexpr int K = 129; // Not a multiple of any vector width.
    uint32_t seed = 100;
    for (const auto &shape : shapes)
    {
        for (const bool with_bias : {false, true})
        {
            for (const int workers : workerCounts())
            {
                SCOPED_TRACE("m=" + std::to_string(shape.m) + " members=" + std::to_string(shape.n.size()) +
                             " bias=" + std::to_string(with_bias) + " workers=" + std::to_string(workers));
                omp_set_num_threads(workers);
                auto input = TestTensorFactory::createFP32Random(
                    {static_cast<size_t>(shape.m), static_cast<size_t>(K)}, -1.0f, 1.0f, ++seed);
                std::vector<Member> members;
                std::vector<ITensorGemm::TensorProjectionDesc> descs;
                for (const int n : shape.n)
                {
                    members.push_back(makeMember(n, K, with_bias, ++seed));
                    auto &member = members.back();
                    member.fused_output = TestTensorFactory::createFP32Zeros(
                        {static_cast<size_t>(shape.m), static_cast<size_t>(n)});
                    member.direct_output = TestTensorFactory::createFP32Zeros(
                        {static_cast<size_t>(shape.m), static_cast<size_t>(n)});
                }
                for (size_t index = 0; index < members.size(); ++index)
                    descs.emplace_back(members[index].engine.get(), members[index].fused_output.get(),
                                       shape.n[index], members[index].bias.get(), "member");
                ASSERT_TRUE(members.front().engine->multiply_fused_tensor(input.get(), descs, shape.m, K));
                for (size_t index = 0; index < members.size(); ++index)
                {
                    auto &member = members[index];
                    ASSERT_TRUE(member.engine->multiply_tensor(
                        input.get(), member.direct_output.get(), shape.m, shape.n[index], K,
                        /*transpose_B=*/true, 1.0f, 0.0f, member.bias.get()));
                    EXPECT_TRUE(sameBytes(member.fused_output->data(), member.direct_output->data(),
                                          static_cast<size_t>(shape.m) * shape.n[index]))
                        << "member " << index;
                }
            }
        }
    }
    omp_set_num_threads(saved_threads);
}

/**
 * @test A pooled fused call forks one team; per-member calls fork one each.
 * @details Observed through the production fork authority on the calling
 *          thread, outside any enclosing team.
 */
TEST(FusedSkinnyFP32Pooling, FusedCallForksOneTeam)
{
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(std::max(2, omp_get_num_procs()));
    constexpr int K = 256;
    for (const int m : {1, 4, 12})
    {
        for (const size_t count : {2u, 3u, 4u})
        {
            SCOPED_TRACE("m=" + std::to_string(m) + " members=" + std::to_string(count));
            auto input = TestTensorFactory::createFP32Random(
                {static_cast<size_t>(m), static_cast<size_t>(K)}, -1.0f, 1.0f, 7u + m);
            std::vector<Member> members;
            std::vector<ITensorGemm::TensorProjectionDesc> descs;
            for (size_t index = 0; index < count; ++index)
            {
                members.push_back(makeMember(48, K, false, 300u + static_cast<uint32_t>(index)));
                members.back().fused_output = TestTensorFactory::createFP32Zeros({static_cast<size_t>(m), 48});
            }
            for (auto &member : members)
                descs.emplace_back(member.engine.get(), member.fused_output.get(), 48);
            {
                const llaminar::v2::OpenMPTeamForkScope scope;
                ASSERT_TRUE(members.front().engine->multiply_fused_tensor(input.get(), descs, m, K));
                EXPECT_EQ(scope.forks(), 1u);
            }
            {
                const llaminar::v2::OpenMPTeamForkScope scope;
                for (auto &member : members)
                    ASSERT_TRUE(member.engine->multiply_tensor(
                        input.get(), member.fused_output.get(), m, 48, K));
                EXPECT_EQ(scope.forks(), count);
            }
        }
    }
    omp_set_num_threads(saved_threads);
}

/**
 * @test The pooled primitive's beta epilogue equals the single-member primitive.
 * @details Outputs start from identical nonzero contents, so the beta read of
 *          the preceding value is exercised for every member.
 */
TEST(FusedSkinnyFP32Pooling, PooledBetaEpilogueMatchesSingleMemberPrimitive)
{
    constexpr int M = 6;
    constexpr int K = 97;
    const std::array<int, 3> widths{5, 48, 64};
    auto input = TestTensorFactory::createFP32Random({M, K}, -1.0f, 1.0f, 11u);
    std::array<std::unique_ptr<FP32Tensor>, 3> weights;
    std::array<std::vector<float>, 3> pooled;
    std::array<std::vector<float>, 3> single;
    std::array<gemm::SkinnyFP32Member, 3> members{};
    for (size_t index = 0; index < widths.size(); ++index)
    {
        const int n = widths[index];
        weights[index] = TestTensorFactory::createFP32Random(
            {static_cast<size_t>(n), static_cast<size_t>(K)}, -0.5f, 0.5f, 40u + static_cast<uint32_t>(index));
        pooled[index].resize(static_cast<size_t>(M) * n);
        for (size_t word = 0; word < pooled[index].size(); ++word)
            pooled[index][word] = 0.125f * static_cast<float>(word % 17) - 1.0f;
        single[index] = pooled[index];
        members[index] = gemm::SkinnyFP32Member{weights[index]->data(), pooled[index].data(), n, nullptr};
    }
    ASSERT_TRUE(gemm::run_fp32_skinny_matmul_members(input->data(), members.data(), 3, M, K, 0.75f, 0.5f));
    for (size_t index = 0; index < widths.size(); ++index)
    {
        ASSERT_TRUE(gemm::run_fp32_skinny_matmul(input->data(), weights[index]->data(), single[index].data(),
                                                 M, widths[index], K, /*transpose_B=*/true, 0.75f, 0.5f));
        EXPECT_TRUE(sameBytes(pooled[index].data(), single[index].data(), pooled[index].size()))
            << "member " << index;
    }
}

/**
 * @test Overlapping pooled storage and oversized transactions are rejected before any write.
 */
TEST(FusedSkinnyFP32Pooling, OverlappingStorageIsRejected)
{
    constexpr int M = 2;
    constexpr int K = 8;
    std::vector<float> input(static_cast<size_t>(M) * K, 1.0f);
    std::vector<float> weights(static_cast<size_t>(4) * K, 1.0f);
    std::vector<float> outputs(static_cast<size_t>(M) * 8, 0.0f);
    const std::array<gemm::SkinnyFP32Member, 2> overlapping{
        gemm::SkinnyFP32Member{weights.data(), outputs.data(), 4, nullptr},
        gemm::SkinnyFP32Member{weights.data(), outputs.data() + 4, 4, nullptr}};
    EXPECT_THROW(gemm::run_fp32_skinny_matmul_members(input.data(), overlapping.data(), 2, M, K),
                 std::invalid_argument);
    EXPECT_TRUE(std::all_of(outputs.begin(), outputs.end(), [](float value) { return value == 0.0f; }));

    const std::array<gemm::SkinnyFP32Member, 1> aliasing{
        gemm::SkinnyFP32Member{weights.data(), input.data(), 4, nullptr}};
    EXPECT_THROW(gemm::run_fp32_skinny_matmul_members(input.data(), aliasing.data(), 1, M, K),
                 std::invalid_argument);

    std::array<gemm::SkinnyFP32Member, gemm::kSkinnyFP32MaxPooledMembers + 1> oversized{};
    EXPECT_THROW(gemm::run_fp32_skinny_matmul_members(input.data(), oversized.data(),
                                                      static_cast<int>(oversized.size()), M, K),
                 std::invalid_argument);
}
