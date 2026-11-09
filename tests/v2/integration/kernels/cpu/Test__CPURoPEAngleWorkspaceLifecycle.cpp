/**
 * @file Test__CPURoPEAngleWorkspaceLifecycle.cpp
 * @brief Lifecycle of the admitted CPU RoPE angle workspace across stages, resets and replays.
 *
 * A serial graph family shares one `cpu_rope_angles` buffer between its
 * decode (M=1), grouped-verifier and prefill RoPE stages. These tests bind one
 * admitted workspace to all three stage kinds, run them in sequence, reset
 * their session state and replay. The buffer address must never change, the
 * declaration must come only from graph rows and head width, and every output
 * must equal an independent row-by-row serial reference. An unbound stage and
 * an undersized binding must fail before any store. Device-free.
 */
#include <gtest/gtest.h>

#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "kernels/rope/RoPEWorkspaceContract.h"
#include "tensors/Tensors.h"
#include "../../../utils/CPURoPETestBinding.h"

#include <cstring>
#include <memory>
#include <vector>

namespace llaminar2::test
{
namespace
{
    constexpr int kHeads = 3;
    constexpr int kKVHeads = 1;
    constexpr int kHeadDim = 64;
    constexpr float kTheta = 10000.0f;

    /** @return A seeded FP32 tensor of rows x cols. */
    std::unique_ptr<FP32Tensor> seededTensor(int rows, int cols, int salt)
    {
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)}, DeviceId::cpu());
        float *data = tensor->mutable_data();
        for (int i = 0; i < rows * cols; ++i)
            data[i] = float((i % 251) * 73 % 251 - 125 + salt) * 0.03125f;
        return tensor;
    }

    /** @return Bytes of an FP32 tensor. */
    std::vector<float> values(const FP32Tensor &tensor)
    {
        return {tensor.data(), tensor.data() + tensor.numel()};
    }

    /**
     * @brief Independent serial reference: one bound kernel, one row per call.
     * @param positions Absolute position per row; negative rows stay untouched.
     */
    void serialReference(std::vector<float> &q, std::vector<float> &k, const std::vector<int> &positions)
    {
        CPURoPEKernelT<ActivationPrecision::FP32> kernel;
        const CPURoPETestBinding binding(kernel, 1, kHeadDim);
        for (size_t row = 0; row < positions.size(); ++row)
        {
            if (positions[row] < 0)
                continue;
            ASSERT_TRUE(kernel.apply_typed(q.data() + row * kHeads * kHeadDim, k.data() + row * kKVHeads * kHeadDim,
                                           &positions[row], 1, kHeads, kKVHeads, kHeadDim, kTheta));
        }
    }

    /** @brief One RoPE stage over its own Q/K tensors. */
    struct StageCase
    {
        std::unique_ptr<FP32Tensor> q, k;
        std::vector<float> q_original, k_original;
        std::vector<int> positions;
        std::unique_ptr<RoPEStage> stage;
    };

    StageCase makeStage(std::vector<int> positions, bool verifier)
    {
        StageCase c;
        const int rows = static_cast<int>(positions.size());
        c.q = seededTensor(rows, kHeads * kHeadDim, 3);
        c.k = seededTensor(rows, kKVHeads * kHeadDim, 5);
        c.q_original = values(*c.q);
        c.k_original = values(*c.k);
        c.positions = std::move(positions);
        RoPEStage::Params params;
        params.device_id = DeviceId::cpu();
        params.Q = c.q.get();
        params.K = c.k.get();
        params.n_heads = kHeads;
        params.n_kv_heads = kKVHeads;
        params.head_dim = kHeadDim;
        params.theta_base = kTheta;
        params.seq_len = rows;
        params.position_ids = c.positions.data();
        params.pos_offset = c.positions.front();
        params.force_decode_equivalent_verifier_prefill = verifier;
        c.stage = std::make_unique<RoPEStage>(params);
        return c;
    }

    /** @brief Restore the original rows so a replay starts from identical inputs. */
    void restore(StageCase &c)
    {
        std::memcpy(c.q->mutable_data(), c.q_original.data(), c.q_original.size() * sizeof(float));
        std::memcpy(c.k->mutable_data(), c.k_original.data(), c.k_original.size() * sizeof(float));
    }

    /**
     * @brief Publish one request's positions, execute, and compare against the serial reference.
     *
     * A session reset clears request-local positions. As in production, where
     * the orchestrator republishes every request through updateDynamicPositionIds
     * or updateDynamicParams before execution, each replay publishes its own
     * positions first; the admitted workspace binding is the state that survives.
     */
    void executeAndCompare(StageCase &c, CPUDeviceContext &ctx, const std::vector<int> &positions)
    {
        c.stage->updateDynamicPositionIds(positions.data(), static_cast<int>(positions.size()));
        auto q = c.q_original, k = c.k_original;
        serialReference(q, k, positions);
        ASSERT_TRUE(c.stage->execute(&ctx));
        EXPECT_EQ(std::memcmp(c.q->data(), q.data(), q.size() * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(c.k->data(), k.data(), k.size() * sizeof(float)), 0);
    }
}

/**
 * @test Decode, grouped-verifier and prefill stages reuse one binding across reset and replay.
 *
 * The family's admitted extent comes from the largest stage's rows and the head
 * width. Every stage borrows the same buffer address before and after a
 * session reset. Each replay republishes a different request's positions, as
 * production does after a reset, and reproduces the serial reference bytes for
 * those positions; verifier padding rows stay untouched.
 */
TEST(CPURoPEAngleWorkspaceLifecycle, SerialFamilyReusesOneBindingAcrossResetAndReplay)
{
    CPUDeviceContext ctx(DeviceId::cpu(), 4);
    std::vector<StageCase> stages;
    stages.push_back(makeStage({17231}, false));                               // ordinary decode, M=1
    stages.push_back(makeStage({17232, 17233, -1, 17235}, true));              // grouped verifier rows
    std::vector<int> prefill(33);
    for (int row = 0; row < 33; ++row) prefill[row] = 900 + row;
    stages.push_back(makeStage(prefill, false));                                // prefill rows

    WorkspaceRequirements family;
    for (auto &c : stages)
        family.merge(c.stage->getWorkspaceRequirements(static_cast<int>(c.positions.size())));
    ASSERT_EQ(family.buffers.size(), 1u);
    EXPECT_EQ(family.buffers.front().name, RoPEWorkspaceBuffers::CPU_ANGLES);
    EXPECT_EQ(family.buffers.front().size_bytes,
              primitives::rope_angle_layout(33, kHeadDim / 2).bytes);

    CPUProjectionTestWorkspace workspace(family);
    for (auto &c : stages)
        c.stage->bindWorkspace(workspace.get());
    void *const address = workspace.get()->getBuffer(RoPEWorkspaceBuffers::CPU_ANGLES);
    ASSERT_NE(address, nullptr);

    for (int replay = 0; replay < 3; ++replay)
    {
        for (auto &c : stages)
        {
            // A new request per replay: shift every live position, keep padding.
            std::vector<int> positions = c.positions;
            for (int &position : positions)
                if (position >= 0)
                    position += 1000 * replay;
            restore(c);
            executeAndCompare(c, ctx, positions);
            EXPECT_EQ(c.stage->getWorkspace(), workspace.get());
            EXPECT_EQ(workspace.get()->getBuffer(RoPEWorkspaceBuffers::CPU_ANGLES), address);
        }
        for (auto &c : stages)
            c.stage->resetSessionState();
    }
}

/** @test A stage without its admitted binding fails before mutating Q or K. */
TEST(CPURoPEAngleWorkspaceLifecycle, UnboundStageIsFatalBeforeAnyStore)
{
    CPUDeviceContext ctx(DeviceId::cpu(), 4);
    auto c = makeStage({5, 6, 7}, false);
    EXPECT_THROW((void)c.stage->execute(&ctx), std::invalid_argument);
    EXPECT_EQ(values(*c.q), c.q_original);
    EXPECT_EQ(values(*c.k), c.k_original);
}

/** @test A binding admitted for fewer rows than the stage executes fails before any store. */
TEST(CPURoPEAngleWorkspaceLifecycle, UndersizedBindingIsFatalBeforeAnyStore)
{
    CPUDeviceContext ctx(DeviceId::cpu(), 4);
    auto c = makeStage({5, 6, 7, 8, 9}, false);
    CPUProjectionTestWorkspace workspace(c.stage->getWorkspaceRequirements(2));
    c.stage->bindWorkspace(workspace.get());
    EXPECT_THROW((void)c.stage->execute(&ctx), std::invalid_argument);
    EXPECT_EQ(values(*c.q), c.q_original);
    EXPECT_EQ(values(*c.k), c.k_original);
}

/** @test The CPU declaration depends only on graph rows and the effective rotated width. */
TEST(CPURoPEAngleWorkspaceLifecycle, DeclarationFollowsRowsAndRotatedWidthOnly)
{
    CPURoPEKernelT<ActivationPrecision::FP32> kernel;
    const auto requirements = kernel.getWorkspaceRequirements(512, 256);
    ASSERT_EQ(requirements.buffers.size(), 1u);
    EXPECT_EQ(requirements.buffers.front().size_bytes, primitives::rope_angle_layout(512, 128).bytes);
    EXPECT_TRUE(kernel.getWorkspaceRequirements(512, 0).buffers.empty());
    EXPECT_EQ(kernel.getWorkspaceRequirements(512, 256).total_bytes_with_alignment(),
              cpu_rope_workspace::requirements({.graph_rows = 512, .rotary_dim = 256}).total_bytes_with_alignment());

    // A partial-RoPE stage declares only its rotated prefix and executes within it.
    CPUDeviceContext ctx(DeviceId::cpu(), 4);
    auto c = makeStage({40, 41, 42}, false);
    RoPEStage::Params params = c.stage->getParams();
    params.partial_rotary_factor = 0.25f;
    RoPEStage partial(params);
    const auto declared = partial.getWorkspaceRequirements(3);
    ASSERT_EQ(declared.buffers.size(), 1u);
    EXPECT_EQ(declared.buffers.front().size_bytes, primitives::rope_angle_layout(3, kHeadDim / 8).bytes);
    CPUProjectionTestWorkspace workspace(declared);
    partial.bindWorkspace(workspace.get());
    EXPECT_TRUE(partial.execute(&ctx));
}
}
