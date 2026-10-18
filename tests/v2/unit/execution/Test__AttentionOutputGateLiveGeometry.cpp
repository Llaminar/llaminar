/**
 * @file Test__AttentionOutputGateLiveGeometry.cpp
 * @brief Prove gate execution and snapshots use live packed data, not capacity.
 *
 * Phase-split TP retains full decode/GDN storage while prefill owns fewer heads.
 * These device-free regressions exercise the production CPU stage with poisoned
 * spare rows/columns, in-place output, unequal SIMD tails and large-to-small
 * reuse. Native backends consume the same checked element count at dispatch.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

/**
 * @brief Compare the packed live prefix and prove every remaining byte is retained.
 * @param input Immutable complete input backing before this invocation.
 * @param gate Immutable complete gate backing before this invocation.
 * @param before Complete output backing before this invocation.
 * @param output Observed output from the real stage.
 * @param live_count Sole valid packed extent, independent of physical row width.
 */
static void expectLiveOutputAndUntouchedTail(
    const std::vector<float> &input, const std::vector<float> &gate,
    const std::vector<float> &before, const FP32Tensor &output, size_t live_count)
{
    const float *actual = output.data();
    for (size_t index = 0; index < live_count; ++index)
    {
        const float expected = input[index] / (1.0f + std::exp(-gate[index]));
        EXPECT_NEAR(actual[index], expected, std::abs(expected) * 0.005f + 1.0e-6f)
            << "live packed index=" << index;
    }
    ASSERT_EQ(std::memcmp(actual + live_count, before.data() + live_count,
                          (before.size() - live_count) * sizeof(float)), 0)
        << "Unused retained capacity must never become a gate write extent";
}

/** @brief Wider rows and spare row capacity must remain outside live execution. */
TEST(AttentionOutputGateLiveGeometry, PackedPartialRowsLeaveAllUnusedCapacityUntouched)
{
    constexpr size_t capacity_rows = 7;
    constexpr size_t capacity_columns = 4096;
    const size_t capacity = capacity_rows * capacity_columns;
    CPUDeviceContext context(DeviceId::cpu(), 1);

    for (const int live_rows : {1, 3, 5})
    for (const int live_columns : {17, 2048, 4096})
    for (const bool in_place : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << live_rows << "x" << live_columns
                                         << " in_place=" << in_place);
        FP32Tensor input({capacity_rows, capacity_columns}, DeviceId::cpu());
        FP32Tensor gate({capacity_rows, capacity_columns}, DeviceId::cpu());
        FP32Tensor destination({capacity_rows, capacity_columns}, DeviceId::cpu());
        FP32Tensor *output = in_place ? &input : &destination;
        const size_t live_count = static_cast<size_t>(live_rows) * live_columns;
        std::fill_n(input.mutable_data(), capacity, -777.0f);
        std::fill_n(gate.mutable_data(), capacity, std::numeric_limits<float>::quiet_NaN());
        std::fill_n(destination.mutable_data(), capacity, 12345.0f);
        for (size_t index = 0; index < live_count; ++index)
        {
            input.mutable_data()[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.125f;
            gate.mutable_data()[index] = static_cast<float>(static_cast<int>(index % 7) - 3) * 0.5f;
        }
        const std::vector<float> original_input(input.data(), input.data() + capacity);
        const std::vector<float> original_gate(gate.data(), gate.data() + capacity);
        const std::vector<float> before(output->data(), output->data() + capacity);

        AttentionOutputGateStage stage({.input = &input, .gate = &gate, .output = output,
                                       .seq_len = live_rows, .feature_dim = live_columns});
        ASSERT_TRUE(stage.execute(&context));
        expectLiveOutputAndUntouchedTail(original_input, original_gate, before, *output, live_count);
        EXPECT_EQ(std::memcmp(gate.data(), original_gate.data(), capacity * sizeof(float)), 0);
        if (!in_place)
            EXPECT_EQ(std::memcmp(input.data(), original_input.data(), capacity * sizeof(float)), 0);

        const auto dump = stage.buildDumpInfoImpl();
        ASSERT_EQ(dump.inputs.size(), 2u);
        ASSERT_EQ(dump.outputs.size(), 1u);
        for (const auto &entry : dump.inputs)
        {
            EXPECT_EQ(entry.rows, static_cast<size_t>(live_rows));
            EXPECT_EQ(entry.cols, static_cast<size_t>(live_columns));
            EXPECT_EQ(entry.byte_size, live_count * sizeof(float));
        }
        EXPECT_EQ(dump.outputs[0].tensor, output);
        EXPECT_EQ(dump.outputs[0].rows, static_cast<size_t>(live_rows));
        EXPECT_EQ(dump.outputs[0].cols, static_cast<size_t>(live_columns));
        EXPECT_EQ(dump.outputs[0].byte_size, live_count * sizeof(float));
        EXPECT_EQ(stage.estimatedFlops(), live_count * 5);
        EXPECT_EQ(stage.estimatedMemoryBytes(), live_count * 3 * sizeof(float));
    }
}

/** @brief Retained full-decode and sharded-prefill stages reuse the same backing. */
TEST(AttentionOutputGateLiveGeometry, FullDecodeAndShardedPrefillReuseLargeToSmall)
{
    constexpr size_t capacity = 5 * 32;
    FP32Tensor input({5, 32}, DeviceId::cpu());
    FP32Tensor gate({5, 32}, DeviceId::cpu());
    FP32Tensor output({5, 32}, DeviceId::cpu());
    std::fill_n(gate.mutable_data(), capacity, 0.0f);
    for (size_t index = 0; index < capacity; ++index)
        input.mutable_data()[index] = static_cast<float>(index + 1);
    std::fill_n(output.mutable_data(), capacity, -999.0f);
    CPUDeviceContext context(DeviceId::cpu(), 1);

    AttentionOutputGateStage full_prefill({.input = &input, .gate = &gate, .output = &output,
                                          .seq_len = 5, .feature_dim = 32});
    AttentionOutputGateStage sharded_prefill({.input = &input, .gate = &gate, .output = &output,
                                             .seq_len = 3, .feature_dim = 16});
    AttentionOutputGateStage full_decode({.input = &input, .gate = &gate, .output = &output,
                                         .seq_len = 1, .feature_dim = 32});
    const std::vector<float> original_input(input.data(), input.data() + capacity);
    const std::vector<float> original_gate(gate.data(), gate.data() + capacity);
    for (auto *stage : {&full_prefill, &sharded_prefill, &full_decode, &sharded_prefill})
    {
        const std::vector<float> before(output.data(), output.data() + capacity);
        ASSERT_TRUE(stage->execute(&context));
        const auto &params = stage->getParams();
        expectLiveOutputAndUntouchedTail(original_input, original_gate, before, output,
                                         static_cast<size_t>(params.seq_len) * params.feature_dim);
    }
}

/** @brief Missing geometry or insufficient backing fails before touching output. */
TEST(AttentionOutputGateLiveGeometry, InvalidGeometryAndCapacityFailBeforeWrites)
{
    EXPECT_THROW(AttentionOutputGateStage(AttentionOutputGateStage::Params{}), std::invalid_argument);
    EXPECT_THROW(AttentionOutputGateStage((AttentionOutputGateStage::Params{
                     .seq_len = 1, .feature_dim = 0})), std::invalid_argument);
    EXPECT_THROW(AttentionOutputGateStage((AttentionOutputGateStage::Params{
                     .seq_len = 0, .feature_dim = 8})), std::invalid_argument);
    EXPECT_THROW(AttentionOutputGateStage((AttentionOutputGateStage::Params{
                     .seq_len = std::numeric_limits<int>::max(), .feature_dim = 2})), std::invalid_argument);
    CPUDeviceContext context(DeviceId::cpu(), 1);
    for (const int short_role : {0, 1, 2})
    {
        SCOPED_TRACE(short_role);
        FP32Tensor input({short_role == 0 ? 1u : 2u, 8}, DeviceId::cpu());
        FP32Tensor gate({short_role == 1 ? 1u : 2u, 8}, DeviceId::cpu());
        FP32Tensor output({short_role == 2 ? 1u : 2u, 8}, DeviceId::cpu());
        std::fill_n(output.mutable_data(), output.numel(), 123.0f);
        AttentionOutputGateStage stage({.input = &input, .gate = &gate, .output = &output,
                                       .seq_len = 2, .feature_dim = 8});
        EXPECT_THROW(stage.execute(&context), std::invalid_argument);
        EXPECT_THROW(stage.buildDumpInfoImpl(), std::invalid_argument);
        for (size_t index = 0; index < output.numel(); ++index)
            EXPECT_EQ(output.data()[index], 123.0f);
    }
}

/** @brief The generic post-Wo schema passes model width rather than buffer width. */
TEST(AttentionOutputGateLiveGeometry, DeclarativePostWoGateUsesResolvedModelWidth)
{
    FP32Tensor input({5, 32}, DeviceId::cpu());
    FP32Tensor gate({5, 32}, DeviceId::cpu());
    FP32Tensor output({5, 32}, DeviceId::cpu());
    ResolvedGraphSpec spec;
    spec.stages.push_back({.name = "post_wo_gate", .type = StageType::AttentionOutputGate,
                           .inputs = {&input, &gate}, .outputs = {&output},
                           .int_params = {{"seq_len", 3}, {"d_model", 7}}});
    auto graph = GraphBuilder::build(spec);
    auto *node = graph.getNode("post_wo_gate");
    ASSERT_NE(node, nullptr);
    auto *stage = dynamic_cast<AttentionOutputGateStage *>(node->stage.get());
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->getParams().feature_dim, 7);
    EXPECT_EQ(stage->buildDumpInfoImpl().outputs.at(0).byte_size, 3u * 7 * sizeof(float));
}
