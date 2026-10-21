/**
 * @file Test__Qwen4ExpGDNOutputStage.cpp
 * @brief Real CPU-prepared Stage6 policy, compact rows and capture-identity contracts.
 *
 * This graph-construction fixture uses unchanged original Q8_0 N2560/K6144 bytes
 * and an actual prepared engine. It executes no model/kernel and creates no GPU
 * context. Native counted query descriptors belong to the explicit backend cases.
 */
#include "../../../utils/Qwen4ExpGDNOutputStageFixture.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
using namespace llaminar2::test;

/** @test CPU compact rows retain exact shape/source and all four policies select only the intended Stage6 arithmetic. */
TEST(Qwen4ExpGDNOutputStage, CPUCompactUsesRealPreparedSourceAndIndependentPolicy)
{
    Qwen4ExpGDNOutputStageFixture source(DeviceId::cpu());
    for (const int rows : {1, 15, 512, 1024, 2048})
    {
        Qwen4ExpGDNOutputActivationFixture activation(rows);
        const std::vector<int> lengths{rows};
        ForwardInput input;
        input.execution_phase=ForwardExecutionPhase::Prefill;
        input.batch_size=1;input.seq_len=rows;input.real_seq_len=rows;input.device=DeviceId::cpu();
        std::array<std::uint64_t,2> corrected_identity{};
        for (const auto hc : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
        for (const auto gdn : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
        for (const bool explicit_lengths : {false, true})
        {
            input.sequence_lengths = explicit_lengths ? &lengths : nullptr;
            auto config = source.config;
            config.qfn_hc_up_activation = hc;
            config.qfn_gdn_output_activation = gdn;
            auto builder = source.builder(config);
            ComputeGraph graph;
            activation.addProducer(graph, rows, DeviceId::cpu());
            ASSERT_EQ(builder->append(graph, input, activation.buffers, source.prepared.binding), "layer0_gdn_out_proj");
            const auto *node = graph.getNode("layer0_gdn_out_proj");
            ASSERT_NE(node, nullptr);
            const auto *stage = dynamic_cast<const GEMMStage *>(node->stage.get());
            ASSERT_NE(stage, nullptr);
            const auto &params = stage->getParams();
            EXPECT_EQ(params.B, source.weight.get());
            ASSERT_TRUE(params.prepared_ref);
            EXPECT_EQ(params.prepared_ref->binding_id, source.prepared.ref.binding_id);
            EXPECT_EQ(params.prepared_store, source.prepared.store.get());
            EXPECT_EQ(params.quantized_operand_precision, gdn);
            EXPECT_EQ(params.m, rows); EXPECT_EQ(params.n, 2560); EXPECT_EQ(params.k, 6144);
            EXPECT_EQ(params.alpha, 1.0f); EXPECT_EQ(params.beta, 0.0f);
            if (gdn == QuantizedActivationOperandPrecision::TwoTermQ8)
            {
                ASSERT_TRUE(params.verifier_row_range);
                EXPECT_EQ(params.verifier_row_range->countOwner(), explicit_lengths ? lengths.data() : nullptr);
                EXPECT_EQ(params.verifier_row_range->physicalRows(), rows);
                const auto signature = stage->graphCaptureVariantSignature();
                ASSERT_NE(signature, 0u);
                auto &same_authority=corrected_identity[explicit_lengths ? 1 : 0];
                if (same_authority) EXPECT_EQ(signature, same_authority);
                same_authority = signature;
            }
            else EXPECT_FALSE(params.verifier_row_range);
        }
        EXPECT_NE(corrected_identity[0],0u);
        EXPECT_NE(corrected_identity[1],0u);
        EXPECT_NE(corrected_identity[0],corrected_identity[1]);
    }
    EXPECT_TRUE(source.sourceUnchanged());
}

/** @test Missing prepared source and padded/inconsistent CPU invocation reject before Stage6 insertion. */
TEST(Qwen4ExpGDNOutputStage, CPURejectsMissingSourceAndPaddedQueryBeforeStage6)
{
    Qwen4ExpGDNOutputStageFixture source(DeviceId::cpu());
    Qwen4ExpGDNOutputActivationFixture activation(512);
    ForwardInput input;
    input.execution_phase=ForwardExecutionPhase::Prefill;
    input.batch_size=1;input.seq_len=512;input.real_seq_len=512;input.device=DeviceId::cpu();
    auto builder = source.builder(source.config);
    for (int mutation = 0; mutation != 6; ++mutation)
    {
        auto invalid = input;
        auto binding = source.prepared.binding;
        if (mutation == 0) invalid.real_seq_len = 9;
        if (mutation == 1) invalid.batch_size = 2;
        if (mutation == 2) binding.slice.col_count /= 2;
        if (mutation == 3) builder->setPreparedWeightStore(nullptr);
        if (mutation == 4) invalid.execution_role = static_cast<ForwardExecutionRole>(255);
        if (mutation == 5) invalid.execution_phase = static_cast<ForwardExecutionPhase>(255);
        ComputeGraph graph;
        activation.addProducer(graph, 512, DeviceId::cpu());
        EXPECT_THROW(builder->append(graph, invalid, activation.buffers, binding), std::exception);
        EXPECT_EQ(graph.size(), 1u);
        EXPECT_EQ(graph.getNode("layer0_gdn_out_proj"), nullptr);
        builder->setPreparedWeightStore(source.prepared.store.get());
    }
    EXPECT_TRUE(source.sourceUnchanged());
}

/** @test The CPU's original host valid-row vector must prove compactness; device-owned lengths cannot replace it. */
TEST(Qwen4ExpGDNOutputStage, CPURejectsNoncompactHostAuthorityBeforeStage6)
{
    Qwen4ExpGDNOutputStageFixture source(DeviceId::cpu());
    Qwen4ExpGDNOutputActivationFixture activation(512);
    ForwardInput input;
    input.execution_phase=ForwardExecutionPhase::Prefill;
    input.batch_size=1;input.seq_len=512;input.real_seq_len=512;input.device=DeviceId::cpu();
    auto builder = source.builder(source.config);
    for (const auto lengths : {std::vector<int>{}, std::vector<int>{0}, std::vector<int>{9},
        std::vector<int>{511}, std::vector<int>{-1}, std::vector<int>{513}, std::vector<int>{512,512}})
    {
        input.sequence_lengths = &lengths;
        ComputeGraph graph;
        activation.addProducer(graph,512,DeviceId::cpu());
        EXPECT_THROW(builder->append(graph,input,activation.buffers,source.prepared.binding),std::exception);
        EXPECT_EQ(graph.size(),1u);
        EXPECT_EQ(graph.getNode("layer0_gdn_out_proj"),nullptr);
    }
    const int32_t forbidden_device_owner = 512;
    input.sequence_lengths = nullptr;
    input.sequence_lengths_device = &forbidden_device_owner;
    ComputeGraph graph;
    activation.addProducer(graph,512,DeviceId::cpu());
    EXPECT_THROW(builder->append(graph,input,activation.buffers,source.prepared.binding),std::exception);
    EXPECT_EQ(graph.size(),1u);
    EXPECT_EQ(graph.getNode("layer0_gdn_out_proj"),nullptr);
    EXPECT_TRUE(source.sourceUnchanged());
}
