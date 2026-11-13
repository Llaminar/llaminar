/**
 * @file Test__Qwen4ExpFAOutputStage.cpp
 * @brief Real CPU-prepared Stage5 policy, compact rows and capture-identity contracts.
 *
 * This graph-construction fixture uses unchanged original Q8_0 N2560/K6144 bytes
 * and an actual prepared engine. It executes no model/kernel and creates no GPU
 * context. Native counted query descriptors belong to the explicit backend cases.
 */
#include "../../../utils/Qwen4ExpFAOutputStageFixture.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
using namespace llaminar2::test;

/** @test CPU compact rows retain exact shape/source and all eight policies select only the intended Stage5 arithmetic. */
TEST(Qwen4ExpFAOutputStage, CPUCompactUsesRealPreparedSourceAndIndependentPolicy)
{
    Qwen4ExpFAOutputStageFixture source(DeviceId::cpu());
    for (const int rows : {1, 15, 512, 1024, 2048})
    {
        Qwen4ExpFAOutputActivationFixture activation(rows);
        const std::vector<int> lengths{rows};
        ForwardInput input;
        input.execution_phase=ForwardExecutionPhase::Prefill;
        input.batch_size=1;input.seq_len=rows;input.real_seq_len=rows;input.device=DeviceId::cpu();
        std::array<std::uint64_t,2> corrected_identity{};
        for (const auto hc : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
        for (const auto gdn : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
        for (const auto fa : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
        for (const bool explicit_lengths : {false, true})
        {
            input.sequence_lengths = explicit_lengths ? &lengths : nullptr;
            auto config = source.source.config;
            config.qfn_hc_up_activation = hc;
            config.qfn_gdn_output_activation = gdn;
            config.qfn_fa_output_activation = fa;
            auto builder = source.builder(config);
            ComputeGraph graph;
            activation.addProducer(graph, rows, DeviceId::cpu());
            ASSERT_EQ(builder->append(graph, input, activation.buffers, source.source.prepared.binding), "layer3_wo_proj");
            const auto *node = graph.getNode("layer3_wo_proj");
            ASSERT_NE(node, nullptr);
            const auto *stage = dynamic_cast<const GEMMStage *>(node->stage.get());
            ASSERT_NE(stage, nullptr);
            const auto &params = stage->getParams();
            EXPECT_EQ(params.B, source.source.weight.get());
            ASSERT_TRUE(params.prepared_ref);
            EXPECT_EQ(params.prepared_ref->binding_id, source.source.prepared.ref.binding_id);
            EXPECT_EQ(params.prepared_store, source.source.prepared.store.get());
            EXPECT_EQ(params.quantized_operand_precision, fa);
            EXPECT_EQ(params.m, rows); EXPECT_EQ(params.n, 2560); EXPECT_EQ(params.k, 6144);
            EXPECT_EQ(params.alpha, 1.0f); EXPECT_EQ(params.beta, 0.0f);
            if (fa == QuantizedActivationOperandPrecision::TwoTermQ8)
            {
                ASSERT_TRUE(params.verifier_row_range);
                EXPECT_EQ(params.verifier_row_range->countOwner(), explicit_lengths ? lengths.data() : nullptr);
                EXPECT_EQ(params.verifier_row_range->physicalRows(), rows);
                const auto signature = stage->graphCaptureVariantSignature();
                ASSERT_NE(signature, 0u);
                auto &same_authority=corrected_identity[explicit_lengths ? 1 : 0];
                if (same_authority) { EXPECT_EQ(signature, same_authority); }
                same_authority = signature;
            }
            else { EXPECT_FALSE(params.verifier_row_range); }
        }
        EXPECT_NE(corrected_identity[0],0u);
        EXPECT_NE(corrected_identity[1],0u);
        EXPECT_NE(corrected_identity[0],corrected_identity[1]);
    }
    EXPECT_TRUE(source.source.sourceUnchanged());
}

/** @test Missing prepared source and padded/inconsistent CPU invocation reject before Stage5 insertion. */
TEST(Qwen4ExpFAOutputStage, CPURejectsMissingSourceAndPaddedQueryBeforeStage5)
{
    Qwen4ExpFAOutputStageFixture source(DeviceId::cpu());
    Qwen4ExpFAOutputActivationFixture activation(512);
    ForwardInput input;
    input.execution_phase=ForwardExecutionPhase::Prefill;
    input.batch_size=1;input.seq_len=512;input.real_seq_len=512;input.device=DeviceId::cpu();
    auto builder = source.builder(source.source.config);
    for (int mutation = 0; mutation != 6; ++mutation)
    {
        auto invalid = input;
        auto binding = source.source.prepared.binding;
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
        EXPECT_EQ(graph.getNode("layer3_wo_proj"), nullptr);
        builder->setPreparedWeightStore(source.source.prepared.store.get());
    }
    EXPECT_TRUE(source.source.sourceUnchanged());
}

/** @test The CPU's original host valid-row vector must prove compactness; device-owned lengths cannot replace it. */
TEST(Qwen4ExpFAOutputStage, CPURejectsNoncompactHostAuthorityBeforeStage5)
{
    Qwen4ExpFAOutputStageFixture source(DeviceId::cpu());
    Qwen4ExpFAOutputActivationFixture activation(512);
    ForwardInput input;
    input.execution_phase=ForwardExecutionPhase::Prefill;
    input.batch_size=1;input.seq_len=512;input.real_seq_len=512;input.device=DeviceId::cpu();
    auto builder = source.builder(source.source.config);
    for (const auto lengths : {std::vector<int>{}, std::vector<int>{0}, std::vector<int>{9},
        std::vector<int>{511}, std::vector<int>{-1}, std::vector<int>{513}, std::vector<int>{512,512}})
    {
        input.sequence_lengths = &lengths;
        ComputeGraph graph;
        activation.addProducer(graph,512,DeviceId::cpu());
        EXPECT_THROW(builder->append(graph,input,activation.buffers,source.source.prepared.binding),std::exception);
        EXPECT_EQ(graph.size(),1u);
        EXPECT_EQ(graph.getNode("layer3_wo_proj"),nullptr);
    }
    const int32_t forbidden_device_owner = 512;
    input.sequence_lengths = nullptr;
    input.sequence_lengths_device = &forbidden_device_owner;
    ComputeGraph graph;
    activation.addProducer(graph,512,DeviceId::cpu());
    EXPECT_THROW(builder->append(graph,input,activation.buffers,source.source.prepared.binding),std::exception);
    EXPECT_EQ(graph.size(),1u);
    EXPECT_EQ(graph.getNode("layer3_wo_proj"),nullptr);
    EXPECT_TRUE(source.source.sourceUnchanged());
}

/** @test Foreign original roles, global sources and prepared intervals reject before actual Stage5 insertion. */
TEST(Qwen4ExpFAOutputStage, OriginalSourceAndPreparedOwnershipAreExact)
{
    Qwen4ExpFAOutputStageFixture source(DeviceId::cpu());
    Qwen4ExpFAOutputActivationFixture activation(15);
    ForwardInput input;input.device=DeviceId::cpu();input.batch_size=1;input.seq_len=15;
    input.real_seq_len=15;input.execution_phase=ForwardExecutionPhase::Prefill;
    auto builder=source.builder(source.source.config);
    for(int mutation=0;mutation<9;++mutation)
    {
        auto binding=source.source.prepared.binding;
        if(mutation==0)binding.immutable=false;
        if(mutation==1)binding.identity.role=WeightRole::Other;
        if(mutation==2)binding.identity.layer=0;
        if(mutation==3)binding.identity.canonical_name="blk.48.attn_output.weight";
        if(mutation==4)binding.tensor=nullptr;
        if(mutation==5)binding.slice.source_cols=6112;
        if(mutation==6)binding.slice.row_count/=2;
        if(mutation==7)binding.slice.col_start=1;
        if(mutation==8)binding.identity.model_id=ModelContextId{71127};
        ComputeGraph graph;activation.addProducer(graph,15,DeviceId::cpu());
        EXPECT_THROW(builder->append(graph,input,activation.buffers,binding),std::exception);
        EXPECT_EQ(graph.size(),1u);EXPECT_EQ(graph.getNode("layer3_wo_proj"),nullptr);
    }
    EXPECT_TRUE(source.source.sourceUnchanged());
}
/** @test Geometry-only setup retains an explicit non-executable graph capability. */
TEST(Qwen4ExpFAOutputStage, WorkspaceDeclarationCannotExecute)
{
    Qwen4ExpFAOutputStageFixture source(DeviceId::cpu());
    Qwen4ExpFAOutputActivationFixture activation(512);
    ForwardInput input;input.device=DeviceId::cpu();input.batch_size=1;input.seq_len=512;
    input.graph_submission_intent=ForwardGraphSubmissionIntent::WorkspaceDeclaration;
    auto builder=source.builder(source.source.config);
    ComputeGraph graph;activation.addProducer(graph,512,DeviceId::cpu());
    EXPECT_EQ(builder->append(graph,input,activation.buffers,source.source.prepared.binding),"layer3_wo_proj");
    EXPECT_TRUE(graph.isWorkspaceDeclaration());EXPECT_THROW(graph.requireExecutable(),std::invalid_argument);
}
