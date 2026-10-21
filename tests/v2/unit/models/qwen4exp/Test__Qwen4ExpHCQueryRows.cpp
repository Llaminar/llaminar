/**
 * @file Test__Qwen4ExpHCQueryRows.cpp
 * @brief Actual CPU-prepared HC branch/head Params retain the original compact query authority.
 *
 * The tests construct real production HC sites and inspect their actual stages.
 * They execute no model/projection and initialize no accelerator. Count changes
 * exercise immutable recording identity, not claimed arithmetic/replay evidence.
 */
#include "../../../utils/Qwen4ExpHCQueryFixture.h"
#include <gtest/gtest.h>
using namespace llaminar2;
using namespace llaminar2::test;

/** @test All main sites retain exact up-only rows without changing down/inject SingleQ8 policy. */
TEST(Qwen4ExpHCQueryRows, ActualCPUBranchReadOnlyAndHeadRetainOriginalAuthority)
{
    HCQuerySource source(DeviceId::cpu());
    for(const int rows:{1,15,512,1024,2048})
    {
        HCQueryActivations activation(rows);
        std::vector<int> lengths{rows};
        for(const auto site:hcQuerySites)
        for(const bool corrected:{false,true})
        for(const bool counted:{false,true})
        {
            auto config=source.config;
            config.qfn_hc_up_activation=corrected?QuantizedActivationOperandPrecision::TwoTermQ8:
                QuantizedActivationOperandPrecision::SingleQ8;
            auto builder=source.builder(config);
            ForwardInput input;
            input.device=DeviceId::cpu(); input.seq_len=rows; input.real_seq_len=rows;
            input.sequence_lengths=counted?&lengths:nullptr;
            ComputeGraph graph;
            const auto prefix=builder->append(graph,input,activation,site);
            const auto &up=hcQueryStage(graph,prefix+"read_up");
            const auto &params=up.getParams();
            EXPECT_EQ(params.m,rows);EXPECT_EQ(params.n,10240);EXPECT_EQ(params.k,320);
            EXPECT_EQ(params.quantized_operand_precision,config.qfn_hc_up_activation);
            EXPECT_EQ(params.B,source.upBinding(site).tensor);
            ASSERT_TRUE(params.prepared_ref);
            EXPECT_EQ(params.prepared_ref->binding_id,source.upBinding(site).prepared->binding_id);
            EXPECT_FALSE(params.force_decode_equivalent_verifier_prefill);
            const auto &down=hcQueryStage(graph,prefix+"read_down").getParams();
            EXPECT_EQ(down.quantized_operand_precision,QuantizedActivationOperandPrecision::SingleQ8);
            EXPECT_FALSE(down.force_decode_equivalent_verifier_prefill);
            EXPECT_FALSE(down.verifier_row_range);
            if(site==HCQuerySite::Attention || site==HCQuerySite::FeedForward)
            {
                const auto &inject=hcQueryStage(graph,prefix+"write_projection").getParams();
                EXPECT_EQ(inject.quantized_operand_precision,QuantizedActivationOperandPrecision::SingleQ8);
                EXPECT_FALSE(inject.force_decode_equivalent_verifier_prefill);
                EXPECT_FALSE(inject.verifier_row_range);
            }
            if(corrected)
            {
                ASSERT_TRUE(params.verifier_row_range);
                EXPECT_EQ(params.verifier_row_range->countOwner(),counted?lengths.data():nullptr);
                EXPECT_EQ(params.verifier_row_range->physicalRows(),rows);
                const auto signature=up.graphCaptureVariantSignature();ASSERT_NE(signature,0u);
                if(counted)
                {
                    for(const int live:{rows,0,std::min(9,rows),rows/2,0,rows})
                    {
                        lengths[0]=live;
                        EXPECT_EQ(params.verifier_row_range->activeRowsFor(live),live);
                        EXPECT_EQ(up.graphCaptureVariantSignature(),signature);
                    }
                    lengths[0]=rows;
                }
            }
            else EXPECT_FALSE(params.verifier_row_range);
        }
    }
}

/** @test Padded/foreign CPU frames reject before any HC graph site is published, without changing SingleQ8. */
TEST(Qwen4ExpHCQueryRows, CPUAuthorityRejectsPaddedQueriesAndKeepsSingleQ8Unchanged)
{
    HCQuerySource source(DeviceId::cpu());HCQueryActivations activation(512);
    for(const auto site:hcQuerySites)
    {
        auto builder=source.builder(source.config);
        for(const auto lengths:{std::vector<int>{},std::vector<int>{0},std::vector<int>{9},
            std::vector<int>{511},std::vector<int>{-1},std::vector<int>{513},std::vector<int>{512,512}})
        {
            ForwardInput input;input.device=DeviceId::cpu();input.seq_len=512;input.real_seq_len=512;
            input.sequence_lengths=&lengths;
            ComputeGraph graph;
            EXPECT_THROW(builder->append(graph,input,activation,site),std::exception);
            EXPECT_EQ(graph.size(),0u);
        }
        for(int mutation=0;mutation!=5;++mutation)
        {
            ForwardInput input;input.device=DeviceId::cpu();input.seq_len=512;input.real_seq_len=512;
            int32_t foreign=512;
            if(mutation==0)input.sequence_lengths_device=&foreign;
            if(mutation==1)input.real_seq_len=9;
            if(mutation==2)input.batch_size=2;
            if(mutation==3)input.execution_role=static_cast<ForwardExecutionRole>(255);
            if(mutation==4)input.execution_phase=static_cast<ForwardExecutionPhase>(255);
            ComputeGraph graph;
            EXPECT_THROW(builder->append(graph,input,activation,site),std::exception);
            EXPECT_EQ(graph.size(),0u);
        }
        auto single=source.config;single.qfn_hc_up_activation=QuantizedActivationOperandPrecision::SingleQ8;
        auto unchanged=source.builder(single);
        std::vector<int> short_rows{9};
        ForwardInput input;input.device=DeviceId::cpu();input.seq_len=512;input.real_seq_len=512;
        input.sequence_lengths=&short_rows;
        ComputeGraph graph;const auto prefix=unchanged->append(graph,input,activation,site);
        EXPECT_FALSE(hcQueryStage(graph,prefix+"read_up").getParams().verifier_row_range);
    }
}

/** @test Actual HC constructors permit geometry-only manifests, but every resulting graph stays non-executable. */
TEST(Qwen4ExpHCQueryRows, WorkspaceOnlySitesDeclareCapacityWithoutClaimingLiveRows)
{
    HCQuerySource source(DeviceId::cpu());
    for(const int rows:{1,512})
        for(const auto site:hcQuerySites)
        {
            HCQueryActivations activation(rows);auto builder=source.builder(source.config);
            std::vector<int> not_admitted{0};
            ForwardInput input;input.device=DeviceId::cpu();input.seq_len=rows;
            input.graph_submission_intent=ForwardGraphSubmissionIntent::WorkspaceDeclaration;
            input.sequence_lengths=&not_admitted;
            ComputeGraph declaration;const auto prefix=builder->append(declaration,input,activation,site);
            ASSERT_TRUE(declaration.isWorkspaceDeclaration());
            const auto &up=hcQueryStage(declaration,prefix+"read_up").getParams();
            ASSERT_TRUE(up.verifier_row_range);
            EXPECT_EQ(up.verifier_row_range->physicalRows(),rows);
            EXPECT_EQ(up.verifier_row_range->countOwner(),not_admitted.data());
            DeviceGraphExecutor executor;
            EXPECT_THROW(executor.execute(declaration,nullptr),std::invalid_argument);
            EXPECT_THROW(executor.captureRetainedGraphFragment(declaration,nullptr,nullptr),std::invalid_argument);
            // Identical geometry with execution intent still rejects an unready
            // count. Workspace permission never relaxes the actual CPU query.
            input.graph_submission_intent=ForwardGraphSubmissionIntent::Execute;
            ComputeGraph runtime;EXPECT_THROW(builder->append(runtime,input,activation,site),std::invalid_argument);
            EXPECT_EQ(runtime.size(),0u);
        }
}
