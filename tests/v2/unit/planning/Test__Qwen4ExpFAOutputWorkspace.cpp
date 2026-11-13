/**
 * @file Test__Qwen4ExpFAOutputWorkspace.cpp
 * @brief Source-main FA selection and exact shared residual-bank admission contracts.
 *
 * Device-free source metadata and canonical workspace factories test the same
 * named BOM used by actual stages. No backend, model or preparation is started.
 */
#include "../../utils/Qwen4ExpHCWorkspaceFixture.h"
#include "models/qwen4exp/Qwen4ExpProjectionWorkspace.h"
#include "planning/MemoryPlanner.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <gtest/gtest.h>
#include <map>
using namespace llaminar2;
namespace
{
constexpr auto single=QuantizedActivationOperandPrecision::SingleQ8;
constexpr auto two=QuantizedActivationOperandPrecision::TwoTermQ8;
/** @return Metadata for full main48 and physically distinct predictor48, without payloads. */
ModelMemoryProfile source()
{
    auto p=test::qfnHCWorkspaceProfile();
    p.n_layers=48; p.mtp_layer_count=1;
    p.flash_next_predictor=Qwen4ExpPredictorMetadata{48,Qwen4ExpPredictorAttention::Dense,
        Qwen4ExpPredictorHiddenNorm::PerBranch,Qwen4ExpPredictorSource::SharedGGUF};
    p.d_model=2560;p.d_ff=640;p.n_heads=24;p.n_kv_heads=2;p.head_dim=256;
    p.gdn_group_count=16;p.gdn_time_step_rank=48;p.gdn_state_size=128;
    p.gdn_inner_size=6144;p.gdn_conv_kernel_size=4;
    p.flash_next->residual_width=10240;p.flash_next->terminal_head_width=2560;p.flash_next->residual_gate_rank=320;
    p.tensors.clear();
    p.tensors.push_back({.name="blk.0.hc_attn_up.weight",.native_bytes=10240ull*320/32*34,
        .quant_type="Q8_0",.elements=10240ull*320,.K=320,.layer_index=0});
    for(const int layer:{1,3,47,48})
        p.tensors.push_back({.name="blk."+std::to_string(layer)+(layer==1?".ssm_out.weight":".attn_output.weight"),
            .native_bytes=2560ull*6144/32*34,.quant_type="Q8_0",.elements=2560ull*6144,.K=6144,.layer_index=layer});
    return p;
}
/** @return Exact canonical residual names beyond ordinary source-native arithmetic. */
WorkspaceRequirements residual(DeviceId device,int m,int k)
{
    if(device.is_gpu())return gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(m,2560,k);
    WorkspaceRequirements r;
    for(const auto &b:cpuQ8ResidualCorrectionWorkspaceRequirements(m,2560,k).buffers)
        if(b.name==kCPUProjectionResidualQ8)r.buffers.push_back(b);
    return r;
}
/** @return Positive source-local participant geometry without runtime queries. */
WorkspaceMemoryGeometry geometry(DeviceId d,int m)
{
    auto g=test::qfnHCWorkspaceGeometry(d,m);g.local_d_ff=640;
    g.local_query_heads=24;g.local_kv_heads=2;g.last_layer=47;g.mtp_target_query_rows=15;
    g.mtp_dense_layout=MTPShiftedKVHeadLayout::FullModelPerParticipant;return g;
}
}
/** @test Only original main FA sources qualify; malformed/foreign names and predictor never cross-match. */
TEST(Qwen4ExpFAOutputWorkspace, OriginalMainOnlyAndStrictPolicy)
{
    EXPECT_EQ(DevicePlanConfig{}.qfn_fa_output_activation,single);
    EXPECT_EQ(WorkspaceMemoryGeometry{}.qfn_fa_output_activation,single);
    for(const auto name:{"blk.3.attn_output.weight","blk.47.attn_output.weight"})
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::faOutputPrecision(two,true,WeightRole::AttentionWO,name,48),two);
    for(const auto name:{"blk.48.attn_output.weight","blk.49.attn_output.weight","blk.-1.attn_output.weight",
        "blk.x.attn_output.weight","blk.-0.attn_output.weight","blk.03.attn_output.weight","blk.3.ssm_out.weight","blk.3.attn_output.bias","output.weight"})
        EXPECT_EQ(Qwen4ExpProjectionWorkspace::faOutputPrecision(two,true,WeightRole::AttentionWO,name,48),single);
    EXPECT_EQ(Qwen4ExpProjectionWorkspace::faOutputPrecision(two,false,WeightRole::AttentionWO,"blk.3.attn_output.weight",48),single);
    EXPECT_EQ(Qwen4ExpProjectionWorkspace::faOutputPrecision(two,true,WeightRole::Other,"blk.3.attn_output.weight",48),single);
    EXPECT_THROW(Qwen4ExpProjectionWorkspace::faOutputPrecision(two,true,WeightRole::AttentionWO,"blk.3.attn_output.weight",0),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpProjectionWorkspace::faOutputPrecision(static_cast<QuantizedActivationOperandPrecision>(255),false,WeightRole::Other,{},48),std::invalid_argument);
    const WeightSliceSpec full{.source_rows=2560,.source_cols=6144,.row_count=2560,.col_count=6144};
    for(int mutation=0;mutation<6;++mutation)
    {
        auto slice=full;int requests=1;
        if(mutation==0)slice.row_start=1;
        if(mutation==1)slice.col_start=1;
        if(mutation==2)slice.row_count/=2;
        if(mutation==3)slice.col_count/=2;
        if(mutation==4)slice.source_cols=0;
        if(mutation==5)requests=2;
        EXPECT_THROW(Qwen4ExpProjectionWorkspace::validateFAOutputInvocation(two,true,WeightRole::AttentionWO,
            "blk.3.attn_output.weight",48,requests,slice),std::invalid_argument);
    }
    EXPECT_NO_THROW(Qwen4ExpProjectionWorkspace::validateFAOutputInvocation(two,true,WeightRole::AttentionWO,
        "blk.3.attn_output.weight",48,1,full));
    auto full_alias=full;full_alias.inner_is_presliced=true;
    EXPECT_NO_THROW(Qwen4ExpProjectionWorkspace::validateFAOutputInvocation(two,true,WeightRole::AttentionWO,
        "blk.3.attn_output.weight",48,1,full_alias));
}
/**
 * @test Every policy union owns one maximum bank within a serial graph family.
 * Equal-row MTP families retain that exact addition after max+rollback: the
 * same named bank is added to both envelopes, while rollback is unchanged.
 */
TEST(Qwen4ExpFAOutputWorkspace, EightPoliciesMergeExactCanonicalBanks)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int rows : {1, 15, 512, 1024, 2048})
    for (const bool retained_family : {false, true})
    {
        const auto profile = source();
        auto plan = geometry(device, rows);
        plan.mtp_target_query_rows = retained_family ? rows : 0;
        const auto ordinary = WorkspaceMemoryEstimator::estimate(profile, plan);
        for (const auto hc : {single, two})
        for (const auto gdn : {single, two})
        for (const auto fa : {single, two})
        {
            SCOPED_TRACE(::testing::Message()
                << device.toString() << " M=" << rows
                << " compact=" << plan.mtp_target_query_rows
                << " HC=" << static_cast<int>(hc)
                << " GDN=" << static_cast<int>(gdn)
                << " FA=" << static_cast<int>(fa));
            plan.qfn_hc_up_activation = hc;
            plan.qfn_gdn_output_activation = gdn;
            plan.qfn_fa_output_activation = fa;
            WorkspaceRequirements expected;
            // Only main layer0 owns HC in this source inventory. A wider
            // predictor HC invocation may not be inferred from MTP capacity.
            if (hc == two)
                expected.merge(residual(device, rows, 320));
            if (gdn == two || fa == two)
                expected.merge(residual(device, rows, 6144));
            EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, plan),
                ordinary + expected.total_bytes_with_alignment());
        }
    }
}
/**
 * @test Unequal main/compact rows preserve proven named-bank inclusion identities.
 * GDN and FA select the same K6144 bank at each family's original row extent;
 * that bank contains the K320 main HC bank. No scalar max-envelope is inferred.
 */
TEST(Qwen4ExpFAOutputWorkspace, FixedDepthMixedFamiliesPreserveNamedBankInclusion)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    for (const int rows : {1, 15, 512, 1024, 2048})
    {
        const auto profile = source();
        const auto base = geometry(device, rows);
        const auto ordinary = WorkspaceMemoryEstimator::estimate(profile, base);
        const auto increment = [&](QuantizedActivationOperandPrecision hc,
                                   QuantizedActivationOperandPrecision gdn,
                                   QuantizedActivationOperandPrecision fa)
        {
            auto plan = base;
            plan.qfn_hc_up_activation = hc;
            plan.qfn_gdn_output_activation = gdn;
            plan.qfn_fa_output_activation = fa;
            return WorkspaceMemoryEstimator::estimate(profile, plan);
        };
        const auto hc_only = increment(two, single, single);
        const auto gdn_only = increment(single, two, single);
        const auto fa_only = increment(single, single, two);
        SCOPED_TRACE(::testing::Message() << device.toString()
            << " M=" << rows << " compact=15");
        ASSERT_GE(hc_only, ordinary);
        ASSERT_GE(gdn_only, ordinary);
        ASSERT_GE(fa_only, ordinary);
        EXPECT_EQ(gdn_only, fa_only);
        // Prove containment independently at both actual invocation widths.
        for (const int family_rows : {rows, 15})
        {
            auto bank = residual(device, family_rows, 6144);
            const auto bytes = bank.total_bytes_with_alignment();
            bank.merge(residual(device, family_rows, 320));
            EXPECT_EQ(bank.total_bytes_with_alignment(), bytes);
        }
        for (const auto hc : {single, two})
        for (const auto gdn : {single, two})
        for (const auto fa : {single, two})
        {
            SCOPED_TRACE(::testing::Message()
                << "HC=" << static_cast<int>(hc)
                << " GDN=" << static_cast<int>(gdn)
                << " FA=" << static_cast<int>(fa));
            const auto actual = increment(hc, gdn, fa);
            EXPECT_GE(actual, ordinary);
            if (hc == two)
            {
                EXPECT_GE(actual, hc_only);
            }
            if (gdn == two)
            {
                EXPECT_GE(actual, gdn_only);
            }
            if (fa == two)
            {
                EXPECT_GE(actual, fa_only);
            }
            EXPECT_EQ(actual, gdn == two || fa == two
                ? gdn_only : hc == two ? hc_only : ordinary);
        }
    }
}
/** @test Original PP source intervals control presence, and predictor FA never borrows the main policy. */
TEST(Qwen4ExpFAOutputWorkspace, OwnedIntervalsFormatsAndFullDenseSource)
{
    for(const auto d:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
    for(const int layer:{0,1,3,47,48})
    {
        SCOPED_TRACE(::testing::Message()
            << d.toString() << " M=512 compact=15 layer=" << layer
            << " HC=SingleQ8 GDN=SingleQ8 FA=TwoTermQ8");
        auto p=source();auto g=geometry(d,512);g.first_layer=g.last_layer=layer;
        const auto ordinary=WorkspaceMemoryEstimator::estimate(p,g);g.qfn_fa_output_activation=two;
        std::size_t additional = 0;
        if (layer == 3 || layer == 47)
        {
            // FA-only participants have no recurrent rollback contract that
            // proves transient aliasing between main and compact namespaces.
            // Preserve both complete envelopes, including their distinct M.
            const auto compact = residual(d, 15, 6144).total_bytes_with_alignment();
            EXPECT_EQ(compact, 103680u);
            additional = residual(d, 512, 6144).total_bytes_with_alignment() + compact;
        }
        EXPECT_EQ(WorkspaceMemoryEstimator::estimate(p,g),ordinary+additional);
        p.tensors[2].quant_type="F16";
        if(layer==3){EXPECT_THROW(WorkspaceMemoryEstimator::estimate(p,g),std::invalid_argument);}
        else{EXPECT_NO_THROW(WorkspaceMemoryEstimator::estimate(p,g));}
    }
    auto p=source();auto g=geometry(DeviceId::rocm(0),15);g.qfn_fa_output_activation=two;
    g.batch_size=2;EXPECT_THROW(WorkspaceMemoryEstimator::estimate(p,g),std::invalid_argument);
    g.batch_size=1;g.total_shards=2;g.local_query_heads=12;g.local_kv_heads=1;
    EXPECT_THROW(WorkspaceMemoryEstimator::estimate(p,g),std::invalid_argument);
    g.qfn_fa_output_activation=single;EXPECT_NO_THROW(WorkspaceMemoryEstimator::estimate(p,g));
    g=geometry(DeviceId::rocm(0),15);g.qfn_fa_output_activation=two;
    p.tensors[2].elements=2560ull*6112;p.tensors[2].K=6112;
    EXPECT_THROW(WorkspaceMemoryEstimator::estimate(p,g),std::invalid_argument);
}

/** @test Canonical PMA admits the complete corrected workspace exactly, and rejects one byte less. */
TEST(Qwen4ExpFAOutputWorkspace, ExactFitAndOneByteShortUseCanonicalPMA)
{
    for(const auto d:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
    {
        auto p=source();auto g=geometry(d,15);g.qfn_fa_output_activation=two;
        const auto workspace=WorkspaceMemoryEstimator::estimate(p,g);
        constexpr std::size_t weights=4096;
        for(const std::size_t shortfall:{0u,1u})
        {
            PhysicalMemoryPlanBuilder plan;
            PhysicalMemoryResource resource{.world_rank=0,.device=d,.total_bytes=workspace+weights,
                .admission_available_bytes=workspace+weights-shortfall};
            plan.add(resource,PhysicalMemoryOwner::PrimaryModelWeights,weights);
            plan.add(resource,PhysicalMemoryOwner::ExecutionWorkspace,workspace);
            EXPECT_EQ(plan.build().fits(),shortfall==0);
        }
    }
}
