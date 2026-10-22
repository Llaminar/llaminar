/**
 * @file Test__CapturedComputeIsolation.cpp
 * @brief Fast device-free adversarial ownership/closure checks for isolated capture.
 *
 * Integer spans in this suite test pure metadata validation only. They never
 * impersonate native residency or PMA admission; actual tensor/device/prepared
 * engine checks belong to the concrete adapter's backend integration proof.
 */
#include "execution/diagnostics/PreparedGemmIsolationAdapter.h"
#include "execution/diagnostics/PreparedGDNIsolationAdapter.h"
#include "../../performance/kernels/recorded/RecordedGDNSourceGeometry.h"
#include "loaders/WeightSlicer.h"
#include "models/qwen35/Qwen35Schema.h"
#include <gtest/gtest.h>

namespace
{
using namespace llaminar2;
using namespace llaminar2::diagnostics;
/** @return One retained metadata owner, not an allocation or GPU capability. */
std::shared_ptr<const void> owner(){return std::make_shared<int>(1);}
/** @return Exact original/local metadata used only by pure validation. */
IsolationGemmShape shape(){return {1,256,640,64,64,1,4};}
/** @return Explicit fixture provenance distinct from qualified inference input bytes. */
IsolationInputProvenance provenance(){return {IsolationInputOrigin::SyntheticFixture,"fixture-observation","upstream_A",1,owner()};}
/** @return Actual shape of a simple producer contract: input A and output C. */
StageBufferContract stage(){return StageBufferContract::build().addInput(BufferId::NORMALIZED).addOutput(BufferId::GATE_PROJ);}
/** @return Disjoint exact spans; tests do not dereference these integer addresses. */
std::vector<IsolationBuffer> bindings(){return {{BufferId::NORMALIZED,0x10000,0x20000,2560,owner(),owner()},
    {BufferId::GATE_PROJ,0x30000,0,256,owner(),{}}};}
/** @brief Admit only pure metadata; native adapter independently derives real pointers and sizes. */
void admit(IsolationGemmShape axes,IsolationInputProvenance input,ComputeStageType kind,
    const StageBufferContract &contract,std::vector<IsolationBuffer> buffers)
{ CapturedComputeIsolationContract value(DeviceId::rocm(0),axes,std::move(input),kind,contract,std::move(buffers)); }
}

/** @brief Reads require retained immutable baselines while write-only output has no fabricated reset input. */
TEST(CapturedComputeIsolation, CompleteClosureAndFixedSourceTopology)
{
    CapturedComputeIsolationContract value(DeviceId::rocm(0),shape(),provenance(),ComputeStageType::GEMM,stage(),bindings());
    EXPECT_EQ(value.resets().size(),1u); EXPECT_EQ(value.bindings().size(),2u);
    EXPECT_EQ(value.shape().first_column,64); EXPECT_EQ(value.shape().local_columns,64);
    EXPECT_EQ(value.provenance().origin,IsolationInputOrigin::SyntheticFixture);
    for(auto bad:{IsolationGemmShape{1,256,640,0,64,1,4},{1,257,640,64,64,1,4},
                  {1,256,640,64,64,4,4},{0,256,640,64,64,1,4}})
        EXPECT_THROW(admit(bad,provenance(),ComputeStageType::GEMM,stage(),bindings()),std::invalid_argument);
    EXPECT_THROW((void)CapturedComputeIsolationContract(DeviceId::cpu(),shape(),provenance(),ComputeStageType::GEMM,stage(),bindings()),std::invalid_argument);
}
/** @brief Missing recorded input identities or ownership cannot become qualified inference evidence. */
TEST(CapturedComputeIsolation, RejectsMissingProvenanceAndResetOwnership)
{
    auto input=provenance();input.generation=0;
    EXPECT_THROW(admit(shape(),input,ComputeStageType::GEMM,stage(),bindings()),std::invalid_argument);
    input=provenance();input.owner.reset();
    EXPECT_THROW(admit(shape(),input,ComputeStageType::GEMM,stage(),bindings()),std::invalid_argument);
    input=provenance();input.observation.clear();
    EXPECT_THROW(admit(shape(),input,ComputeStageType::GEMM,stage(),bindings()),std::invalid_argument);
    for(int failure=0;failure<3;++failure)
    {
        auto buffers=bindings();
        if(failure==0)buffers[0].baseline=0;
        if(failure==1)buffers[0].baseline_owner.reset();
        if(failure==2)buffers[0].owner.reset();
        EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,stage(),buffers),std::invalid_argument);
    }
}
/** @brief Actual inout state must be reset; output-only storage cannot masquerade as fixed input state. */
TEST(CapturedComputeIsolation, MutableReadClosureRequiresReset)
{
    auto contract=stage();contract.addInOut(BufferId::FFN_OUTPUT);
    auto buffers=bindings();buffers.push_back({BufferId::FFN_OUTPUT,0x40000,0,256,owner(),{}});
    EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,contract,buffers),std::invalid_argument);
    buffers.back().baseline=0x50000;buffers.back().baseline_owner=owner();
    EXPECT_NO_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,contract,buffers));
    buffers[1].baseline=0x60000;buffers[1].baseline_owner=owner();
    EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,contract,buffers),std::invalid_argument);
}
/** @brief Missing/extra bindings and every partial overlap fail before enqueue or cache access. */
TEST(CapturedComputeIsolation, RejectsAliasesOverflowAndIncompleteClosure)
{
    auto buffers=bindings();buffers.pop_back();
    EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,stage(),buffers),std::invalid_argument);
    buffers=bindings();buffers[1].id=buffers[0].id;
    EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,stage(),buffers),std::invalid_argument);
    buffers=bindings();buffers[1].address=buffers[0].address+4;
    EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,stage(),buffers),std::invalid_argument);
    buffers=bindings();buffers[0].baseline=buffers[1].address+4;
    EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,stage(),buffers),std::invalid_argument);
    buffers=bindings();buffers[0].address=std::numeric_limits<std::uintptr_t>::max()-4;
    EXPECT_THROW(admit(shape(),provenance(),ComputeStageType::GEMM,stage(),buffers),std::invalid_argument);
}
/** @brief Protocol/service/control stages cannot use this plain compute adapter by changing a filter string. */
TEST(CapturedComputeIsolation, RejectsCommunicationServiceRoutingAndStatefulStages)
{
    for(auto kind:{ComputeStageType::ALLREDUCE,ComputeStageType::NATIVE_ALLGATHER,
        ComputeStageType::DEVICE_COUNTED_ALLGATHER,ComputeStageType::TP_KV_CACHE_STATE_ALLGATHER,
        ComputeStageType::LOCAL_PP_TRANSFER,ComputeStageType::MOE_ROUTER,ComputeStageType::MOE_PROJECTION_PHASE,
        ComputeStageType::QWEN4EXP_NGRAM_HASH,ComputeStageType::KV_CACHE_APPEND,ComputeStageType::ATTENTION})
        EXPECT_THROW(admit(shape(),provenance(),kind,stage(),bindings()),std::invalid_argument);
}
/** @brief Params supplement the stage contract: beta*C, bias, fusion, counts and MPI are never silently omitted. */
TEST(CapturedComputeIsolation, PlainGemmRejectsHiddenReadsAndCommunication)
{
    GEMMStage::Params params;
    EXPECT_TRUE(plainGemmIsolationFormSupported(params));
    params.beta=1; EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.beta=0;
    params.alpha=.5f;EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.alpha=1;
    params.needs_allreduce=true;EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.needs_allreduce=false;
    params.do_swiglu=true;EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.do_swiglu=false;
    params.bias_required=true;EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.bias_required=false;
    params.bias=reinterpret_cast<const float*>(0x10000);EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.bias=nullptr;
    params.mpi_ctx=reinterpret_cast<const IMPIContext*>(0x10000);EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.mpi_ctx=nullptr;
    params.gate_input=reinterpret_cast<const ITensor*>(0x10000);EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.gate_input=nullptr;
    params.output_range={0,32};EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.output_range={};
    params.request_rows=DeviceRequestRowRanges::fullyActive(1);EXPECT_FALSE(plainGemmIsolationFormSupported(params));params.request_rows.reset();
    params.verifier_row_range=DeviceRowRange::fullyActive(1);EXPECT_FALSE(plainGemmIsolationFormSupported(params));
}

/** @brief Serial source policy is immutable identity; physical width and arithmetic width remain distinct. */
TEST(CapturedComputeIsolation, OutputPartitionPolicyIsExplicitAndBounded)
{
    for(int width:{0,1,32,64})
    {
        CapturedComputeIsolationContract value(DeviceId::rocm(0),shape(),provenance(),ComputeStageType::GEMM,
            stage(),bindings(),IsolationOutputPartitionPolicy{.serial_columns=width});
        EXPECT_EQ(value.outputPartitionPolicy().serial_columns,width);
    }
    for(int width:{-1,65,256})
        EXPECT_THROW((void)CapturedComputeIsolationContract(DeviceId::rocm(0),shape(),provenance(),ComputeStageType::GEMM,
            stage(),bindings(),IsolationOutputPartitionPolicy{.serial_columns=width}),std::invalid_argument);
}

/** @brief Complete ordinary GDN supports no borrowed counts, corrected QKV or hidden verifier policy. */
TEST(CapturedComputeIsolation, GDNBundleFormRejectsUnimplementedRowArithmetic)
{
    GDNProjectionStage::Params p;p.m=1;p.k=2560;p.n_qkv=2560;p.n_z=1536;p.n_a=12;p.n_b=12;
    EXPECT_TRUE(staticGDNIsolationFormSupported(p));
    p.mpi_ctx=reinterpret_cast<const IMPIContext *>(0x10000);EXPECT_FALSE(staticGDNIsolationFormSupported(p));p.mpi_ctx=nullptr;
    p.qkv_operand_precision=QuantizedActivationOperandPrecision::TwoTermQ8;
    EXPECT_FALSE(staticGDNIsolationFormSupported(p));p.qkv_operand_precision=QuantizedActivationOperandPrecision::SingleQ8;
    p.qkv_row_range=DeviceRowRange::fullyActive(1);EXPECT_FALSE(staticGDNIsolationFormSupported(p));p.qkv_row_range.reset();
    p.verifier_row_range=DeviceRowRange::fullyActive(1);EXPECT_FALSE(staticGDNIsolationFormSupported(p));p.verifier_row_range.reset();
    p.force_decode_equivalent_verifier_prefill=true;EXPECT_FALSE(staticGDNIsolationFormSupported(p));p.force_decode_equivalent_verifier_prefill=false;
    p.n_b=0;EXPECT_FALSE(staticGDNIsolationFormSupported(p));p.n_b=12;
    p.k=2559;EXPECT_FALSE(staticGDNIsolationFormSupported(p));
}
/** @brief A complete bundle's only activation read and all four independent writes must be owned. */
TEST(CapturedComputeIsolation, GDNBundleClosureRejectsMissingSiblingAndResetAliases)
{
    auto contract=StageBufferContract::build().addInput(BufferId::NORMALIZED)
        .addOutput(BufferId::GDN_QKV).addOutput(BufferId::GDN_Z).addOutput(BufferId::GDN_ALPHA).addOutput(BufferId::GDN_BETA);
    std::vector<IsolationBuffer> spans={{BufferId::NORMALIZED,0x10000,0x20000,10240,owner(),owner()},
        {BufferId::GDN_QKV,0x30000,0,10240,owner(),{}},{BufferId::GDN_Z,0x40000,0,6144,owner(),{}},
        {BufferId::GDN_ALPHA,0x50000,0,48,owner(),{}},{BufferId::GDN_BETA,0x60000,0,48,owner(),{}}};
    EXPECT_NO_THROW(validateIsolationBufferClosure(contract,spans));
    auto bad=spans;bad.pop_back();EXPECT_THROW(validateIsolationBufferClosure(contract,bad),std::invalid_argument);
    bad=spans;bad[4].address=bad[3].address;EXPECT_THROW(validateIsolationBufferClosure(contract,bad),std::invalid_argument);
    bad=spans;bad[4].address=bad[0].baseline;EXPECT_THROW(validateIsolationBufferClosure(contract,bad),std::invalid_argument);
    bad=spans;bad[0].baseline_owner.reset();EXPECT_THROW(validateIsolationBufferClosure(contract,bad),std::invalid_argument);
}

/** @brief TP1 must retain actual ordinary source axes so the production slicer recognizes asymmetric GDN QKV. */
TEST(CapturedComputeIsolation, GDNBundleOriginalDimensionsRetainTP1PackedSource)
{
    using namespace llaminar2::test::recorded_gdn;
    const Qwen4ExpSparseAttentionGeometry attention{.query_heads=24,.kv_heads=2,.head_dimension=256};
    const Qwen4ExpDeltaNetGeometry gdn{.key_heads=16,.value_heads=48,.head_dimension=128,.value_width=6144,.convolution_kernel=4};
    const auto dimensions=originalSourceDimensions(attention,gdn);
    EXPECT_EQ(dimensions.n_heads,24);EXPECT_EQ(dimensions.n_kv_heads,2);EXPECT_EQ(dimensions.head_dim,256);
    for(int degree:{1,2,4})
    {
        const auto assignments=originalSourceAssignments(dimensions,degree);
        auto tp=std::make_shared<TensorParallelConfig>(assignments);
        WeightSlicer slicer(dimensions,Qwen35SchemaFactory{}.getWeightShardingConfig(),tp);
        for(int rank=0;rank<degree;++rank)
        {
            const auto slice=slicer.computeFusedQKVSliceForAssignment("blk.0.attn_qkv.weight",10240,assignments[rank]);
            ASSERT_TRUE(slice.has_value());EXPECT_TRUE(slice->modulo_linked_gdn);
            EXPECT_EQ(slice->q_total,2048u);EXPECT_EQ(slice->k_total,2048u);EXPECT_EQ(slice->v_total,6144u);
            const auto expected=GDNHeadAssignment::forEqualRank(16,48,rank,degree);
            const auto key=expected.keyElementSpan(128);EXPECT_EQ(slice->q.start,std::size_t(key.start));EXPECT_EQ(slice->q.count,std::size_t(key.count));
            const auto values=expected.valueElementSpans(128);ASSERT_EQ(slice->v.size(),values.size());
            for(std::size_t i=0;i<values.size();++i){EXPECT_EQ(slice->v[i].start,std::size_t(values[i].start));EXPECT_EQ(slice->v[i].count,std::size_t(values[i].count));}
        }
    }
    auto incomplete=attention;incomplete.head_dimension=0;
    EXPECT_THROW(originalSourceDimensions(incomplete,gdn),std::invalid_argument);
    incomplete=attention;incomplete.query_heads=0;EXPECT_THROW(originalSourceDimensions(incomplete,gdn),std::invalid_argument);
    auto missing=dimensions;missing.n_heads=0;EXPECT_THROW(originalSourceAssignments(missing,1),std::invalid_argument);
}

/** @brief Immutable pool descriptors use implicit capacity, unlike reusable slots. */
TEST(CapturedComputeIsolation,GDNBundleQ8AllocationRetainsImmutablePoolContract)
{
    DeviceNativeVNNIMatrixDesc native{};
    native.payload=reinterpret_cast<const std::uint8_t *>(0x10000);
    native.scales=reinterpret_cast<const void *>(0x20000);
    native.n=640;native.k=640;native.blocks_per_row=20;
    native.codebook_id=19;native.source_codebook_id=19;native.source_identity_present=1;
    EXPECT_NO_THROW(validateGDNIsolationQ8Allocation(native));
    native.allocation_payload_bytes_per_block=32;
    EXPECT_NO_THROW(validateGDNIsolationQ8Allocation(native));
    for(const auto capacity:{16,64})
    {
        auto bad=native;bad.allocation_payload_bytes_per_block=capacity;
        EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
    }
    for(const bool explicit_capacity:{false,true})
    {
        auto original=native;original.allocation_payload_bytes_per_block=explicit_capacity?32:0;
        auto bad=original;bad.allocation_has_mins=1;
        EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
        bad=original;bad.allocation_has_emins=1;
        EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
        bad=original;bad.mins=reinterpret_cast<const void *>(0x30000);
        EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
        bad=original;bad.emins=reinterpret_cast<const void *>(0x40000);
        EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
    }
    auto bad=native;bad.blocks_per_row=19;
    EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
    bad=native;bad.source_identity_present=0;
    EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
    bad=native;bad.source_is_superblock=1;
    EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
    bad=native;bad.codebook_id=23;
    EXPECT_THROW(validateGDNIsolationQ8Allocation(bad),std::invalid_argument);
}
