/**
 * @file Test__CapturedComputeIsolation.cpp
 * @brief Fast device-free adversarial ownership/closure checks for isolated capture.
 *
 * Integer spans in this suite test pure metadata validation only. They never
 * impersonate native residency or PMA admission; actual tensor/device/prepared
 * engine checks belong to the concrete adapter's backend integration proof.
 */
#include "execution/diagnostics/PreparedGemmIsolationAdapter.h"
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
