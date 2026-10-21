/**
 * @file Test__Qwen4ExpHCUpWorkspace.cpp
 * @brief Device-free native HC scratch factories, semantic selection and name reuse.
 */
#include "models/qwen4exp/Qwen4ExpHCUpWorkspace.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
namespace {
using Precision=QuantizedActivationOperandPrecision;
/** @return Exact role-specific named contribution used by production admission. */
WorkspaceRequirements contribution(DeviceId device,int rows,std::string_view name="blk.0.hc_attn_up.weight",
    std::string_view format="Q8_0",Precision policy=Precision::TwoTermQ8,
    WeightRole role=WeightRole::HyperConnectionProjection,bool qfn=true)
{
    return Qwen4ExpHCUpWorkspace::requirements(device,policy,qfn,role,name,format,rows,10240,320);
}
/** @return Capacity of one required native name; absence is a test failure. */
std::size_t bytes(const WorkspaceRequirements &r,const std::string &name)
{
    for(const auto &b:r.buffers)if(b.name==name)return b.size_bytes;
    ADD_FAILURE()<<"Missing workspace name "<<name;return 0;
}
}
/** @test The small and large row buckets own exactly two native CPU operands. */
TEST(Qwen4ExpHCUpWorkspace, CPUNamedBanksCoverEveryAdmittedRow)
{
    for(int m:{1,2,15,16,127,512,1024,2048,8192}){
        const auto r=contribution(DeviceId::cpu(),m);
        ASSERT_EQ(r.buffers.size(),2u);
        EXPECT_EQ(bytes(r,kCPUProjectionQ8),std::size_t(m)*10*sizeof(Q8_1Block));
        EXPECT_EQ(bytes(r,kCPUProjectionResidualQ8),std::size_t(m)*10*sizeof(Q8_1Block));
        for(const auto &b:r.buffers)EXPECT_EQ(b.alignment,64u);
    }
}
/** @test CUDA and HIP use identical named capacities despite distinct primary arithmetic. */
TEST(Qwen4ExpHCUpWorkspace, GPUContractsAreBackendSymmetricAndRowScaled)
{
    for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})for(int m:{1,15,512,1024,2048,8192}){
        const auto r=contribution(device,m);ASSERT_EQ(r.buffers.size(),4u);
        EXPECT_EQ(bytes(r,kGPUProjectionPrimaryQ8),std::size_t(m)*320);
        EXPECT_EQ(bytes(r,kGPUProjectionResidualQ8),std::size_t(m)*320);
        EXPECT_EQ(bytes(r,kGPUProjectionPrimaryQ8Scales),std::size_t(m)*40);
        EXPECT_EQ(bytes(r,kGPUProjectionResidualQ8Scales),std::size_t(m)*40);
        for(const auto &b:r.buffers)EXPECT_EQ(b.alignment,256u);
    }
}
/** @test Remote CUDA admission adds only the new names; runtime merging remains idempotent. */
TEST(Qwen4ExpHCUpWorkspace, MetadataOnlyResidualBanksMatchCompleteFactory)
{
    for(int m:{1,15,512,1024,2048,8192}){
        const auto additional=gpuQ8ResidualCorrectionAdditionalWorkspaceRequirements(m,10240,320);
        const auto full=gpuQ8ResidualCorrectionWorkspaceRequirements(m,10240,320);
        ASSERT_EQ(additional.buffers.size(),2u);
        for(const auto &b:additional.buffers){
            EXPECT_TRUE(b.name==kGPUProjectionResidualQ8||b.name==kGPUProjectionResidualQ8Scales);
            EXPECT_EQ(b.size_bytes,bytes(full,b.name));
            EXPECT_EQ(b.alignment,256u);
        }
        auto runtime=full;runtime.merge(additional);
        EXPECT_EQ(runtime.total_bytes_with_alignment(),full.total_bytes_with_alignment());
        EXPECT_EQ(runtime.buffers.size(),full.buffers.size());
    }
}
/** @test Main terminal and predictor heads are included, with unrelated roles unchanged. */
TEST(Qwen4ExpHCUpWorkspace, SemanticConsumersIncludeBothTerminalHeads)
{
    for(auto name:{"blk.0.hc_attn_up.weight","blk.47.hc_ffn_up.weight","output_hc_up.weight",
                   "blk.48.hc_attn_up.weight","blk.48.nextn.hc_head_up.weight"})
        EXPECT_FALSE(contribution(DeviceId::cpu(),15,name).buffers.empty())<<name;
    for(auto name:{"blk.0.hc_attn_down.weight","blk.0.hc_attn_inject.weight","blk.0.hc_ffn_inject.weight",
                   "blk.48.nextn.eh_proj.weight","blk.0.ffn_up.weight","output.weight"})
        EXPECT_TRUE(contribution(DeviceId::cpu(),15,name).buffers.empty())<<name;
    EXPECT_TRUE(contribution(DeviceId::cpu(),15,"blk.0.hc_attn_up.weight","Q8_0",
        Precision::TwoTermQ8,WeightRole::Other).buffers.empty());
}
/** @test Explicit candidate selection cannot silently change native source precision. */
TEST(Qwen4ExpHCUpWorkspace, RejectsForeignSourceGeometryDeviceAndPolicy)
{
    for(auto format:{"F32","F16","BF16","Q6_K","Q8_1","IQ4_NL"})
        EXPECT_THROW((void)contribution(DeviceId::cpu(),15,"blk.0.hc_attn_up.weight",format),std::invalid_argument)<<format;
    EXPECT_THROW((void)contribution(DeviceId::invalid(),15),std::invalid_argument);
    EXPECT_THROW((void)contribution(DeviceId::cpu(),0),std::invalid_argument);
    EXPECT_THROW((void)contribution(DeviceId::cpu(),15,"blk.0.hc_attn_up.weight","Q8_0",static_cast<Precision>(9)),std::invalid_argument);
    EXPECT_THROW((void)cpuQ8ResidualCorrectionWorkspaceRequirements(1,33,319),std::invalid_argument);
    EXPECT_THROW((void)gpuQ8ResidualCorrectionWorkspaceRequirements(1,33,319),std::invalid_argument);
    // Complete INT32 dimensions still fit the64-bit metadata extent. The
    // physical capacity authority, rather than this geometry helper, admits bytes.
    EXPECT_NO_THROW(validateQ8ResidualCorrectionGeometry(std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),std::numeric_limits<int>::max()-31));
}
/** @test Ordinary arithmetic and other model families add no correction bytes. */
TEST(Qwen4ExpHCUpWorkspace, SingleQ8AndOrdinaryModelsKeepExistingRequirements)
{
    for(auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)}){
        EXPECT_TRUE(contribution(device,1024,"blk.0.hc_attn_up.weight","Q8_0",Precision::SingleQ8).buffers.empty());
        EXPECT_TRUE(contribution(device,1024,"blk.0.hc_attn_up.weight","Q8_0",Precision::TwoTermQ8,
            WeightRole::HyperConnectionProjection,false).buffers.empty());
    }
    EXPECT_EQ(WorkspaceMemoryGeometry{}.qfn_hc_up_activation,Precision::SingleQ8);
}
/** @test Serial layers and main/predictor graphs share stable names at the largest capacity. */
TEST(Qwen4ExpHCUpWorkspace, LayerCountDoesNotMultiplyPhysicalScratch)
{
    for(auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)}){
        auto r=contribution(device,512);const auto expected=r.total_bytes_with_alignment();
        for(int layer=0;layer<49;++layer){
            r.merge(contribution(device,15,"blk."+std::to_string(layer)+".hc_attn_up.weight"));
            r.merge(contribution(device,1,"blk."+std::to_string(layer)+".hc_ffn_up.weight"));
        }
        r.merge(contribution(device,1,"output_hc_up.weight"));
        r.merge(contribution(device,15,"blk.48.nextn.hc_head_up.weight"));
        EXPECT_EQ(r.total_bytes_with_alignment(),expected);
        r.merge(contribution(device,1024));
        EXPECT_EQ(r.total_bytes_with_alignment(),contribution(device,1024).total_bytes_with_alignment());
    }
}
