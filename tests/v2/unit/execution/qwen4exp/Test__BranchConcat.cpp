/**
 * @file Test__BranchConcat.cpp
 * @brief Device-free Dense0 branch layout, live extent and setup authority proofs.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpBranchConcatStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cstring>
using namespace llaminar2;
namespace {
/** @brief Actual CPU arena operands; no tensor row views or synthetic GPU aliases. */
struct Fixture {
    int m,h,count;
    FP32Tensor hidden,embedding,output;
    CPUDeviceContext context{DeviceId::cpu(),1};
    /** @brief Populate distinct special payloads so byte identity is observable. */
    Fixture(int rows,int width):m(rows),h(width),count(rows),hidden({rows,width*4}),embedding({rows,width}),output({rows,width*8}) {
        constexpr std::array<std::uint32_t,8>bits{0u,0x80000000u,1u,0x7fc00123u,0xffc00456u,0x3f800000u,0x7f800000u,0xff800000u};
        for(std::size_t i=0;i<hidden.size_bytes()/4;++i)hidden.mutable_data()[i]=std::bit_cast<float>(bits[i%8]);
        for(std::size_t i=0;i<embedding.size_bytes()/4;++i)embedding.mutable_data()[i]=std::bit_cast<float>(bits[(i+3)%8]);
        std::memset(output.raw_mutable_data(),0xcd,output.size_bytes());
    }
    /** @brief Seal exact token-row authority and canonical arena identities. */
    Qwen4ExpBranchConcatStage::Params params(DeviceRowRange rows) {
        return {.device_id=DeviceId::cpu(),.rows=rows,.hidden_dim=h,
            .hidden={hidden,BufferId::MTP_NORM_HIDDEN},.embedding={embedding,BufferId::MTP_NORM_EMBEDDING},.output={output,BufferId::MTP_CONCAT}};
    }
    /** @brief Check copied half-row payloads and untouched inactive capacity. */
    void verify(int live) {
        for(int row=0;row<live;++row)for(int branch=0;branch<4;++branch){
            const float *p=output.data()+(row*4+branch)*h*2;
            EXPECT_EQ(std::memcmp(p,embedding.data()+row*h,h*4),0);
            EXPECT_EQ(std::memcmp(p+h,hidden.data()+(row*4+branch)*h,h*4),0);
        }
        const auto *p=static_cast<const unsigned char*>(output.raw_data());
        for(std::size_t i=std::size_t(live)*h*32;i<output.size_bytes();++i)ASSERT_EQ(p[i],0xcd);
    }
};
}
/** @test Production shape sweep preserves NaN payloads, signed zero and branch order. */
TEST(BranchConcat, NativeCPUStageBitsAndSerialGroupedRows) {
    llaminar::v2::ThreadCountGuard threads(2);
    for(int m:{1,15,512,1024})for(int h:{1,7,128}){
        Fixture f(m,h);Qwen4ExpBranchConcatStage stage(f.params(DeviceRowRange::fullyActive(m)));
        ASSERT_TRUE(stage.execute(&f.context));f.verify(m);
        std::vector<float>serial(std::size_t(m)*h*8);
        for(int row=0;row<m;++row)qwen4exp::cpuBranchConcat({DeviceRowRange::fullyActive(1),h,f.hidden.data()+row*h*4,f.embedding.data()+row*h,serial.data()+row*h*8},nullptr);
        EXPECT_EQ(std::memcmp(serial.data(),f.output.raw_data(),f.output.size_bytes()),0);
    }
}
/** @test Large-to-small and empty replay leave dirty dead capacity unchanged. */
TEST(BranchConcat, CountedPartialEmptyAndCorruptCount) {
    llaminar::v2::ThreadCountGuard threads(1);Fixture f(15,13);
    Qwen4ExpBranchConcatStage stage(f.params(DeviceRowRange::deviceCounted(f.m,&f.count)));
    for(int n:{15,2,0,7}){f.count=n;std::memset(f.output.raw_mutable_data(),0xcd,f.output.size_bytes());ASSERT_TRUE(stage.execute(&f.context));f.verify(n);}
    for(int n:{-1,16}){f.count=n;std::memset(f.output.raw_mutable_data(),0xcd,f.output.size_bytes());EXPECT_THROW(stage.execute(&f.context),std::invalid_argument);f.verify(0);}
}
/** @test Slice counts retain original coordinates while payload pointers are exact local subviews. */
TEST(BranchConcat, OriginalCountSliceAndFullCaptureIdentity) {
    Fixture f(4,7);int count=6;
    auto rows=DeviceRowRange::deviceCounted(10,&count).slice(3,4);
    Qwen4ExpBranchConcatStage stage(f.params(rows));const auto signature=stage.graphCaptureVariantSignature();
    ASSERT_TRUE(stage.execute(&f.context));f.verify(3);count=0;EXPECT_EQ(signature,stage.graphCaptureVariantSignature());
    auto p=f.params(DeviceRowRange::deviceCounted(10,&count).slice(2,4));Qwen4ExpBranchConcatStage other(p);EXPECT_NE(signature,other.graphCaptureVariantSignature());
    int second=0;p=f.params(DeviceRowRange::deviceCounted(10,&second).slice(3,4));Qwen4ExpBranchConcatStage owner(p);EXPECT_NE(signature,owner.graphCaptureVariantSignature());
}
/** @test Invalid setup never reaches execution or creates a replacement count/operand. */
TEST(BranchConcat, RejectRepeatedWrongTypeCapacityAndAliasing) {
    Fixture f(4,7);auto p=f.params(DeviceRowRange::fullyActive(1).repeatEachRow(4));EXPECT_THROW(Qwen4ExpBranchConcatStage{p},std::invalid_argument);
    p=f.params(DeviceRowRange::fullyActive(4));p.output=p.hidden;EXPECT_THROW(Qwen4ExpBranchConcatStage{p},std::invalid_argument);
    p=f.params(DeviceRowRange::fullyActive(4));p.hidden_dim=8;EXPECT_THROW(Qwen4ExpBranchConcatStage{p},std::invalid_argument);
    p=f.params(DeviceRowRange::fullyActive(4));p.output.byteOffset=2;EXPECT_THROW(Qwen4ExpBranchConcatStage{p},std::invalid_argument);
    p=f.params(DeviceRowRange::fullyActive(4));p.hidden_dim=0;EXPECT_THROW(Qwen4ExpBranchConcatStage{p},std::invalid_argument);
    INT32Tensor wrong({4,56});p=f.params(DeviceRowRange::fullyActive(4));p.output={wrong,BufferId::MTP_CONCAT};EXPECT_THROW(Qwen4ExpBranchConcatStage{p},std::invalid_argument);
    EXPECT_THROW(qwen4exp::cpuBranchConcat({DeviceRowRange::fullyActive(1),7,f.hidden.data(),f.embedding.data(),f.output.mutable_data()},reinterpret_cast<void*>(1)),std::invalid_argument);
}
/** @test Arena byte offsets, output diagnostics and immutable policies remain explicit. */
TEST(BranchConcat, ByteSubviewAndBufferContract) {
    Fixture f(4,7);FP32Tensor backing({4,57});auto p=f.params(DeviceRowRange::fullyActive(4));p.output={backing,BufferId::MTP_CONCAT,4};
    Qwen4ExpBranchConcatStage stage(p);ASSERT_TRUE(stage.execute(&f.context));
    EXPECT_EQ(stage.bufferContract().inputs.size(),2);EXPECT_EQ(stage.bufferContract().outputs.size(),1);
    auto dump=stage.buildDumpInfoImpl();ASSERT_EQ(dump.outputs.size(),1);EXPECT_EQ(dump.outputs[0].tensor_byte_offset,4);
    EXPECT_EQ(stage.estimatedFlops(),0);EXPECT_EQ(stage.estimatedMemoryBytes(),std::size_t(4*7*4*13));
}

/** @test Cold allocation sees distinct output roots while explicit root aliases remain InOut. */
TEST(BranchConcat, ColdDistinctOutputRequiresOrdinaryPreparation) {
    Fixture f(3,7);Qwen4ExpBranchConcatStage stage(f.params(DeviceRowRange::fullyActive(f.m)));
    const auto contract=stage.bufferContract();ASSERT_EQ(contract.outputs.size(),1u);
    EXPECT_EQ(contract.outputs[0].id,BufferId::MTP_CONCAT);EXPECT_TRUE(contract.outputs[0].prepare_write_storage);
    ASSERT_EQ(contract.writesRequiringPrepare().size(),1u);
    FP32Tensor shared({3*7*13});auto p=f.params(DeviceRowRange::fullyActive(f.m));
    p.hidden={shared,BufferId::MTP_NORM_HIDDEN};p.embedding={shared,BufferId::MTP_NORM_EMBEDDING,3*7*4*4};
    p.output={shared,BufferId::MTP_NORM_HIDDEN,3*7*5*4};Qwen4ExpBranchConcatStage alias(p);
    const auto aliased=alias.bufferContract();EXPECT_TRUE(aliased.outputs.empty());ASSERT_EQ(aliased.inouts.size(),1u);
    EXPECT_FALSE(aliased.inouts[0].prepare_write_storage);EXPECT_TRUE(aliased.writesRequiringPrepare().empty());
}
