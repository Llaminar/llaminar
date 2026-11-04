/**
 * @file Test__CPUWeightPreparationMemory.cpp
 * @brief Device-free totality and checked geometry for CPU preparation BOM terms.
 *
 * Actual all-format prepared allocation equality belongs to the CPU planning
 * integration group. These tests exercise admission inputs without preparing
 * weights, invoking kernels, or acquiring devices.
 */
#include "kernels/cpu/gemm/CPUWeightPreparationMemory.h"
#include "tensors/NativeVnniFormatInfo.h"
#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "tensors/AlignedVector.h"
#include <gtest/gtest.h>
#include <limits>
#include <atomic>

using namespace llaminar2;

TEST(CPUWeightPreparationMemory, FloatingNativeBytesNeedNoPackingShadow)
{
    for (const auto format : {"F16", "BF16", "F32"})
    {
        const auto demand = CPUWeightPreparationMemory::sourceNative(format, 65, 259);
        const size_t payload=65u*259u*(std::string_view(format)=="F32"?4:2);
        EXPECT_EQ(demand.persistent_payload_bytes,payload);
        EXPECT_EQ(demand.persistent_bytes,AlignedVector<uint8_t>::requiredAllocationBytes(payload));
        EXPECT_EQ(demand.temporary_bytes, 0u);
    }
}

TEST(CPUWeightPreparationMemory, EveryNativeCodebookHasPaddedAllocationGeometry)
{
    for (const auto &format : native_vnni_formats::kAllSourceFormats)
    {
        SCOPED_TRACE(format.quant_type);
        const auto full = CPUWeightPreparationMemory::sourceNative(format.quant_type, 64, 256);
        const auto tail = CPUWeightPreparationMemory::sourceNative(format.quant_type, 65, 256);
        EXPECT_GT(full.persistent_bytes, 0u);
        EXPECT_EQ(tail.persistent_payload_bytes,2*full.persistent_payload_bytes);
        EXPECT_EQ(full.persistent_bytes,AlignedVector<uint8_t>::requiredAllocationBytes(full.persistent_payload_bytes));
        EXPECT_EQ(tail.persistent_bytes,AlignedVector<uint8_t>::requiredAllocationBytes(tail.persistent_payload_bytes));
        EXPECT_EQ(tail.temporary_bytes, full.temporary_bytes);
    }
}

TEST(CPUWeightPreparationMemory, NibbleSetupPayloadsRetireBeforePublication)
{
    const auto q4 = CPUWeightPreparationMemory::sourceNative("Q4_0", 64, 32);
    EXPECT_EQ(q4.persistent_bytes,1280u);
    EXPECT_EQ(q4.persistent_payload_bytes,1280u);
    EXPECT_EQ(q4.temporary_bytes,64u*cpu::native_vnni::NativePackGroupScratch::checked(*native_vnni_formats::forQuantType("Q4_0")).bytes);
    const auto q6 = CPUWeightPreparationMemory::sourceNative("Q6_K", 64, 256);
    EXPECT_EQ(q6.persistent_payload_bytes,8u*1792u);
    EXPECT_EQ(q6.persistent_bytes,AlignedVector<uint8_t>::requiredAllocationBytes(8u*1792u));
    EXPECT_EQ(q6.temporary_bytes,64u*cpu::native_vnni::NativePackGroupScratch::checked(*native_vnni_formats::forQuantType("Q6_K")).bytes);
    EXPECT_GT(CPUWeightPreparationMemory::sourceNative("IQ2_S", 64, 256).temporary_bytes,0u);
}

TEST(CPUWeightPreparationMemory, RejectsInvalidSourceBeforeAllocation)
{
    for (const auto &[n, k] : {std::pair<size_t, size_t>{0, 256}, {256, 0},
            {std::numeric_limits<size_t>::max(), 256}, {256, std::numeric_limits<size_t>::max()}})
        EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("F32", n, k), std::invalid_argument);
    EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("unknown", 256, 256), std::invalid_argument);
    EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("Q4_0", 256, 31), std::invalid_argument);
    EXPECT_THROW(CPUWeightPreparationMemory::sourceNative("Q6_K", 256, 32), std::invalid_argument);
}

namespace {
/** @brief Observe original staging ownership at the actual source codec boundary. */
class StagingObservedQ4 final : public Q4_0Tensor {
public:
    /** @brief Retain real native source bytes; the probe owns no alternate pack implementation. */
    StagingObservedQ4(size_t rows,size_t columns,std::vector<uint8_t> bytes,
        std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId allocator)
        : Q4_0Tensor({rows,columns},bytes),memory_(std::move(memory)),allocator_(allocator) {}
    /** @brief Sample the original PMA exactly once while the production append reads source blocks. */
    void packVnniBlock(const VnniPackContext& context,int source_n,int destination_n,int block) const override {
        if(!sampled_.test_and_set()) observed_bytes.store(memory_->claimedBytes(allocator_,
            PhysicalMemoryOwner::WeightLoadStaging,PhysicalMemoryMaterializationKind::NewAllocation));
        Q4_0Tensor::packVnniBlock(context,source_n,destination_n,block);
    }
    mutable std::atomic<size_t> observed_bytes{0};
private:
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    DeviceId allocator_;
    mutable std::atomic_flag sampled_=ATOMIC_FLAG_INIT;
};
}

/** @test Immediate packing uses bounded admitted append workspace and releases it before publication. */
TEST(CPUWeightPreparationMemory, ImmediateNativePackingChargesBoundedOriginalScratch)
{
    struct BudgetRestore {
        int previous=mutableDebugEnv().rocm.repack_budget_mb;
        ~BudgetRestore(){mutableDebugEnv().rocm.repack_budget_mb=previous;}
    } restore;
    mutableDebugEnv().rocm.repack_budget_mb=1;
    constexpr size_t n=65,k=16384,source_rows=80;
    const DeviceId allocator{DeviceType::CPU,1};
    const auto bytes=CPUWeightPreparationMemory::sourceNative("Q4_0",n,k);
    ASSERT_LE(bytes.temporary_bytes,1024u*1024u);
    ASSERT_EQ(bytes.temporary_bytes,CPUWeightPreparationMemory::sourceNative("Q4_0",4096,k).temporary_bytes);
    PhysicalMemoryPlanBuilder budget;
    const auto total=bytes.persistent_bytes+bytes.temporary_bytes;
    const PhysicalMemoryResource resource{0,allocator,total,total};
    budget.add(resource,PhysicalMemoryOwner::PrimaryModelWeights,bytes.persistent_bytes);
    budget.add(resource,PhysicalMemoryOwner::WeightLoadStaging,bytes.temporary_bytes);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),0);
    std::vector<uint8_t> raw(source_rows*(k/32)*sizeof(Q4_0Block),0x31);
    for(size_t offset=0;offset<raw.size();offset+=sizeof(Q4_0Block)) {
        const uint16_t scale=0x3c00;std::memcpy(raw.data()+offset,&scale,sizeof(scale));
    }
    StagingObservedQ4 source(source_rows,k,std::move(raw),memory,allocator);
    cpu::native_vnni::CPUNativeVNNIPackedWeights packed;
    ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(&source,packed,7,72,nullptr,
        CPUWeightStoragePlacement::accounted(memory,allocator,PhysicalMemoryOwner::PrimaryModelWeights,
            CPUWeightStoragePlacement::local())));
    EXPECT_EQ(source.observed_bytes.load(),bytes.temporary_bytes);
    EXPECT_EQ(memory->claimedBytes(allocator,PhysicalMemoryOwner::WeightLoadStaging,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(memory->claimedBytes(allocator,PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),bytes.persistent_bytes);
    cpu::native_vnni::CPUNativeVNNIPackedWeights oracle;
    ASSERT_TRUE(cpu::native_vnni::packWeightsCPUNativeVNNI(&source,oracle,7,72));
    ASSERT_EQ(packed.native_interleaved.size(),oracle.native_interleaved.size());
    EXPECT_EQ(std::memcmp(packed.native_interleaved.data(),oracle.native_interleaved.data(),
        packed.native_interleaved.size()),0);
}
