/**
 * @file Test__ProjectionRowWorkBinding.cpp
 * @brief Device-free cold ownership, publication and reset regressions for query row work.
 *
 * Small INT32 tensor fixtures model original publications. Workspace storage
 * uses the canonical PMA-backed CPU manager, with no model, GPU, native kernel
 * or profiler. The CUDA/ROCm-shaped missing-stream cases stop before allocation
 * or backend lookup. These tests prove ownership independently of GPU execution.
 */
#include "execution/compute_stages/ProjectionRowWorkBinding.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/BackendManager.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
#include "kernels/qwen4exp/QwenSparseTransactionPrior.h"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>

namespace llaminar2::test {
namespace {
/** @brief One genuine small CPU allocation with controllable authority and replacement identity. */
class OriginalPublication final : public INT32Tensor {
public:
    /** @brief Allocate ordinary tensor metadata and two stable test-owned source banks. */
    OriginalPublication():INT32Tensor({6}){}
    /** @brief Model publication authority without invoking an accelerator or copying live values. */
    TensorCoherenceState coherenceState() const override {return authority;}
    /** @brief Select a new allocation-shaped address while retaining the tensor identity and size. */
    void replaceStorage(){selected=1-selected;}
    /** @return CPU-owned original words for the active test publication. */
    std::array<std::int32_t,6> &words(){return banks[selected];}
    TensorCoherenceState authority=TensorCoherenceState::HOST_ONLY;
protected:
    /** @return Exact active original bank; cold preparation may borrow its address only. */
    void *raw_host_data_ptr() override {return banks[selected].data();}
    /** @return Exact const original bank, with no authority repair or live count copy. */
    const void *raw_host_data_ptr() const override {return banks[selected].data();}
private:
    std::array<std::array<std::int32_t,6>,2> banks{{{2048,6,0,0,0,0},{4096,2,0,0,0,0}}};
    int selected=0;
};
/** @brief Test-owned workspace using the same exact allocation authority as production. */
class OriginalWorkspace final {
public:
    /** @brief Admit a bounded CPU workspace and allocate only its declared names. */
    explicit OriginalWorkspace(const WorkspaceRequirements &requirements) {
        if(!hasCPUBackend())initCPUBackend(-1);
        constexpr std::size_t budget=16384;
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank=0,.device=DeviceId::cpu(),.total_bytes=budget,.admission_available_bytes=budget},
            PhysicalMemoryOwner::ExecutionWorkspace,budget);
        memory=std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
        manager=std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(),budget,memory);
        if(!manager->allocate(requirements))throw std::runtime_error("Original publication fixture workspace allocation failed");
    }
    /** @return Current PMA-owned physical bytes, including any surviving retained publication. */
    std::size_t claimed() const {
        return memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,
            PhysicalMemoryMaterializationKind::NewAllocation);
    }
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    std::unique_ptr<DeviceWorkspaceManager> manager;
};
/** @return Canonical single-request declaration over the original first two scalar words. */
ProjectionRowWorkDeclaration singleDeclaration(const TensorBase &source) {
    return {8,8192,2051,{{&source,BufferId::QFN_QSA_REPLAY,0,sizeof(std::int32_t)}}};
}
/** @return Arbitrary independent owners, forcing the immutable address-table layout. */
ProjectionRowWorkDeclaration separateDeclaration(const TensorBase &first,const TensorBase &second) {
    return {8,8192,2051,{{&first,BufferId::QFN_QSA_REPLAY,0,sizeof(std::int32_t)},
        {&second,BufferId::QFN_QSA_TRANSACTION_PRIOR,0,sizeof(std::int32_t)}}};
}
} // namespace

TEST(ProjectionRowWorkBinding, PackedOriginalAndRetainedHeadersNeedNoTableBOM) {
    for(const int stride:{int(sizeof(qwen4exp::QwenSparseReplay)/sizeof(std::int32_t)),
                         int(sizeof(qwen4exp::QwenSparseTransactionPrior)/sizeof(std::int32_t))}) {
        INT32Tensor source({std::size_t(3*stride)});
        auto *words=static_cast<std::int32_t *>(source.raw_mutable_data());
        ProjectionRowWorkDeclaration declaration{8,8192,2051,{}};
        for(int request=0;request<3;++request) {
            words[request*stride]=request==0?512:2048;words[request*stride+1]=request==0?0:6;
            declaration.sources.push_back({&source,BufferId::QFN_QSA_REPLAY,
                std::size_t(request*stride)*4,std::size_t(request*stride+1)*4});
        }
        ProjectionRowWorkBinding binding(declaration,DeviceId::cpu(),24);
        WorkspaceRequirements requirements;binding.appendWorkspaceRequirements(requirements);
        EXPECT_TRUE(requirements.buffers.empty());EXPECT_FALSE(binding.prepared());
        binding.prepare(nullptr,nullptr);binding.requirePublishedInputs();
        EXPECT_EQ(binding.table(),nullptr);EXPECT_EQ(binding.work().requests(),3);
        EXPECT_EQ(binding.work().resolveHost(0).size(),0);EXPECT_EQ(binding.work().resolveHost(1).first,11);
        const auto identity=binding.work().captureIdentity();
        words[stride]=4096;words[stride+1]=1;
        binding.prepare(nullptr,nullptr);
        EXPECT_EQ(binding.work().captureIdentity(),identity);EXPECT_EQ(binding.work().resolveHost(1).first,8);
    }
}
TEST(ProjectionRowWorkBinding, ArbitraryOwnersDeclareExactExclusiveTableAndReusePublication) {
    OriginalPublication first,second;second.words()={4096,2,0,0,0,0};
    const auto declaration=separateDeclaration(first,second);
    ProjectionRowWorkBinding binding(declaration,DeviceId::cpu(),16);
    WorkspaceRequirements requirements;binding.appendWorkspaceRequirements(requirements);
    ASSERT_EQ(requirements.buffers.size(),1u);const auto &spec=requirements.buffers.front();
    EXPECT_EQ(spec.size_bytes,2*sizeof(ProjectionRequestPublication));EXPECT_TRUE(spec.required);
    EXPECT_EQ(spec.content_lifetime,WorkspaceContentLifetime::SerialGraphFamily);
    EXPECT_EQ(spec.regime,WorkspaceExecutionRegime::CompactDecodeOnly);
    OriginalWorkspace workspace(requirements);binding.prepare(workspace.manager.get(),nullptr);
    const auto table=binding.table();ASSERT_NE(table,nullptr);
    EXPECT_EQ(table->entries().size(),2u);EXPECT_EQ(table->storage().sizeBytes(),spec.size_bytes);
    EXPECT_EQ(table->data(),workspace.manager->getBuffer(spec.name));
    const auto charge=workspace.claimed();const auto identity=binding.work().captureIdentity();
    const std::array<ProjectionRequestPublication,2> bytesBefore{{table->entries()[0],table->entries()[1]}};
    first.words()[0]=2050;first.words()[1]=2;second.words()[1]=0;
    binding.prepare(workspace.manager.get(),nullptr);binding.requirePublishedInputs();
    EXPECT_EQ(binding.table(),table);EXPECT_EQ(workspace.claimed(),charge);
    EXPECT_EQ(binding.work().captureIdentity(),identity);
    EXPECT_EQ(binding.work().observeHost(0).span.first,1);EXPECT_EQ(binding.work().observeHost(1).span.size(),0);
    EXPECT_EQ(std::memcmp(table->data(),bytesBefore.data(),spec.size_bytes),0);
    ProjectionRowWorkBinding sibling(declaration,DeviceId::cpu(),16);
    sibling.prepare(workspace.manager.get(),nullptr);
    EXPECT_EQ(sibling.table(),table);EXPECT_EQ(workspace.claimed(),charge);
}
TEST(ProjectionRowWorkBinding, ColdPreparationBindsAddressesAndPublicationGateRequiresCPUAuthority) {
    OriginalPublication source;source.authority=TensorCoherenceState::INVALID;
    ProjectionRowWorkBinding binding(singleDeclaration(source),DeviceId::cpu(),8);
    EXPECT_THROW(binding.requirePublishedInputs(),std::logic_error);
    EXPECT_NO_THROW(binding.prepare(nullptr,nullptr));
    EXPECT_THROW(binding.requirePublishedInputs(),std::logic_error);
    source.authority=TensorCoherenceState::DEVICE_AUTHORITATIVE;
    EXPECT_THROW(binding.requirePublishedInputs(),std::logic_error);
    source.words()[0]=2048;source.words()[1]=6;source.authority=TensorCoherenceState::HOST_AUTHORITATIVE;
    EXPECT_NO_THROW(binding.requirePublishedInputs());EXPECT_EQ(binding.work().resolveHost().first,3);
}
TEST(ProjectionRowWorkBinding, SourceReplacementAndWorkspaceReplacementFailWithoutRepublishing) {
    OriginalPublication first,second;
    ProjectionRowWorkBinding binding(separateDeclaration(first,second),DeviceId::cpu(),16);
    WorkspaceRequirements requirements;binding.appendWorkspaceRequirements(requirements);
    OriginalWorkspace workspace(requirements);binding.prepare(workspace.manager.get(),nullptr);
    const auto table=binding.table();const auto claimed=workspace.claimed();
    first.replaceStorage();
    EXPECT_THROW(binding.requirePrepared(workspace.manager.get(),nullptr),std::logic_error);
    EXPECT_THROW(binding.prepare(workspace.manager.get(),nullptr),std::logic_error);
    EXPECT_EQ(binding.table(),table);EXPECT_EQ(workspace.claimed(),claimed);
    first.replaceStorage();
    EXPECT_NO_THROW(binding.requirePrepared(workspace.manager.get(),nullptr));
    auto larger=requirements;larger.buffers.front().size_bytes*=2;
    ASSERT_TRUE(workspace.manager->extend(larger));
    EXPECT_NE(workspace.manager->getBuffer(requirements.buffers.front().name),table->data());
    EXPECT_THROW(binding.requirePrepared(workspace.manager.get(),nullptr),std::logic_error);
    EXPECT_THROW(binding.prepare(workspace.manager.get(),nullptr),std::logic_error);
    EXPECT_EQ(binding.table(),table);
}
TEST(ProjectionRowWorkBinding, RetiredWorkspaceKeepsOneOriginalClaimUntilFinalCapturedLeaseRetires) {
    OriginalPublication first,second;
    auto binding=std::make_unique<ProjectionRowWorkBinding>(separateDeclaration(first,second),DeviceId::cpu(),16);
    WorkspaceRequirements requirements;binding->appendWorkspaceRequirements(requirements);
    OriginalWorkspace workspace(requirements);binding->prepare(workspace.manager.get(),nullptr);
    auto table=binding->table();const auto claimed=workspace.claimed();ASSERT_GT(claimed,0u);
    workspace.manager->release();
    EXPECT_EQ(workspace.claimed(),claimed);EXPECT_FALSE(workspace.manager->isAllocated());
    EXPECT_THROW(binding->requirePrepared(workspace.manager.get(),nullptr),std::logic_error);
    binding.reset();EXPECT_EQ(workspace.claimed(),claimed);
    table.reset();EXPECT_EQ(workspace.claimed(),0u);
}
TEST(ProjectionRowWorkBinding, MalformedColdDeclarationsFailBeforeAnyPublication) {
    OriginalPublication first,second;
    const auto declaration=separateDeclaration(first,second);
    for(int mutation=0;mutation<6;++mutation) {
        auto bad=declaration;
        if(mutation==0)bad.sources[0].buffer=BufferId::_COUNT;
        if(mutation==1)bad.sources[0].frontierByteOffset=1;
        if(mutation==2)bad.sources[0].liveByteOffset=bad.sources[0].frontierByteOffset;
        if(mutation==3)bad.sources[0].liveByteOffset=first.size_bytes();
        if(mutation==4)bad.sources[1].tensor=nullptr;
        if(mutation==5)bad.rowsPerRequest=9;
        EXPECT_THROW((void)ProjectionRowWorkBinding(bad,DeviceId::cpu(),16),std::invalid_argument);
    }
    int count=16;
    EXPECT_THROW((void)ProjectionRowWorkBinding(declaration,DeviceId::cpu(),16,
        DeviceRowRange::deviceCounted(16,&count)),std::invalid_argument);
    EXPECT_THROW((void)ProjectionRowWorkBinding(declaration,DeviceId::cpu(),16,
        DeviceRowRange::fullyActive(24).slice(1,16)),std::invalid_argument);
    EXPECT_THROW((void)ProjectionRowWorkBinding(declaration,DeviceId::cpu(),16,
        DeviceRowRange::fullyActive(8).repeatEachRow(2)),std::invalid_argument);
    EXPECT_NO_THROW((void)ProjectionRowWorkBinding(declaration,DeviceId::cpu(),16,DeviceRowRange::fullyActive(16)));
}
TEST(ProjectionRowWorkBinding, InheritedMismatchFailsOnlyAtOrderedObservationAndNeverChangesIdentity) {
    OriginalPublication source;int inherited=6;
    ProjectionRowWorkBinding binding(singleDeclaration(source),DeviceId::cpu(),8,DeviceRowRange::deviceCounted(8,&inherited));
    binding.prepare(nullptr,nullptr);const auto identity=binding.signature();
    binding.requirePublishedInputs();EXPECT_EQ(binding.work().observeHost().live,6);
    inherited=5;EXPECT_EQ(binding.signature(),identity);
    EXPECT_THROW((void)binding.work().observeHost(),std::invalid_argument);
    source.words()[1]=5;EXPECT_NO_THROW((void)binding.work().observeHost());
    EXPECT_EQ(binding.signature(),identity);
}
TEST(ProjectionRowWorkBinding, MissingDefaultOrWrongParticipantStreamFailsBeforeBackendLookup) {
    OriginalPublication source;
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})for(std::uintptr_t stream:{0u,1u,2u}) {
        ProjectionRowWorkBinding binding(singleDeclaration(source),device,8);
        EXPECT_THROW(binding.prepare(nullptr,reinterpret_cast<void *>(stream)),std::logic_error);
        EXPECT_FALSE(binding.prepared());
    }
    ProjectionRowWorkBinding cpu(singleDeclaration(source),DeviceId::cpu(),8);
    EXPECT_THROW(cpu.prepare(nullptr,reinterpret_cast<void *>(0x4000)),std::logic_error);
    {GraphCaptureGuard capture;EXPECT_THROW(cpu.prepare(nullptr,nullptr),std::logic_error);}
    EXPECT_FALSE(cpu.prepared());
}
TEST(ProjectionRowWorkBinding, SourceAndPolicyIdentityChangesButMutableWordsDoNot) {
    OriginalPublication source;
    const auto declaration=singleDeclaration(source);
    ProjectionRowWorkBinding original(declaration,DeviceId::cpu(),8);const auto identity=original.signature();
    source.words()[0]=4096;source.words()[1]=1;EXPECT_EQ(original.signature(),identity);
    for(int mutation=0;mutation<4;++mutation) {
        auto changed=declaration;
        if(mutation==0)changed.firstRequiredLogicalRow++;
        if(mutation==1)changed.sourceCapacity++;
        if(mutation==2)changed.sources[0].buffer=BufferId::QFN_QSA_TRANSACTION_PRIOR;
        if(mutation==3){changed.sources[0].frontierByteOffset=8;changed.sources[0].liveByteOffset=12;}
        EXPECT_NE(ProjectionRowWorkBinding(changed,DeviceId::cpu(),8).signature(),identity);
    }
}
TEST(ProjectionRowWorkspaceMetadata, ExactCPUCopyLeavesCapacityPaddingAndRejectsInvalidSetup) {
    WorkspaceRequirements requirements;requirements.buffers.emplace_back("original_metadata",64,8,true,
        WorkspaceExecutionRegime::Any,WorkspaceContentLifetime::SerialGraphFamily);
    OriginalWorkspace workspace(requirements);auto lease=workspace.manager->retainBuffer("original_metadata",8,16);
    auto *storage=static_cast<unsigned char *>(workspace.manager->getBuffer("original_metadata"));
    std::fill_n(storage,64,0xa5);const std::array<std::byte,8> payload{};
    TransferEngine engine;engine.initializeWorkspaceMetadata(*lease,payload,nullptr);
    for(int i=0;i<64;++i)EXPECT_EQ(storage[i],i>=8&&i<16?0:0xa5);
    EXPECT_THROW(engine.initializeWorkspaceMetadata(*lease,{},nullptr),std::invalid_argument);
    const std::array<std::byte,17> oversized{};
    EXPECT_THROW(engine.initializeWorkspaceMetadata(*lease,oversized,nullptr),std::invalid_argument);
    EXPECT_THROW(engine.initializeWorkspaceMetadata(*lease,payload,reinterpret_cast<void *>(0x4000)),std::invalid_argument);
    {GraphCaptureGuard capture;EXPECT_THROW(engine.initializeWorkspaceMetadata(*lease,payload,nullptr),std::invalid_argument);}
}
} // namespace llaminar2::test
