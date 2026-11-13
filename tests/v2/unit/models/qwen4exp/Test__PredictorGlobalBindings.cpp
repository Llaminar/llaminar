/**
 * @file Test__PredictorGlobalBindings.cpp
 * @brief Device-free target-source/terminal-layout authentication without native tensor impersonation.
 *
 * Pure metadata validation cannot issue a native graph binding. The actual
 * factory separately borrows real immutable tensors and prepared vocab ranges.
 */
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPredictorGlobalBindings.h"
#include <gtest/gtest.h>
#include <array>
using namespace llaminar2;
namespace {
/** @brief Exact metadata only; no fake TensorBase or device allocation exists. */
struct DescriptorFixture {
    WeightIdentity main{{7},31,40,"token_embd.weight",WeightRole::Embedding};
    WeightIdentity selected=main;
    std::array<size_t,2> shape{64,16};
    uint64_t binding=91;
    PreparedWeightRef prepared{{7},91,PreparedWeightKind::PreparedEmbedding,DeviceId(DeviceType::CPU,1)};
    /** @return Metadata projected by the real production factory. */
    Qwen4ExpPredictorGlobalDescriptor descriptor()const{return {selected,binding,prepared,TensorType::Q8_0,shape};}
    /** @return Original main identity; selected instance may differ but logical source may not. */
    Qwen4ExpPredictorGlobalDescriptor target()const{return {main,22,std::nullopt,TensorType::Q8_0,shape};}
};
}
TEST(PredictorGlobalBindings, DraftSourcePolicyIsIndependentOfRows)
{
    using S=Qwen4ExpPredictorGlobalSource;
    for (bool replicated:{false,true})for(bool mirrored:{false,true}) {
        const auto plan=Qwen4ExpPredictorGlobalBindings::select(Qwen4ExpPredictorInvocation::Draft,true,replicated,mirrored,
            MTPTerminalLogitsLayout::FullVocabularyPerParticipant,true);
        EXPECT_EQ(plan.embedding,replicated||mirrored?S::PreparedReplica:S::Primary);
        EXPECT_EQ(plan.head,S::PreparedReplica);EXPECT_TRUE(plan.mirrored_head);
    }
}
TEST(PredictorGlobalBindings, ShiftedPrefillConsumesOnlyPrimaryEmbedding)
{
    for(bool replicated:{false,true})for(bool mirrored:{false,true}) {
        const auto plan=Qwen4ExpPredictorGlobalBindings::select(Qwen4ExpPredictorInvocation::ShiftedPrefill,true,replicated,mirrored,
            MTPTerminalLogitsLayout::FullVocabularyPerParticipant,true);
        EXPECT_EQ(plan.embedding,Qwen4ExpPredictorGlobalSource::Primary);
        EXPECT_EQ(plan.head,Qwen4ExpPredictorGlobalSource::NotConsumed);EXPECT_FALSE(plan.mirrored_head);
    }
}
TEST(PredictorGlobalBindings, ExplicitShardAndSingleParticipantKeepPrimaryHead)
{
    const auto sharded=Qwen4ExpPredictorGlobalBindings::select(Qwen4ExpPredictorInvocation::Draft,true,false,false,
        MTPTerminalLogitsLayout::VocabularyShardPerParticipant,true);
    EXPECT_EQ(sharded.head,Qwen4ExpPredictorGlobalSource::Primary);EXPECT_FALSE(sharded.mirrored_head);
    const auto single=Qwen4ExpPredictorGlobalBindings::select(Qwen4ExpPredictorInvocation::Draft,false,true,true,
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant,false);
    EXPECT_EQ(single.embedding,Qwen4ExpPredictorGlobalSource::Primary);EXPECT_EQ(single.head,Qwen4ExpPredictorGlobalSource::Primary);
}
TEST(PredictorGlobalBindings, UnknownPolicyNeverSelectsAnotherSource)
{
    EXPECT_THROW((void)Qwen4ExpPredictorGlobalBindings::select(static_cast<Qwen4ExpPredictorInvocation>(9),false,false,false,
        MTPTerminalLogitsLayout::FullVocabularyPerParticipant,false),std::invalid_argument);
    EXPECT_THROW((void)Qwen4ExpPredictorGlobalBindings::select(Qwen4ExpPredictorInvocation::Draft,false,false,false,
        static_cast<MTPTerminalLogitsLayout>(9),false),std::invalid_argument);
}
TEST(PredictorGlobalBindings, OriginalNativeTargetIdentityAndReplicaLogicalSourceAreAccepted)
{
    DescriptorFixture f;
    EXPECT_NO_THROW(Qwen4ExpPredictorGlobalBindings::requireDescriptor(WeightRole::Embedding,f.descriptor(),f.target(),DeviceId(DeviceType::CPU,1),64,16));
    f.selected.instance_id=987;f.selected.derivation=WeightDerivationKind::DeviceClone;
    EXPECT_NO_THROW(Qwen4ExpPredictorGlobalBindings::requireDescriptor(WeightRole::Embedding,f.descriptor(),f.target(),DeviceId(DeviceType::CPU,1),64,16));
}
TEST(PredictorGlobalBindings, SidecarRetargetForeignModelAndWrongRoleAreRejected)
{
    for(int mutation=0;mutation<6;++mutation) {
        DescriptorFixture f;
        if(mutation==0)f.selected.layer=48;
        if(mutation==1)f.selected.model_id.value=8;
        if(mutation==2)f.selected.logical_id=32;
        if(mutation==3)f.selected.canonical_name="nextn.shared_head.weight";
        if(mutation==4)f.selected.role=WeightRole::LMHead;
        if(mutation==5)f.main.logical_id=0;
        EXPECT_THROW(Qwen4ExpPredictorGlobalBindings::requireDescriptor(WeightRole::Embedding,f.descriptor(),f.target(),DeviceId(DeviceType::CPU,1),64,16),std::invalid_argument)<<mutation;
    }
}
TEST(PredictorGlobalBindings, PreparedParticipantKindBindingFormatAndShapeAreAuthenticated)
{
    for(int mutation=0;mutation<8;++mutation) {
        DescriptorFixture f;
        if(mutation==0)f.prepared.device=DeviceId::cpu();
        if(mutation==1)f.prepared.model_id.value=8;
        if(mutation==2)f.prepared.binding_id=92;
        if(mutation==3)f.prepared.kind=PreparedWeightKind::CpuPackedGemm;
        if(mutation==4)f.binding=0;
        if(mutation==5)f.shape[0]=63;
        if(mutation==6)f.shape[1]=17;
        auto descriptor=f.descriptor();if(mutation==7)descriptor.native_type=TensorType::FP32;
        EXPECT_THROW(Qwen4ExpPredictorGlobalBindings::requireDescriptor(WeightRole::Embedding,descriptor,f.target(),DeviceId(DeviceType::CPU,1),64,16),std::invalid_argument)<<mutation;
    }
}
TEST(PredictorGlobalBindings, TerminalHeadRequiresExistingNativePreparedGemm)
{
    DescriptorFixture f;f.main.role=f.selected.role=WeightRole::LMHead;
    f.main.canonical_name=f.selected.canonical_name="output.weight";
    for(auto kind:{PreparedWeightKind::CpuPackedGemm,PreparedWeightKind::CudaInt8PackedGemm,PreparedWeightKind::RocmInt8PackedGemm}) {
        f.prepared.kind=kind;
        EXPECT_NO_THROW(Qwen4ExpPredictorGlobalBindings::requireDescriptor(WeightRole::LMHead,f.descriptor(),f.target(),DeviceId(DeviceType::CPU,1),64,16));
    }
    f.prepared.kind=PreparedWeightKind::PreparedEmbedding;
    EXPECT_THROW(Qwen4ExpPredictorGlobalBindings::requireDescriptor(WeightRole::LMHead,f.descriptor(),f.target(),DeviceId(DeviceType::CPU,1),64,16),std::invalid_argument);
}
