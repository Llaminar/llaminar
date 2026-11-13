/**
 * @file Test__WeightManagerCpuHostRelease.cpp
 * @brief Actual bounded source retirement and final CPU consumer lifetime contracts.
 *
 * The original source is a genuine tensor-backed reader. Every final consumer
 * is declared before admission and committed through WeightManager/Store. Native
 * GEMMs retain only their packed root; floating GEMMs, embeddings, norms and
 * biases retain their actual final readable root. No placeholder prepared
 * reference or synthetic full-parent pointer authorizes source retirement.
 */
#include <gtest/gtest.h>
#include <cstring>
#include "loaders/WeightManager.h"
#include "loaders/PreparedWeightStore.h"
#include "mocks/MockModelLoader.h"
#include "utils/CPUWeightPlanTestAdmission.h"
#include "utils/CPUProjectionTestWorkspace.h"
using namespace llaminar2;
using namespace llaminar2::test;

/** @brief Original source and immutable admission used by each complete CPU population. */
class Test__WeightManagerCpuHostRelease : public ::testing::Test {
protected:
    /** @brief Initialize only the CPU backend used by actual final consumers. */
    void SetUp() override {if(!hasCPUBackend())initCPUBackend(-1);}
    std::shared_ptr<MockModelLoader> loader=MockModelLoader::createMinimal();
    /**
     * @brief Declare an original source rectangle without loading its parent.
     * @param name Canonical original weight name.
     * @param role Intended final CPU consumer.
     * @param rows Selected source rows, or zero for the complete original tensor.
     * @return Requirement retaining the original source geometry and consumer.
     */
    WeightRequirement request(std::string name,WeightRole role,size_t rows=0) {
        WeightRequirement value;value.canonical_name=std::move(name);value.role=role;
        value.layer=inferWeightLayer(value.canonical_name);value.target_device=DeviceId::cpu();
        if(rows) {value.derivation=WeightDerivationKind::RowSlice;value.slice.row_count=rows;
            value.slice.inner_is_presliced=true;}
        if(role==WeightRole::Other)value.expected_prepared_kind=PreparedWeightKind::CPUResidentTensor;
        return value;
    }
    /**
     * @brief Commit declared final owners under their actual memory authority.
     * @param manager Production loader and Store receiving the original plan.
     * @param requests Complete consumer requirements to admit before population.
     * @return Frozen original bindings to the committed final CPU owners.
     */
    FrozenModelWeightSet populate(WeightManager& manager,std::vector<WeightRequirement> requests) {
        InferenceStrategy original;original.model_id={98504};original.devices={DeviceId::cpu()};
        WeightPlan plan(original);for(auto& value:requests)plan.add(std::move(value));
        if(!manager.preparedWeightStore()->bindModelIdIfUnset(original.model_id))
            throw std::logic_error("CPU lifetime fixture changed model authority");
        admitCPUWeightPlansForTest(manager,*loader,{&plan});
        return manager.materialize(plan);
    }
    /**
     * @brief Verify real execution after original source payload retirement.
     * @param manager Production Store retaining the prepared consumer.
     * @param binding Original committed CPU binding being exercised.
     * @param source_name Original input payload to release before repeat execution.
     */
    void proveEngineLifetime(WeightManager& manager,const WeightBinding& binding,const std::string& source_name) {
        ASSERT_TRUE(binding.prepared);
        auto engine=manager.preparedWeightStore()->gemmKernelLifetime(*binding.prepared);
        ASSERT_TRUE(engine);ASSERT_TRUE(engine->weights_converted());
        const int n=int(binding.tensor->shape()[0]),k=int(binding.tensor->shape()[1]);
        CPUProjectionTestWorkspace workspace(1,k,cpuProjectionTestRequirements(1,{engine.get()}));
        FP32Tensor input({1,size_t(k)}),before({1,size_t(n)}),after({1,size_t(n)});
        std::fill_n(input.mutable_data(),k,0.25f);
        ASSERT_TRUE(engine->multiply_tensor(&input,&before,1,n,k,true,1.f,0.f,nullptr,nullptr,-1,workspace.get()));
        auto original=loader->loadTensor(source_name);ASSERT_TRUE(original);
        original->release_host_weight_data();ASSERT_TRUE(original->is_raw_data_released());
        const auto reads=loader->nativeSourceReadCallCount();
        manager.clearCache();
        ASSERT_TRUE(engine->multiply_tensor(&input,&after,1,n,k,true,1.f,0.f,nullptr,nullptr,-1,workspace.get()));
        EXPECT_EQ(std::memcmp(before.data(),after.data(),size_t(n)*sizeof(float)),0);
        EXPECT_EQ(loader->nativeSourceReadCallCount(),reads);
    }
};

TEST_F(Test__WeightManagerCpuHostRelease, ReleasesQuantizedTensorSliceAfterCpuPreparation) {
    loader->addQ4_0RandomTensor("blk.0.attn_q.weight",{256,64});
    WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_q.weight",WeightRole::AttentionQ,128)});
    ASSERT_EQ(frozen.bindings().size(),1u);const auto& binding=frozen.bindings().front();
    ASSERT_TRUE(binding.native_source);EXPECT_EQ(binding.native_source->original.original_shape,(std::vector<size_t>{64,256}));
    EXPECT_EQ(binding.native_source->range.rows,128u);
    EXPECT_EQ(binding.tensor->shape(),(std::vector<size_t>{128,64}));
    EXPECT_TRUE(binding.tensor->is_raw_data_released());EXPECT_EQ(binding.tensor->raw_data(),nullptr);
    const auto reads=loader->nativeSourceReadCallCount();
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));
    EXPECT_EQ(loader->nativeSourceReadCallCount(),reads);
    proveEngineLifetime(manager,binding,"blk.0.attn_q.weight");
}
TEST_F(Test__WeightManagerCpuHostRelease, RetainsFP32TensorAfterCpuPreparation) {
    loader->addFP32RandomTensor("blk.0.attn_output.weight",{64,128});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_output.weight",WeightRole::AttentionWO)});
    const auto& binding=frozen.bindings().front();ASSERT_TRUE(binding.tensor->raw_data());
    const auto* root=binding.tensor->raw_data();ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));
    EXPECT_FALSE(binding.tensor->is_raw_data_released());EXPECT_EQ(binding.tensor->raw_data(),root);
    proveEngineLifetime(manager,binding,"blk.0.attn_output.weight");
}
TEST_F(Test__WeightManagerCpuHostRelease, RetainsFP32TensorSliceAfterCpuPreparation) {
    loader->addFP32RandomTensor("blk.0.attn_q.weight",{48,64});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_q.weight",WeightRole::AttentionQ,24)});
    const auto& binding=frozen.bindings().front();EXPECT_EQ(binding.tensor->shape(),(std::vector<size_t>{24,64}));
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));
    EXPECT_FALSE(binding.tensor->is_raw_data_released());ASSERT_TRUE(binding.tensor->raw_data());
    proveEngineLifetime(manager,binding,"blk.0.attn_q.weight");
}
TEST_F(Test__WeightManagerCpuHostRelease, RetainsEmbeddingWeightAfterCpuPreparation) {
    loader->addQ4_0RandomTensor("token_embd.weight",{256,64});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("token_embd.weight",WeightRole::Embedding,128)});
    const auto& binding=frozen.bindings().front();const auto* root=binding.tensor->raw_data();ASSERT_TRUE(root);
    const auto ref=manager.preparedWeightStore()->prepareEmbedding(binding,64);
    ASSERT_TRUE(manager.preparedWeightStore()->embeddingHandle(ref));
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));
    EXPECT_EQ(binding.tensor->raw_data(),root);EXPECT_FALSE(binding.tensor->is_raw_data_released());
}
TEST_F(Test__WeightManagerCpuHostRelease, RetainsNormWeightAfterCpuPreparation) {
    loader->addFP32RandomTensor("blk.0.attn_norm.weight",{128});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_norm.weight",WeightRole::Norm)});
    const auto& binding=frozen.bindings().front();const auto* root=binding.tensor->raw_data();ASSERT_TRUE(root);
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));
    EXPECT_EQ(binding.tensor->raw_data(),root);EXPECT_FALSE(binding.tensor->is_raw_data_released());
}
TEST_F(Test__WeightManagerCpuHostRelease, SkipsBindingsWithoutPreparedField) {
    loader->addQ4_0RandomTensor("blk.0.attn_q.weight",{256,64});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_q.weight",WeightRole::AttentionQ,128)});
    const auto original=frozen.bindings().front();auto missing=original;missing.prepared.reset();
    FrozenModelWeightSet descriptor(frozen.strategy(),{missing});
    // Removing a descriptor's ref does not revoke its actual committed Store owner or reload its source.
    const auto reads=loader->nativeSourceReadCallCount();
    ASSERT_TRUE(manager.prepareWeightsForDevice(descriptor,DeviceId::cpu()));
    EXPECT_EQ(loader->nativeSourceReadCallCount(),reads);
    EXPECT_EQ(manager.preparedWeightStore()->gemmKernelLifetime(*original.prepared).get(),
        manager.preparedWeightStore()->gemmKernelLifetime(*frozen.bindings().front().prepared).get());
    EXPECT_TRUE(original.tensor->is_raw_data_released());
}
TEST_F(Test__WeightManagerCpuHostRelease, SkipsBindingsPreparedForDifferentDevice) {
    loader->addQ4_0RandomTensor("blk.0.attn_q.weight",{128,64});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_q.weight",WeightRole::AttentionQ)});
    const auto& binding=frozen.bindings().front();const auto ref=*binding.prepared;
    const auto reads=loader->nativeSourceReadCallCount();
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId(DeviceType::CPU,1)));
    EXPECT_EQ(loader->nativeSourceReadCallCount(),reads);
    EXPECT_TRUE(manager.preparedWeightStore()->gemmKernelLifetime(ref));
    EXPECT_FALSE(manager.preparedWeightStore()->preparedRefForBinding(binding.binding_id,DeviceId(DeviceType::CPU,1),PreparedWeightKind::CpuPackedGemm));
}
TEST_F(Test__WeightManagerCpuHostRelease, MixedBindingsReleasesOnlyQuantized) {
    loader->addQ4_0RandomTensor("blk.0.attn_q.weight",{256,64});
    loader->addQ4_0RandomTensor("blk.0.ffn_gate.weight",{512,64});
    loader->addFP32RandomTensor("blk.0.attn_output.weight",{64,128});
    loader->addQ4_0RandomTensor("token_embd.weight",{256,64});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_q.weight",WeightRole::AttentionQ,128),
        request("blk.0.ffn_gate.weight",WeightRole::FFNGate,256),request("blk.0.attn_output.weight",WeightRole::AttentionWO),
        request("token_embd.weight",WeightRole::Embedding,128)});
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));
    for(const auto& binding:frozen.bindings()) {
        const bool packed=binding.identity.role==WeightRole::AttentionQ || binding.identity.role==WeightRole::FFNGate;
        EXPECT_EQ(binding.tensor->is_raw_data_released(),packed);
        EXPECT_EQ(binding.tensor->raw_data()==nullptr,packed);
        if(packed)proveEngineLifetime(manager,binding,binding.identity.canonical_name);
    }
}
TEST_F(Test__WeightManagerCpuHostRelease, RetainsBiasWeightAfterCpuPreparation) {
    loader->addFP32RandomTensor("blk.0.attn_q.bias",{128});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_q.bias",WeightRole::Bias)});
    const auto& binding=frozen.bindings().front();ASSERT_TRUE(binding.tensor->raw_data());
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));EXPECT_FALSE(binding.tensor->is_raw_data_released());
}
TEST_F(Test__WeightManagerCpuHostRelease, Skips1DTensorAfterCpuPreparation) {
    loader->addFP32RandomTensor("blk.0.some_1d.weight",{128});WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.some_1d.weight",WeightRole::Other)});
    const auto& binding=frozen.bindings().front();ASSERT_TRUE(binding.prepared);
    EXPECT_EQ(binding.prepared->kind,PreparedWeightKind::CPUResidentTensor);
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));EXPECT_FALSE(binding.tensor->is_raw_data_released());
    EXPECT_EQ(manager.preparedWeightStore()->gemmKernelLifetime(*binding.prepared),nullptr);
}
TEST_F(Test__WeightManagerCpuHostRelease, ReleasesMultipleLayersQuantizedWeights) {
    std::vector<WeightRequirement> requests;
    for(int layer=0;layer<4;++layer) {const auto name="blk."+std::to_string(layer)+".attn_q.weight";
        loader->addQ4_0RandomTensor(name,{256,64});requests.push_back(request(name,WeightRole::AttentionQ,128));}
    WeightManager manager(*loader);auto frozen=populate(manager,std::move(requests));
    ASSERT_EQ(frozen.bindings().size(),4u);ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu()));
    for(const auto& binding:frozen.bindings()) {EXPECT_TRUE(binding.tensor->is_raw_data_released());
        proveEngineLifetime(manager,binding,binding.identity.canonical_name);}
}
TEST_F(Test__WeightManagerCpuHostRelease, IncludeExpertJobsFalseSkipsRoutedExpertBindings) {
    loader->addQ4_0RandomTensor("blk.0.attn_q.weight",{128,64});
    const size_t expert_bytes=2*128*64/Q4_0Block::BLOCK_SIZE*sizeof(Q4_0Block);
    loader->addTensor("blk.0.ffn_gate_exps.weight",std::make_shared<Q4_0Tensor>(
        std::vector<size_t>{128,64,2},std::vector<uint8_t>(expert_bytes,0)));
    WeightManager manager(*loader);
    auto frozen=populate(manager,{request("blk.0.attn_q.weight",WeightRole::AttentionQ),
        request("blk.0.ffn_gate_exps.weight",WeightRole::MoEExpertGate)});
    std::vector<std::shared_ptr<ITensorGemm>> retained;
    for(const auto& binding:frozen.bindings()) {ASSERT_TRUE(binding.prepared);
        retained.push_back(manager.preparedWeightStore()->gemmKernelLifetime(*binding.prepared));ASSERT_TRUE(retained.back());}
    const auto reads=loader->nativeSourceReadCallCount();
    ASSERT_TRUE(manager.prepareWeightsForDevice(frozen,DeviceId::cpu(),false));
    EXPECT_EQ(loader->nativeSourceReadCallCount(),reads);
    for(size_t index=0;index<frozen.bindings().size();++index) {
        EXPECT_EQ(manager.preparedWeightStore()->gemmKernelLifetime(*frozen.bindings()[index].prepared),retained[index]);
        EXPECT_TRUE(frozen.bindings()[index].tensor->is_raw_data_released());}
}

TEST_F(Test__WeightManagerCpuHostRelease, FinalFloatingRootSurvivesBothHostReleaseSweeps) {
    loader->addFP32RandomTensor("blk.0.attn_q.weight",{16,64});WeightManager manager(*loader);
    auto declared=request("blk.0.attn_q.weight",WeightRole::AttentionQ);
    declared.host_policy=WeightHostPolicy::ReleasableAfterPreparation;
    auto frozen=populate(manager,{declared});const auto& binding=frozen.bindings().front();
    EXPECT_EQ(binding.residency.host_policy,WeightHostPolicy::RequiredForCPUExecution);
    const auto retained=manager.getWeightForDevice("blk.0.attn_q.weight",DeviceId::cpu());
    ASSERT_EQ(retained,binding.tensor_owner);ASSERT_TRUE(retained->hasPreparedDeviceState());
    const auto* root=retained->raw_data();ASSERT_TRUE(root);retained->setHostResident();
    manager.markMaterializationComplete();manager.markDevicePreparationComplete();manager.markGraphMaterializationComplete();
    EXPECT_EQ(manager.releaseAllHostWeightData(),0u);
    EXPECT_EQ(manager.releaseHostResidentWeightData(),0u);
    EXPECT_EQ(retained->raw_data(),root);EXPECT_FALSE(retained->is_raw_data_released());
    // The original failing policy discards the root. Do not dereference freed
    // bytes in the negative control; the failed ownership assertions suffice.
    if(!retained->raw_data())return;
    proveEngineLifetime(manager,binding,"blk.0.attn_q.weight");
}
TEST_F(Test__WeightManagerCpuHostRelease, TiedAliasPreservesRequiredFinalSourceLifetime) {
    loader->addFP32RandomTensor("token_embd.weight",{32,64});WeightManager manager(*loader);
    auto original=request("token_embd.weight",WeightRole::Embedding);
    auto alias=request("output.weight",WeightRole::LMHead);
    alias.source_name="token_embd.weight";alias.derivation=WeightDerivationKind::TiedAlias;
    alias.host_policy=WeightHostPolicy::ReleasableAfterPreparation;
    auto frozen=populate(manager,{original,alias});
    const auto& source=frozen.global("token_embd.weight");const auto& head=frozen.global("output.weight");
    ASSERT_EQ(source.tensor_owner,head.tensor_owner);ASSERT_TRUE(head.prepared);
    EXPECT_EQ(source.residency.host_policy,WeightHostPolicy::RequiredForCPUExecution);
    EXPECT_EQ(head.residency.host_policy,source.residency.host_policy);
    const auto retained=manager.getWeightForDevice("token_embd.weight",DeviceId::cpu());
    ASSERT_EQ(retained,source.tensor_owner);ASSERT_TRUE(retained->hasPreparedDeviceState());
    retained->setHostResident();const auto* root=retained->raw_data();ASSERT_TRUE(root);
    manager.markMaterializationComplete();manager.markDevicePreparationComplete();manager.markGraphMaterializationComplete();
    EXPECT_EQ(manager.releaseAllHostWeightData(),0u);EXPECT_EQ(manager.releaseHostResidentWeightData(),0u);
    EXPECT_EQ(retained->raw_data(),root);EXPECT_FALSE(retained->is_raw_data_released());
    if(!retained->raw_data())return;
    proveEngineLifetime(manager,head,"token_embd.weight");
    const auto reads=loader->nativeSourceReadCallCount();
    // Reuse cannot weaken the already committed final owner's required policy.
    InferenceStrategy strategy;strategy.model_id={98504};strategy.devices={DeviceId::cpu()};
    WeightPlan repeated(strategy);repeated.add(original);repeated.add(alias);
    const auto reused=manager.materialize(repeated);
    EXPECT_EQ(reused.global("output.weight").tensor_owner,head.tensor_owner);
    EXPECT_EQ(reused.global("output.weight").residency.host_policy,WeightHostPolicy::RequiredForCPUExecution);
    EXPECT_EQ(loader->nativeSourceReadCallCount(),reads);
}
