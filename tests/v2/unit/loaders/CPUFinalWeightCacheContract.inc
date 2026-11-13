/**
 * @file CPUFinalWeightCacheContract.inc
 * @brief Actual final CPU handle lookup, original replica identity and index retirement.
 *
 * All source/final bytes use the existing real tensor-backed reader and original
 * immutable plan/PMA. Logical rank sweeps are device-free metadata coverage,
 * not a claim about native MPI processes or physical socket availability.
 */
#include "../../utils/CPUWeightPlanTestAdmission.h"
#include "loaders/PreparedWeightStore.h"
#include <unordered_set>
namespace {
constexpr const char* cacheNormName="output_norm.weight";
/** @return Original single-vector source declaration, with an optional original TP participant. */
llaminar2::WeightRequirement cacheNormRequirement(int rank=-1) {
    llaminar2::WeightRequirement requirement;
    requirement.canonical_name=cacheNormName;
    requirement.role=llaminar2::WeightRole::OutputNorm;
    requirement.target_device=llaminar2::DeviceId::cpu();
    if(rank>=0) {requirement.tp_domain=0;requirement.tp_rank_or_device_index=rank;}
    return requirement;
}
/** @brief Install exactly the final/source population declared by this original test model. */
void admitCacheNormPlan(llaminar2::WeightManager& manager,llaminar2::IModelLoader& loader,
    const llaminar2::WeightPlan& plan) {
    ASSERT_TRUE(manager.preparedWeightStore()->bindModelIdIfUnset(plan.strategy().model_id));
    llaminar2::test::admitCPUWeightPlansForTest(manager,loader,{&plan});
}
}
TEST(CPUFinalWeightCacheContract, SecondLookupCompletesAndKeepsTheOriginalFinalOwner) {
    using namespace llaminar2;
    llaminar2::test::MockModelLoader loader;
    loader.addFP32OnesTensor(cacheNormName,{32});
    WeightManager manager(loader);
    InferenceStrategy strategy;strategy.model_id={97501};strategy.devices={DeviceId::cpu()};
    WeightPlan plan(strategy);plan.add(cacheNormRequirement());
    admitCacheNormPlan(manager,loader,plan);
    const auto first=manager.getWeightForDevice(cacheNormName,DeviceId::cpu());
    ASSERT_TRUE(first);
    const auto reads=loader.nativeSourceReadCallCount();
    ASSERT_GT(reads,0u);
    // The original failing body recursively acquired the manager mutex here.
    // The canonical process guardian supplies the ordinary 30-second deadline.
    const auto second=manager.getWeightForDevice(cacheNormName,DeviceId::cpu());
    ASSERT_TRUE(second);EXPECT_EQ(first,second);EXPECT_EQ(loader.nativeSourceReadCallCount(),reads);
    EXPECT_EQ(manager.cacheSize(),1u);
    manager.clearCache();EXPECT_EQ(manager.cacheSize(),0u);
    const auto after_clear=manager.getWeightForDevice(cacheNormName,DeviceId::cpu());
    EXPECT_EQ(after_clear,first);EXPECT_EQ(loader.nativeSourceReadCallCount(),reads);
    EXPECT_EQ(manager.cacheSize(),1u);
}
TEST(CPUFinalWeightCacheContract, OriginalRanksRetainIndependentRootsAcrossOneTwoFourEightParticipants) {
    using namespace llaminar2;
    for(const int degree:{1,2,4,8}) {
        SCOPED_TRACE(degree);
        llaminar2::test::MockModelLoader loader;
        loader.addFP32OnesTensor(cacheNormName,{32});
        WeightManager manager(loader);
        Qwen2SchemaFactory schema;manager.setWeightShardingConfig(schema.getWeightShardingConfig());
        const auto original=std::make_shared<TensorParallelConfig>(TensorParallelConfig::equalSplit(
            degree,degree*2,degree,degree*64,degree*64,std::vector<DeviceId>(size_t(degree),DeviceId::cpu())));
        manager.setTensorParallelConfig(original);
        InferenceStrategy strategy;strategy.model_id={uint64_t(97510+degree)};
        strategy.devices={DeviceId::cpu()};strategy.tp_degree=degree;
        WeightPlan plan(strategy);
        for(int rank=0;rank<degree;++rank)plan.add(cacheNormRequirement(rank));
        admitCacheNormPlan(manager,loader,plan);
        std::vector<std::shared_ptr<TensorBase>> roots(static_cast<size_t>(degree));
        std::unordered_set<const void*> physical;
        for(int rank=0;rank<degree;++rank) {
            roots[size_t(rank)]=manager.getShardedWeightForAssignment(cacheNormName,DeviceId::cpu(),original->forRank(rank),-1);
            ASSERT_TRUE(roots[size_t(rank)]);
            EXPECT_TRUE(physical.insert(roots[size_t(rank)]->raw_data()).second);
            EXPECT_EQ(manager.getShardedWeightForAssignment(cacheNormName,DeviceId::cpu(),original->forRank(rank),-1),roots[size_t(rank)]);
            const auto metadata=manager.weightMetadataRegistry()->metadata(roots[size_t(rank)].get());
            ASSERT_TRUE(metadata);EXPECT_EQ(metadata->identity.tp_rank_or_device_index,rank);
            EXPECT_EQ(metadata->identity.tp_domain,0);
        }
        EXPECT_EQ(manager.cacheSize(),size_t(degree));
        const auto reads=loader.nativeSourceReadCallCount();
        EXPECT_EQ(reads,size_t(degree));
        manager.clearCache();EXPECT_EQ(manager.cacheSize(),0u);
        for(int rank=degree-1;rank>=0;--rank)
            EXPECT_EQ(manager.getShardedWeightForAssignment(cacheNormName,DeviceId::cpu(),original->forRank(rank),-1),roots[size_t(rank)]);
        EXPECT_EQ(loader.nativeSourceReadCallCount(),reads);
        EXPECT_EQ(manager.cacheSize(),size_t(degree));
    }
}
TEST(CPUFinalWeightCacheContract, MissingOrWithdrawnStoreAuthorityCannotRereadIntoAFreshRoot) {
    using namespace llaminar2;
    llaminar2::test::MockModelLoader loader;loader.addFP32OnesTensor(cacheNormName,{32});
    WeightManager manager(loader);
    EXPECT_THROW(manager.getWeightForDevice(cacheNormName,DeviceId::cpu()),std::logic_error);
    EXPECT_FALSE(manager.physicalMemoryAuthority());EXPECT_EQ(loader.nativeSourceReadCallCount(),0u);
    InferenceStrategy strategy;strategy.model_id={97530};strategy.devices={DeviceId::cpu()};
    WeightPlan plan(strategy);plan.add(cacheNormRequirement());admitCacheNormPlan(manager,loader,plan);
    auto retained=manager.getWeightForDevice(cacheNormName,DeviceId::cpu());ASSERT_TRUE(retained);
    const auto reads=loader.nativeSourceReadCallCount();const auto memory=manager.physicalMemoryAuthority();
    const auto bytes=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation);
    ASSERT_GT(bytes,0u);manager.preparedWeightStore()->clear();
    EXPECT_THROW(manager.getWeightForDevice(cacheNormName,DeviceId::cpu()),std::logic_error);
    EXPECT_EQ(loader.nativeSourceReadCallCount(),reads);
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),bytes);
    manager.clearCache();retained.reset();
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}

TEST(CPUFinalWeightCacheContract, RetiredResidentBytesCannotMasqueradeAsAValidFinalHandle) {
    using namespace llaminar2;
    llaminar2::test::MockModelLoader loader;loader.addFP32OnesTensor(cacheNormName,{32});
    WeightManager manager(loader);
    InferenceStrategy strategy;strategy.model_id={97540};strategy.devices={DeviceId::cpu()};
    WeightPlan plan(strategy);plan.add(cacheNormRequirement());admitCacheNormPlan(manager,loader,plan);
    const auto final=manager.getWeightForDevice(cacheNormName,DeviceId::cpu());ASSERT_TRUE(final);
    const auto reads=loader.nativeSourceReadCallCount();
    final->release_host_weight_data();
    ASSERT_TRUE(final->is_raw_data_released());
    EXPECT_THROW(manager.getWeightForDevice(cacheNormName,DeviceId::cpu()),std::logic_error);
    EXPECT_EQ(loader.nativeSourceReadCallCount(),reads);
}

TEST(CPUFinalWeightCacheContract, DecodeIdentityKeepsExactFractionsAndOriginalCPUAllocators) {
    using namespace llaminar2;
    constexpr const char* name="blk.0.attn_q.weight";
    constexpr size_t rows=16,columns=8;
    const DeviceId cpu0=DeviceId::cpu(),cpu1{DeviceType::CPU,1};
    // CPU:1 is an explicit allocator identity, not a NUMA or MPI rank.
    auto original=std::make_shared<FP32Tensor>(std::vector<size_t>{rows,columns});
    for(size_t index=0;index<rows*columns;++index)original->mutable_data()[index]=float(index);
    llaminar2::test::MockModelLoader loader;loader.addTensor(name,original);
    loader.addFP32OnesTensor(cacheNormName,{32});
    WeightManager manager(loader);
    Qwen2SchemaFactory schema;manager.setWeightShardingConfig(schema.getWeightShardingConfig());
    const float half=0.5f,above=std::nextafter(half,1.0f),same_geometry=std::nextafter(above,1.0f);
    ASSERT_EQ(std::to_string(half),std::to_string(above));
    InferenceStrategy strategy;strategy.model_id={97550};strategy.devices={cpu0,cpu1};
    WeightPlan plan(strategy);plan.add(cacheNormRequirement());
    const auto declaration=[&](DeviceId device,size_t count) {
        WeightRequirement requirement;
        requirement.canonical_name=name;requirement.role=WeightRole::AttentionQ;
        requirement.layer=0;requirement.target_device=device;requirement.derivation=WeightDerivationKind::DecodeShard;
        requirement.slice.source_rows=rows;requirement.slice.source_cols=columns;
        requirement.slice.row_start=rows-count;requirement.slice.row_count=count;
        requirement.slice.col_count=columns;requirement.slice.inner_is_presliced=true;
        return requirement;
    };
    plan.add(declaration(cpu0,8));plan.add(declaration(cpu0,9));plan.add(declaration(cpu1,8));
    admitCacheNormPlan(manager,loader,plan);
    const auto first=manager.getDecodeWeight(name,cpu0,half,0);
    const auto second=manager.getDecodeWeight(name,cpu0,above,0);
    const auto equal_source=manager.getDecodeWeight(name,cpu0,same_geometry,0);
    const auto other_cpu=manager.getDecodeWeight(name,cpu1,half,0);
    ASSERT_TRUE(first);ASSERT_TRUE(second);ASSERT_TRUE(other_cpu);
    EXPECT_EQ(first->shape(),(std::vector<size_t>{8,columns}));
    EXPECT_EQ(second->shape(),(std::vector<size_t>{9,columns}));
    EXPECT_NE(first,second);EXPECT_EQ(second,equal_source);
    EXPECT_NE(first,other_cpu);EXPECT_EQ(first->home_device(),cpu0);EXPECT_EQ(other_cpu->home_device(),cpu1);
    EXPECT_EQ(manager.decodeCacheSize(),4u);
    for(const auto& tensor:{first,second,other_cpu}) {
        const size_t start=rows-tensor->shape()[0];
        const auto* values=static_cast<const float*>(tensor->raw_data());ASSERT_TRUE(values);
        for(size_t row=0;row<tensor->shape()[0];++row)for(size_t col=0;col<columns;++col)
            EXPECT_EQ(values[row*columns+col],original->data()[(start+row)*columns+col]);
    }
    EXPECT_EQ(loader.loadTensorCallCount(),0u);
    EXPECT_EQ(loader.nativeSourceReadCallCount(),3u);
    const auto norm=manager.getWeightForDevice(cacheNormName,cpu0);ASSERT_TRUE(norm);
    const auto reads=loader.nativeSourceReadCallCount();
    EXPECT_EQ(manager.getDecodeWeight(cacheNormName,cpu0,0.25f,0),norm);
    EXPECT_EQ(manager.getDecodeWeight(cacheNormName,cpu0,0.5f,0),norm);
    EXPECT_EQ(loader.nativeSourceReadCallCount(),reads);
    EXPECT_THROW(manager.getDecodeWeight(name,cpu0,std::numeric_limits<float>::quiet_NaN(),0),std::runtime_error);
    EXPECT_EQ(loader.nativeSourceReadCallCount(),reads);
}
