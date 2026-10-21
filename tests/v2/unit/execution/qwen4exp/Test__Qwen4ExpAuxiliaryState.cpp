/**
 * @file Test__Qwen4ExpAuxiliaryState.cpp
 * @brief Cache-owned accepted-state reset, compact archive and rollback regressions.
 *
 * Fixtures use real CPU cache frontiers and admitted tensor subviews. They prove
 * request isolation, live-tail copy extents, append-only pool preservation and
 * fixed lexical histories without a host mirror or an auxiliary allocator.
 */
#include "execution/qwen4exp/Qwen4ExpAuxiliaryState.h"
#include "execution/prefix_cache/PrefixStateSnapshot.h"
#include "execution/prefix_cache/PrefixAuxiliaryStateIO.h"
#include "execution/prefix_cache/RamPrefixStorageBackend.h"
#include "execution/prefix_cache/ResidualRowGeometry.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
using namespace llaminar2;
namespace
{
/** @brief Two independent admitted request banks with one sparse and one lexical owner. */
struct AuxiliaryFixture
{
    MPIContext mpi{0,1,MPI_COMM_WORLD};
    std::shared_ptr<IKVCache> cache=createCPURingKVCache(ActivationPrecision::FP32,mpi,2,2,16,1,128,DeviceId::cpu());
    FP32Tensor pools{{2,4,128}},partials{{2,4,128}},convolution{{2,18,8}};
    INT32Tensor tokens{{2,4}};
    Qwen4ExpAuxiliaryState *owner=nullptr;
    AuxiliaryFixture()
    {
        Qwen4ExpAuxiliaryState::Params params{.cache=*cache,.device=DeviceId::cpu(),.requestCapacity=2,.eosToken=42};
        for(int request=0;request<2;++request) {
            params.sparseLayers.push_back({0,0,request,16,
                {pools,BufferId::QFN_QSA_POOLED_KEYS,std::size_t(request)*4*128*4},
                {partials,BufferId::QFN_QSA_PARTIAL_KEYS,std::size_t(request)*4*128*4}});
            params.pleHistories.push_back({1,1,request,8,
                {tokens,BufferId::QFN_PLE_TOKEN_HISTORY,std::size_t(request)*16},
                {tokens,BufferId::QFN_PLE_TOKEN_HISTORY,std::size_t(request)*16+8},
                {convolution,BufferId::QFN_PLE_CONV_HISTORY,std::size_t(request)*18*8*4},
                {convolution,BufferId::QFN_PLE_CONV_HISTORY,std::size_t(request)*18*8*4+9*8*4}});
        }
        std::fill_n(pools.mutable_data(),pools.numel(),-77.0f);
        auto state=std::make_unique<Qwen4ExpAuxiliaryState>(std::move(params));owner=state.get();
        cache->attachAuxiliaryState(std::move(state),nullptr);
    }
};
}
TEST(Qwen4ExpAuxiliaryState, CanonicalResetIsRequestAndLayerScoped)
{
    AuxiliaryFixture f;
    for(int i=0;i<f.tokens.numel();++i)EXPECT_EQ(static_cast<const int *>(f.tokens.raw_data())[i],42);
    std::fill_n(f.partials.mutable_data(),f.partials.numel(),11.0f);
    std::fill_n(f.convolution.mutable_data(),f.convolution.numel(),12.0f);
    std::fill_n(static_cast<int *>(f.tokens.raw_mutable_data()),f.tokens.numel(),13);
    const auto boundary=IKVCache::StateResetContext::testReinitialization(nullptr,"qfn_request_isolation");
    ASSERT_TRUE(f.cache->resetSequenceState(0,boundary));
    for(int i=0;i<4*128;++i)EXPECT_EQ(f.partials.data()[i],0.0f);
    for(int i=4*128;i<8*128;++i)EXPECT_EQ(f.partials.data()[i],11.0f);
    for(int i=0;i<4;++i)EXPECT_EQ(static_cast<const int *>(f.tokens.raw_data())[i],42);
    for(int i=4;i<8;++i)EXPECT_EQ(static_cast<const int *>(f.tokens.raw_data())[i],13);
    ASSERT_TRUE(f.cache->resetLayerSequenceState(0,1,boundary));
    for(int i=4*128;i<8*128;++i)EXPECT_EQ(f.partials.data()[i],0.0f);
    for(int i=4;i<8;++i)EXPECT_EQ(static_cast<const int *>(f.tokens.raw_data())[i],13);
    ASSERT_TRUE(f.cache->resetLayerState(1,boundary));
    for(int i=0;i<f.tokens.numel();++i)EXPECT_EQ(static_cast<const int *>(f.tokens.raw_data())[i],42);
    for(int i=0;i<f.pools.numel();++i)EXPECT_EQ(f.pools.data()[i],-77.0f);
}
TEST(Qwen4ExpAuxiliaryState, RollbackAndArchiveCopyOnlyTheirDeclaredLiveData)
{
    AuxiliaryFixture f;
    std::vector<float> kv(7*128,0.0f);
    ASSERT_TRUE(f.cache->importLogicalBlock({0,0,0,7,nullptr},kv.data(),kv.data()));
    for(int i=0;i<f.partials.numel();++i)f.partials.mutable_data()[i]=float(i)+0.25f;
    for(int i=0;i<f.convolution.numel();++i)f.convolution.mutable_data()[i]=float(i)+0.5f;
    for(int i=0;i<f.pools.numel();++i)f.pools.mutable_data()[i]=float(i)+0.75f;
    auto *tokens=static_cast<int *>(f.tokens.raw_mutable_data());tokens[0]=123456789;tokens[1]=2147483646;
    const auto partial=std::vector<float>(f.partials.data(),f.partials.data()+3*128);
    const auto conv=std::vector<float>(f.convolution.data(),f.convolution.data()+9*8);
    std::vector<std::uint32_t> checkpoint((f.owner->checkpointCapacity(0)+3)/4,0xdeadbeefu);
    f.owner->captureCheckpoint(0,{checkpoint.data(),checkpoint.size()*4,CacheAuxiliaryMemoryDomain::Host},nullptr);
    ASSERT_EQ(checkpoint[0],3u);
    EXPECT_EQ(checkpoint[1],0xdeadbeefu);EXPECT_EQ(checkpoint[2],0xdeadbeefu);EXPECT_EQ(checkpoint[3],0xdeadbeefu);
    std::fill_n(f.partials.mutable_data(),4*128,-11.0f);std::fill_n(f.convolution.mutable_data(),9*8,-12.0f);tokens[0]=tokens[1]=-1;
    f.owner->restoreCheckpoint(0,{checkpoint.data(),checkpoint.size()*4,CacheAuxiliaryMemoryDomain::Host},nullptr);
    EXPECT_EQ(std::memcmp(partial.data(),f.partials.data(),partial.size()*4),0);
    EXPECT_EQ(f.partials.data()[3*128],-11.0f); // Inactive fourth raw key was never captured/restored.
    EXPECT_EQ(std::memcmp(conv.data(),f.convolution.data(),conv.size()*4),0);
    EXPECT_EQ(tokens[0],123456789);EXPECT_EQ(tokens[1],2147483646);
    const CacheAuxiliaryBlockRange range{0,0,3,6};ASSERT_EQ(f.owner->blockArchiveBytes(range),2*128*4u);
    std::vector<float> archived(2*128,-1.0f);
    f.owner->exportBlock(range,{archived.data(),archived.size()*4,CacheAuxiliaryMemoryDomain::Host},nullptr);
    EXPECT_EQ(std::memcmp(archived.data(),f.pools.data(),archived.size()*4),0);
    const auto endpointBytes=f.owner->endpointArchiveBytes(0,7);
    EXPECT_EQ(endpointBytes,3*128*4u+8+9*8*4u);
    std::vector<std::uint32_t> endpoint(endpointBytes/4);
    f.owner->exportEndpoint(0,7,{endpoint.data(),endpointBytes,CacheAuxiliaryMemoryDomain::Host},nullptr);
    std::fill_n(f.partials.mutable_data(),3*128,0.0f);tokens[0]=tokens[1]=0;
    f.owner->importEndpoint(0,7,{endpoint.data(),endpointBytes,CacheAuxiliaryMemoryDomain::Host},nullptr);
    EXPECT_EQ(std::memcmp(partial.data(),f.partials.data(),partial.size()*4),0);EXPECT_EQ(tokens[1],2147483646);
    EXPECT_EQ(f.cache->get_cached_tokens(0,0),7); // Archive/rollback never invents a second frontier.
    auto duplicate=f.owner->bindings();duplicate.sparseLayers[1].partialKeys=duplicate.sparseLayers[0].partialKeys;
    EXPECT_THROW(Qwen4ExpAuxiliaryState(std::move(duplicate)),std::invalid_argument);
}
/** @test Snapshot aliases exclude slot reuse and preserve accepted state independently of pool capacity. */
TEST(Qwen4ExpAuxiliaryState, HostRollbackPoolRetainsSnapshotAliasesAndRejectsForeignOwners)
{
    AuxiliaryFixture f,foreign;
    std::vector<float> kv(7*128,0.0f);
    ASSERT_TRUE(f.cache->importLogicalBlock({0,0,0,7,nullptr},kv.data(),kv.data()));
    ASSERT_TRUE(f.cache->importLogicalBlock({1,0,0,7,nullptr},kv.data(),kv.data()));
    std::fill_n(f.partials.mutable_data(),3*128,17.25f);
    std::fill_n(f.convolution.mutable_data(),9*8,19.5f);
    auto *tokens=static_cast<int *>(f.tokens.raw_mutable_data());tokens[0]=2147483646;tokens[1]=123456789;
    HostCacheAuxiliaryCheckpointPool pool;
    EXPECT_THROW(pool.capture(-1,0,*f.owner),std::logic_error);
    auto lease=std::make_shared<int>(1);
    const auto bytes=f.owner->checkpointCapacity(0);
    pool.add(*f.owner,lease,std::make_shared<std::vector<std::uint8_t>>(bytes,0xcd));
    pool.seal();
    EXPECT_THROW(pool.add(*f.owner,lease,std::make_shared<std::vector<std::uint8_t>>(bytes)),std::invalid_argument);
    EXPECT_THROW(pool.capture(-1,0,*foreign.owner),std::runtime_error);
    PrefixStateSnapshot original;
    original.host_auxiliary_checkpoints.push_back(pool.capture(-1,0,*f.owner));
    const auto *address=original.host_auxiliary_checkpoints[0].storage->data();
    EXPECT_EQ(*reinterpret_cast<const std::uint32_t *>(address),3u);
    EXPECT_EQ(*reinterpret_cast<const std::uint32_t *>(address+4),0xcdcdcdcdu);
    EXPECT_THROW(pool.capture(-1,0,*f.owner),std::runtime_error);
    EXPECT_THROW(original.host_auxiliary_checkpoints[0].validate(*foreign.owner,0),std::invalid_argument);
    EXPECT_THROW(original.host_auxiliary_checkpoints[0].validate(*f.owner,1),std::invalid_argument);
    auto alias=original;
    PrefixStateSnapshot moved=std::move(original);
    moved.host_auxiliary_checkpoints.clear();
    EXPECT_THROW(pool.capture(-1,0,*f.owner),std::runtime_error);
    std::fill_n(f.partials.mutable_data(),4*128,-1.0f);
    std::fill_n(f.convolution.mutable_data(),9*8,-2.0f);tokens[0]=tokens[1]=-1;
    ASSERT_TRUE(f.cache->importLogicalBlock({0,0,7,3,nullptr},kv.data(),kv.data()));
    ASSERT_TRUE(f.cache->importLogicalBlock({1,0,7,3,nullptr},kv.data(),kv.data()));
    ASSERT_TRUE(f.cache->truncateSequence(0,7,nullptr));
    alias.host_auxiliary_checkpoints[0].restore(*f.owner,0);
    for(int i=0;i<3*128;++i)EXPECT_EQ(f.partials.data()[i],17.25f);
    EXPECT_EQ(f.partials.data()[3*128],-1.0f);
    for(int i=0;i<9*8;++i)EXPECT_EQ(f.convolution.data()[i],19.5f);
    EXPECT_EQ(tokens[0],2147483646);EXPECT_EQ(tokens[1],123456789);
    EXPECT_EQ(f.pools.data()[0],-77.0f);
    alias.host_auxiliary_checkpoints.clear();
    auto reused=pool.capture(2,0,*f.owner);
    EXPECT_EQ(reused.storage->data(),address);EXPECT_EQ(reused.cache_depth,2);
}
/** @test Residual publication preserves the entire4H trunk and exact request row offsets. */
TEST(Qwen4ExpAuxiliaryState, ResidualRowsRejectHeadWidthNarrowing)
{
    const ResidualRowGeometry trunk(10240),head(2560);
    const std::array<std::size_t,2> source{15,10240},narrowed{15,2560};
    EXPECT_EQ(trunk.rowBytes(),40960u);
    EXPECT_EQ(trunk.elementOffset(source,14),14*10240u);
    EXPECT_THROW(trunk.validate(narrowed),std::invalid_argument);
    EXPECT_THROW(head.validate(source),std::invalid_argument);
    EXPECT_THROW(trunk.elementOffset(source,15),std::invalid_argument);
    EXPECT_THROW(ResidualRowGeometry(0),std::invalid_argument);
}
/** @test Full auxiliary archives retain only completed pools and compact endpoint bytes under PMA. */
TEST(Qwen4ExpAuxiliaryState, CompactArchiveRetainsPhysicalAuthorityAfterTransientIndexRetirement)
{
    AuxiliaryFixture f;
    PrefixPayloadLayout layout;layout.device=DeviceId::cpu();layout.fa_layers=1;
    applyPrefixAuxiliaryLayout(layout,*f.cache,nullptr,0,0,7,7,true);
    const auto bytes=layout.modelAuxiliaryBytes();
    ASSERT_EQ(bytes,128*4u+3*128*4u+8+9*8*4u);
    ASSERT_NE(bytes,f.owner->checkpointCapacity(0));
    PhysicalMemoryPlanBuilder plan;
    plan.add(PhysicalMemoryResource{.world_rank=0,.device=DeviceId::cpu(),
        .total_bytes=4096,.admission_available_bytes=4096},PhysicalMemoryOwner::PrefixHostTier,bytes);
    auto admission=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build());
    auto authority=std::make_shared<PhysicalMemoryAuthority>(std::move(admission),0);
    auto tier=RamPrefixStorageBackend::create(DeviceId::cpu(),bytes,authority);
    ASSERT_TRUE(tier);
    const auto key=makePrefixCacheKey(0x515341,0,0,0,{1,2,3,4,5,6,7});
    auto archive=tier->allocate(key,layout);ASSERT_TRUE(archive.valid());
    std::fill_n(f.pools.mutable_data(),128,12.5f);
    std::fill_n(f.partials.mutable_data(),3*128,13.5f);
    static_cast<int *>(f.tokens.raw_mutable_data())[0]=2147483646;
    exportPrefixAuxiliaryState(*f.cache,0,PrefixAuxiliaryLane::Main,archive,nullptr);
    ASSERT_TRUE(tier->release(archive));EXPECT_EQ(tier->usedBytes(),0u);
    EXPECT_FALSE(tier->canStore(bytes)); // Snapshot's retained physical lease remains authoritative.
    auto alias=archive;archive={};
    std::fill_n(f.pools.mutable_data(),4*128,-1.0f);
    std::fill_n(f.partials.mutable_data(),4*128,-2.0f);
    static_cast<int *>(f.tokens.raw_mutable_data())[0]=-1;
    importPrefixAuxiliaryState(*f.cache,0,PrefixAuxiliaryLane::Main,alias,nullptr);
    for(int i=0;i<128;++i)EXPECT_EQ(f.pools.data()[i],12.5f);
    EXPECT_EQ(f.pools.data()[128],-1.0f);
    for(int i=0;i<3*128;++i)EXPECT_EQ(f.partials.data()[i],13.5f);
    EXPECT_EQ(f.partials.data()[3*128],-2.0f);
    EXPECT_EQ(static_cast<const int *>(f.tokens.raw_data())[0],2147483646);
    alias={};EXPECT_TRUE(tier->canStore(bytes));
}
