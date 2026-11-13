/**
 * @file Test__QueryKVGeometryObservation.cpp
 * @brief Device-free actual admission-input and cache-interface observation guards.
 *
 * The cache is authored metadata, not hardware/storage evidence. It throws if
 * a producer reads token state or invokes execution. Native positive cache
 * qualification remains in the actual integration/model gate.
 */
#include <gtest/gtest.h>
#include "execution/diagnostics/QueryKVGeometryObservation.h"
#include <memory>
#include <stdexcept>
namespace llaminar2::test
{
namespace
{
/** @brief Cache metadata fixture; no live state operation is legal. */
class MetadataCache final : public IKVCache
{
public:
    int first=1,count=1;
    /** @copydoc IKVCache::k_precision */
    ActivationPrecision k_precision()const override{return ActivationPrecision::FP32;}
    /** @copydoc IKVCache::get_cached_tokens */
    int get_cached_tokens(int,int=0)const override{throw std::logic_error("live token read forbidden");}
    /** @copydoc IKVCache::max_seq_len */
    int max_seq_len()const override{return 64;}
    /** @copydoc IKVCache::n_layers */
    int n_layers()const override{return 1;}
    /** @copydoc IKVCache::local_n_kv_heads */
    int local_n_kv_heads()const override{return count;}
    /** @copydoc IKVCache::kv_head_start */
    int kv_head_start()const override{return first;}
    /** @copydoc IKVCache::logicalBlockLayout */
    KVCacheLogicalBlockLayout logicalBlockLayout(int,int)const override
    {return {.local_kv_heads=count,.kv_head_start=first,.head_dim=4,.k_bytes=16,.v_bytes=16};}
    /** @copydoc IKVCache::get_kv */
    bool get_kv(int,int,ITensor **,ITensor **,int * =nullptr)override{throw std::logic_error("payload read forbidden");}
    /** @copydoc IKVCache::get_kv */
    bool get_kv(int,int,const ITensor **,const ITensor **,int * =nullptr)const override{throw std::logic_error("payload read forbidden");}
    /** @copydoc IKVCache::append */
    bool append(int,int,const ITensor *,const ITensor *,int)override{throw std::logic_error("execution forbidden");}
    /** @copydoc IKVCache::resetRequestState */
    bool resetRequestState(const StateResetContext &)override{return false;}
    /** @copydoc IKVCache::resetSequenceState */
    bool resetSequenceState(int,const StateResetContext &)override{return false;}
    /** @copydoc IKVCache::resetLayerSequenceState */
    bool resetLayerSequenceState(int,int,const StateResetContext &)override{return false;}
    /** @copydoc IKVCache::resetLayerState */
    bool resetLayerState(int,const StateResetContext &)override{return false;}
};
/** @return Actual pure canonical nonordinal assignment plus local graph coordinates. */
GraphConfig config()
{
    GraphConfig result;
    result.tp_config=std::make_shared<TensorParallelConfig>(TensorParallelConfig::equalSplit(4,24,2,640,248320,
        std::vector<DeviceId>{DeviceId::rocm(19),DeviceId::rocm(7),DeviceId::rocm(11),DeviceId::rocm(3)},
        KVHeadCoveragePolicy::QueryInterval));
    result.d_model=512;result.n_heads=24;result.n_kv_heads=2;result.head_dim=4;result.tp_device_idx=2;result.local_rank=2;
    const auto &a=result.tp_config->forRank(2);
    result.head_start=a.head_start;result.local_n_heads=a.head_count;result.local_n_kv_heads=a.kv_head_count;
    return result;
}
/** @return Synthetic certificate; bytes are authored, never inferred from head count. */
std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> admission(DeviceId device)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add({.world_rank=0,.device=device,.total_bytes=128,.admission_available_bytes=128},
        PhysicalMemoryOwner::KVCache,128);
    return std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build());
}
}
/** @test Selected input uses exact certificate bytes and rejects foreign or absent physical input. */
TEST(QueryKVGeometryObservation, InstalledInputUsesActualAssignmentAndCertificate)
{
    const auto graph=config();const auto device=DeviceId::rocm(11);const auto proof=admission(device);
    DevicePlanConfig actual;actual.world_rank=0;actual.device=device;actual.shard_index=2;actual.total_shards=4;
    actual.bindTensorParallelAssignment(graph.tp_config->forRank(2));
    ModelMemoryProfile source;source.d_model=512;source.n_heads=24;source.n_kv_heads=2;source.head_dim=4;
    const auto tags=query_kv_geometry::installedAssignmentTags(source,actual,*proof);
    EXPECT_EQ(tags.at("kv_first"),"1");EXPECT_EQ(tags.at("kv_count"),"1");
    EXPECT_EQ(tags.at("kv_planned_bytes"),"128");
    EXPECT_EQ(tags.at("admission_identity"),query_kv_geometry::admissionIdentity(proof.get()));
    actual.tensor_parallel_assignment->kv_head_start=0;
    EXPECT_THROW(query_kv_geometry::installedAssignmentTags(source,actual,*proof),std::invalid_argument);
    actual.tensor_parallel_assignment=graph.tp_config->forRank(2);
    actual.device=DeviceId::rocm(3);
    EXPECT_THROW(query_kv_geometry::installedAssignmentTags(source,actual,*proof),std::invalid_argument);
    actual.device=device;actual.tensor_parallel_assignment.reset();
    EXPECT_THROW(query_kv_geometry::installedAssignmentTags(source,actual,*proof),std::invalid_argument);
}
/** @test Actual cache-interface getters, lease and coherent PMA are used without token/payload reads. */
TEST(QueryKVGeometryObservation, CacheTupleRequiresGeometryAndLeaseWithoutLiveReads)
{
    auto graph=config();const auto device=DeviceId::rocm(11);auto proof=admission(device);
    PhysicalMemoryAuthority owner(proof,0);MetadataCache cache;
    EXPECT_THROW(query_kv_geometry::cacheTags(graph,device,cache,owner),std::invalid_argument);
    // Synthetic pure ledger claim; this supplies no hardware/storage qualification.
    cache.bindPhysicalMemoryLease(std::make_shared<PhysicalMemoryAllocationLease>(
        owner.claimNewAllocation(device,PhysicalMemoryOwner::KVCache,128)));
    const auto tags=query_kv_geometry::cacheTags(graph,device,cache,owner);
    EXPECT_EQ(tags.at("cache_first"),"1");EXPECT_EQ(tags.at("payload_first"),"1");
    EXPECT_EQ(tags.at("kv_materialized_bytes"),"128");EXPECT_EQ(tags.at("per_instance_replay_proven"),"false");
    cache.first=0;EXPECT_THROW(query_kv_geometry::cacheTags(graph,device,cache,owner),std::invalid_argument);
    cache.first=1;graph.tp_config=std::make_shared<TensorParallelConfig>(TensorParallelConfig::equalSplit(4,24,2,640,248320));
    EXPECT_THROW(query_kv_geometry::cacheTags(graph,device,cache,owner),std::invalid_argument);
}
} // namespace llaminar2::test
