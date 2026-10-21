/**
 * @file Test__Qwen4ExpPrefixCapacity.cpp
 * @brief Device-free exact compact-prefix slot bounds across QFN range residues.
 *
 * A brute-force immutable-record oracle checks the constant-size production
 * residue search. These tests instantiate CPU/CUDA/ROCm metadata only and
 * prove that retained capacity never changes a record's live transfer extent.
 */
#include "execution/prefix_cache/PrefixPayloadLayout.h"
#include "execution/config/RuntimeConfig.h"
#include "planning/MemoryPlanner.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "execution/qwen4exp/Qwen4ExpAuxiliaryState.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "../../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>

using namespace llaminar2;
using Geometry = qwen4exp::Qwen4ExpArchiveGeometry;

namespace
{
/** @brief Independent literal equation for one lane's actual compact payload. */
size_t compactBytes(Geometry geometry, int first, int count, int prefix)
{
    const size_t pools = (first + count) / 4 - first / 4;
    return geometry.sparse_layers * (pools + prefix % 4) * 128 * sizeof(float) +
        (geometry.ple_channels ? 2 * sizeof(int32_t) + 9 * geometry.ple_channels * sizeof(float) : 0);
}
/** @brief Common section inventory; no table/arena/backend is allocated. */
PrefixPayloadLayout baseLayout(DeviceId device, int block, Geometry main, bool mtp)
{
    PrefixPayloadLayout base;
    base.device = device;
    base.block_size = block;
    base.fa_layers = main.sparse_layers;
    base.bytes_per_fa_layer_k = 7;
    base.bytes_per_fa_layer_v = 11;
    base.hybrid_state_bytes = 97;
    base.includes_hybrid_state = true;
    base.mtp_layers = mtp ? 1 : 0;
    base.mtp_kv_bytes = mtp ? 23 : 0;
    base.includes_mtp_state = mtp;
    base.terminal_hidden_bytes = mtp ? 10240 * sizeof(float) : 0;
    base.includes_terminal_hidden = mtp;
    base.terminal_logits_bytes = 13;
    base.includes_terminal_logits = true;
    return base;
}
} // namespace

/** @test All supported range residues and PP owners resolve the exact maximum actual record. */
TEST(Qwen4ExpPrefixCapacity, ConstantSizeSearchEqualsAllActualOrdinaryAndShiftedRecords)
{
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        for (const Geometry main : {Geometry{}, Geometry{.ple_channels=10240},
                                   Geometry{.sparse_layers=1}, Geometry{12,10240}})
            for (const bool mtp : {false, true})
                for (const int block : {1,2,3,4,5,6,7,8,15,16,17,63,64,65})
                    for (const int context : {1,2,3,4,5,7,9,15,16,17,31,63,64,65,127,128,129})
                    {
                        SCOPED_TRACE(device.toString()+":"+std::to_string(main.sparse_layers)+":"+
                            std::to_string(main.ple_channels)+":"+std::to_string(mtp)+":"+
                            std::to_string(block)+":"+std::to_string(context));
                        const auto base = baseLayout(device, block, main, mtp);
                        const Geometry shifted{.sparse_layers=mtp?1u:0u};
                        size_t expected = 0;
                        for (int first=0;first<context;first+=block)
                            for (int count=1;count<=std::min(block,context-first);++count)
                            {
                                const int prefix=first+count;
                                const size_t actual=compactBytes(main,first,count,prefix)+
                                    compactBytes(shifted,first?first-1:0,first?count:count-1,prefix-1);
                                expected=std::max(expected,actual);
                            }
                        const auto maximum=qwen4ExpPrefixCapacityLayout(base,main,shifted,context);
                        EXPECT_EQ(maximum.totalBytes(),base.totalBytes()+expected);
                        EXPECT_EQ(maximum.modelAuxiliaryBytes(),expected);
                        EXPECT_EQ(base.modelAuxiliaryBytes(),0u) << "Capacity selection must not mutate live record layouts";
                    }
}

/** @test Partial main endpoints and later shifted pools require more space than the first full block. */
TEST(Qwen4ExpPrefixCapacity, PartialMainAndLaterShiftedBlocksFitOneCanonicalSlab)
{
    const Geometry main{12,10240},mtp{1,0};
    const auto base=baseLayout(DeviceId::cuda(0),64,main,true);
    const auto maximum=qwen4ExpPrefixCapacityLayout(base,main,mtp,512);
    const auto first_full=compactBytes(main,0,64,64)+compactBytes(mtp,0,63,63);
    const auto later_partial=compactBytes(main,64,63,127)+compactBytes(mtp,63,63,126);
    EXPECT_GT(later_partial,first_full);
    EXPECT_EQ(maximum.modelAuxiliaryBytes(),later_partial);
    EXPECT_EQ(maximum.modelAuxiliaryBytes(),488456u);
    const auto slot=maximum.totalBytes();
    const auto budget=slot*3+slot-1;
    EXPECT_EQ(prefixCacheWholeBlockReservationBytes(budget,slot),slot*3);
    // A short record still transfers only its real pool rows/tail/history.
    EXPECT_LT(compactBytes(main,0,9,9)+compactBytes(mtp,0,8,8),maximum.modelAuxiliaryBytes());
}

/** @test Invalid geometry and byte overflow cannot become a small admitted slot. */
TEST(Qwen4ExpPrefixCapacity, RejectsForeignOwnershipAndOverflow)
{
    auto base=baseLayout(DeviceId::cpu(),64,{1,0},false);
    EXPECT_THROW((void)qwen4ExpPrefixCapacityLayout(base,{2,0},{},512),std::invalid_argument);
    EXPECT_THROW((void)qwen4ExpPrefixCapacityLayout(base,{1,0},{0,10240},512),std::invalid_argument);
    EXPECT_THROW((void)qwen4ExpPrefixCapacityLayout(base,{1,0},{},0),std::invalid_argument);
    EXPECT_THROW((void)Geometry::blockBytes(std::numeric_limits<size_t>::max(),1),std::overflow_error);
    const Geometry enormous{std::numeric_limits<size_t>::max(),0};
    EXPECT_THROW((void)enormous.endpointBytes(3),std::overflow_error);
    base.terminal_hidden_bytes=std::numeric_limits<size_t>::max();
    base.includes_terminal_hidden=true;
    EXPECT_THROW((void)base.totalBytes(),std::overflow_error);
}

/** @test Actual metadata-only planner BOM and runtime layout reserve the identical device-hot slab. */
TEST(Qwen4ExpPrefixCapacity, PlannerAdmissionUsesTheSameParticipantLocalCapacityLayout)
{
    auto profile=ModelMemoryProfile::fromGGUF(test::qwen4MetadataFixture(false));
    // Weight residency is orthogonal to this allocation contract. Keep exact
    // model geometry while avoiding unrelated synthetic weight-format inputs.
    profile.tensors.clear();
    profile.total_native_bytes=0;
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
        for(const auto interval:{std::pair{0,1},std::pair{3,3},std::pair{40,47},std::pair{0,47}})
        {
            SCOPED_TRACE(device.toString()+":"+std::to_string(interval.first)+":"+std::to_string(interval.second));
            DevicePlanConfig cfg;
            cfg.device=device;
            cfg.device_compute_units=60;
            cfg.device_total_bytes=cfg.device_free_bytes=64ull<<30;
            cfg.associated_host_memory=PhysicalMemoryResource{.world_rank=-1,.device=DeviceId::cpu(),
                .total_bytes=64ull<<30,.admission_available_bytes=64ull<<30};
            cfg.first_layer=interval.first;
            cfg.last_layer=interval.second;
            cfg.max_seq_len=512;
            cfg.activation_seq_len=16;
            cfg.kv_precision="fp16";
            cfg.prefix_cache.enabled=true;
            cfg.prefix_cache.storage_mode=PrefixCacheStorageMode::Tiered;
            cfg.prefix_cache.block_size=64;
            cfg.prefix_cache.ram_budget_bytes=256u<<20;
            cfg.prefix_cache.device_budget_bytes=(512u<<20)+1234;
            cfg.prefix_cache.disk_budget_bytes=0;
            const auto state=PersistentStateMemoryEstimator::estimate(profile,device,1,512,2,2,0,24,
                interval.first,interval.second,"fp16",MTPStateRole::Disabled);
            const auto kv=KVCacheMemoryEstimator::estimateGPULogicalBlock(state.main_kv_family,64,2,256,"fp16",device);
            PrefixPayloadLayout base;
            base.device=device;
            base.block_size=64;
            base.fa_layers=state.main_full_attention_layers;
            base.bytes_per_fa_layer_k=kv.k_bytes;
            base.bytes_per_fa_layer_v=kv.v_bytes;
            base.hybrid_state_bytes=state.prefix_hybrid_device_state_bytes;
            base.includes_hybrid_state=base.hybrid_state_bytes>0;
            base.terminal_logits_bytes=interval.second==47?248320u*sizeof(float):0;
            base.includes_terminal_logits=base.terminal_logits_bytes>0;
            const Geometry geometry{static_cast<size_t>(state.main_full_attention_layers),
                interval.first<=1 && interval.second>=1?10240u:0u};
            const auto capacity=qwen4ExpPrefixCapacityLayout(base,geometry,{},512);
            const auto plan=MemoryPlanner::plan(profile,{cfg});
            const auto *device_plan=plan.physicalPlan().find({-1,device});
            ASSERT_NE(device_plan,nullptr);
            EXPECT_EQ(device_plan->bytes(PhysicalMemoryOwner::PrefixDeviceTier),
                prefixCacheWholeBlockReservationBytes(cfg.prefix_cache.device_budget_bytes,capacity.totalBytes()));
        }
}

/** @test Dense shifted KV keeps its complete base payload without inventing sparse or lexical sections. */
TEST(Qwen4ExpPrefixCapacity, DenseShiftedKVHasNoAuxiliaryOwner)
{
    const Geometry main{12,10240};
    for (const auto device : {DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        for (const int block : {1,3,4,15,64,65})
            for (const int context : {1,3,16,65,129})
            {
                SCOPED_TRACE(device.toString()+":"+std::to_string(block)+":"+std::to_string(context));
                auto base=baseLayout(device,block,main,true);
                base.mtp_origin=MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero;
                size_t expected=0;
                for(int first=0;first<context;first+=block)
                    for(int count=1;count<=std::min(block,context-first);++count)
                        expected=std::max(expected,compactBytes(main,first,count,first+count));
                const auto capacity=qwen4ExpPrefixCapacityLayout(base,main,{},context);
                EXPECT_EQ(capacity.mtp_layers,1);
                EXPECT_EQ(capacity.mtpKVBytes(),base.mtpKVBytes());
                EXPECT_TRUE(capacity.includes_mtp_state);
                EXPECT_FALSE(capacity.mtp_auxiliary.kind.has_value());
                EXPECT_EQ(capacity.mtp_auxiliary.blockBytes(1),0u);
                EXPECT_EQ(capacity.mtp_auxiliary.endpointBytes(),0u);
                EXPECT_EQ(capacity.modelAuxiliaryBytes(),expected);
                EXPECT_EQ(capacity.totalBytes(),base.totalBytes()+expected);
                // Empty auxiliary is a distinct supported descriptor; an
                // authored nonempty sparse bank must still match exactly.
                EXPECT_THROW((void)qwen4ExpPrefixCapacityLayout(base,main,{2,0},context),std::invalid_argument);
                EXPECT_THROW((void)qwen4ExpPrefixCapacityLayout(base,main,{0,10240},context),std::invalid_argument);
            }
}

/** @test The runtime family publisher observes a real main auxiliary owner and real dense shifted cache. */
TEST(Qwen4ExpPrefixCapacity, RuntimePublisherPreservesDenseShiftedKVWithoutAuxiliaryState)
{
    MPIContext mpi{0,1,MPI_COMM_WORLD};
    auto main=createCPURingKVCache(ActivationPrecision::FP32,mpi,1,1,16,1,128,DeviceId::cpu());
    auto shifted=createCPURingKVCache(ActivationPrecision::FP32,mpi,1,1,16,1,128,DeviceId::cpu());
    shifted->bindStateOwnership({.role=IKVCache::StateRole::MTPShiftedSidecar,.mtp_depth=0,
        .shifted_origin=MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero});
    FP32Tensor pools{{4,128}},partial{{4,128}};
    Qwen4ExpAuxiliaryState::Params params{.cache=*main,.device=DeviceId::cpu(),.requestCapacity=1,.eosToken=42};
    params.sparseLayers.push_back({0,0,0,16,
        {pools,BufferId::QFN_QSA_POOLED_KEYS,0},{partial,BufferId::QFN_QSA_PARTIAL_KEYS,0}});
    main->attachAuxiliaryState(std::make_unique<Qwen4ExpAuxiliaryState>(std::move(params)),nullptr);
    ASSERT_NE(main->auxiliaryState(),nullptr);
    ASSERT_EQ(shifted->auxiliaryState(),nullptr);
    auto base=baseLayout(DeviceId::cpu(),4,{1,0},true);
    base.mtp_origin=shifted->stateOwnership().shifted_origin;
    const auto published=prefixPayloadCapacityLayout(base,*main,shifted.get());
    const auto expected=qwen4ExpPrefixCapacityLayout(base,{1,0},{},16);
    EXPECT_EQ(published.totalBytes(),expected.totalBytes());
    EXPECT_EQ(published.main_auxiliary,expected.main_auxiliary);
    EXPECT_EQ(published.mtp_layers,1);
    EXPECT_EQ(published.mtp_kv_bytes,base.mtp_kv_bytes);
    EXPECT_FALSE(published.mtp_auxiliary.kind.has_value());
    EXPECT_EQ(shifted->auxiliaryState(),nullptr);
}
