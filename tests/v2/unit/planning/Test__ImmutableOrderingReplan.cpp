/**
 * @file Test__ImmutableOrderingReplan.cpp
 * @brief Device-free exact mandatory/optional mixed ngram admission and policy/wire invariants.
 *
 * The canonical original48 source fixture is metadata-only. Tests never load
 * model/native row payload, start a backend, allocate GPU storage or invent a
 * separate capacity resolver. All expected cache/frame sizes come from actual
 * source-owned geometry, and every capacity decision uses complete PMA plans.
 */
#include "planning/MemoryPlanner.h"
#include "fixtures/planning/ImmutableRowTableAdmissionWireReplayAccess.h"
#include "fixtures/planning/RecordedNgramOrdering.h"
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "kernels/qwen4exp/PooledImmutableNativeStorage.h"
#include "transfer/TransferEngine.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "memory/BufferArena.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
using namespace llaminar2;
namespace
{
/** @return Explicit bounded new mode, independent of the unchanged Automatic preference. */
ImmutableRowTablePolicy policy(std::uint64_t rows)
{
    ImmutableRowTablePolicy p;p.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    p.window_bytes=65536;p.window_count=2;
    p.gpu_adaptive={rows,1u<<20,{.aging_accesses=97,.protected_numerator=2,.protected_denominator=5},
        {rows,1u<<20,{.aging_accesses=113,.protected_numerator=1,.protected_denominator=3}}};
    return p;
}
/** @return One full-main actual GPU PLE consumer with exact canonical CPU resource observation. */
DevicePlanConfig consumer(DeviceType backend,std::uint64_t rows=64)
{
    DevicePlanConfig c;c.world_rank=0;c.device={backend,0};c.device_compute_units=64;
    c.device_total_bytes=512ull<<30;c.device_free_bytes=500ull<<30;
    c.associated_host_memory=PhysicalMemoryResource{0,DeviceId::cpu(),512ull<<30,500ull<<30};
    c.max_seq_len=64;c.activation_seq_len=32;c.first_layer=0;c.last_layer=47;c.batch_size=1;
    c.ngram_execution_policy=policy(rows);c.native_ordering_events=test::recordedNgramOrdering(c.device);test::declareRecordedNgramHostServices(c);return c;
}
/** @return Original native source words only; source/file authentication remains a separate real owner duty. */
ImmutableRowTableSource source(const ModelMemoryProfile &profile)
{
    const auto &g=*profile.flash_next;
    return {.rows=g.ple_table_rows,.row_elements=g.ple_row_elements,.format=static_cast<GGUFTensorType>(g.ple_table_gguf_type)};
}
/** @return Canonical immutable request after mandatory graph/source owners are declared. */
ImmutableRowTableAdmissionRequest request(const ModelMemoryProfile &profile,const std::vector<DevicePlanConfig> &configs)
{
    auto originalConfigs=configs;std::vector<DeviceId> originalDevices;
    for(const auto& cfg:configs)if(cfg.device.is_gpu() && std::find(originalDevices.begin(),originalDevices.end(),cfg.device)==originalDevices.end())originalDevices.push_back(cfg.device);
    const auto recorded=test::NativeOrderingEventGroupReplayAccess::projection(originalDevices);
    for(auto& cfg:originalConfigs)cfg.native_ordering_events=recorded;
    test::shareRecordedNgramHostServices(originalConfigs);
    const std::vector<std::vector<DevicePlanConfig>> groups{originalConfigs};
    auto q=MemoryPlanner::immutableRowTableAdmissionRequest(profile,*configs.front().ngram_execution_policy,groups,{},
        configs.front().ngram_host_placement);
    if(!q)throw std::logic_error("Metadata source omitted its declared GPU consumer");
    return std::move(*q);
}
/** @return Exact existing owner line, or zero for an absent optional owner. */
std::size_t bytes(const PhysicalMemoryPlan &plan,PhysicalMemoryAllocatorIdentity id,PhysicalMemoryOwner owner)
{const auto *bom=plan.find(id);return bom?bom->bytes(owner):0;}
}

TEST(ImmutableOrderingReplanLocal, MandatoryOwnersAndFinalOptionalSlotsUseOneCompletePMAPlan)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());
    const auto &profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    {
        auto c=consumer(backend);const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const auto q=request(profile,{c});ASSERT_TRUE(q.execution_provision);
        // FullModel is the existing typed grant for EVERY original routed
        // expert plus dense/shared weights. KV and graph rows are also frozen
        // before the optional cache solve; no zero-weight synthetic base is used.
        EXPECT_EQ(c.weight_residency.kind(),DeviceWeightResidency::Kind::FullModel);
        EXPECT_EQ(profile.expert_count,512);
        EXPECT_GT(bytes(frozen,{0,c.device},PhysicalMemoryOwner::PrimaryModelWeights),0u);
        EXPECT_GT(bytes(frozen,{0,c.device},PhysicalMemoryOwner::KVCache),0u);
        const auto zero=AdaptiveImmutableLookupGeometry::fromLayout(q.input.gpuAdaptiveLayout(q.requested,0,0,
            q.execution_provision->maximumLiveIds()),source(profile),q.requested.gpu_adaptive);
        const auto mandatory=llaminar2::test::recordedNgramMandatory(zero,q.input.consumers().front(),q.input.host());
        for(const auto &bom:mandatory.resources())
        {
            const PhysicalMemoryAllocatorIdentity id{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(frozen,id,PhysicalMemoryOwner::ImmutableLookupStaging),bom.bytes(PhysicalMemoryOwner::ImmutableLookupStaging));
            EXPECT_EQ(bytes(frozen,id,PhysicalMemoryOwner::ImmutableTableStorage),bom.bytes(PhysicalMemoryOwner::ImmutableTableStorage));
        }
        const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
        ASSERT_TRUE(selected.placement->isGPUAdaptive());EXPECT_TRUE(selected.physical_plan.fits());
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),source(profile),q.requested.gpu_adaptive);
        EXPECT_EQ(geometry.physical().owners().front().slots,64u);EXPECT_EQ(geometry.host().slots(),64u);
        const auto optional=llaminar2::test::recordedNgramOptional(geometry,q.input.consumers().front(),q.input.host());
        for(const auto &bom:frozen.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);const PhysicalMemoryAllocatorIdentity id{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(selected.physical_plan,id,owner),bom.bytes(owner)+bytes(optional,id,owner));
        }
        auto resolved=c;resolved.ngram_table_placement=selected.placement;
        const auto projected=MemoryPlanner::plan(profile,{resolved}).physicalPlan();
        EXPECT_FALSE(projected.requiredFootprintMismatch(selected.physical_plan));
        EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(projected));
    }
}

TEST(ImmutableOrderingReplanLocal, NgramCanonicalHostPreservesRanklessPlanningIdentity){
    const PhysicalMemoryResource host{-1,DeviceId::cpu(),8u<<20,8u<<20},gpu{-1,DeviceId::cuda(0),4u<<20,4u<<20};
    const std::array consumers{gpu};
    const auto input=ImmutableRowTablePlacementInput::forConsumers(ImmutableRowTableExtent::fromNative(GGUFTensorType::F32,1024,16),host,consumers,ImmutableRowTableHostPlacement::onNode(1));
    EXPECT_EQ(input.host().device,DeviceId::cpu());EXPECT_EQ(input.host().world_rank,-1);EXPECT_EQ(input.hostPlacement(),ImmutableRowTableHostPlacement::onNode(1));
    auto invented=host;invented.device=DeviceId{DeviceType::CPU,1};
    EXPECT_THROW((void)ImmutableRowTablePlacementInput::forConsumers(input.extent(),invented,consumers),std::invalid_argument);
}
