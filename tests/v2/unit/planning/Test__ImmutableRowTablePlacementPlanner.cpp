/**
 * @file Test__ImmutableRowTablePlacementPlanner.cpp
 * @brief Device-free total-priority and exact-consumer table admission regressions.
 *
 * Real immutable PMA plans prove lowest-priority selection, with no backend,
 * allocation, source payload or alternate live ledger. Every explicit preference
 * is a single candidate; Automatic cannot alter the frozen higher-priority BOM.
 */
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "loaders/ImmutableRowTable.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
namespace llaminar2::test
{
namespace
{
constexpr std::size_t GiB = std::size_t{1} << 30;
/** @return An exact observed physical resource, independent of NUMA affinity. */
PhysicalMemoryResource resource(DeviceId device, std::size_t available = GiB)
{
    return {3, device, GiB, available};
}
/** @return A large native extent with cheap bounded mmap compared with full residency. */
ImmutableRowTableExtent extent()
{
    return ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0), 131072u, 4096u);
}
/** @return A concrete host policy using exactly one page-sized mapping window. */
ImmutableRowTablePolicy request(ImmutableRowStoragePreference preference = ImmutableRowStoragePreference::Automatic)
{
    return {preference, 65536u, 1u};
}
/** @return Synthetic checked eligibility with no incremental staging allocation. */
ImmutableRowTableModeContributions eligible()
{
    return {.vram=PhysicalMemoryPlan{}};
}
/** @return Fitting base with one explicitly frozen owner on each actual consumer. */
PhysicalMemoryPlan base(std::span<const PhysicalMemoryResource> consumers,
    PhysicalMemoryOwner owner = PhysicalMemoryOwner::PrimaryModelWeights, std::size_t bytes = 1u)
{
    PhysicalMemoryPlanBuilder builder;
    for (const auto &r : consumers) builder.add(r, owner, bytes);
    return builder.build();
}
/** @return Checked rank-local source/consumer input with canonical CPU0 host authority. */
ImmutableRowTablePlacementInput input(std::span<const PhysicalMemoryResource> consumers,
    std::size_t host_available = GiB)
{
    return ImmutableRowTablePlacementInput::forConsumers(extent(),
        resource(DeviceId::cpu(), host_available), consumers);
}
}
/** @test Every other owner wins capacity before the lowest-priority resident table. */
TEST(ImmutableRowTablePlacementPlanner, EveryHigherPriorityOwnerIsFrozenBeforeAutomaticSelection)
{
    const std::array consumers{resource(DeviceId::cuda(0))};
    for (std::size_t i = 0; i < PhysicalMemoryBOM::ownerCount(); ++i)
    {
        const auto owner = PhysicalMemoryBOM::ownerAt(i);
        if (owner == PhysicalMemoryOwner::ImmutableTableStorage || owner == PhysicalMemoryOwner::ImmutableLookupStaging) continue;
        const auto frozen = base(consumers, owner, GiB - extent().nativeBytes() + 1u);
        const auto result = ImmutableRowTablePlacementPlanner::select(request(), input(consumers), frozen, eligible());
        ASSERT_EQ(result.placement->storage(), ImmutableRowStorage::RAM) << toString(owner);
        const auto *actual = result.physical_plan.find({3, consumers.front().device});
        ASSERT_NE(actual, nullptr);
        EXPECT_EQ(actual->charge(owner).planned_bytes, frozen.find({3, consumers.front().device})->bytes(owner));
        EXPECT_EQ(actual->bytes(PhysicalMemoryOwner::ImmutableTableStorage), 0u);
    }
}
/** @test Every actual GPU gets one resident native copy; another GPU's free space cannot replace it. */
TEST(ImmutableRowTablePlacementPlanner, VRAMUsesEachActualConsumerAndOneTightConsumerRejectsWholeTier)
{
    const std::array consumers{resource(DeviceId::cuda(0)), resource(DeviceId::rocm(1))};
    const auto frozen = base(consumers);
    const auto selected = ImmutableRowTablePlacementPlanner::select(request(), input(consumers), frozen, eligible());
    ASSERT_EQ(selected.placement->storage(), ImmutableRowStorage::VRAM);
    ASSERT_EQ(selected.placement->allocators().size(), 2u);
    EXPECT_EQ(selected.placement->consumers().size(), 2u);
    for (const auto &r : consumers)
    {
        EXPECT_EQ(selected.placement->storageForConsumer({3, r.device}), ImmutableRowStorage::VRAM);
        ASSERT_NE(selected.physical_plan.find({3, r.device}), nullptr);
        EXPECT_EQ(selected.physical_plan.find({3, r.device})->bytes(PhysicalMemoryOwner::ImmutableTableStorage), extent().nativeBytes());
    }
    EXPECT_THROW(selected.placement->storageForConsumer({3, DeviceId::cuda(1)}), std::invalid_argument);
    auto tight = consumers;
    tight[1].admission_available_bytes = extent().nativeBytes(); // primary byte makes this candidate fail.
    const auto hosted = ImmutableRowTablePlacementPlanner::select(request(), input(tight), base(tight), eligible());
    EXPECT_EQ(hosted.placement->storage(), ImmutableRowStorage::RAM);
    ASSERT_EQ(hosted.placement->allocators().size(), 1u);
    EXPECT_EQ(hosted.placement->allocators().front().device, (DeviceId::cpu()));
    EXPECT_EQ(hosted.placement->storageForConsumer({3, DeviceId::rocm(1)}), ImmutableRowStorage::RAM);
}
/** @test CPU consumers exclude VRAM; insufficient shared RAM selects bounded mmap without changing the base. */
TEST(ImmutableRowTablePlacementPlanner, CPUConsumerAndExactHostCapacityChooseRAMThenMmap)
{
    const std::array consumers{resource(DeviceId::cuda(0)), resource(DeviceId::cpu(), 1u << 20)};
    const auto frozen = base(consumers);
    const auto selected = ImmutableRowTablePlacementPlanner::select(request(), input(consumers, 1u << 20), frozen, eligible());
    EXPECT_EQ(selected.placement->storage(), ImmutableRowStorage::MemoryMapped);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::VRAM),
        input(consumers, 1u << 20), frozen, eligible()), std::invalid_argument);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::RAM),
        input(consumers, 1u << 20), frozen, eligible()), PhysicalMemoryCapacityExhausted);
    const auto *host = selected.physical_plan.find({3, DeviceId::cpu()});
    ASSERT_NE(host, nullptr);
    EXPECT_EQ(host->bytes(PhysicalMemoryOwner::ImmutableTableStorage),
        ImmutableRowTable::physicalBytesForNativeExtent(extent().nativeBytes(),
            ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,65536,1)));
}
/** @test Explicit VRAM never retries a fitting RAM candidate after complete resident admission fails. */
TEST(ImmutableRowTablePlacementPlanner, ExplicitCandidateCannotRetryOrShrinkAnyHigherPriorityOwner)
{
    const std::array consumers{resource(DeviceId::cuda(0),extent().nativeBytes())};
    const auto frozen = base(consumers);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::VRAM),
        input(consumers), frozen, eligible()), PhysicalMemoryCapacityExhausted);
    EXPECT_EQ(frozen.find({3,DeviceId::cuda(0)})->bytes(PhysicalMemoryOwner::PrimaryModelWeights),1u);
    EXPECT_EQ(ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::RAM),
        input(consumers), frozen, eligible()).placement->storage(), ImmutableRowStorage::RAM);
}
/** @test Mapped channels, protocol arenas and bounded upload staging participate in the actual candidate fit. */
TEST(ImmutableRowTablePlacementPlanner, ExactModeOwnersParticipateAndNeverLeakAcrossCandidates)
{
    const std::array consumers{resource(DeviceId::cuda(0),extent().nativeBytes()+2u)};
    const auto in=input(consumers);
    PhysicalMemoryPlanBuilder upload;
    upload.add(consumers[0],PhysicalMemoryOwner::WeightLoadStaging,2u);
    PhysicalMemoryPlanBuilder hosted;
    hosted.add(consumers[0],PhysicalMemoryOwner::ActivationArena,7u)
        .add(consumers[0],PhysicalMemoryOwner::NativeGraphExecutable,11u)
        .add(in.host(),PhysicalMemoryOwner::ImmutableLookupStaging,13u);
    const auto result=ImmutableRowTablePlacementPlanner::select(request(),in,base(consumers),
        {upload.build(),hosted.build(),hosted.build()});
    ASSERT_EQ(result.placement->storage(),ImmutableRowStorage::RAM);
    EXPECT_EQ(result.physical_plan.find({3,consumers[0].device})->bytes(PhysicalMemoryOwner::WeightLoadStaging),0u);
    EXPECT_EQ(result.physical_plan.find({3,consumers[0].device})->bytes(PhysicalMemoryOwner::ActivationArena),7u);
    EXPECT_EQ(result.physical_plan.find({3,in.host().device})->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),13u);
}
/** @test Retained credits remain under the same PMA arithmetic rather than another remaining-capacity calculation. */
TEST(ImmutableRowTablePlacementPlanner, CanonicalRetainedCreditAndOverflowStayAuthoritative)
{
    const std::array consumers{resource(DeviceId::cuda(0),extent().nativeBytes()+1u)};
    PhysicalMemoryPlanBuilder retained;
    retained.add(consumers[0],PhysicalMemoryOwner::PrimaryModelWeights,4097u,4096u);
    const auto selected=ImmutableRowTablePlacementPlanner::select(request(),input(consumers),retained.build(),eligible());
    EXPECT_EQ(selected.placement->storage(),ImmutableRowStorage::VRAM);
    EXPECT_EQ(selected.physical_plan.find({3,consumers[0].device})->alreadyResidentBytes(PhysicalMemoryOwner::PrimaryModelWeights),4096u);
    EXPECT_THROW(ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),
        std::numeric_limits<std::uint64_t>::max(),2u),std::overflow_error);
    const PhysicalMemoryResource enormous{3,DeviceId::cuda(0),std::numeric_limits<std::size_t>::max(),std::numeric_limits<std::size_t>::max()};
    const std::array huge{enormous};
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(),input(huge),
        base(huge,PhysicalMemoryOwner::PrimaryModelWeights,std::numeric_limits<std::size_t>::max()),eligible()),std::overflow_error);
}
/** @test Explicit mmap never tries a fitting resident tier; total Automatic exhaustion is fatal. */
TEST(ImmutableRowTablePlacementPlanner, ExplicitMmapAndTotalExhaustionDoNotRetryAtMaterialization)
{
    const std::array consumers{resource(DeviceId::cuda(0))};
    const auto frozen=base(consumers);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::MemoryMapped),
        input(consumers,1u),frozen,eligible()),PhysicalMemoryCapacityExhausted);
    EXPECT_EQ(ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::VRAM),
        input(consumers,1u),frozen,eligible()).placement->storage(),ImmutableRowStorage::VRAM);
    const std::array full{resource(DeviceId::cuda(0),1u)};
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(),input(full,1u),base(full),eligible()),PhysicalMemoryCapacityExhausted);
}
/** @test Missing original upload geometry is setup ineligibility, distinct from VRAM capacity rejection. */
TEST(ImmutableRowTablePlacementPlanner, MissingOriginalUploadSlotSkipsOnlyAutomaticVRAM)
{
    const std::array consumers{resource(DeviceId::cuda(0))};
    const auto frozen=base(consumers);
    EXPECT_EQ(ImmutableRowTablePlacementPlanner::select(request(),input(consumers),frozen,{}).placement->storage(),ImmutableRowStorage::RAM);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::VRAM),
        input(consumers),frozen,{}),std::invalid_argument);
}
/** @test Unsupported identities, mixed accounting and missing consumers fail before any materialization. */
TEST(ImmutableRowTablePlacementPlanner, InvalidOwnershipAndMixedBaseAreRejected)
{
    const std::array consumers{resource(DeviceId::cuda(0))};
    const std::array duplicate{consumers[0],consumers[0]};
    EXPECT_THROW(input(duplicate),std::invalid_argument);
    EXPECT_THROW(input(std::span<const PhysicalMemoryResource>{}),std::invalid_argument);
    auto foreign=consumers;foreign[0].world_rank=4;
    EXPECT_THROW(input(foreign),std::invalid_argument);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(),input(consumers),{},eligible()),std::invalid_argument);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(),input(consumers),
        base(consumers,PhysicalMemoryOwner::ImmutableTableStorage),eligible()),std::invalid_argument);
    PhysicalMemoryPlanBuilder invalid;
    invalid.add(consumers[0],PhysicalMemoryOwner::KVCache,1u);
    EXPECT_THROW(ImmutableRowTablePlacementPlanner::select(request(),input(consumers),base(consumers),
        {invalid.build(),{}, {}}),std::invalid_argument);
}

/** @test Only real capacity pressure reaches the automatic topology rejection callback. */
TEST(ImmutableRowTablePlacementPlanner, CapacityKindDoesNotHideIneligibleOrMalformedInputs)
{
    const std::array consumers{resource(DeviceId::cuda(0),1u)};
    const auto frozen=base(consumers);
    int capacity_rejections=0;
    for(const auto preference:{ImmutableRowStoragePreference::VRAM,ImmutableRowStoragePreference::RAM,
        ImmutableRowStoragePreference::MemoryMapped,ImmutableRowStoragePreference::Automatic})
    {
        try { (void)ImmutableRowTablePlacementPlanner::select(request(preference),input(consumers,1u),frozen,eligible());
            FAIL()<<"An exhausted final table was admitted"; }
        catch(const PhysicalMemoryCapacityExhausted &) {++capacity_rejections;}
        catch(...) {FAIL()<<"Genuine table capacity rejection lost its canonical type";}
        EXPECT_EQ(frozen.find({3,DeviceId::cuda(0)})->bytes(PhysicalMemoryOwner::PrimaryModelWeights),1u);
        EXPECT_EQ(frozen.find({3,DeviceId::cuda(0)})->bytes(PhysicalMemoryOwner::ImmutableTableStorage),0u);
    }
    EXPECT_EQ(capacity_rejections,4);
    try { (void)ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::VRAM),
        input(consumers),frozen,{});FAIL()<<"Missing original upload eligibility was admitted"; }
    catch(const PhysicalMemoryCapacityExhausted &) {FAIL()<<"Ineligible mode must not become memory pressure";}
    catch(const std::invalid_argument &) {}
    try { (void)ImmutableRowTablePlacementPlanner::select(request(),input(consumers),{},eligible());
        FAIL()<<"Missing source consumer owner was admitted"; }
    catch(const PhysicalMemoryCapacityExhausted &) {FAIL()<<"Malformed owner must not become memory pressure";}
    catch(const std::invalid_argument &) {}
}
/** @test Locality changes immutable reuse identity without changing a single PMA owner byte. */
TEST(ImmutableRowTablePlacementPlanner, HostLocalityIsSealedSeparatelyFromPhysicalAllocator)
{
    const std::array consumers{resource(DeviceId::cuda(0))};
    const auto original=input(consumers);
    const auto bound=ImmutableRowTablePlacementInput::forConsumers(original.extent(),original.host(),
        original.consumers(),ImmutableRowTableHostPlacement::onNode(3));
    const auto task=ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::RAM),original,base(consumers),eligible());
    const auto exact=ImmutableRowTablePlacementPlanner::select(request(ImmutableRowStoragePreference::RAM),bound,base(consumers),eligible());
    EXPECT_EQ(task.physical_plan.summary(),exact.physical_plan.summary());
    EXPECT_NE(*task.placement,*exact.placement);
    EXPECT_TRUE(task.placement->hostPlacement().usesTaskPolicy());
    EXPECT_EQ(exact.placement->hostPlacement(),ImmutableRowTableHostPlacement::onNode(3));
    EXPECT_EQ(task.placement->allocators().front(),exact.placement->allocators().front());
}

}
