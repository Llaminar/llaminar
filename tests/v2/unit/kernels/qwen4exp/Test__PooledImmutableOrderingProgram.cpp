/**
 * @file Test__PooledImmutableOrderingProgram.cpp
 * @brief Actual owner/coordinator/source root recipes define the complete existing event population.
 *
 * These device-free tests project production geometry and native plans without
 * granting a resource certificate or creating any native resources. Native
 * creator and lifetime replays remain separate admission requirements.
 */
#include "kernels/qwen4exp/PooledImmutableNativePlan.h"
#include "kernels/qwen4exp/PooledImmutableSourceGeometry.h"
#include "loaders/ImmutableRowTable.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <vector>
namespace llaminar2::test { namespace {
namespace q=qwen4exp;
/** @brief Original checked root geometry with explicit ordered physical owners. */
auto physical(std::span<const std::uint32_t> slots,DeviceType consumer){
    const PhysicalMemoryAllocatorIdentity coordinator{0,{consumer,7}};
    std::vector<ImmutableRowTableLayoutOwner> roots;
    std::vector<PhysicalMemoryAllocatorIdentity> order;
    bool positive=false;
    for(std::size_t i=0;i<slots.size();++i){
        order.push_back({0,{i%2?consumer:(consumer==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA),static_cast<int>(i)}});
        roots.push_back({{order.back(),ImmutableRowStorage::VRAM},0,slots[i],{}});positive|=slots[i]!=0;
    }
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(
        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,roots,
        {{0,DeviceId::cpu()},coordinator},{},
        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
        {},positive?std::optional(coordinator):std::nullopt,16);
    return q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
}
/** @test Event count comes from the actual nonempty canonical roots for every role. */
TEST(PooledImmutableOrderingProgram, ActualRootMasksMatchStaticAndRuntimeRecipes){
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const auto count:{1u,2u,4u}){
        std::vector<std::uint32_t> slots(count);for(std::size_t i=0;i<count;++i)slots[i]=i%2?5:0;
        for(const bool positive:{false,true}){
            if(positive)slots[0]=17;
            const auto geometry=physical(slots,backend);
            const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(geometry,{65536,1,2},64);
            const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(geometry,policy,701,17);
            std::vector<q::PooledImmutableOwnerNativePlan> owners;
            for(std::uint32_t i=0;i<count;++i){
                owners.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(geometry,policy,i,701,17));
                const auto& plan=owners.back();const auto program=plan.orderingProgram();
                EXPECT_EQ(program,q::PooledImmutableOwnerNativePlan::orderingProgram(geometry,policy,i));
                EXPECT_EQ(program.kind(),NativeOrderingEventProgram::Kind::PooledOwnerInitialization);
                EXPECT_EQ(program.orderingEvents(),slots[i]?4u:3u);
                const auto& p=plan.parameters();EXPECT_NO_THROW(program.requireRootBytes({p.cache.bytes,p.frame.bytes,p.native.bytes}));
                EXPECT_THROW(program.requireRootBytes({p.cache.bytes,p.frame.bytes,0}),std::invalid_argument);
            }
            EXPECT_EQ(coordinator.orderingProgram(),q::PooledImmutableCoordinatorNativePlan::orderingProgram(geometry,policy));
            EXPECT_EQ(coordinator.orderingProgram().orderingEvents(),coordinator.parameters().cache.bytes?4u:3u);
            const ImmutableRowTableSource source{.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
            const auto sourcePlan=q::PooledImmutableSourceGeometry::fromPlans(owners,coordinator,source,1);
            EXPECT_EQ(sourcePlan.orderingProgram(),q::PooledImmutableSourceGeometry::orderingProgram(geometry,policy,source));
            EXPECT_EQ(sourcePlan.orderingProgram().orderingEvents(),3u);
            EXPECT_NE(sourcePlan.orderingProgram(),NativeOrderingEventProgram::pooledOwner(false));
            EXPECT_EQ(NativeOrderingEventProgram::lookupSubmissionRetirement().orderingEvents(),1u);
        }
    }
}
/** @test Repeated large-to-C0 geometry has no stale optional publication event. */
TEST(PooledImmutableOrderingProgram, RepeatedLargeToSmallAndForeignCoordinateReject){
    for(int replay=0;replay<20;++replay)for(const auto slots:{std::array{65u,0u},std::array{1u,2u},std::array{0u,0u}}){
        const auto geometry=physical(slots,DeviceType::ROCm);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(geometry,{65536,1,2},64);
        for(std::uint32_t i=0;i<slots.size();++i){
            const auto program=q::PooledImmutableOwnerNativePlan::orderingProgram(geometry,policy,i);
            EXPECT_EQ(program.nonemptyRootMask(),slots[i]?7u:6u);
        }
        EXPECT_THROW(q::PooledImmutableOwnerNativePlan::orderingProgram(geometry,policy,2),std::out_of_range);
    }
}
/** @test Readiness reconstructs the declared compute-reader population independently of physical owner count. */
TEST(PooledImmutableOrderingProgram, SourceReadinessPreservesDeclaredComputeReaderPopulation)
{
    const ImmutableRowTableSource source{.path="recorded-source-readiness", .tensor_name="rows",
        .file_offset=37, .rows=0x100000100ULL, .row_elements=160, .format=GGUFTensorType::IQ4_NL};
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto storage_owners : {1u, 3u})
    {
        // The source channel, storage-owner population and compute-reader
        // population are independent original coordinates. No resource is created.
        const std::vector<std::uint32_t> slots(storage_owners, 3u);
        const auto geometry=physical(slots,backend);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(geometry,{65536,1,2},64);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(geometry,policy,701,17);
        std::vector<q::PooledImmutableOwnerNativePlan> owners;
        for (std::uint32_t owner=0; owner<storage_owners; ++owner)
            owners.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(geometry,policy,owner,701,17));
        std::vector<std::uint64_t> identities;
        for (const auto readers : {1u, 2u, 4u, 8u})
        {
            SCOPED_TRACE(::testing::Message()<<"backend="<<static_cast<int>(backend)
                         <<" owners="<<storage_owners<<" readers="<<readers);
            const auto original=q::PooledImmutableSourceGeometry::fromPlans(owners,coordinator,source,readers);
            EXPECT_EQ(original.parameters().consumers,readers);
            identities.push_back(original.parameters().identity);
            EXPECT_NO_THROW(original.requireOriginalPlans(owners,coordinator,source,readers));
            for (const auto changed_readers : {1u, 2u, 4u, 8u})
                if (changed_readers!=readers)
                {
                    try {
                        original.requireOriginalPlans(owners,coordinator,source,changed_readers);
                        ADD_FAILURE()<<"Changed compute-reader population was accepted";
                    } catch (const std::logic_error &error) {
                        EXPECT_NE(std::string(error.what()).find("compute-reader population changed"),std::string::npos);
                    }
                }
            auto changed_source=source;
            changed_source.row_elements=128;
            EXPECT_THROW(original.requireOriginalPlans(owners,coordinator,changed_source,readers),std::invalid_argument);
            auto foreign_owners=owners;
            foreign_owners.front()=q::PooledImmutableOwnerNativePlan::fromGeometry(geometry,policy,0,701,18);
            EXPECT_THROW(original.requireOriginalPlans(foreign_owners,coordinator,source,readers),std::invalid_argument);
            const auto changed_coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(geometry,policy,701,18);
            for (std::uint32_t owner=0; owner<storage_owners; ++owner)
                foreign_owners[owner]=q::PooledImmutableOwnerNativePlan::fromGeometry(geometry,policy,owner,701,18);
            EXPECT_THROW(original.requireOriginalPlans(foreign_owners,changed_coordinator,source,readers),std::logic_error);
        }
        std::sort(identities.begin(),identities.end());
        EXPECT_EQ(std::unique(identities.begin(),identities.end()),identities.end());
        EXPECT_THROW(q::PooledImmutableSourceGeometry::fromPlans(owners,coordinator,source,0),std::invalid_argument);
    }
}
}}
