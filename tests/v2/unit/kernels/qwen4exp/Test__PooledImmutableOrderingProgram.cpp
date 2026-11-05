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
auto physical(std::span<const std::uint32_t> slots,DeviceType consumer,bool source_has_cache=false,int source_ordinal=7){
    const PhysicalMemoryAllocatorIdentity coordinator{0,{consumer,source_ordinal}};
    std::vector<ImmutableRowTableLayoutOwner> roots;
    std::vector<PhysicalMemoryAllocatorIdentity> order;
    bool positive=false;
    for(std::size_t i=0;i<slots.size();++i){
        order.push_back(source_has_cache && i==0 ? coordinator : PhysicalMemoryAllocatorIdentity{0,{i%2?consumer:(consumer==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA),static_cast<int>(i)}});
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
/** @test Retention selects actual source or owner observations, never an execution-consumer boundary. */
TEST(PooledImmutableOrderingProgram, RetainedRootProjectionDistinguishesPhysicalOwnersAndReaders)
{
    // Recorded claim bytes exercise only the production metadata projection;
    // they create no PMA observation, root, source boundary or reuse permission.
    struct RecordedClaims {
        PhysicalMemoryAllocatorIdentity original;
        std::size_t table,staging;
        PhysicalMemoryAllocatorIdentity allocator()const{return original;}
    };
    const ImmutableRowTableSource descriptor{.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const bool combined:{false,true})
    {
        const auto shape=physical(std::array{3u,5u,0u},backend,combined);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(shape,{65536,1,2},64);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(shape,policy,701,17);
        std::vector<q::PooledImmutableOwnerNativePlan> owners;
        std::vector<RecordedClaims> recorded;
        for(std::size_t index=0;index<shape.owners().size();++index){
            owners.push_back(q::PooledImmutableOwnerNativePlan::fromGeometry(shape,policy,index,701,17));
            recorded.push_back({owners.back().allocator(),11+index*13,17+index*19});
        }
        const auto geometry=q::PooledImmutableSourceGeometry::fromPlans(owners,coordinator,descriptor,2);
        // Source bytes already include any local owner. The helper must return
        // this unchanged observation, never add that owner's claims again.
        const RecordedClaims source{geometry.allocator(),101,103};
        std::vector<PhysicalMemoryAllocatorIdentity> probes{geometry.allocator()};
        for(const auto& owner:owners)if(owner.allocator()!=geometry.allocator())probes.push_back(owner.allocator());
        const PhysicalMemoryAllocatorIdentity reader_only{0,{backend,11}},unrelated{0,{backend,12}};
        probes.push_back(reader_only);probes.push_back(unrelated);
        for(const auto probe:probes){
            SCOPED_TRACE(::testing::Message()<<"combined="<<combined<<" target="<<probe.device.toString());
            int source_calls=0,owner_calls=0;
            const auto get_source=[&]{
                ++source_calls;
                // Original failing behavior asked this source-only boundary
                // for every consumer, including the foreign shared reader.
                if(probe!=geometry.allocator())throw std::invalid_argument("recorded foreign source boundary");
                return source;
            };
            const auto get_owner=[&](std::size_t index){++owner_calls;return recorded.at(index);};
            if(probe!=geometry.allocator()){
                EXPECT_THROW((void)get_source(),std::invalid_argument);
                source_calls=0;
            }
            const auto observed=geometry.projectRetainedOwnerObservation(probe,get_source,get_owner);
            if(probe==reader_only || probe==unrelated){
                EXPECT_FALSE(observed);EXPECT_EQ(source_calls,0);EXPECT_EQ(owner_calls,0);
            }else{
                ASSERT_TRUE(observed);EXPECT_EQ(observed->allocator(),probe);
                if(probe==geometry.allocator()){
                    EXPECT_EQ(observed->table,101u);EXPECT_EQ(observed->staging,103u);
                    EXPECT_EQ(source_calls,1);EXPECT_EQ(owner_calls,0);
                }else{
                    const auto expected=std::find_if(recorded.begin(),recorded.end(),[&](const auto& row){return row.allocator()==probe;});
                    ASSERT_NE(expected,recorded.end());
                    EXPECT_EQ(observed->table,expected->table);EXPECT_EQ(observed->staging,expected->staging);
                    EXPECT_EQ(source_calls,0);EXPECT_EQ(owner_calls,1);
                }
            }
        }
        // A copied observation from another allocator cannot be silently credited.
        EXPECT_THROW(geometry.projectRetainedOwnerObservation(geometry.allocator(),
            [&]{return RecordedClaims{reader_only,101,103};},[&](std::size_t index){return recorded.at(index);}),std::logic_error);
    }
    // Consumer-local sources remain separate original physical owners. A borrow
    // of source A cannot give source B's rows a second retention credit.
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const int ordinal:{7,8}){
        const auto shape=physical(std::array{3u},backend,true,ordinal);
        const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(shape,{65536,1,2},64);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(shape,policy,701,17);
        const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(shape,policy,0,701,17);
        const auto geometry=q::PooledImmutableSourceGeometry::fromPlans(owner,coordinator,descriptor,1);
        int source_calls=0,owner_calls=0;
        const auto get_source=[&]{++source_calls;return RecordedClaims{geometry.allocator(),211,223};};
        const auto get_owner=[&](std::size_t){++owner_calls;return RecordedClaims{owner.allocator(),31,37};};
        const auto local=geometry.projectRetainedOwnerObservation(geometry.allocator(),get_source,get_owner);
        ASSERT_TRUE(local);EXPECT_EQ(local->table,211u);EXPECT_EQ(local->staging,223u);
        EXPECT_EQ(source_calls,1);EXPECT_EQ(owner_calls,0);
        const auto other=geometry.projectRetainedOwnerObservation({0,{backend,ordinal==7?8:7}},get_source,get_owner);
        EXPECT_FALSE(other);EXPECT_EQ(source_calls,1);EXPECT_EQ(owner_calls,0);
    }
}
}}
