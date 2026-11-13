/**
 * @file Test__GPUAdaptiveImmutableLookupCohortGeometry.cpp
 * @brief Device-free pooled ngram declarations and exact per-allocator admission.
 *
 * These tests use source metadata and the actual native-root BOM factories.
 * They never read table payloads, initialize accelerators or authorize serving.
 * Exact-fit decisions use the canonical topology certificate, and projections
 * cover independent/co-located coordinators and zero-capacity physical owners.
 */
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "fixtures/planning/RecordedNgramOrdering.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

using namespace llaminar2;
namespace
{
namespace q = qwen4exp;
using Allocator = PhysicalMemoryAllocatorIdentity;
using Geometry = AdaptiveImmutableLookupGeometry;

/** @brief Authored metadata fixture, with no runtime/native resource creation. */
struct Cohort
{
    std::vector<Allocator> order;
    Allocator coordinator;
    const Allocator host{0, DeviceId::cpu()};
    ImmutableRowTableSource source;
    AdaptiveVRAMRAMSSDRequest request;
    std::size_t maximum_ids = 16; ///< Complete query envelope, independent of owner slot capacity.

    /** @brief Author homogeneous/mixed GPU owners; a negative local index separates the coordinator. */
    Cohort(std::size_t count, DeviceType backend, bool mixed, int local,
           GGUFTensorType format = GGUFTensorType::Q8_0, bool wide = false)
        : source{.rows = wide ? (1ULL << 32) + 17 : 4096,
                 .row_elements = format == GGUFTensorType::Q4_K ? 256u :
                     (format == GGUFTensorType::IQ4_NL ? 160u : 64u), .format = format}
    {
        for (std::size_t i = 0; i < count; ++i)
            order.push_back({0, {mixed && i % 2 ?
                (backend == DeviceType::CUDA ? DeviceType::ROCm : DeviceType::CUDA) : backend,
                static_cast<int>(i)}});
        coordinator = local < 0 ? Allocator{0, {backend, 31}} : order.at(static_cast<std::size_t>(local));
        request.maximum_gpu_cache_rows = 64;
        request.gpu_cache_physical_byte_limit = 1u << 20;
        request.host.maximum_cache_rows = 7;
        request.host.cache_physical_byte_limit = 1u << 20;
    }

    /** @return Original endpoint membership with independently authored slot capacities. */
    ImmutableRowTableLayoutGeometry layout(std::span<const std::uint32_t> slots) const
    {
        if (slots.size() != order.size()) throw std::invalid_argument("Fixture owner count mismatch");
        std::vector<ImmutableRowTableLayoutOwner> owners;
        for (std::size_t i = 0; i < order.size(); ++i)
            owners.push_back({{order[i], ImmutableRowStorage::VRAM}, 0, slots[i], {}});
        owners.push_back({{host, ImmutableRowStorage::RAM}, 0, 7, {}});
        const std::array tiers{ImmutableRowStorage::VRAM, ImmutableRowStorage::RAM, ImmutableRowStorage::MemoryMapped};
        const bool positive = std::any_of(slots.begin(), slots.end(), [](auto value) { return value != 0; });
        return ImmutableRowTableLayoutGeometry::checked(
            ImmutableRowTableExtent::fromNative(source.format, source.rows, source.rowBytes()),
            ImmutableRowTableTiers::of(tiers), ImmutableRowTableResidency::Adaptive,
            std::move(owners), {coordinator}, {},
            {{host, ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped, 4096, 2)}},
            {}, positive ? std::optional(coordinator) : std::nullopt, maximum_ids);
    }

    /** @return Valid authored resource metadata; no driver observation is performed. */
    static PhysicalMemoryResource resource(Allocator id)
    { return {id.world_rank, id.device, 1ULL << 34, 1ULL << 34}; }

    /** @return Every distinct GPU allocator once, with deliberately unrelated observation ordering. */
    std::vector<PhysicalMemoryResource> devices() const
    {
        std::vector<PhysicalMemoryResource> result;
        for (const auto id : order) result.push_back(resource(id));
        if (std::find(order.begin(), order.end(), coordinator) == order.end())
            result.push_back(resource(coordinator));
        std::reverse(result.begin(), result.end());
        return result;
    }
};

/** @brief Combine actual contributor lines through the canonical aggregate builder. */
void append(PhysicalMemoryPlanBuilder& builder, const PhysicalMemoryPlan& plan)
{ for (const auto& line : plan.resources()) builder.add(line); }

/** @return Complete cache/source plan; transport remains a separate factory obligation. */
PhysicalMemoryPlan complete(const Geometry& geometry, std::span<const PhysicalMemoryResource> devices,
                            PhysicalMemoryResource host)
{
    PhysicalMemoryPlanBuilder builder(llaminar2::test::recordedNgramMandatory(geometry,devices, host));
    append(builder, geometry.cacheBOM(devices, host));
    return builder.build();
}

/** @return Independent composition of the original native runtime factories, without cohort subtotal arithmetic. */
PhysicalMemoryPlan runtimeRoots(const Cohort& fixture, const Geometry& geometry)
{
    PhysicalMemoryPlanBuilder builder;
    const auto& physical = geometry.physical();
    const auto& coordinator = geometry.coordinator();
    for (std::size_t i = 0; i < physical.owners().size(); ++i)
    {
        const auto resource = Cohort::resource(physical.owners()[i].allocator);
        append(builder, physical.mandatoryBOM(static_cast<std::uint32_t>(i), resource));
        append(builder, physical.cacheBOM(static_cast<std::uint32_t>(i), resource));
        append(builder, q::PooledImmutableOwnerNativePlan::mandatoryBOM(
            physical, coordinator, static_cast<std::uint32_t>(i), resource));
    }
    const auto gpu = Cohort::resource(fixture.coordinator);
    append(builder, coordinator.mandatoryBOM(gpu));
    append(builder, coordinator.policyBOM(gpu));
    append(builder, q::PooledImmutableCoordinatorNativePlan::mandatoryBOM(physical, coordinator, gpu));
    append(builder, q::PooledImmutableSourceGeometry::mandatoryBOM(physical, coordinator, fixture.source, gpu));
    const auto host = Cohort::resource(fixture.host);
    const auto projection=llaminar2::test::recordedNgramOrdering(fixture.devices());
    for(std::uint32_t i=0;i<physical.owners().size();++i)
        append(builder,projection.bom(q::PooledImmutableOwnerNativePlan::orderingProgram(physical,coordinator,i),host,Cohort::resource(physical.owners()[i].allocator)));
    append(builder,projection.bom(q::PooledImmutableCoordinatorNativePlan::orderingProgram(physical,coordinator),host,gpu));
    append(builder,projection.bom(NativeOrderingEventProgram::pooledSource(),host,gpu));
    append(builder,projection.bom(NativeOrderingEventProgram::lookupSubmissionRetirement(),host,gpu));
    append(builder, geometry.host().bom(host));
    builder.add(host, PhysicalMemoryOwner::ImmutableLookupStaging,
                q::ImmutableLookupService::hostBytes(geometry.channelLayout()));
    return builder.build();
}

/** @brief Compare every typed physical owner line in both directions. */
void expectSameRoots(const PhysicalMemoryPlan& actual, const PhysicalMemoryPlan& expected)
{
    EXPECT_FALSE(actual.requiredFootprintMismatch(expected)) << actual.summary() << expected.summary();
    EXPECT_FALSE(expected.requiredFootprintMismatch(actual)) << actual.summary() << expected.summary();
}
}

/** @test Every original allocator receives its own roots; shared source/policy roots are charged only once. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, OriginalFactoriesMatchEveryOwnerAndSharedSource)
{
    for (const auto count : {1u, 2u, 4u, 8u})
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            for (bool mixed : {false, true})
                for (const int local : {-1, 0, static_cast<int>(count - 1)})
                {
                    SCOPED_TRACE(std::to_string(count) + "/" + std::to_string(local));
                    Cohort fixture(count, backend, mixed, local);
                    std::vector<std::uint32_t> slots(count);
                    for (std::size_t i = 0; i < count; ++i) slots[i] = i % 3 ? 3 + i : 0;
                    if (count == 1) slots.front() = 3;
                    for (bool empty : {false, true})
                    {
                        if (empty) std::fill(slots.begin(), slots.end(), 0);
                        const auto geometry = Geometry::fromLayout(fixture.layout(slots), fixture.source,
                                                                   fixture.request, fixture.order);
                        const auto devices = fixture.devices();
                        const auto host = Cohort::resource(fixture.host);
                        const auto actual = complete(geometry, devices, host);
                        expectSameRoots(actual, runtimeRoots(fixture, geometry));
                        EXPECT_NO_THROW(PhysicalMemoryPlanAdmissionCertificate{actual});
                        const auto mandatory = llaminar2::test::recordedNgramMandatory(geometry,devices, host);
                        const auto cache = geometry.cacheBOM(devices, host);
                        for (const auto& device : devices)
                        {
                            const Allocator id{device.world_rank, device.device};
                            ASSERT_NE(mandatory.find(id), nullptr);
                            EXPECT_EQ(mandatory.find(id)->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),
                                      geometry.mandatoryDeviceBytes(id));
                            EXPECT_EQ(cache.find(id) ? cache.find(id)->bytes(PhysicalMemoryOwner::ImmutableTableStorage) : 0,
                                      geometry.gpuCacheBytes(id));
                        }
                        EXPECT_EQ(geometry.physical().owners().size(), count);
                        for (std::size_t i = 0; i < count; ++i)
                            EXPECT_EQ(geometry.physical().owners()[i].allocator, fixture.order[i]);
                    }
                }
}

/** @test C0 plus optional growth equals full native roots across replayed capacities and native encodings. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, FixedAndOptionalRootsRemainExactAcrossCapacityReplay)
{
    for (const auto format : {GGUFTensorType::Q8_0, GGUFTensorType::Q4_K, GGUFTensorType::IQ4_NL,
                              GGUFTensorType::F16, GGUFTensorType::BF16, GGUFTensorType::F32})
        for (bool wide : {false, true})
            for (const auto count : {1u, 2u, 4u, 8u})
            {
                Cohort fixture(count, DeviceType::CUDA, true, static_cast<int>(count / 2), format, wide);
                std::reverse(fixture.order.begin(), fixture.order.end());
                const auto devices = fixture.devices();
                const auto host = Cohort::resource(fixture.host);
                const auto zero = Geometry::fromLayout(fixture.layout(std::vector<std::uint32_t>(count)),
                                                       fixture.source, fixture.request, fixture.order);
                const auto fixed = llaminar2::test::recordedNgramMandatory(zero,devices, host);
                for (const auto capacity : {31u, 3u, 0u, 17u, 1u})
                {
                    const auto geometry = Geometry::fromLayout(fixture.layout(std::vector<std::uint32_t>(count, capacity)),
                                                               fixture.source, fixture.request, fixture.order);
                    PhysicalMemoryPlanBuilder candidate(fixed);
                    const auto optional = llaminar2::test::recordedNgramOptional(geometry,devices, host);
                    append(candidate, optional);
                    expectSameRoots(candidate.build(), complete(geometry, devices, host));
                    for (const auto& device : devices)
                    {
                        const Allocator id{device.world_rank, device.device};
                        const auto* line = optional.find(id);
                        EXPECT_EQ(geometry.gpuSlotDependentBytes(id), line ?
                            line->bytes(PhysicalMemoryOwner::ImmutableTableStorage) +
                            line->bytes(PhysicalMemoryOwner::ImmutableLookupStaging) : 0);
                    }
                }
            }
}

/** @test Model/verifier row envelopes scale original frame roots, including larger admitted buckets. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, QueryCapacityUsesOriginalRootsAcrossSmallAndLargeBuckets)
{
    for (const auto count : {1u, 2u, 4u, 8u})
        for (const auto rows : {1u, 2u, 4u, 16u, 64u, 512u, 1024u, 2048u})
            for (const auto format : {GGUFTensorType::IQ4_NL, GGUFTensorType::F32})
            {
                SCOPED_TRACE(std::to_string(count) + "/rows=" + std::to_string(rows));
                Cohort fixture(count, DeviceType::ROCm, true, static_cast<int>(count / 2), format);
                fixture.maximum_ids = 16u * rows;
                const auto devices = fixture.devices();
                const auto host = Cohort::resource(fixture.host);
                const auto zero = Geometry::fromLayout(fixture.layout(std::vector<std::uint32_t>(count)),
                                                       fixture.source, fixture.request, fixture.order);
                const auto geometry = Geometry::fromLayout(fixture.layout(std::vector<std::uint32_t>(count, 31)),
                                                           fixture.source, fixture.request, fixture.order);
                const auto actual = complete(geometry, devices, host);
                expectSameRoots(actual, runtimeRoots(fixture, geometry));
                PhysicalMemoryPlanBuilder candidate(llaminar2::test::recordedNgramMandatory(zero,devices, host));
                append(candidate, llaminar2::test::recordedNgramOptional(geometry,devices, host));
                expectSameRoots(actual, candidate.build());
                EXPECT_NO_THROW(PhysicalMemoryPlanAdmissionCertificate{actual});
                EXPECT_EQ(geometry.maximumIds(), fixture.maximum_ids);
            }
}

/** @test Optional cache roots extend frozen model/expert/KV/graph declarations without reassigning them. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, OptionalStorageCannotBorrowFrozenOwnersOrAnotherDevice)
{
    Cohort fixture(8, DeviceType::ROCm, true, 3);
    const auto devices = fixture.devices();
    const auto host = Cohort::resource(fixture.host);
    const auto zero = Geometry::fromLayout(fixture.layout(std::vector<std::uint32_t>(8)),
                                           fixture.source, fixture.request, fixture.order);
    PhysicalMemoryPlanBuilder frozen(llaminar2::test::recordedNgramMandatory(zero,devices, host));
    constexpr std::array prior_owners{PhysicalMemoryOwner::PrimaryModelWeights,
        PhysicalMemoryOwner::RoutedExpertWeights, PhysicalMemoryOwner::KVCache,
        PhysicalMemoryOwner::ActivationArena, PhysicalMemoryOwner::NativeGraphExecutable};
    for (const auto& device : devices)
        for (const auto owner : prior_owners)
            frozen.add(device, owner, (static_cast<std::size_t>(owner) + 1) << 20);
    frozen.add(host, PhysicalMemoryOwner::PrefixHostTier, 1u << 28);
    const auto base = frozen.build();
    for (const auto capacity : {31u, 1u, 0u, 17u})
    {
        const auto geometry = Geometry::fromLayout(fixture.layout(std::vector<std::uint32_t>(8, capacity)),
                                                   fixture.source, fixture.request, fixture.order);
        PhysicalMemoryPlanBuilder candidate(base);
        append(candidate, llaminar2::test::recordedNgramOptional(geometry,devices, host));
        const auto complete_plan = candidate.build();
        EXPECT_NO_THROW(PhysicalMemoryPlanAdmissionCertificate{complete_plan});
        for (const auto& device : devices)
        {
            const Allocator id{device.world_rank, device.device};
            for (const auto owner : prior_owners)
                EXPECT_EQ(complete_plan.find(id)->bytes(owner), base.find(id)->bytes(owner));
        }
        EXPECT_EQ(complete_plan.find(fixture.host)->bytes(PhysicalMemoryOwner::PrefixHostTier),
                  base.find(fixture.host)->bytes(PhysicalMemoryOwner::PrefixHostTier));
    }
}

/** @test One insufficient physical allocator rejects the whole cohort; other GPUs cannot hide its deficit. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, ExactFitAndEachOneByteShortUseCanonicalAdmission)
{
    for (const int local : {-1, 0, 3})
    {
        Cohort fixture(4, DeviceType::CUDA, true, local);
        const auto geometry = Geometry::fromLayout(fixture.layout(std::array{0u, 3u, 7u, 19u}),
                                                   fixture.source, fixture.request, fixture.order);
        auto devices = fixture.devices();
        auto host = Cohort::resource(fixture.host);
        const auto generous = complete(geometry, devices, host);
        for (auto& device : devices)
            device.admission_available_bytes = generous.find({device.world_rank, device.device})->incrementalBytes();
        host.admission_available_bytes = generous.find(fixture.host)->incrementalBytes();
        EXPECT_NO_THROW(PhysicalMemoryPlanAdmissionCertificate{complete(geometry, devices, host)});
        for (std::size_t i = 0; i <= devices.size(); ++i)
        {
            auto short_devices = devices;
            auto short_host = host;
            auto& pressured = i == devices.size() ? short_host : short_devices[i];
            ASSERT_GT(pressured.admission_available_bytes, 0u);
            --pressured.admission_available_bytes;
            const auto short_plan = complete(geometry, short_devices, short_host);
            EXPECT_THROW((PhysicalMemoryPlanAdmissionCertificate{short_plan}), PhysicalMemoryCapacityExhausted);
        }
    }
}

/** @test Partial/duplicate/foreign observations cannot silently aggregate unrelated physical capacities. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, RejectIncompleteObservationsAndAmbiguousLegacyProjection)
{
    Cohort fixture(4, DeviceType::CUDA, true, 2);
    const auto layout = fixture.layout(std::array{0u, 3u, 7u, 19u});
    const auto geometry = Geometry::fromLayout(layout, fixture.source, fixture.request, fixture.order);
    EXPECT_THROW(geometry.gpuCacheBytes(), std::invalid_argument);
    EXPECT_THROW(geometry.gpuSlotDependentBytes(), std::invalid_argument);
    EXPECT_THROW(geometry.mandatoryDeviceBytes(), std::invalid_argument);
    EXPECT_THROW(Geometry::fromLayout(layout, fixture.source, fixture.request), std::invalid_argument);
    for (unsigned defect = 0; defect < 5; ++defect)
    {
        auto devices = fixture.devices();
        switch (defect)
        {
        case 0: devices.pop_back(); break;
        case 1: devices.back() = devices.front(); break;
        case 2: devices.front().world_rank = 1; break;
        case 3: devices.front().device = DeviceId::cpu(); break;
        case 4: devices.push_back(Cohort::resource({0, DeviceId::cuda(30)})); break;
        }
        EXPECT_THROW(llaminar2::test::recordedNgramMandatory(geometry,devices, Cohort::resource(fixture.host)), std::invalid_argument);
        EXPECT_THROW(geometry.cacheBOM(devices, Cohort::resource(fixture.host)), std::invalid_argument);
        EXPECT_THROW(llaminar2::test::recordedNgramOptional(geometry,devices, Cohort::resource(fixture.host)), std::invalid_argument);
    }
    auto order = fixture.order;
    order.back() = order.front();
    EXPECT_THROW(Geometry::fromLayout(layout, fixture.source, fixture.request, order), std::invalid_argument);
    order = fixture.order;
    order.pop_back();
    EXPECT_THROW(Geometry::fromLayout(layout, fixture.source, fixture.request, order), std::invalid_argument);
    EXPECT_THROW(geometry.gpuCacheBytes({1, fixture.coordinator.device}), std::invalid_argument);
}

/** @test Existing single-GPU callers keep identical root requirements and explicit allocator projections. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, ExistingSingleOwnerSpellingRetainsExactPhysicalRoots)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto slots : {0u, 1u, 31u})
        {
            Cohort fixture(1, backend, false, 0);
            const auto layout = fixture.layout(std::array{slots});
            const auto original = Geometry::fromLayout(layout, fixture.source, fixture.request);
            const auto explicit_cohort = Geometry::fromLayout(layout, fixture.source, fixture.request, fixture.order);
            const auto gpu = Cohort::resource(fixture.coordinator);
            const auto host = Cohort::resource(fixture.host);
            expectSameRoots(llaminar2::test::recordedNgramMandatory(original,gpu, host), llaminar2::test::recordedNgramMandatory(explicit_cohort,fixture.devices(), host));
            expectSameRoots(original.cacheBOM(gpu, host), explicit_cohort.cacheBOM(fixture.devices(), host));
            expectSameRoots(llaminar2::test::recordedNgramOptional(original,gpu, host), llaminar2::test::recordedNgramOptional(explicit_cohort,fixture.devices(), host));
            EXPECT_EQ(original.gpuCacheBytes(), explicit_cohort.gpuCacheBytes(fixture.coordinator));
            EXPECT_EQ(original.gpuSlotDependentBytes(), explicit_cohort.gpuSlotDependentBytes(fixture.coordinator));
            EXPECT_EQ(original.mandatoryDeviceBytes(), explicit_cohort.mandatoryDeviceBytes(fixture.coordinator));
        }
}

/** @test Complete RAM source eligibility preserves exact GPU owner/coordinator and C0/positive slot contracts. */
TEST(GPUAdaptiveImmutableLookupCohortGeometry, FullRAMSourceKeepsOriginalPhysicalOwnerAndCoordinatorGuards){
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const auto count:{1u,2u,4u})for(const bool positive:{false,true}){
        Cohort original(count,backend,false,0);std::vector<std::uint32_t> slots(count,positive?3u:0u);
        const auto ssd=original.layout(slots);auto owners=std::vector<ImmutableRowTableLayoutOwner>(ssd.owners().begin(),ssd.owners().end());
        for(auto& owner:owners)if(owner.key.tier==ImmutableRowStorage::RAM)owner.cache_rows=0;
        const auto ram=ImmutableRowTableLayoutGeometry::checked(ssd.extent(),
            ImmutableRowTableTiers::of(std::array{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM}),ssd.residency(),owners,
            std::vector<Allocator>(ssd.consumers().begin(),ssd.consumers().end()),{},
            {{original.host,ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM)}},{},ssd.gpuCoordinator(),ssd.maximumInflightRows());
        const auto actual=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(ram,original.order,original.coordinator);
        EXPECT_EQ(actual.coordinator(),original.coordinator);ASSERT_EQ(actual.owners().size(),count);
        for(std::size_t index=0;index<count;++index){EXPECT_EQ(actual.owners()[index].allocator,original.order[index]);EXPECT_EQ(actual.owners()[index].slots,slots[index]);}
        EXPECT_THROW(q::PooledImmutablePhysicalOwnerGeometry::fromLayout(ram,original.order,{0,DeviceId(backend,99)}),std::invalid_argument);
        EXPECT_THROW(q::PooledImmutablePhysicalOwnerGeometry::fromLayout(ram,original.order,original.host),std::invalid_argument);
        if(count>1){auto duplicate=original.order;duplicate.back()=duplicate.front();EXPECT_THROW(q::PooledImmutablePhysicalOwnerGeometry::fromLayout(ram,duplicate,original.coordinator),std::invalid_argument);}
    }
}
