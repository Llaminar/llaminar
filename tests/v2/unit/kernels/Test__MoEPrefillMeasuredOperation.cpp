/**
 * @file Test__MoEPrefillMeasuredOperation.cpp
 * @brief Device-free regression for complete-operation MoE measurement identity.
 *
 * The literal installed tables remain unchanged. These tests use their actual
 * keys to prove that a complete FFN measurement cannot certify a split phase,
 * compact column interval, canonical-route publication or selected ownership.
 * The same immutable tag survives ROCm live-route admission without reading
 * counts. This is dispatch provenance coverage, not GPU byte or timing evidence.
 */
#include <gtest/gtest.h>

#include "kernels/common/MoEPrefillProjectionExecution.h"
#include "kernels/common/MoEGroupedOffsetSlots.h"
#include "kernels/rocm/gemm/ROCmMoEProductionPrefillPolicy.h"
#include "kernels/rocm/gemm/ROCmMoEGroupedPrefillRoutePolicy.h"
#include "kernels/cuda/gemm/CUDAMoEProductionPrefillPolicy.h"
#include "kernels/cuda/gemm/CUDAMoEGroupedPrefillKernels.h"

#include <array>
#include <vector>
#include <type_traits>

namespace
{
    using Execution = llaminar2::MoEPrefillProjectionExecution;
    using Layout = llaminar2::MoEPrefillOutputLayout;
    using Scope = llaminar2::MoEPrefillRouteScope;
    using Operation = llaminar2::MoEPrefillMeasuredOperation;
    constexpr auto kComplete = Operation::CompleteFullWidthTokenRowsAllRoutes;

    /**
     * @brief Enumerate operations sharing the old table key but not its meaning.
     * @param hidden Original model width used by every arithmetic policy.
     * @return Gate/up, whole down, every TP2/4/8 down interval, canonical and
     *         selected/unproven complete publications, without reading routes.
     */
    std::vector<Operation> unmeasuredOperations(int hidden)
    {
        std::vector<Operation> operations{
            Execution::gateUp(hidden).measuredOperation(Layout::TokenRows, Scope::AllRoutes),
            Execution::gateUp(hidden).measuredOperation(Layout::TokenRows, Scope::SelectedRoutes),
            Execution::down(hidden, 0, hidden).measuredOperation(Layout::TokenRows, Scope::AllRoutes),
            Execution::complete(hidden).measuredOperation(Layout::OriginalRouteRows, Scope::AllRoutes),
            Execution::complete(hidden).measuredOperation(Layout::TokenRows, Scope::SelectedRoutes),
            Execution::complete(hidden).measuredOperation(Layout::TokenRows, Scope::Unspecified)};
        for (int degree : {2, 4, 8})
            for (int ordinal = 0; ordinal < degree; ++ordinal)
                for (auto layout : {Layout::TokenRows, Layout::OriginalRouteRows})
                    operations.push_back(Execution::down(hidden, ordinal * (hidden / degree), hidden / degree)
                                             .measuredOperation(layout, Scope::AllRoutes));
        return operations;
    }
}

/** @test Duplicate routes and interior sentinel holes retain every original expert row exactly once. */
TEST(MoEPrefillMeasuredOperation, PackedDirectoryOffsetSlotsReplay)
{
    using llaminar2::moe_grouped_directory::WideFloor8OffsetSlots;
    constexpr uint32_t sentinel=UINT32_MAX;
    static_assert(sizeof(int)==sizeof(uint32_t));
    for (const auto counts : {std::vector<int>{0,1,7,8,15,16,17,31,32,33,64,128},
                             std::vector<int>{17,0,33,1,7,8},std::vector<int>{128},std::vector<int>{0,0}})
    {
        int slots=0;for (int count:counts)slots+=count;
        const int capacity=std::max(1,slots);
        std::vector<uint32_t> directory(capacity,sentinel);
        int offset=0,expected_tiles=0;
        for (int expert=0;expert<int(counts.size());++expert)
        {
            const int count=counts[expert];
            const auto range=WideFloor8OffsetSlots::bind(count,offset,capacity);
            const int tiles=count ? (count<8 ? 1 : (count+15)/16) : 0;
            EXPECT_EQ(range.tiles(),tiles);expected_tiles+=tiles;
            for (int tile=0;tile<range.tiles();++tile)
            {
                const int slot=range.slot(tile);ASSERT_GE(slot,offset);ASSERT_LT(slot,offset+count);
                ASSERT_EQ(directory[slot],sentinel);directory[slot]=range.descriptor(expert,int(counts.size()),tile);
            }
            offset+=count;
        }
        std::vector<std::vector<int>> visits;for (int count:counts)visits.emplace_back(count,0);
        int observed=0;
        // Deliberately scan beyond holes: an empty word is not the end of work.
        for (uint32_t word:directory)
        {
            if (word==sentinel)continue;
            ++observed;const int expert=word&1023u,first=word>>10;
            ASSERT_GE(expert,0);ASSERT_LT(expert,int(counts.size()));
            const int step=counts[expert]<8 ? 12 : 16;
            for (int row=first;row<std::min(first+step,counts[expert]);++row)++visits[expert][row];
        }
        EXPECT_EQ(observed,expected_tiles);
        for (const auto &expert:visits)for (int writers:expert)EXPECT_EQ(writers,1);
    }
    for (const auto invalid : {std::array{-1,0,128},std::array{1,-1,128},std::array{129,0,128},
                               std::array{17,112,128},std::array{0,129,128},std::array{0,0,0}})
        EXPECT_THROW((void)WideFloor8OffsetSlots::bind(invalid[0],invalid[1],invalid[2]),std::invalid_argument);
    const auto tail=WideFloor8OffsetSlots::bind(17,111,128);
    EXPECT_EQ(tail.slot(1),127);EXPECT_EQ(tail.descriptor(255,256,1),(16u<<10)|255u);
    EXPECT_THROW((void)tail.slot(2),std::invalid_argument);
    EXPECT_THROW((void)tail.descriptor(256,256,0),std::invalid_argument);
}

/** @test Packed-directory authority remains tied to one successful grouping generation. */
TEST(MoEPrefillMeasuredOperation, PackedDirectoryPublicationLifecycleReplay)
{
    using Owner = llaminar2::MoEPrefillGroupingProvenance;
    using Binding = llaminar2::MoEGroupingPackedDirectory;
    static_assert(!std::is_copy_constructible_v<Owner::DirectoryBorrow>);
    static_assert(!std::is_move_constructible_v<Owner::DirectoryBorrow>);
    std::array<int,256> counts{}, offsets{}, active{};
    std::array<uint32_t,128> words{};
    counts.fill(-917); offsets.fill(-923); active.fill(-929);
    int workspace = 0;
    const auto device = llaminar2::DeviceId::rocm(0);
    const auto stream = llaminar2::ExplicitGPUStream(reinterpret_cast<void *>(0x1000));
    const auto binding = Binding::wideFloor8(counts.data(),offsets.data(),active.data(),words.data(),
        words.size(),16,8,256,device,&workspace,stream);
    Owner owner;
    const auto empty = owner.borrowDirectory();
    EXPECT_FALSE(empty.present());
    {
        auto preparation = owner.beginAllRoutes();
        EXPECT_THROW((void)empty.present(),std::logic_error);
        EXPECT_THROW((void)owner.borrowDirectory(),std::logic_error);
        std::move(preparation).publish(binding);
    }
    const auto first = owner.borrowDirectory();
    ASSERT_TRUE(first.present());
    EXPECT_EQ(first.requireMatching(binding).words(),words.data());
    EXPECT_EQ(first.requireMatching(binding).totalSlots(),128);
    const auto projection = [&](const int *c,const int *o,const uint32_t *w,int rows=16,int routes=8,int experts=256,
        llaminar2::DeviceId target=llaminar2::DeviceId::rocm(0),void *producer=reinterpret_cast<void *>(0x1000)) {
        return first.requireProjection(c,o,w,rows,routes,experts,target,producer).totalSlots();
    };
    EXPECT_EQ(projection(counts.data(),offsets.data(),words.data()),128);
    EXPECT_THROW((void)projection(counts.data()+1,offsets.data(),words.data()),std::logic_error);
    EXPECT_THROW((void)projection(counts.data(),offsets.data()+1,words.data()),std::logic_error);
    EXPECT_THROW((void)projection(counts.data(),offsets.data(),words.data()+1),std::logic_error);
    EXPECT_THROW((void)projection(counts.data(),offsets.data(),words.data(),32,4),std::logic_error);
    EXPECT_THROW((void)projection(counts.data(),offsets.data(),words.data(),16,8,255),std::logic_error);
    EXPECT_THROW((void)projection(counts.data(),offsets.data(),words.data(),16,8,256,llaminar2::DeviceId::rocm(1)),std::logic_error);
    EXPECT_THROW((void)projection(counts.data(),offsets.data(),words.data(),16,8,256,device,reinterpret_cast<void *>(0x2000)),std::logic_error);
    const auto changed_device = Binding::wideFloor8(counts.data(),offsets.data(),active.data(),words.data(),
        words.size(),16,8,256,llaminar2::DeviceId::rocm(1),&workspace,stream);
    EXPECT_THROW((void)first.requireMatching(changed_device),std::logic_error);
    const auto changed_stream = Binding::wideFloor8(counts.data(),offsets.data(),active.data(),words.data(),
        words.size(),16,8,256,device,&workspace,llaminar2::ExplicitGPUStream(reinterpret_cast<void *>(0x2000)));
    EXPECT_THROW((void)first.requireMatching(changed_stream),std::logic_error);
    const auto changed_shape = Binding::wideFloor8(counts.data(),offsets.data(),active.data(),words.data(),
        words.size(),9,8,256,device,&workspace,stream);
    EXPECT_THROW((void)first.requireMatching(changed_shape),std::logic_error);
    int replacement_workspace = 0;
    const auto changed_workspace = Binding::wideFloor8(counts.data(),offsets.data(),active.data(),words.data(),
        words.size(),16,8,256,device,&replacement_workspace,stream);
    EXPECT_THROW((void)first.requireMatching(changed_workspace),std::logic_error);
    {
        auto abandoned = owner.beginAllRoutes();
        EXPECT_THROW((void)first.requireMatching(binding),std::logic_error);
    }
    EXPECT_FALSE(owner.borrowDirectory().present());
    {
        auto reset_during_preparation = owner.beginAllRoutes();
        owner.invalidate();
        EXPECT_THROW(std::move(reset_during_preparation).publish(binding),std::logic_error);
    }
    EXPECT_FALSE(owner.borrowDirectory().present());
    {
        auto selected = owner.beginSelectedRoutes();
        EXPECT_THROW(std::move(selected).publish(binding),std::logic_error);
    }
    EXPECT_FALSE(owner.borrowDirectory().present());
    for (int replay = 0; replay < 3; ++replay)
    {
        auto preparation = owner.beginAllRoutes();
        std::move(preparation).publish(binding);
        const auto current = owner.borrowDirectory();
        EXPECT_EQ(current.requireMatching(binding).counts(),counts.data());
        owner.invalidate();
        EXPECT_THROW((void)current.present(),std::logic_error);
        EXPECT_THROW((void)first.present(),std::logic_error);
    }
    // Binding and revocation never inspect or repair live device count/offset data.
    {
        auto packed = owner.beginAllRoutes();
        std::move(packed).publish(binding);
        const auto prior = owner.borrowDirectory();
        auto ordinary = owner.beginAllRoutes();
        std::move(ordinary).publish();
        EXPECT_THROW((void)prior.present(),std::logic_error);
        EXPECT_FALSE(owner.borrowDirectory().present());
    }
    EXPECT_EQ(counts.front(),-917); EXPECT_EQ(offsets.back(),-923); EXPECT_EQ(active.front(),-929);
}

/** @test Expired owners and same-address replacements cannot resurrect a lexical borrow. */
TEST(MoEPrefillMeasuredOperation, PackedDirectoryOwnerLifetimeReplay)
{
    using Owner = llaminar2::MoEPrefillGroupingProvenance;
    const auto dangling = [] {
        Owner local;
        return local.borrowDirectory();
    }();
    EXPECT_THROW((void)dangling.present(),std::logic_error);
    auto abandoned_owner = [] {
        Owner local;
        return local.beginAllRoutes();
    }();
    EXPECT_THROW(std::move(abandoned_owner).publish(),std::logic_error);
    alignas(Owner) std::byte storage[sizeof(Owner)];
    Owner *owner = std::construct_at(reinterpret_cast<Owner *>(storage));
    const auto original = owner->borrowDirectory();
    EXPECT_FALSE(original.present());
    std::destroy_at(owner);
    EXPECT_THROW((void)original.present(),std::logic_error);
    owner = std::construct_at(reinterpret_cast<Owner *>(storage));
    EXPECT_THROW((void)original.present(),std::logic_error);
    EXPECT_FALSE(owner->borrowDirectory().present());
    auto all = owner->beginAllRoutes();
    std::move(all).publish();
    const auto revoked = owner->borrowDirectory();
    owner->revokeDirectory();
    EXPECT_EQ(owner->scope(),Scope::AllRoutes);
    EXPECT_THROW((void)revoked.present(),std::logic_error);
    std::destroy_at(owner);
}

/** @test Original bank bounds, device and stream must be valid before publication can exist. */
TEST(MoEPrefillMeasuredOperation, PackedDirectoryBindingRejectsInvalidStorageReplay)
{
    using Binding = llaminar2::MoEGroupingPackedDirectory;
    std::array<int,256> counts{}, offsets{}, active{};
    std::array<uint32_t,128> words{};
    int workspace = 0;
    const auto bind = [&](const int *c,const int *o,const int *a,uint32_t *w,size_t capacity,
        int rows=16,int top_k=8,int experts=256,llaminar2::DeviceId device=llaminar2::DeviceId::rocm(0),
        void *stream=reinterpret_cast<void *>(0x1000)) {
        return Binding::wideFloor8(c,o,a,w,capacity,rows,top_k,experts,device,&workspace,
            llaminar2::ExplicitGPUStream(stream));
    };
    EXPECT_NO_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),128));
    EXPECT_THROW((void)bind(nullptr,offsets.data(),active.data(),words.data(),128),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),counts.data()+1,active.data(),words.data(),128),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),reinterpret_cast<uint32_t *>(active.data()+1),128),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),127),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),128,0),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),128,16,0),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),128,16,8,1025),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),128,
        std::numeric_limits<int>::max(),2),std::invalid_argument);
    EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),128,16,8,256,
        llaminar2::DeviceId::cpu()),std::invalid_argument);
    for (uintptr_t legacy : {uintptr_t{0},uintptr_t{1},uintptr_t{2}})
        EXPECT_THROW((void)bind(counts.data(),offsets.data(),active.data(),words.data(),128,16,8,256,
            llaminar2::DeviceId::rocm(0),reinterpret_cast<void *>(legacy)),std::invalid_argument);
}

/** @test Shared verifier origin is explicit, count-independent, revocable and complete-operation-only. */
TEST(MoEPrefillMeasuredOperation, SharedVerifierOriginReplay)
{
    using llaminar2::DeviceRowRange;
    using Kind = llaminar2::MoESharedExpertGroupKind;
    llaminar2::MoEPrefillGroupingProvenance provenance;
    int32_t unpublished = std::numeric_limits<int32_t>::min();
    const auto counted = DeviceRowRange::deviceCounted(16, &unpublished);
    for (const auto rows : {counted, DeviceRowRange::fullyActive(16), counted.slice(0, 9),
                           counted.slice(1, 15), counted.repeatEachRow(2), counted})
    {
        auto publication = provenance.beginSharedExpert(Kind::DecodeEquivalentVerifier, rows);
        EXPECT_EQ(provenance.scope(), Scope::Unspecified);
        EXPECT_THROW((void)provenance.beginAllRoutes(), std::logic_error);
        std::move(publication).publish();
        const bool original = rows.captureIdentity() == counted.captureIdentity();
        EXPECT_EQ(provenance.scope(), original ? Scope::SharedVerifierCountedOriginalRows : Scope::Unspecified);
        EXPECT_EQ(provenance.scopeForProjection(rows.physicalRows(), 1), provenance.scope());
        if (original)
        {
            EXPECT_THROW((void)provenance.scopeForProjection(9, 1), std::logic_error);
            EXPECT_THROW((void)provenance.scopeForProjection(32, 1), std::logic_error);
            EXPECT_THROW((void)provenance.scopeForProjection(16, 2), std::logic_error);
        }
        EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::TokenRows, provenance.scope()),
            original ? Operation::CompleteFullWidthSharedVerifierTokenRows : Operation::Unmeasured);
        EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::OriginalRouteRows, provenance.scope()), Operation::Unmeasured);
        for (auto execution : {Execution::gateUp(2048), Execution::down(2048, 0, 2048), Execution::down(2048, 0, 1024)})
            EXPECT_EQ(execution.measuredOperation(Layout::TokenRows, provenance.scope()), Operation::Unmeasured);
        EXPECT_EQ(unpublished, std::numeric_limits<int32_t>::min());
    }
    {
        auto failed = provenance.beginSharedExpert(Kind::Ordinary, counted);
    }
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    {
        auto ordinary = provenance.beginSharedExpert(Kind::Ordinary, counted);
        std::move(ordinary).publish();
        EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    }
    {
        auto verifier = provenance.beginSharedExpert(Kind::DecodeEquivalentVerifier, counted);
        std::move(verifier).publish();
    }
    EXPECT_THROW((void)provenance.beginSharedExpert(static_cast<Kind>(255), counted), std::invalid_argument);
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    {
        auto wider = provenance.beginSharedExpert(Kind::DecodeEquivalentVerifier, DeviceRowRange::deviceCounted(32, &unpublished));
        std::move(wider).publish();
        EXPECT_THROW((void)provenance.scopeForProjection(16, 1), std::logic_error);
        EXPECT_EQ(provenance.scopeForProjection(32, 1), Scope::SharedVerifierCountedOriginalRows);
    }
}

/** The actual producer metadata revokes old evidence before failed setup. */
TEST(MoEPrefillMeasuredOperation, GroupingPreparationFailureAndResetRevokePriorOrigin)
{
    llaminar2::MoEPrefillGroupingProvenance provenance;
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    {
        auto publication = provenance.beginAllRoutes();
        EXPECT_EQ(provenance.scope(), Scope::Unspecified);
        std::move(publication).publish();
        EXPECT_EQ(provenance.scope(), Scope::AllRoutes);
        EXPECT_THROW(std::move(publication).publish(), std::logic_error);
    }
    {
        // A production early return abandons this same typed transaction.
        auto failed = provenance.beginSelectedRoutes();
        EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    }
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    try
    {
        auto failed = provenance.beginAllRoutes();
        throw std::runtime_error("injected setup failure before publication");
    }
    catch (const std::runtime_error &) {}
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    {
        auto publication = provenance.beginAllRoutes();
        std::move(publication).publish();
    }
    provenance.invalidate(); // Same operation used by reset/raw/workspace paths.
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
    EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::TokenRows, provenance.scope()),
              Operation::Unmeasured);
}

/** Masked-to-unmasked replacement publishes the origin of the new producer. */
TEST(MoEPrefillMeasuredOperation, GroupingReplacementPublishesOnlySuccessfulNewOrigin)
{
    llaminar2::MoEPrefillGroupingProvenance provenance;
    {
        auto selected = provenance.beginSelectedRoutes();
        std::move(selected).publish();
    }
    EXPECT_EQ(provenance.scope(), Scope::SelectedRoutes);
    EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::TokenRows, provenance.scope()),
              Operation::Unmeasured);
    {
        auto all = provenance.beginAllRoutes();
        // The raw count/scan/scatter helper invalidates any stale private map.
        provenance.invalidate();
        std::move(all).publish(); // Its enclosing complete producer succeeded.
    }
    EXPECT_EQ(provenance.scope(), Scope::AllRoutes);
    EXPECT_EQ(Execution::complete(2048).measuredOperation(Layout::TokenRows, provenance.scope()), kComplete);
    {
        auto failed = provenance.beginSelectedRoutes();
    }
    EXPECT_EQ(provenance.scope(), Scope::Unspecified);
}

/** Overlapping or consumed tokens cannot restore an obsolete all-route origin. */
TEST(MoEPrefillMeasuredOperation, GroupingPreparationRejectsOverlapAndConsumedReuse)
{
    llaminar2::MoEPrefillGroupingProvenance provenance;
    auto all = provenance.beginAllRoutes();
    EXPECT_THROW((void)provenance.beginSelectedRoutes(), std::logic_error);
    EXPECT_THROW((void)provenance.beginAllRoutes(), std::logic_error);
    provenance.invalidate(); // The raw helper may invalidate within its producer.
    EXPECT_THROW((void)provenance.beginSelectedRoutes(), std::logic_error);
    std::move(all).publish(); // Rejected overlap did not steal this authority.
    EXPECT_EQ(provenance.scope(), Scope::AllRoutes);
    {
        auto selected = provenance.beginSelectedRoutes();
        EXPECT_THROW(std::move(all).publish(), std::logic_error);
        EXPECT_THROW((void)provenance.beginAllRoutes(), std::logic_error);
        std::move(selected).publish(); // A consumed token did not release B.
        EXPECT_EQ(provenance.scope(), Scope::SelectedRoutes);
    }
    {
        auto abandoned = provenance.beginAllRoutes();
        EXPECT_THROW((void)provenance.beginSelectedRoutes(), std::logic_error);
    }
    auto after_failure = provenance.beginSelectedRoutes();
    std::move(after_failure).publish(); // Failure released the active authority.
    EXPECT_EQ(provenance.scope(), Scope::SelectedRoutes);
}

/** Existing complete rows require every static operation fact simultaneously. */
TEST(MoEPrefillMeasuredOperation, QualifiesOnlyCompleteFullWidthTokenRowsAllRoutes)
{
    for (int hidden : {2048, 2560, 3072})
    {
        EXPECT_EQ(Execution::complete(hidden).measuredOperation(Layout::TokenRows, Scope::AllRoutes), kComplete);
        for (auto operation : unmeasuredOperations(hidden))
            EXPECT_EQ(operation, Operation::Unmeasured);
    }
    EXPECT_THROW((void)Execution::complete(2048).measuredOperation(
                     static_cast<Layout>(255), Scope::AllRoutes), std::invalid_argument);
    EXPECT_THROW((void)Execution::complete(2048).measuredOperation(
                     Layout::TokenRows, static_cast<Scope>(255)), std::invalid_argument);
}

/** Every ROCm installed config survives complete selection byte-for-field. */
TEST(MoEPrefillMeasuredOperation, ROCmInstalledCompleteRowsRetainEveryConfig)
{
    using namespace llaminar2::rocm;
    for (const auto &entry : generated::kROCmMoEProductionPrefillOverlayEntries)
    {
        generated::ROCmMoEProductionPrefillOverlayConfig selected{};
        ASSERT_TRUE(selectROCmMoEProductionPrefillOverlayForOperation(
            kComplete, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
            entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
        EXPECT_EQ(selected.gateup_tile_m, entry.config.gateup_tile_m);
        EXPECT_EQ(selected.gateup_tile_n, entry.config.gateup_tile_n);
        EXPECT_EQ(selected.down_tile_m, entry.config.down_tile_m);
        EXPECT_EQ(selected.down_tile_n, entry.config.down_tile_n);
    }
}

/** A failed exact lookup must not contaminate the caller's generic selection. */
TEST(MoEPrefillMeasuredOperation, ROCmSplitAndSelectedOperationsCannotInheritExactRows)
{
    using namespace llaminar2::rocm;
    for (const auto &entry : generated::kROCmMoEProductionPrefillOverlayEntries)
        for (auto operation : unmeasuredOperations(entry.hidden_size))
        {
            generated::ROCmMoEProductionPrefillOverlayConfig selected{3, 17, 5, 19};
            EXPECT_FALSE(selectROCmMoEProductionPrefillOverlayForOperation(
                operation, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
                entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
            EXPECT_EQ(selected.gateup_tile_m, 3);
            EXPECT_EQ(selected.gateup_tile_n, 17);
            EXPECT_EQ(selected.down_tile_m, 5);
            EXPECT_EQ(selected.down_tile_n, 19);
        }
}

/** Every CUDA ordered exact width, including non-generic widths, is retained. */
TEST(MoEPrefillMeasuredOperation, CUDAInstalledCompleteRowsRetainEveryConfig)
{
    using namespace llaminar2::cuda;
    for (const auto &entry : generated::kCUDAMoEProductionPrefillOverlayEntries)
    {
        int selected = -71;
        ASSERT_TRUE(selectCUDAMoEProductionPrefillOverlayForOperation(
            kComplete, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
            entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
        EXPECT_EQ(selected, entry.gateup_tile_n);
    }
}

/** Same old key with a different operation is no longer measured CUDA evidence. */
TEST(MoEPrefillMeasuredOperation, CUDASplitAndSelectedOperationsCannotInheritExactRows)
{
    using namespace llaminar2::cuda;
    for (const auto &entry : generated::kCUDAMoEProductionPrefillOverlayEntries)
        for (auto operation : unmeasuredOperations(entry.hidden_size))
        {
            int selected = -71;
            EXPECT_FALSE(selectCUDAMoEProductionPrefillOverlayForOperation(
                operation, entry.gateup_codebook, entry.down_codebook, entry.hidden_size,
                entry.expert_width, entry.expert_count, entry.top_k, entry.m, selected));
            EXPECT_EQ(selected, -71);
        }
}

/** The separate inline IMMA regimes obey the same qualification as ordered rows. */
TEST(MoEPrefillMeasuredOperation, CUDAInlineImmaExceptionsRequireCompleteOperation)
{
    using namespace llaminar2::cuda::moe;
    for (const auto &regime : kQwen35BGroupedImmaCodebookRegimes)
        for (int rows : {64, 512, 768, 1024, 4096})
            for (int down : {regime.down_codebook,
                             regime.down_codebook == kQwenIQ4ExecutionCodebook ? 4 : regime.down_codebook})
            {
                const auto complete = selectGroupedImmaLaunchPolicy(
                    kComplete, regime.gate_up_codebook, down, rows, 2048, 512, 256, 8);
                EXPECT_TRUE(complete.exact_overlay);
                EXPECT_EQ(complete.gate_up_columns, rows <= regime.columns32_max_rows
                                                       ? GroupedImmaColumns::Columns32
                                                       : GroupedImmaColumns::Columns64);
                for (auto operation : unmeasuredOperations(2048))
                {
                    const auto selected = selectGroupedImmaLaunchPolicy(
                        operation, regime.gate_up_codebook, down, rows, 2048, 512, 256, 8);
                    EXPECT_FALSE(selected.exact_overlay);
                    EXPECT_EQ(selected.gate_up_columns, GroupedImmaColumns::Columns32);
                    EXPECT_EQ(selected.down_columns, GroupedImmaColumns::Columns32);
                    EXPECT_EQ(selected.gate_up_schedule, GroupedImmaGateUpSchedule::PairedProjections);
                }
            }
}

/** Sparse endpoint exceptions belong to the measured operation, not H/F/M alone. */
TEST(MoEPrefillMeasuredOperation, ROCmRouteExceptionsRequireCompleteOperation)
{
    using namespace llaminar2::rocm;
    for (const auto &entry : kROCmMoEGroupedPrefillExactRoutePolicies)
    {
        EXPECT_TRUE(selectROCmMoEGroupedPrefillRouteStrategy(entry.key).exact);
        EXPECT_EQ(selectROCmMoEGroupedPrefillRouteStrategy(entry.key).strategy, entry.strategy);
        for (auto operation : unmeasuredOperations(entry.key.hidden_size))
        {
            auto key = entry.key;
            key.operation = operation;
            const auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
            EXPECT_FALSE(decision.exact);
            EXPECT_EQ(decision.strategy, ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
        }
    }
}

/** Device live-row projection preserves qualification rather than minting it. */
TEST(MoEPrefillMeasuredOperation, ROCmLiveAdmissionCannotResurrectExactProvenance)
{
    using namespace llaminar2::rocm;
    int unread_device_word = -919;
    auto key = kROCmMoEGroupedPrefillExactRoutePolicies.front().key;
    key.rows = 64;
    const ROCmMoEGroupedRouteAdmission complete(key, &unread_device_word, &unread_device_word);
    EXPECT_EQ(complete.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned), 32);
    for (auto operation : unmeasuredOperations(key.hidden_size))
    {
        key.operation = operation;
        const ROCmMoEGroupedRouteAdmission selected(key, &unread_device_word, &unread_device_word);
        EXPECT_TRUE(selected.maySelect(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned));
        EXPECT_TRUE(selected.maySelect(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled));
        EXPECT_EQ(selected.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::RouteOwned), 8);
        EXPECT_EQ(selected.maximumSlotsFor(ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled), 64);
        for (int slots = 0; slots <= 64; ++slots)
            EXPECT_EQ(selected.strategyForSlots(slots), slots <= 8
                          ? ROCmMoEGroupedPrefillRouteStrategy::RouteOwned
                          : ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    }
    EXPECT_EQ(unread_device_word, -919);
}

/** Unseen complete geometries and source families keep their existing exact miss. */
TEST(MoEPrefillMeasuredOperation, UnseenShapesAndFamiliesRemainExactMisses)
{
    llaminar2::rocm::generated::ROCmMoEProductionPrefillOverlayConfig rocm{3, 17, 5, 19};
    int cuda = -71;
    for (auto operation : {kComplete, Operation::Unmeasured})
    {
        EXPECT_FALSE(llaminar2::rocm::selectROCmMoEProductionPrefillOverlayForOperation(
            operation, 5, 19, 2560, 640, 512, 10, 512, rocm));
        EXPECT_FALSE(llaminar2::cuda::selectCUDAMoEProductionPrefillOverlayForOperation(
            operation, 5, 19, 2560, 640, 512, 10, 512, cuda));
        EXPECT_FALSE(llaminar2::rocm::selectROCmMoEProductionPrefillOverlayForOperation(
            operation, 23, 23, 2048, 512, 256, 8, 512, rocm));
        EXPECT_FALSE(llaminar2::cuda::selectCUDAMoEProductionPrefillOverlayForOperation(
            operation, 23, 23, 2048, 512, 256, 8, 512, cuda));
    }
    EXPECT_EQ(cuda, -71);
    EXPECT_EQ(rocm.gateup_tile_n, 17);
}
