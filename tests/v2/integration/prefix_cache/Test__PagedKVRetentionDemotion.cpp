/**
 * @file Test__PagedKVRetentionDemotion.cpp
 * @brief §8.4 demotion decision planner contracts (P3 M8).
 *
 * The planner is the DECISION DRIVER: victim order from the retention owner,
 * tail-first page selection (a hit starts at token zero, so partial demotion
 * keeps the leading pages resident), deficit-only page-rounded selection,
 * move-once for shared pages, real-overflow cascade verdict from the composed
 * route's quota, and all-or-none refusal. The mechanical byte movement
 * (snapshot -> store.put -> binding re-anchor) is the runner-side wiring M;
 * these tests use the REAL disk store for the extent identity + admission
 * parts so the plan's extents are real store-backed values.
 */
#include "execution/prefix_cache/PagedKVRetentionDemotion.h"
#include "planning/PagedKVInstanceFactory.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <unistd.h>

namespace llaminar2
{
namespace
{
struct TempDir
{
    std::string path;
    TempDir()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = (std::filesystem::temp_directory_path() /
            ("llaminar_demotion_" + std::to_string(stamp) + "_" +
             std::to_string(::getpid()))).string();
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

RetentionValueScore score(std::uint64_t recency)
{
    RetentionValueScore v;
    v.recency = recency;
    return v;
}

PagedPrefixTierEntry entry(std::uint64_t ns, std::uint64_t prefix,
    RetentionEntryKey owner, std::initializer_list<std::uint64_t> pages)
{
    PagedPrefixTierEntry e;
    e.namespace_id = ns;
    e.prefix_id = prefix;
    e.owner = owner;
    e.page_indices = pages;
    return e;
}

PagedKVArchiveExtentKey ext(std::uint64_t ns, std::uint64_t prefix,
    std::uint64_t page, std::size_t bytes = 4096, std::uint64_t digest = 7)
{
    return {prefix, ns, digest, bytes};
}

struct Composition
{
    TempDir dir;
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PagedKVPoolBudget budget;
    PagedKVInstancePolicy policy{8, 2, 4, 4};
    std::optional<PagedKVInstance> instance;
    std::size_t membership = 0;
    Composition()
    {
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 64 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 4;
        const auto device = DeviceId::cpu();
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
        if (!plan) throw std::logic_error("demotion fixture base plan refused");
        membership = plan->pool_geometry.frames + 4;
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank=0, .device=device,
            .total_bytes=plan->total_bytes,
            .admission_available_bytes=plan->total_bytes},
            PhysicalMemoryOwner::KVCache, plan->total_bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
        PagedKVTierConfig tiers;
        tiers.disk_enabled = true;
        tiers.disk_cap_bytes = 1 << 20;
        tiers.disk_journal_membership = membership;
        tiers.disk_dir = dir.path;
        auto outcome = admitConfiguredPagedKVInstance(authority, tiers, budget, policy, device);
        if (!outcome.ok()) throw std::logic_error("demotion fixture composed refuse: " + outcome.refusal);
        instance = std::move(*outcome.instance);
    }
    PagedKVArchiveStore *store() { return instance->diskStore(); }
};
} // namespace

/** @test Victim order drives the plan: the retention owner's first victim's
 * TAIL page is selected, deficit-only (one page for deficit one). */
TEST(PagedKVRetentionDemotion, TailFirstDeficitOnlyVictimOrder)
{
    Composition fx;
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::InMemory, 0}, 0, score(1), false));
    ASSERT_TRUE(owner.admit({1, 2}, {RetentionClass::Default, 0}, 0, score(2), false));
    auto page = [&](std::uint64_t ns, std::uint64_t prefix, std::uint64_t idx) {
        return std::optional<PagedKVArchiveExtentKey>(ext(ns, prefix, idx));
    };
    auto admit = [](const PagedKVArchiveExtentKey &) { return true; };
    std::vector<PagedPrefixTierEntry> entries{
        entry(1, 11, {1, 1}, {0, 1, 2}), // InMemory, 3 pages.
        entry(1, 12, {1, 2}, {0, 1})};   // Default, 2 pages.
    auto plan = planTailFirstDemotion(entries, owner, /*now=*/1,
        /*deficit_pages=*/1, page, admit);
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    ASSERT_EQ(plan.moves.size(), 1u);
    EXPECT_EQ(plan.moves[0].namespace_id, 1u);
    EXPECT_EQ(plan.moves[0].prefix_id, 11u);   // InMemory victimized first.
    EXPECT_EQ(plan.moves[0].page_index, 2u);   // TAIL page (last) first.
    EXPECT_EQ(plan.moves[0].extent.digest, 7u);
}

/** @test F2 MOVE-ONCE on PHYSICAL identity: two prefixes sharing ONE extent
 * (same resolved digest+bytes) move it exactly once; the deficit then fills
 * from the second prefix's distinct page. */
TEST(PagedKVRetentionDemotion, SharedExtentMovesExactlyOnce)
{
    Composition fx;
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::Default, 0}, 0, score(5), false));
    ASSERT_TRUE(owner.admit({1, 2}, {RetentionClass::Default, 0}, 0, score(4), false));
    // Both prefixes SHARE the physical page 1 (same resolved extent: digest
    // 111, bytes 4096); page 0 differs per prefix.
    std::vector<PagedPrefixTierEntry> entries{
        entry(1, 11, {1, 1}, {0, 1}),
        entry(1, 12, {1, 2}, {0, 1})};
    auto page = [](std::uint64_t ns, std::uint64_t prefix, std::uint64_t idx) {
        (void)ns; (void)prefix;
        return std::optional<PagedKVArchiveExtentKey>(
            ext(1, 0, idx, 4096, idx == 1 ? 1000 : (100 + prefix + idx)));
    };
    auto admit = [](const PagedKVArchiveExtentKey &) { return true; };
    // DEFICIT 3 over two prefixes: victim 12 (lower value) first, its tail
    // page 1 (shared) counted once, its page 0, then victim 11's page 1 is
    // DEDUPED (same physical extent) and its page 0 closes the deficit.
    auto plan = planTailFirstDemotion(entries, owner, 1, /*deficit=*/3, page, admit);
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    ASSERT_EQ(plan.moves.size(), 3u);
    EXPECT_EQ(plan.moves[0].prefix_id, 12u);
    EXPECT_EQ(plan.moves[0].page_index, 1u);
    EXPECT_EQ(plan.moves[2].prefix_id, 11u);
    EXPECT_EQ(plan.moves[2].page_index, 0u);
    // The shared physical page (digest 111) appears EXACTLY once.
    std::size_t shared = 0;
    for (const auto &m : plan.moves)
        if (m.extent.digest == 1000u) ++shared;
    EXPECT_EQ(shared, 1u);
}

/** @test F2 NEGATIVE: two prefixes with the SAME page index but DIFFERENT
 * physical content (different resolved digest) are never deduped — both move. */
TEST(PagedKVRetentionDemotion, SameIndexDifferentExtentMovesBoth)
{
    Composition fx;
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::Default, 0}, 0, score(5), false));
    ASSERT_TRUE(owner.admit({1, 2}, {RetentionClass::Default, 0}, 0, score(4), false));
    std::vector<PagedPrefixTierEntry> entries{
        entry(1, 11, {1, 1}, {0, 1}),
        entry(1, 12, {1, 2}, {0, 1})};
    auto page = [](std::uint64_t ns, std::uint64_t prefix, std::uint64_t idx) {
        // Same index 1 in the same namespace, DIFFERENT content per prefix.
        return std::optional<PagedKVArchiveExtentKey>(
            ext(ns, prefix, idx, 4096, (prefix == 11 ? 111u : 222u) + idx));
    };
    auto admit = [](const PagedKVArchiveExtentKey &) { return true; };
    auto plan = planTailFirstDemotion(entries, owner, 1, /*deficit=*/3, page, admit);
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    ASSERT_EQ(plan.moves.size(), 3u);
    // Victim 12 tail (page 1, its own digest 222+1), its page 0, then victim
    // 11's page 1 with its OWN digest (111+1) — the logical-index pair is NOT
    // the dedup key.
    EXPECT_EQ(plan.moves[2].prefix_id, 11u);
    EXPECT_EQ(plan.moves[2].page_index, 1u);
    EXPECT_EQ(plan.moves[2].extent.digest, 112u);
}

/** @test Live/protected prefixes are never victimized (retention owner
 * exclusion), so a plan over only-live entries refuses precisely. */
TEST(PagedKVRetentionDemotion, LiveEntriesNeverDemoted)
{
    Composition fx;
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::InMemory, 0}, 0, score(1), true));
    std::vector<PagedPrefixTierEntry> entries{entry(1, 11, {1, 1}, {0, 1})};
    auto page = [](std::uint64_t, std::uint64_t, std::uint64_t idx) {
        return std::optional<PagedKVArchiveExtentKey>(ext(1, 1, idx));
    };
    auto admit = [](const PagedKVArchiveExtentKey &) { return true; };
    auto plan = planTailFirstDemotion(entries, owner, 1, 1, page, admit);
    EXPECT_FALSE(plan.ok());
    EXPECT_FALSE(plan.refusal.empty());
}

/** @test REAL-OVERFLOW: a refuse from the composed quota turns the plan
 * into cascade_required with NO partial moves (all-or-none). */
TEST(PagedKVRetentionDemotion, RealOverflowCascadesAndNeverPartials)
{
    Composition fx;
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::Default, 0}, 0, score(1), false));
    std::vector<PagedPrefixTierEntry> entries{entry(1, 11, {1, 1}, {0, 1, 2})};
    auto page = [](std::uint64_t ns, std::uint64_t prefix, std::uint64_t idx) {
        return std::optional<PagedKVArchiveExtentKey>(ext(ns, prefix, idx));
    };
    auto admit = [](const PagedKVArchiveExtentKey &) { return false; }; // Full.
    auto plan = planTailFirstDemotion(entries, owner, 1, 2, page, admit);
    EXPECT_TRUE(plan.cascade_required);
    EXPECT_TRUE(plan.moves.empty()); // All-or-none.
    EXPECT_FALSE(plan.refusal.empty());
}

/** @test F1 MID-PLAN negative: the TAIL page (page 1) admits but the NEXT
 * selected page (page 0) refuses — the plan publishes NO moves and marks
 * cascade (all-or-none output, not a partial decision set). */
TEST(PagedKVRetentionDemotion, MidPlanRefusalDropsAllDecisions)
{
    Composition fx;
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::Default, 0}, 0, score(1), false));
    std::vector<PagedPrefixTierEntry> entries{entry(1, 11, {1, 1}, {0, 1})};
    auto page = [](std::uint64_t ns, std::uint64_t prefix, std::uint64_t idx) {
        return std::optional<PagedKVArchiveExtentKey>(ext(ns, prefix, idx));
    };
    auto admit = [](const PagedKVArchiveExtentKey &e) {
        return e.digest != 7u; // Page index 1 (digest 7) admits; page 0 refuses.
    };
    auto plan = planTailFirstDemotion(entries, owner, 1, 2, page, admit);
    EXPECT_TRUE(plan.cascade_required);
    EXPECT_TRUE(plan.moves.empty()); // Page 1's earlier decision is DROPPED.
    EXPECT_FALSE(plan.refusal.empty());
}

/** @test A selected tail page with no storable extent refuses (the page was
 * never exported; the deficit is unsatisfiable from this tier). */
TEST(PagedKVRetentionDemotion, UnresolvableExtentRefuses)
{
    Composition fx;
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::Default, 0}, 0, score(1), false));
    std::vector<PagedPrefixTierEntry> entries{entry(1, 11, {1, 1}, {0, 1})};
    auto page = [](std::uint64_t, std::uint64_t, std::uint64_t) {
        return std::optional<PagedKVArchiveExtentKey>{}; // Never exported.
    };
    auto admit = [](const PagedKVArchiveExtentKey &) { return true; };
    auto plan = planTailFirstDemotion(entries, owner, 1, 1, page, admit);
    EXPECT_TRUE(plan.refusal.find("no storable extent") != std::string::npos);
    EXPECT_TRUE(plan.moves.empty()); // F1: no partial decision leaks.
}

/** @test Zero deficit and zero-page rounding: both are no-ops. */
TEST(PagedKVRetentionDemotion, ZeroDeficitAndRounding)
{
    Composition fx;
    EXPECT_EQ(roundDeficitToPages(0, 4096), 0u);
    EXPECT_EQ(roundDeficitToPages(1, 4096), 1u);
    EXPECT_EQ(roundDeficitToPages(4096, 4096), 1u);
    EXPECT_EQ(roundDeficitToPages(4097, 4096), 2u);
    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({1, 1}, {RetentionClass::Default, 0}, 0, score(1), false));
    std::vector<PagedPrefixTierEntry> entries{entry(1, 11, {1, 1}, {0})};
    auto page = [](std::uint64_t ns, std::uint64_t prefix, std::uint64_t idx) {
        return std::optional<PagedKVArchiveExtentKey>(ext(ns, prefix, idx));
    };
    auto admit = [](const PagedKVArchiveExtentKey &) { return true; };
    auto plan = planTailFirstDemotion(entries, owner, 1, 0, page, admit);
    EXPECT_TRUE(plan.ok());
    EXPECT_TRUE(plan.moves.empty());
}

/** @test COMPOSED REAL FLOW: the plan's extent identities are REAL disk-store
 * extents (live page exported from the composed pool, store.put), the
 * wouldAdmit overflow gate is the composed quota, and after the planned move
 * the extent really is admitted in the store (the claim boundary: byte
 * movement is the wiring M, but extent ownership is real here). */
TEST(PagedKVRetentionDemotion, ComposedRealExtentFlow)
{
    Composition fx;
    auto *store = fx.store();
    ASSERT_TRUE(store != nullptr);
    // One live page exported from the pooled table through the R23 seam.
    auto grant = fx.instance->pool().tryAdmit(1);
    ASSERT_TRUE(grant);
    auto frame = grant->assignFrame();
    std::vector<float> rows(8 * 2 * 1 * 8);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(i % 7 - 3) * 0.25f;
    auto write = grant->prepareFreshWrite(frame, rows.size() * sizeof(float));
    ASSERT_TRUE(write);
    write->copyFrom(std::as_bytes(std::span(rows)));
    auto page = CPUFP32KVPage(write->commit(), 1, 8);
    auto reservation = fx.instance->table().reserveAcceptedWork(1);
    ASSERT_TRUE(reservation);
    auto update = fx.instance->table().tryBeginReserved(*reservation);
    ASSERT_TRUE(update);
    update->append(std::move(page));
    auto snapshot = update->commit();
    auto archive = PagedKVArchive::fromView(snapshot.view());
    std::string refusal;
    auto key = store->put(archive, /*generation=*/17, /*namespace_id=*/3, &refusal);
    ASSERT_TRUE(key.has_value()) << refusal;

    RetentionOwner owner(16);
    ASSERT_TRUE(owner.admit({3, 42}, {RetentionClass::Default, 0}, 0, score(1), false));
    std::vector<PagedPrefixTierEntry> entries{entry(3, 42, {3, 42}, {7})};
    auto page_extent = [&](std::uint64_t ns, std::uint64_t prefix, std::uint64_t idx) {
        (void)ns; (void)prefix;
        if (idx != 7) return std::optional<PagedKVArchiveExtentKey>{};
        return std::optional<PagedKVArchiveExtentKey>(*key);
    };
    auto admit = [&](const PagedKVArchiveExtentKey &e) {
        // The route admits a NEW extent, or an already-stored one costs
        // nothing (move-once dedup: no second charge).
        const DiskExtentKey de{e.generation, e.namespace_id, e.digest, e.bytes};
        return store->quota().contains(de) || store->quota().wouldAdmit(de);
    };
    auto plan = planTailFirstDemotion(entries, owner, 1, 1, page_extent, admit);
    ASSERT_TRUE(plan.ok()) << plan.refusal;
    ASSERT_EQ(plan.moves.size(), 1u);
    EXPECT_EQ(plan.moves[0].extent, *key);
    EXPECT_EQ(plan.moves[0].extent.digest, key->digest);
    // The extent really is admitted in the composed store; retire it cleanly.
    EXPECT_TRUE(store->quota().contains(
        DiskExtentKey{key->generation, key->namespace_id, key->digest, key->bytes}));
    EXPECT_TRUE(store->retire(*key));
    EXPECT_EQ(store->quota().bytesAdmitted(), 0u);
}
} // namespace llaminar2
