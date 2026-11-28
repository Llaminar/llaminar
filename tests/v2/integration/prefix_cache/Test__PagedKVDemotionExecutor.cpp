/**
 * @file Test__PagedKVDemotionExecutor.cpp
 * @brief Mechanical demotion executor contracts (P3 M9).
 */
#include "execution/prefix_cache/PagedKVDemotionExecutor.h"
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
            ("llaminar_executor_" + std::to_string(stamp) + "_" +
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
        if (!plan) throw std::logic_error("executor fixture base plan refused");
        // The authority admits TWO instance BOMs (one composed, one plain) so
        // the disk-less refusal test can admit its own instance.
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
        if (!outcome.ok()) throw std::logic_error("executor fixture refuse: " + outcome.refusal);
        instance = std::move(*outcome.instance);
    }
};

/** @brief Export one pooled page's serialized section (R23 oracle seam). */
PagedKVArchive exportPage(PagedKVInstance &instance, int heads = 1,
    int dimension = 8, int seed = 1)
{
    auto grant = instance.pool().tryAdmit(1);
    if (!grant) throw std::logic_error("executor grant refused");
    auto frame = grant->assignFrame();
    const auto width = static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    std::vector<float> rows(static_cast<size_t>(8) * 2 * width);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>((static_cast<int>(i + seed) % 11) - 5) * 0.5f;
    auto write = grant->prepareFreshWrite(frame, rows.size() * sizeof(float));
    if (!write) throw std::logic_error("executor write refused");
    write->copyFrom(std::as_bytes(std::span(rows)));
    auto page = CPUFP32KVPage(write->commit(), heads, dimension);
    auto reservation = instance.table().reserveAcceptedWork(1);
    if (!reservation) throw std::logic_error("executor reservation refused");
    auto update = instance.table().tryBeginReserved(*reservation);
    if (!update) throw std::logic_error("executor update refused");
    update->append(std::move(page));
    auto snapshot = update->commit();
    return PagedKVArchive::fromView(snapshot.view());
}

PagedDemotionPlan oneMovePlan(std::uint64_t ns, std::uint64_t prefix,
    std::uint64_t page, const PagedKVArchiveExtentKey &extent)
{
    PagedDemotionPlan plan;
    plan.moves.push_back({ns, prefix, page, extent});
    return plan;
}
} // namespace

/** @test Execute one real demotion move: a live pooled page's bytes publish
 * through the composed store (durable, quota-charged, real extent identity). */
TEST(PagedKVDemotionExecutor, ExecutesRealMoveThroughComposedStore)
{
    Composition fx;
    auto *store = fx.instance->diskStore();
    ASSERT_TRUE(store != nullptr);
    auto section = exportPage(*fx.instance);
    const auto serialized = section.serialize();
    const auto planned = PagedKVArchiveExtentKey{100, 3, 42, 4096};
    auto plan = oneMovePlan(3, 100, 7, planned);
    auto bytes = [&](const PagedDemotionMove &) {
        return std::optional<std::vector<std::byte>>(serialized);
    };
    auto execution = executePagedDemotionPlan(plan, *fx.instance, bytes);
    ASSERT_TRUE(execution.ok()) << execution.refusal;
    ASSERT_EQ(execution.executed.size(), 1u);
    EXPECT_TRUE(store->quota().contains(DiskExtentKey{
        execution.executed[0].stored_extent.generation,
        execution.executed[0].stored_extent.namespace_id,
        execution.executed[0].stored_extent.digest,
        execution.executed[0].stored_extent.bytes}));
    EXPECT_EQ(store->quota().bytesAdmitted(),
        execution.executed[0].stored_extent.bytes);
    // The stored extent really reads back byte-exact.
    std::string refusal;
    auto loaded = store->get(execution.executed[0].stored_extent, &refusal);
    ASSERT_TRUE(loaded.has_value()) << refusal;
    EXPECT_EQ(loaded->serialize().size(), serialized.size());
    EXPECT_EQ(std::memcmp(loaded->serialize().data(), serialized.data(),
        serialized.size()), 0);
    EXPECT_TRUE(store->retire(execution.executed[0].stored_extent));
    EXPECT_EQ(store->quota().bytesAdmitted(), 0u);
}

/** @test ALL-OR-NONE at the storage level: a mid-plan unavailable page rolls
 * back every extent already published (no partial persistence, no residue). */
TEST(PagedKVDemotionExecutor, MidPlanUnavailableRollsBackAll)
{
    Composition fx;
    auto *store = fx.instance->diskStore();
    ASSERT_TRUE(store != nullptr);
    auto section = exportPage(*fx.instance);
    const auto serialized = section.serialize();
    PagedDemotionPlan plan;
    plan.moves.push_back({3, 100, 1, {100, 3, 42, 4096}});
    plan.moves.push_back({3, 100, 0, {100, 3, 43, 4096}});
    auto bytes = [&](const PagedDemotionMove &m) {
        if (m.page_index != 1) return std::optional<std::vector<std::byte>>{};
        return std::optional<std::vector<std::byte>>(serialized);
    };
    auto execution = executePagedDemotionPlan(plan, *fx.instance, bytes);
    EXPECT_FALSE(execution.ok());
    EXPECT_TRUE(execution.executed.empty());
    // The FIRST move's extent was published then ROLLED BACK: no residue.
    EXPECT_EQ(store->quota().bytesAdmitted(), 0u);
    std::error_code ec;
    EXPECT_EQ(std::filesystem::directory_iterator(fx.dir.path, ec) ==
              std::filesystem::directory_iterator(), true);
}

/** @test A non-ok plan and a disk-less instance both refuse precisely. */
TEST(PagedKVDemotionExecutor, NonOkPlanAndNoStoreRefuse)
{
    Composition fx;
    PagedDemotionPlan bad;
    bad.cascade_required = true;
    auto bytes = [](const PagedDemotionMove &) {
        return std::optional<std::vector<std::byte>>{};
    };
    EXPECT_FALSE(executePagedDemotionPlan(bad, *fx.instance, bytes).ok());
    // Disk-disabled instance: its own plain authority (no composed route),
    // plan ok, but no store to execute against.
    TempDir dir2;
    auto [auth2, plan2] = ([&] {
        const auto device = DeviceId::cpu();
        auto p = planPagedKVInstance(PagedKVTierConfig{}, fx.budget, fx.policy, device);
        if (!p) throw std::logic_error("executor plain plan refused");
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank=0, .device=device,
            .total_bytes=p->total_bytes,
            .admission_available_bytes=p->total_bytes},
            PhysicalMemoryOwner::KVCache, p->total_bytes);
        return std::make_pair(
            std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
                    builder.build()), 0),
            *p);
    }());
    auto outcome = admitConfiguredPagedKVInstance(auth2, PagedKVTierConfig{},
        fx.budget, fx.policy, DeviceId::cpu());
    ASSERT_TRUE(outcome.ok()) << outcome.refusal;
    PagedDemotionPlan good = oneMovePlan(3, 100, 7, {100, 3, 42, 4096});
    auto execution = executePagedDemotionPlan(good, *outcome.instance, bytes);
    EXPECT_FALSE(execution.ok());
    EXPECT_NE(execution.refusal.find("no disk route"), std::string::npos);
}
} // namespace llaminar2
