/**
 * @file Test__ComposedPagedKVRamDiskRoute.cpp
 * @brief One parser/planner path for the composed RAM+disk tier route (P3 §11.3).
 *
 * The composed route is NOT a RAM experiment next to a disk experiment: a
 * single admitted instance carries the RAM pool AND the validated disk route
 * under one PMA authority, the planner sizes and checks the disk membership
 * from the admitted frame geometry, and actual reuse (export -> put -> get ->
 * retire) moves bytes through the SAME instance. Impossible composed routes
 * (membership below the geometry's need, quota below one rounded extent,
 * wrong directory) fail closed with a precise diagnostic and no partial
 * instance.
 */
#include "execution/prefix_cache/PagedKVArchiveStore.h"
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
            ("llaminar_composed_" + std::to_string(stamp) + "_" +
             std::to_string(::getpid()))).string();
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

struct ComposedFixture
{
    TempDir dir;
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PagedKVPoolBudget budget;
    PagedKVInstancePolicy policy{8, 2, 4, 4};
    std::size_t membership = 0; ///< Geometry-typed membership the planner needs.
    ComposedFixture()
    {
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 64 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 4;
        const auto device = DeviceId::cpu();
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
        if (!plan) throw std::logic_error("Composed fixture base plan refused");
        membership = plan->pool_geometry.frames + 4;
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank=0, .device=device,
            .total_bytes=plan->total_bytes,
            .admission_available_bytes=plan->total_bytes},
            PhysicalMemoryOwner::KVCache, plan->total_bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()), 0);
    }
    PagedKVTierConfig tier() const
    {
        PagedKVTierConfig tiers;
        tiers.disk_enabled = true;
        tiers.disk_cap_bytes = 1 << 20;
        tiers.disk_journal_membership = membership;
        tiers.disk_dir = dir.path;
        return tiers;
    }
};

/** @brief Export one page from the instance's table as an archive section,
 * so the composed route's reuse is real bytes from the admitted pool. */
PagedKVArchive exportPage(PagedKVInstance &instance, int heads = 1, int dimension = 8)
{
    auto grant = instance.pool().tryAdmit(1);
    if (!grant) throw std::logic_error("Composed fixture grant refused");
    auto frame = grant->assignFrame();
    const auto width = static_cast<size_t>(heads) * static_cast<size_t>(dimension);
    constexpr int tokens = 8;
    std::vector<float> rows(static_cast<size_t>(tokens) * 2 * width);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.5f;
    auto plan = grant->prepareFreshWrite(frame, rows.size() * sizeof(float));
    if (!plan) throw std::logic_error("Composed fixture write refused");
    plan->copyFrom(std::as_bytes(std::span(rows)));
    auto page = CPUFP32KVPage(plan->commit(), heads, dimension);
    auto reservation = instance.table().reserveAcceptedWork(1);
    if (!reservation) throw std::logic_error("Composed fixture reservation refused");
    auto update = instance.table().tryBeginReserved(*reservation);
    if (!update) throw std::logic_error("Composed fixture update refused");
    update->append(std::move(page));
    auto snapshot = update->commit();
    return PagedKVArchive::fromView(snapshot.view());
}
} // namespace

/** @test The composed route admits as ONE instance: RAM pool + validated disk
 * route under one authority, store opened from the planned geometry. */
TEST(ComposedPagedKVRamDiskRoute, ComposedRouteAdmitsAsOneInstance)
{
    ComposedFixture fx;
    const auto device = DeviceId::cpu();
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, fx.tier(),
        fx.budget, fx.policy, device);
    ASSERT_TRUE(outcome.ok()) << outcome.refusal;
    EXPECT_TRUE(outcome.instance->diskStore() != nullptr);
    EXPECT_GT(outcome.instance->diskStore()->quota().capacity(), 0u);
    EXPECT_EQ(outcome.instance->diskStore()->quota().capacity(),
        fx.membership);
    EXPECT_EQ(outcome.instance->authority().get(), fx.authority.get());
}

/** @test ACTUAL REUSE through the composed route: export a live page from the
 * admitted pool, put the section through the same instance's store (quota
 * charged), get it back byte-exact, retire it (quota released, file gone). */
TEST(ComposedPagedKVRamDiskRoute, ComposedRouteRamDiskActualReuse)
{
    ComposedFixture fx;
    const auto device = DeviceId::cpu();
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, fx.tier(),
        fx.budget, fx.policy, device);
    ASSERT_TRUE(outcome.ok()) << outcome.refusal;
    auto &instance = *outcome.instance;
    auto archive = exportPage(instance);
    auto *store = instance.diskStore();
    ASSERT_TRUE(store != nullptr);
    std::string refusal;
    auto key = store->put(archive, /*generation=*/1, /*namespace_id=*/3, &refusal);
    ASSERT_TRUE(key.has_value()) << refusal;
    EXPECT_GT(store->quota().bytesAdmitted(), 0u);
    auto loaded = store->get(*key, &refusal);
    ASSERT_TRUE(loaded.has_value()) << refusal;
    const auto a = archive.serialize();
    const auto b = loaded->serialize();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);
    EXPECT_TRUE(store->retire(*key));
    EXPECT_EQ(store->quota().bytesAdmitted(), 0u);
    // Retirement removed the only extent file: the engine-owned directory is
    // empty again (no debris, no foreign surprise).
    std::error_code ec;
    EXPECT_TRUE(std::filesystem::is_empty(store->directory(), ec));
}

/** @test A membership cap below the planned geometry's typed extent need
 * refuses BEFORE any instance exists, with the precise number. */
TEST(ComposedPagedKVRamDiskRoute, MembershipBelowGeometryRefuses)
{
    ComposedFixture fx;
    auto tiers = fx.tier();
    tiers.disk_journal_membership = fx.membership - 1;
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, tiers,
        fx.budget, fx.policy, DeviceId::cpu());
    EXPECT_FALSE(outcome.ok());
    EXPECT_NE(outcome.refusal.find("membership"), std::string::npos);
    EXPECT_FALSE(outcome.instance.has_value());
    // No instance was admitted: the authority still holds nothing.
    EXPECT_EQ(fx.authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test A quota that cannot hold one rounded extent of the admitted frame
 * geometry refuses with the exact byte numbers. */
TEST(ComposedPagedKVRamDiskRoute, QuotaBelowOneExtentRefuses)
{
    ComposedFixture fx;
    auto tiers = fx.tier();
    tiers.disk_cap_bytes = pagedKVArchiveExtentBytes(fx.budget.frame_bytes) - 1;
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, tiers,
        fx.budget, fx.policy, DeviceId::cpu());
    EXPECT_FALSE(outcome.ok());
    EXPECT_NE(outcome.refusal.find("one rounded extent"), std::string::npos);
    EXPECT_EQ(fx.authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test A wrong (missing) directory refuses the whole composed instance —
 * no pool, no table, no disk store partials. */
TEST(ComposedPagedKVRamDiskRoute, WrongDirectoryRefusesWholeInstance)
{
    ComposedFixture fx;
    auto tiers = fx.tier();
    tiers.disk_dir = fx.dir.path + "/does-not-exist";
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, tiers,
        fx.budget, fx.policy, DeviceId::cpu());
    EXPECT_FALSE(outcome.ok());
    EXPECT_NE(outcome.refusal.find("composed disk route refused"),
        std::string::npos);
    EXPECT_EQ(fx.authority->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::KVCache,
        PhysicalMemoryMaterializationKind::NewAllocation), 0u);
}

/** @test Disk disabled (default tier intent): no disk store, no route, and
 * nothing is ever admitted to a directory. */
TEST(ComposedPagedKVRamDiskRoute, DisabledTierHasNoStore)
{
    ComposedFixture fx;
    auto outcome = admitConfiguredPagedKVInstance(fx.authority, PagedKVTierConfig{},
        fx.budget, fx.policy, DeviceId::cpu());
    ASSERT_TRUE(outcome.ok()) << outcome.refusal;
    EXPECT_TRUE(outcome.instance->diskStore() == nullptr);
}
} // namespace llaminar2
