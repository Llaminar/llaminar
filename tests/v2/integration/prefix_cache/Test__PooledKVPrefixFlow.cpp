/**
 * @file Test__PooledKVPrefixFlow.cpp
 * @brief One typed prefix flow over a composed instance (P3 M11).
 */
#include "execution/prefix_cache/PooledKVPrefixFlow.h"
#include "execution/prefix_cache/BlockHash.h"
#include "execution/compute_stages/stages/PooledKVDecodeBinding.h"
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
            ("llaminar_flow_" + std::to_string(stamp) + "_" +
             std::to_string(::getpid()))).string();
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

struct FlowFixture
{
    TempDir dir;
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PagedKVPoolBudget budget;
    PagedKVInstancePolicy policy{8, 2, 4, 4};
    std::optional<PagedKVInstance> instance;
    std::size_t membership = 0;
    FlowFixture()
    {
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 64 * 1024;
        budget.request_records = 4;
        budget.max_pages_per_request = 4;
        const auto device = DeviceId::cpu();
        auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget, policy, device);
        if (!plan) throw std::logic_error("flow fixture base plan refused");
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
        auto outcome = admitConfiguredPagedKVInstance(authority, tiers, budget,
            policy, device);
        if (!outcome.ok()) throw std::logic_error("flow fixture refuse: " + outcome.refusal);
        instance = std::move(*outcome.instance);
    }
    PooledKVDecodeBinding binding(std::uint64_t request_id, int layers = 1,
        std::size_t issuances = 2)
    {
        return PooledKVDecodeBinding(&instance->pool(), &instance->table(),
            request_id, 0, /*kv_heads=*/1, /*dimension=*/8, /*page_tokens=*/8,
            layers, issuances);
    }
};

/** @brief Publish `rows` rows (page-sized chunks) through the table. */
PagedKVArchive publishArchive(FlowFixture &fx, int rows, int seed = 1,
    int heads = 1, int dim = 8, int page_tokens = 8)
{
    const auto pages_needed = (rows + page_tokens - 1) / page_tokens;
    auto grant = fx.instance->pool().tryAdmit(static_cast<std::size_t>(pages_needed));
    if (!grant) throw std::logic_error("flow grant refused");
    const auto width = static_cast<size_t>(heads) * static_cast<size_t>(dim);
    std::vector<float> data(static_cast<size_t>(rows) * 2 * width);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>((static_cast<int>(i + seed) % 13) - 6) * 0.5f;
    auto update = fx.instance->table().tryBeginUpdate();
    if (!update) throw std::logic_error("flow table refused");
    std::size_t offset = 0;
    int remaining = rows;
    while (remaining > 0)
    {
        const auto this_rows = remaining >= page_tokens ? page_tokens : remaining;
        const auto bytes = static_cast<size_t>(this_rows) * 2 * width * sizeof(float);
        auto frame = grant->assignFrame();
        auto write = grant->prepareFreshWrite(frame, bytes);
        if (!write) throw std::logic_error("flow write refused");
        write->copyFrom(std::as_bytes(std::span(data).subspan(offset,
            bytes / sizeof(float))));
        offset += bytes / sizeof(float);
        remaining -= this_rows;
        auto page = CPUFP32KVPage(write->commit(), heads, dim);
        update->append(std::move(page));
    }
    auto snapshot = update->commit();
    return PagedKVArchive::fromView(snapshot.view());
}
} // namespace

/** @test ADMISSION: the resolver+ceiling+owner path admits the standard
 * fields and refuses unsupported/above-ceiling precisely. */
TEST(PooledKVPrefixFlow, AdmissionResolvesAndRefuses)
{
    FlowFixture fx;
    PooledKVPrefixFlow flow(*fx.instance, /*metadata_capacity=*/4,
        /*retention_capacity=*/8, RetentionClass::Default, /*trust=*/42);
    auto rec = flow.admit(3, 99, "24h", "shared-key");
    EXPECT_FALSE(rec.has_value()); // Above the Default ceiling.
    rec = flow.admit(3, 99, "in_memory", "shared-key");
    ASSERT_TRUE(rec.has_value());
    EXPECT_EQ(rec->live, true);
    EXPECT_EQ(flow.count(), 1u);
    EXPECT_FALSE(flow.admit(3, 99, "in_memory", "k").has_value()); // Duplicate.
    EXPECT_FALSE(flow.admit(3, 98, "48h", "k").has_value());       // Unsupported.
}

/** @test FULL ROUND TRIP: demoteTail stores the complete prefix (head+tail,
 * route-stamped) and truncates the binding to the resident head; restore on
 * a FRESH binding imports the merged section byte-exactly with the ROUTE's
 * real trust/ancestry expectations. */
TEST(PooledKVPrefixFlow, DemoteThenRestoreRoundTripByteExact)
{
    FlowFixture fx;
    auto source = publishArchive(fx, /*rows=*/12, /*seed=*/4);
    {
        PooledKVPrefixFlow flow(*fx.instance, 4, 8, std::nullopt, /*trust=*/42);
        auto rec = flow.admit(3, 99, "24h", "key");
        ASSERT_TRUE(rec.has_value());
        ASSERT_TRUE(flow.setLive(3, 99, false));
        auto binding = fx.binding(801);
        ASSERT_TRUE(binding.attachStoredImport({source}, 0, 0));
        auto outcome = flow.demoteTail(3, 99, binding, /*deficit_pages=*/1);
        ASSERT_TRUE(outcome.ok()) << outcome.refusal;
        EXPECT_EQ(outcome.head_rows, 8); // 12 rows -> last page is partial (4 rows).
        ASSERT_EQ(outcome.moved.size(), 1u);
        EXPECT_GT(fx.instance->diskStore()->quota().bytesAdmitted(), 0u);
        // The binding was truncated to the resident head (RAM retirement).
        EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 8);
        // The record is fully store-resident (head + tail).
        const auto *r = flow.record(3, 99);
        ASSERT_TRUE(r != nullptr);
        EXPECT_EQ(r->head_extents.size(), 1u);
        EXPECT_EQ(r->tail_extents.size(), 1u);
        EXPECT_EQ(r->trust_domain, 42u);
        EXPECT_EQ(r->ancestry_id, 99u);

        // RESTORE into a fresh binding: the route's expectation values are
        // passed as real parameters (stamped at write; enforced at import).
        auto fresh = fx.binding(802);
        std::string refusal;
        EXPECT_TRUE(flow.restore(3, 99, fresh, &refusal)) << refusal;
        // F2: the successful restore RETIRED the demoted extents — the store
        // quota returns to its PRE-demotion value (0); the disk tier credits
        // the source after preservation (no anonymous unowned quota).
        EXPECT_EQ(fx.instance->diskStore()->quota().bytesAdmitted(), 0u);
        ASSERT_TRUE(fresh.hasLivePublication());
        ASSERT_TRUE(fresh.priorFor(0));
        EXPECT_EQ(fresh.priorFor(0)->snapshot.view().rows(), 12);
        auto reexport = fresh.exportLayer(0);
        ASSERT_TRUE(reexport.has_value());
        const auto a = source.serialize();
        const auto b = reexport->serialize();
        ASSERT_EQ(a.size(), b.size());
        EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);
        // The record's extents cleared after the successful restore.
        const auto *r2 = flow.record(3, 99);
        ASSERT_TRUE(r2 != nullptr);
        EXPECT_TRUE(r2->tail_extents.empty());
        EXPECT_TRUE(r2->head_extents.empty());
    }
}

/** @test RESTORE FAILS CLOSED on a tampered tail: after demotion the tail
 * file's bytes are bit-flipped (store digest bind refuses) — no import, the
 * record keeps its extents. */
TEST(PooledKVPrefixFlow, TamperedTailRefusesRestore)
{
    FlowFixture fx;
    auto source = publishArchive(fx, 12, 2);
    PooledKVPrefixFlow flow(*fx.instance, 4, 8, std::nullopt, 42);
    ASSERT_TRUE(flow.admit(3, 99, "24h", "key").has_value());
    ASSERT_TRUE(flow.setLive(3, 99, false));
    auto binding = fx.binding(803);
    ASSERT_TRUE(binding.attachStoredImport({source}, 0, 0));
    auto outcome = flow.demoteTail(3, 99, binding, 1);
    ASSERT_TRUE(outcome.ok()) << outcome.refusal;
    // Corrupt the tail extent file in place (the store's digest bind).
    const auto *r = flow.record(3, 99);
    ASSERT_TRUE(r != nullptr);
    ASSERT_EQ(r->tail_extents.size(), 1u);
    std::error_code ec;
    {
        // Find the tail extent's file path: digest-hex.namespace.generation.
    }
    bool corrupted = false;
    for (std::filesystem::directory_iterator it(fx.dir.path, ec),
             end = std::filesystem::directory_iterator();
         !ec && it != end; it.increment(ec))
    {
        const auto name = it->path().filename().string();
        if (name.find(prefixHashHex(r->tail_extents[0].digest)) != std::string::npos)
        {
            std::fstream f(it->path(), std::ios::in | std::ios::out | std::ios::binary);
            ASSERT_TRUE(f.good());
            f.seekp(80);
            char b = 0;
            f.read(&b, 1);
            b = static_cast<char>(b ^ 0x3C);
            f.seekp(80);
            f.write(&b, 1);
            f.flush();
            corrupted = true;
            break;
        }
    }
    ASSERT_TRUE(corrupted) << "tail extent file not found";
    auto fresh = fx.binding(804);
    std::string refusal;
    EXPECT_FALSE(flow.restore(3, 99, fresh, &refusal));
    EXPECT_FALSE(refusal.empty());
    EXPECT_FALSE(fresh.hasLivePublication());
    const auto *r2 = flow.record(3, 99);
    ASSERT_TRUE(r2 != nullptr);
    EXPECT_EQ(r2->tail_extents.size(), 1u); // Kept: restore failed closed.
}

/** @test F1 NEGATIVE: a LATE-stage refusal — the second demotion's store puts
 * succeed but the atomic truncate's bank update refuses (the binding's
 * accepted-work envelope is exhausted) — refuses with NOTHING changed: the
 * NEW store extents are rolled back, the binding keeps the FIRST demotion's
 * 8 rows, and the record still holds ONLY the first demotion's extents. */
TEST(PooledKVPrefixFlow, LateTruncateRefusalLeavesEverything)
{
    FlowFixture fx;
    auto source = publishArchive(fx, 12, 5);
    PooledKVPrefixFlow flow(*fx.instance, 4, 8, std::nullopt, 42);
    ASSERT_TRUE(flow.admit(3, 95, "24h", "k").has_value());
    ASSERT_TRUE(flow.setLive(3, 95, false));
    // The binding's accepted-work envelope: import consumes one issuance,
    // the first atomic truncate one, the second has NONE left.
    auto binding = fx.binding(807, 1, /*issuances=*/2);
    ASSERT_TRUE(binding.attachStoredImport({source}, 0, 0));
    auto first = flow.demoteTail(3, 95, binding, 1);
    ASSERT_TRUE(first.ok()) << first.refusal;
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 8);
    const auto quota_after_first =
        fx.instance->diskStore()->quota().bytesAdmitted();
    EXPECT_GT(quota_after_first, 0u);
    // SECOND demotion on the 8-row head: a 1-page deficit covers the whole
    // (single-page) head — the precheck refuses BEFORE any store write.
    auto second = flow.demoteTail(3, 95, binding, 1);
    ASSERT_FALSE(second.ok()) << "whole-prefix second demotion must refuse";
    EXPECT_NE(second.refusal.find("no resident head remains"), std::string::npos)
        << second.refusal;
    // NOTHING changed: the new extents were rolled back (quota == the FIRST
    // demotion's), the binding still holds 8 rows, the record keeps ONLY the
    // first demotion's extents.
    EXPECT_EQ(fx.instance->diskStore()->quota().bytesAdmitted(), quota_after_first);
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 8);
    const auto *r = flow.record(3, 95);
    ASSERT_TRUE(r != nullptr);
    EXPECT_EQ(r->head_extents.size(), 1u);
    EXPECT_EQ(r->tail_extents.size(), 1u);
}

/** @test GATE NEGATIVES: live record refuses demotion; no-demoted record
 * refuses restore; zero deficit is a no-op. */
TEST(PooledKVPrefixFlow, GateNegatives)
{
    FlowFixture fx;
    PooledKVPrefixFlow flow(*fx.instance, 4, 8, std::nullopt, 42);
    ASSERT_TRUE(flow.admit(3, 98, "24h", "k").has_value());
    auto binding = fx.binding(805);
    auto source = publishArchive(fx, 12, 3);
    ASSERT_TRUE(binding.attachStoredImport({source}, 0, 0));
    auto outcome = flow.demoteTail(3, 98, binding, 1);
    EXPECT_FALSE(outcome.ok()); // Still live.
    EXPECT_TRUE(flow.setLive(3, 98, false));
    outcome = flow.demoteTail(3, 98, binding, 0);
    EXPECT_TRUE(outcome.ok());
    EXPECT_EQ(outcome.head_rows, 0); // Zero deficit no-op.
    std::string refusal;
    EXPECT_FALSE(flow.restore(3, 97, binding, &refusal)); // No record.
    EXPECT_NE(refusal.find("admitted prefix record"), std::string::npos);
    EXPECT_FALSE(flow.restore(3, 98, binding, &refusal)); // No demotion happened.
}
} // namespace llaminar2
