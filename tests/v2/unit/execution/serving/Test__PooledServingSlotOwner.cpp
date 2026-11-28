/**
 * @file Test__PooledServingSlotOwner.cpp
 * @brief R52 serving-side pooled slot lifecycle: the wiring glue (identity
 * policy, miss/hit admission, retirement demotion, slot release) over the
 * REAL composed instance + flow, and the ServingSession call-order wiring
 * (pooledAdmit replaces the ring restore; pooledDemote at retirement).
 */
#include <gtest/gtest.h>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>

#include "execution/serving/PooledServingSlotOwner.h"
#include "execution/serving/ServingSession.h"
#include "kernels/cpu/attention/PagedKVArchive.h"
#include "kernels/cpu/attention/CPUFP32PagedKVTable.h"
#include "planning/PagedKVInstanceFactory.h"
#include "planning/PagedKVInstancePlan.h"

using namespace llaminar2;

namespace
{
struct TempDir
{
    std::string path;
    TempDir()
    {
        char pattern[] = "/tmp/llaminar-pooled-slot-XXXXXX";
        char *dir = ::mkdtemp(pattern);
        EXPECT_TRUE(dir != nullptr);
        path = dir ? dir : "/tmp";
    }
    ~TempDir()
    {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;
};

struct SlotFixture
{
    TempDir dir;
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PagedKVPoolBudget budget;
    PagedKVInstancePolicy policy{8, 2, 4, 4};
    std::optional<PagedKVInstance> instance;
    std::size_t membership = 0;
    SlotFixture()
    {
        budget.remaining_budget_bytes = 2 * 1024 * 1024;
        budget.frame_bytes = 64 * 1024;
        budget.request_records = 24;
        budget.max_pages_per_request = 16;
        const auto device = DeviceId::cpu();
        // The instance plan seats 12 issuances (banks/headroom from the
        // request policy) with widened request records: the slot lifecycle
        // holds the import grant + the demotion's atomic-truncate grant
        // simultaneously (the composition's algebra, not the single-binding
        // one).
        const auto policy = PagedKVInstancePolicy{/*page_tokens=*/8,
            /*bank_count=*/12 *12, /*readers_per_bank=*/4,
            /*request_records=*/16};
        const auto plan = planPagedKVInstance(PagedKVTierConfig{}, budget,
            policy, device);
        EXPECT_TRUE(plan.has_value()) << "slot fixture base plan refused";
        membership = plan->pool_geometry.frames + 4;
        PhysicalMemoryPlanBuilder builder0;
        builder0.add({.world_rank=0, .device=device,
            .total_bytes=plan->total_bytes,
            .admission_available_bytes=plan->total_bytes},
            PhysicalMemoryOwner::KVCache, plan->total_bytes);
        auto authority0 = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder0.build()), 0);
        PagedKVTierConfig tiers;
        tiers.disk_enabled = true;
        tiers.disk_cap_bytes = 1 << 20;
        tiers.disk_journal_membership = membership;
        tiers.disk_dir = dir.path;
        auto outcome = admitConfiguredPagedKVInstance(authority0, tiers, budget,
            policy, device);
        EXPECT_TRUE(outcome.ok()) << "slot fixture refuse: " << outcome.refusal;
        authority = std::move(authority0);
        instance = std::move(*outcome.instance);
    }
};

/** @brief Publish `rows` rows (page-sized chunks) through the REAL table
 * (same fixture pattern as the prefix-flow suite: grant -> fresh write ->
 * page append -> committed snapshot -> archive view). */
PagedKVArchive publishArchive(SlotFixture &fx, int rows, int seed = 1)
{
    const auto pages_needed = (rows + 7) / 8;
    auto grant = fx.instance->pool().tryAdmit(static_cast<std::size_t>(pages_needed));
    if (!grant) throw std::logic_error("slot fixture grant refused");
    constexpr int heads = 1;
    constexpr int dim = 8;
    const auto width = static_cast<size_t>(heads) * static_cast<size_t>(dim);
    std::vector<float> data(static_cast<size_t>(rows) * 2 * width);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>((static_cast<int>(i + seed) % 13) - 6) * 0.5f;
    auto update = fx.instance->table().tryBeginUpdate();
    if (!update) throw std::logic_error("slot fixture table refused");
    std::size_t offset = 0;
    int remaining = rows;
    while (remaining > 0)
    {
        const auto this_rows = remaining >= 8 ? 8 : remaining;
        const auto bytes = static_cast<size_t>(this_rows) * 2 * width * sizeof(float);
        auto frame = grant->assignFrame();
        auto write = grant->prepareFreshWrite(frame, bytes);
        if (!write) throw std::logic_error("slot fixture write refused");
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

/** @test R52 wiring: MISS -> publish via the binding -> retire (demote) ->
 * HIT (restore, byte-identical row count) -> retire again; the slot binding
 * is fresh after every release; the store quota and directory follow the
 * owner's lifecycle (no anonymous quota). */
TEST(PooledServingSlotOwner, MissDemoteHitDemoteRoundTrip)
{
    SlotFixture fx;
    PooledServingSlotOwner owner(*fx.instance, /*slots=*/2, /*kv_heads=*/1,
        /*head_dim=*/8, /*layers=*/1, /*page_tokens=*/8,
        /*admitted_issuances=*/12, /*readers_per_bank=*/4,
        std::nullopt, /*trust_domain=*/42, /*metadata_capacity=*/4,
        /*retention_capacity=*/8);

    const std::vector<int32_t> prompt{1, 3, 5, 7, 9, 11, 13, 15};
    std::string error;
    int restored = -1;
    // MISS.
    ASSERT_TRUE(owner.admitSlot(0, 801, "24h", "k", prompt, restored, error))
        << error;
    EXPECT_EQ(restored, 0);
    EXPECT_TRUE(owner.bindingFor(0) != nullptr);
    // Publish a live publication into the slot binding (table-level import;
    // the same authority the DGO path uses — byte-exact seeded rows).
    auto source = publishArchive(fx, 12);
    ASSERT_TRUE(owner.bindingFor(0)->attachStoredImport({source}, 0, 0));
    EXPECT_EQ(owner.bindingFor(0)->priorFor(0)->snapshot.view().rows(), 12);
    // RETIREMENT demotes 1 of 2 pages (tail-first), releases the slot.
    ASSERT_TRUE(owner.demoteSlot(0, error)) << error;
    EXPECT_TRUE(owner.bindingFor(0) == nullptr);
    EXPECT_GT(fx.instance->diskStore()->quota().bytesAdmitted(), 0u);
    // HIT on slot 1 with the SAME prompt + key: restored 12 rows.
    ASSERT_TRUE(owner.admitSlot(1, 802, "24h", "k", prompt, restored, error))
        << error;
    EXPECT_EQ(restored, 12);
    EXPECT_TRUE(owner.bindingFor(1) != nullptr);
    EXPECT_EQ(owner.bindingFor(1)->priorFor(0)->snapshot.view().rows(), 12);
    EXPECT_EQ(fx.instance->diskStore()->quota().bytesAdmitted(), 0u);
    // Byte identity of the restored publication vs the original section.
    auto reexport = owner.bindingFor(1)->exportLayer(0);
    ASSERT_TRUE(reexport.has_value());
    const auto a = source.serialize();
    const auto b = reexport->serialize();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);
    // Retire the hit: demote again (12 rows = 2 pages -> 1 page demoted).
    ASSERT_TRUE(owner.demoteSlot(1, error)) << error;
    EXPECT_TRUE(owner.bindingFor(1) == nullptr);
    EXPECT_GT(fx.instance->diskStore()->quota().bytesAdmitted(), 0u);
}

/** @test R52 wiring identity policy: the same prompt under a DIFFERENT key is
 * a DIFFERENT namespace (no cross-key hit); the same key with a DIFFERENT
 * prompt is a different prefix (miss). */
TEST(PooledServingSlotOwner, IdentityPolicyIsByteIdentityPerNamespace)
{
    SlotFixture fx;
    PooledServingSlotOwner owner(*fx.instance, /*slots=*/3, 1, 8, 1, 8, 12, 4,
        std::nullopt, 42, 4, 8);
    const std::vector<int32_t> prompt{1, 3, 5, 7, 9, 11, 13, 15};
    std::string error;
    int restored = -1;
    ASSERT_TRUE(owner.admitSlot(0, 811, "24h", "tenant-a", prompt, restored, error)) << error;
    auto source = publishArchive(fx, 12);
    ASSERT_TRUE(owner.bindingFor(0)->attachStoredImport({source}, 0, 0));
    ASSERT_TRUE(owner.demoteSlot(0, error)) << error;
    // Same prompt, different key -> different namespace -> MISS.
    ASSERT_TRUE(owner.admitSlot(1, 812, "24h", "tenant-b", prompt, restored, error)) << error;
    EXPECT_EQ(restored, 0);
    // Same key, different prompt -> different prefix -> MISS.
    const std::vector<int32_t> other{2, 4, 6, 8, 10, 12, 14, 16};
    ASSERT_TRUE(owner.admitSlot(2, 813, "24h", "tenant-a", other, restored, error)) << error;
    EXPECT_EQ(restored, 0);
}

/** @test R52 wiring retirement (F1(a)): a single-page prefix is stored
 * WHOLE on the disk tier at retirement (head-only extents, store quota
 * charged) and the SAME prompt's FOLLOW-UP request on a fresh slot is a HIT
 * that restores every row — never a zombie record, never a duplicate
 * refusal. */
TEST(PooledServingSlotOwner, SinglePagePrefixIsStoreResidentAndHitsLater)
{
    SlotFixture fx;
    PooledServingSlotOwner owner(*fx.instance, 2, 1, 8, 1, 8, 12, 4,
        std::nullopt, 42, 4, 8);
    const std::vector<int32_t> prompt{1, 3, 5, 7};
    std::string error;
    int restored = -1;
    ASSERT_TRUE(owner.admitSlot(0, 821, "24h", "k", prompt, restored, error)) << error;
    auto source = publishArchive(fx, 8); // Exactly one full page.
    ASSERT_TRUE(owner.bindingFor(0)->attachStoredImport({source}, 0, 0));
    ASSERT_TRUE(owner.demoteSlot(0, error)) << error;
    EXPECT_GT(fx.instance->diskStore()->quota().bytesAdmitted(), 0u);
    EXPECT_TRUE(owner.bindingFor(0) == nullptr);
    // FOLLOW-UP: the same prefix on a fresh slot must HIT (restore 8 rows).
    ASSERT_TRUE(owner.admitSlot(1, 822, "24h", "k", prompt, restored, error))
        << error;
    EXPECT_EQ(restored, 8);
    EXPECT_EQ(fx.instance->diskStore()->quota().bytesAdmitted(), 0u);
    EXPECT_EQ(owner.bindingFor(1)->priorFor(0)->snapshot.view().rows(), 8);
    // Byte identity again: head-only store restores the exact section.
    auto reexport = owner.bindingFor(1)->exportLayer(0);
    ASSERT_TRUE(reexport.has_value());
    const auto a = source.serialize();
    const auto b = reexport->serialize();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);
}

/** @test R52 wiring: the operator ceiling refuses at slot admission (the
 * flow's resolver is the single typed path; the HTTP admission enforcement
 * is the outer layer's and remains). */
TEST(PooledServingSlotOwner, CeilingRefusesAboveClass)
{
    SlotFixture fx;
    PooledServingSlotOwner owner(*fx.instance, 1, 1, 8, 1, 8, 12, 4,
        RetentionClass::InMemory, 42, 4, 8);
    const std::vector<int32_t> prompt{1, 3, 5, 7};
    std::string error;
    int restored = -1;
    EXPECT_FALSE(owner.admitSlot(0, 831, "24h", "k", prompt, restored, error));
    EXPECT_NE(error.find("ceiling"), std::string::npos) << error;
    EXPECT_TRUE(owner.bindingFor(0) == nullptr);
}

namespace
{
/** @brief Session-level wiring probe: pooledAdmit/pooledDemote call order. */
class PooledScriptedRows final : public IServingRowExecutor
{
public:
    static constexpr int kVocab = 8;
    int pooled_admit_calls = 0;
    int pooled_demote_calls = 0;
    int ring_restore_calls = 0;
    bool pooled = true;
    bool pooled_mode = true; // configured pooled (mode); pooled=false rejects
    bool forwardServingRows(const std::vector<int> &, const std::vector<std::vector<int>> &rows) override
    {
        logits_.assign(rows.size() * kVocab, 0.0f);
        for (std::size_t r = 0; r < rows.size(); ++r)
            logits_[r * kVocab + 3] = 1.0f;
        return true;
    }
    const float *servingRowLogits(int row) const override { return logits_.data() + row * kVocab; }
    bool resetServingRow(int) override { return true; }
    int restoreServingRowPrefix(int, const std::vector<int32_t> &, PrefixLookupResult &admission) override
    {
        ++ring_restore_calls;
        admission = {};
        return 0;
    }
    bool harvestServingRowPrefix(int, const PrefixLookupResult &, const std::vector<int32_t> &) override
    {
        return true;
    }
    bool pooledConfigured() const override { return pooled_mode; }
    bool pooledAdmit(int, std::uint64_t, std::string_view, std::string_view,
                     const std::vector<int32_t> &, int &restored) override
    {
        ++pooled_admit_calls;
        restored = pooled ? 2 : 0; // scripted restored prefix rows
        return pooled;
    }
    bool pooledDemote(int) override
    {
        ++pooled_demote_calls;
        return true;
    }
    std::string servingRowError() const override { return {}; }

private:
    std::vector<float> logits_;
};
} // namespace

/** @test R52 session wiring (F2): a POOLED refusal is a precise request
 * error — restoreServingRowPrefix is NEVER called inside a pooled session
 * (no silent mode change); the request row errors and the slot frees. */
TEST(PooledServingSlotOwner, PooledRefusalNeverFallsBackToRing)
{
    PooledScriptedRows rows;
    rows.pooled = false; // scripted pooled refusal
    ServingSession session(rows, 64, PooledScriptedRows::kVocab);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error));
    ServingBatchRequest request;
    request.id = 11;
    request.prompt = {1, 2, 3, 4, 5};
    request.max_new_tokens = 3;
    request.sampling.temperature = 0.0f;
    request.sampling.top_k = 1;
    request.sampling.seed = 1;
    request.prompt_cache_retention = "24h";
    request.prompt_cache_key = "k";
    ASSERT_TRUE(session.submit(std::move(request), error) != ServingSubmissionOutcome::Refused);
    ServingBatchEvent terminal;
    for (int step = 0; step < 8 && terminal.finish == ServingFinishReason::None; ++step)
        for (const auto &event : session.step())
            if (event.finished()) terminal = event;
    EXPECT_NE(terminal.finish, ServingFinishReason::None);
    EXPECT_EQ(terminal.finish, ServingFinishReason::Error);
    EXPECT_EQ(rows.pooled_admit_calls, 1);
    EXPECT_EQ(rows.ring_restore_calls, 0); // NEVER the ring inside pooled mode
    EXPECT_EQ(rows.pooled_demote_calls, 0);
    ASSERT_TRUE(session.idle());
}

/** @test R52 session wiring: with a pooled executor, promotion calls
 * pooledAdmit INSTEAD of the ring restore (the ring restore is not called),
 * and retirement calls pooledDemote before the reset. */
TEST(PooledServingSlotOwner, ServingSessionCallsPooledHooksInsteadOfRing)
{
    PooledScriptedRows rows;
    ServingSession session(rows, 64, PooledScriptedRows::kVocab);
    std::string error;
    ASSERT_TRUE(session.begin(1, 1, error));
    ServingBatchRequest request;
    request.id = 9;
    request.prompt = {1, 2, 3, 4, 5};
    request.max_new_tokens = 3;
    request.sampling.temperature = 0.0f;
    request.sampling.top_k = 1;
    request.sampling.seed = 1;
    request.prompt_cache_retention = "24h";
    request.prompt_cache_key = "k";
    ASSERT_TRUE(session.submit(std::move(request), error) != ServingSubmissionOutcome::Refused);
    for (int step = 0; step < 12 && !session.idle(); ++step)
        (void)session.step();
    ASSERT_TRUE(session.idle());
    EXPECT_EQ(rows.pooled_admit_calls, 1);
    EXPECT_EQ(rows.ring_restore_calls, 0); // pooledAdmit replaced the ring restore
    EXPECT_EQ(rows.pooled_demote_calls, 1); // retirement demoted exactly once
}
