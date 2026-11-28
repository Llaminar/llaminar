/**
 * @file Test__PooledBindingImport.cpp
 * @brief Binding-level nonresident import (attachStoredImport) contracts (P3 M10).
 */
#include "execution/compute_stages/stages/PooledKVDecodeBinding.h"
#include "kernels/cpu/attention/PagedKVArchive.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <gtest/gtest.h>

namespace llaminar2
{
namespace
{
struct ImportFixture
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    std::shared_ptr<PhysicalPagePool> pool;
    std::shared_ptr<CPUFP32PagedKVTable> table;
    int page_tokens = 8;
    int kv_heads = 1;
    int dimension = 8;
    explicit ImportFixture(int dim = 8) : dimension(dim)
    {
        const auto device = DeviceId::cpu();
        const PhysicalPagePoolGeometry pool_geometry{24,
            static_cast<std::size_t>(page_tokens) * 2 * kv_heads * dimension *
                sizeof(float),
            /*request_records=*/6, 4};
        const CPUFP32PagedKVTableGeometry table_geometry{8, 4, 4, page_tokens,
            32, 8};
        const auto bytes = PhysicalPagePool::physicalBytes(pool_geometry, device) +
            CPUFP32PagedKVTable::physicalBytes(table_geometry, device);
        PhysicalMemoryPlanBuilder plan;
        plan.add({.world_rank=0, .device=device, .total_bytes=bytes,
            .admission_available_bytes=bytes},
            PhysicalMemoryOwner::KVCache, bytes);
        authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        pool = authority->createCPUPagePool(device, pool_geometry, CPUStoragePlacement::local());
        table = CPUFP32PagedKVTable::create(authority, device, table_geometry);
    }
    PooledKVDecodeBinding binding(std::uint64_t request_id, int layers = 1,
        std::size_t issuances = 4)
    {
        return PooledKVDecodeBinding(pool.get(), table.get(), request_id, 0,
            kv_heads, dimension, page_tokens, layers, issuances);
    }
};

/** @brief Publish `rows` native rows through the table in page-sized chunks
 * (the fixture frame is exactly one page); returns the archive. */
PagedKVArchive publishArchive(ImportFixture &fx, int rows, int seed = 1)
{
    const auto pages_needed = (rows + fx.page_tokens - 1) / fx.page_tokens;
    auto grant = fx.pool->tryAdmit(static_cast<std::size_t>(pages_needed));
    if (!grant) throw std::logic_error("import fixture grant refused");
    const auto width = static_cast<size_t>(fx.kv_heads) * static_cast<size_t>(fx.dimension);
    std::vector<float> data(static_cast<size_t>(rows) * 2 * width);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>((static_cast<int>(i + seed) % 13) - 6) * 0.5f;
    auto update = fx.table->tryBeginUpdate();
    if (!update) throw std::logic_error("import fixture table refused");
    std::size_t offset = 0;
    int remaining = rows;
    while (remaining > 0)
    {
        const auto this_rows = remaining >= fx.page_tokens
            ? fx.page_tokens : remaining;
        const auto bytes = static_cast<size_t>(this_rows) * 2 * width *
            sizeof(float);
        auto frame = grant->assignFrame();
        auto write = grant->prepareFreshWrite(frame, bytes);
        if (!write) throw std::logic_error("import fixture write refused");
        write->copyFrom(std::as_bytes(std::span(data).subspan(offset, bytes /
            sizeof(float))));
        offset += bytes / sizeof(float);
        remaining -= this_rows;
        auto page = CPUFP32KVPage(write->commit(), fx.kv_heads, fx.dimension);
        update->append(std::move(page));
    }
    auto snapshot = update->commit();
    return PagedKVArchive::fromView(snapshot.view());
}
} // namespace

/** @test A stored section imports into a fresh binding: byte-exact rows,
 * provenance recorded, epoch bumped, cursor usable. */
TEST(PooledBindingImport, StoredSectionImportsByteExact)
{
    ImportFixture fx;
    const auto archive = publishArchive(fx, /*rows=*/12, /*seed=*/3);
    auto binding = fx.binding(/*request_id=*/701, /*layers=*/1);
    EXPECT_FALSE(binding.hasLivePublication());
    const auto epoch_before = binding.cursorEpoch();
    EXPECT_TRUE(binding.attachStoredImport({archive},
        archive.meta().trust_domain, archive.meta().ancestry_id));
    ASSERT_TRUE(binding.hasLivePublication());
    EXPECT_EQ(binding.cursorEpoch(), epoch_before + 1);
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 12);
    // Round trip: the imported publication re-exports to the SAME bytes.
    auto reexport = binding.exportLayer(0);
    ASSERT_TRUE(reexport.has_value());
    const auto a = archive.serialize();
    const auto b = reexport->serialize();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);
    // Provenance record present (collision-resolving ancestry identity).
    ASSERT_EQ(binding.importAncestry().size(), 1u);
    EXPECT_EQ(binding.importAncestry()[0].rows, 12u);
    EXPECT_EQ(binding.importAncestry()[0].heads, fx.kv_heads);
}

/** @test Import accepts a MID-PAGE tail (partial final page) exactly. */
TEST(PooledBindingImport, PartialTailImportExact)
{
    ImportFixture fx;
    const auto archive = publishArchive(fx, /*rows=*/11, /*seed=*/5); // Mid-page.
    auto binding = fx.binding(702);
    EXPECT_TRUE(binding.attachStoredImport({archive},
        archive.meta().trust_domain, archive.meta().ancestry_id));
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 11);
    auto reexport = binding.exportLayer(0);
    ASSERT_TRUE(reexport.has_value());
    const auto a = archive.serialize();
    const auto b = reexport->serialize();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size()), 0);
}

/** @test Multi-layer import requires one archive per layer, equal extents
 * (a per-layer skew refuses before any mutation), and succeeds when exact. */
TEST(PooledBindingImport, MultiLayerSkewRefusesAndExactSucceeds)
{
    ImportFixture fx;
    const auto archive_a = publishArchive(fx, 12, 1);
    const auto archive_b = publishArchive(fx, 9, 2); // Different extent.
    auto binding = fx.binding(703, /*layers=*/2);
    // Layer count mismatch refuses.
    EXPECT_FALSE(binding.attachStoredImport({archive_a}, 0, 0));
    EXPECT_FALSE(binding.hasLivePublication());
    // Skewed extents refuse.
    EXPECT_FALSE(binding.attachStoredImport({archive_a, archive_b}, 0, 0));
    EXPECT_FALSE(binding.hasLivePublication());
    // Exact pair succeeds; both layers byte-exact (the FLOW supplies the
    // route expectations; this binding-level self-consistency case passes
    // the archive's own record).
    const auto archive_b2 = publishArchive(fx, 12, 8);
    EXPECT_TRUE(binding.attachStoredImport({archive_a, archive_b2},
        archive_a.meta().trust_domain, archive_a.meta().ancestry_id));
    ASSERT_TRUE(binding.hasLivePublication());
    ASSERT_EQ(binding.importAncestry().size(), 2u);
    const auto a = archive_a.serialize();
    const auto b1 = binding.exportLayer(0)->serialize();
    const auto b2 = binding.exportLayer(1)->serialize();
    EXPECT_EQ(std::memcmp(a.data(), b1.data(), a.size()), 0);
    EXPECT_EQ(std::memcmp(archive_b2.serialize().data(), b2.data(),
        archive_b2.serialize().size()), 0);
}

/** @test Gate negatives: live destination, geometry mismatch, cancelled
 * binding — all refuse WITHOUT mutation (epoch unchanged, no live pub). */
TEST(PooledBindingImport, GateNegativesRefuseWithoutMutation)
{
    ImportFixture fx;
    const auto archive = publishArchive(fx, 12, 1);
    // Fresh-history refusal: first import succeeds, second refuses.
    auto binding = fx.binding(704);
    SCOPED_TRACE("case A: fresh destination");
    EXPECT_TRUE(binding.attachStoredImport({archive},
        archive.meta().trust_domain, archive.meta().ancestry_id));
    const auto epoch = binding.cursorEpoch();
    SCOPED_TRACE("case B: duplicate attach refusal");
    EXPECT_FALSE(binding.attachStoredImport({archive}, 0, 0));
    EXPECT_EQ(binding.cursorEpoch(), epoch);
    // Geometry mismatch: a binding with a DIFFERENT dimension (its own frame
    // geometry admits its envelope) imports the archive and refuses.
    ImportFixture wide; // dimension 16 => two-row pages of 1024 bytes.
    SCOPED_TRACE("case C: wide fixture ctor");
    auto other = PooledKVDecodeBinding(wide.pool.get(), wide.table.get(), 705, 0,
        1, 16, 8, 1, 4);
    EXPECT_FALSE(other.attachStoredImport({archive}, 0, 0)); // heads/dim mismatch.
    EXPECT_FALSE(other.hasLivePublication());
    // Cancelled binding refuses.
    // Fresh pool/table for the cancelled case: the shared fixture's tail
    // envelope is seated for the bindings above; isolation is not the point
    // here (the import gate is).
    ImportFixture cancelled_fx;
    auto cancelled = PooledKVDecodeBinding(cancelled_fx.pool.get(),
        cancelled_fx.table.get(), 706, 0, 1, 8, 8, 1, 4);
    cancelled.cancelRequest();
    EXPECT_FALSE(cancelled.attachStoredImport({archive}, 0, 0));
    // NOTE: a VALID archive on another authority of the same geometry does
    // import legitimately — the binding restores bytes only; the
    // trust-domain/namespace gate is the FLOW's job (store route + resolver
    // ownership), never a binding-level content refusal.
    ImportFixture other_fx;
    auto foreign = PooledKVDecodeBinding(other_fx.pool.get(), other_fx.table.get(),
        707, 0, fx.kv_heads, fx.dimension, fx.page_tokens, 1, 4);
    EXPECT_TRUE(foreign.attachStoredImport({archive},
        archive.meta().trust_domain, archive.meta().ancestry_id));
    ASSERT_TRUE(foreign.hasLivePublication());
}
/** @test F1 (atomic truncate) negative DIRECT: an out-of-range rows request
 * refuses in the PRECHECK with NO layer changed and NO epoch bump; a valid
 * request after it succeeds on every layer. */
TEST(PooledBindingImport, AtomicTruncatePrecheckRefusesWithoutMutation)
{
    ImportFixture fx;
    auto a = publishArchive(fx, 12, 1);
    auto b = publishArchive(fx, 12, 2);
    auto binding = fx.binding(710, /*layers=*/2);
    ASSERT_TRUE(binding.attachStoredImport({a, b}, 0, 0));
    // Shrink both layers to 8, then ask for 10: the precheck must refuse
    // BEFORE any layer changes.
    ASSERT_TRUE(binding.truncateAllLayers(8));
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 8);
    EXPECT_EQ(binding.priorFor(1)->snapshot.view().rows(), 8);
    const auto epoch = binding.cursorEpoch();
    EXPECT_FALSE(binding.truncateAllLayers(10));
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 8); // Unchanged.
    EXPECT_EQ(binding.priorFor(1)->snapshot.view().rows(), 8); // Unchanged.
    EXPECT_EQ(binding.cursorEpoch(), epoch);                    // No bump.
    // A valid following request still works -> no poisoned state.
    EXPECT_TRUE(binding.truncateAllLayers(4));
    EXPECT_EQ(binding.priorFor(0)->snapshot.view().rows(), 4);
    EXPECT_EQ(binding.priorFor(1)->snapshot.view().rows(), 4);
}

/** @test CARRY G1 primitive-level negative: a non-matching trust domain or
 * ancestry identity makes the table's authenticated attach THROW before any
 * byte is staged (the binding call site forward is tautological today; the
 * primitive itself enforces). */
TEST(PooledBindingImport, PrimitiveRefusesMismatchedTrustAndAncestry)
{
    ImportFixture fx;
    const auto archive = publishArchive(fx, 8, 1);
    auto grant = fx.pool->tryAdmit(1);
    ASSERT_TRUE(grant);
    auto update = fx.table->tryBeginUpdate();
    ASSERT_TRUE(update);
    EXPECT_THROW(update->attachArchivedPrefix(archive, *grant, fx.page_tokens,
        archive.meta().trust_domain + 1 /* mismatch */,
        archive.meta().ancestry_id), std::invalid_argument);
    EXPECT_THROW(update->attachArchivedPrefix(archive, *grant, fx.page_tokens,
        archive.meta().trust_domain,
        archive.meta().ancestry_id + 1 /* mismatch */), std::invalid_argument);
    // Matching identity stages cleanly (the trust/ancestry self-consistency
    // case the binding call site uses today).
    EXPECT_NO_THROW(update->attachArchivedPrefix(archive, *grant, fx.page_tokens,
        archive.meta().trust_domain, archive.meta().ancestry_id));
}

} // namespace llaminar2
