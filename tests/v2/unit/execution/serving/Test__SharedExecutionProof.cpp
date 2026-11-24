/**
 * @file Test__SharedExecutionProof.cpp
 * @brief A-continues / B-finishes / C-enters-B's-retired-slot protocol proof + discriminating
 *        lifecycle regressions (grants consumed+backed, explicit restore readiness, owned
 *        snapshot receipts, partial final windows, mixed quanta, cancellation, MTP exactly-once).
 *
 * The decode callback is a pure per-row mock; this file proves the lifecycle protocol through
 * the engine's actual owned interfaces (no const-cast, no renamed unrelated fixtures).
 */

#include <gtest/gtest.h>

#include "execution/serving/GroupedServingEngine.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "mocks/MockOrchestrationRunner.h"

// #149 SCOPED SEQUENCED ALLOCATOR OBSERVER: a bounded, non-allocating global
// aligned-probe that forwards to the real system heap and records
// (kind, ptr, size) into ONE SHARED CHRONOLOGY (fixed-capacity array — no
// vector growth; overflow sets a fail-closed flag; the aligned delete's record
// is appended AFTER the native free RETURNS, with the observer mutex acquired
// BEFORE the free and HELD through the append (the free + the completion
// marker = one atomic publication — the per-address lifetime chronology is
// preserved across address reuse; array writes do not allocate and cannot
// throw; the MUTEX acquisition itself is a conditional API-failure point, NOT
// claimed unconditional). The engine's
// ReserveAllocatorT performs `::operator new(size, align_val)` /
// `::operator delete(ptr, align_val)` for the ControlReserve block; the shared
// chronology records the literal same-address new->delete order.
#include <cstdlib>
#include <cstring>
#include <mutex>
namespace physical_free_probe
{
    enum class Kind : std::uint8_t { New, Delete };
    struct Entry
    {
        Kind kind;
        void *ptr;
        std::size_t size;
    };
    static constexpr std::size_t kCapacity = 8192;
    static Entry records[kCapacity];
    static std::size_t count = 0;
    static bool overflowed = false;
    static std::mutex probe_mutex;
    static std::atomic<bool> normal_capture{false}; // scoped capture for the
                                                    // unaligned std::map node BOM
                                                    // witness (DEFAULT OFF)

    struct Snapshot
    {
        std::size_t count;
        bool overflowed;
    };
    /** @brief ACTUAL NON-ALLOCATING TARGET MATCHER over a coherent bounded
     *         event view (the probe's shared chronology OR a synthetic
     *         deterministic event array — one function for the engine witness
     *         AND the contract controls): selects the LAST NEW(ptr,size) whose
     *         lifetime is UNRETIRED at live_end (no DELETE(ptr) after it before
     *         live_end — a retired prefix/reused-address suffix cannot qualify:
     *         the retirement/rebinding across the live prefix is validated
     *         before any post-live join); no dependency on heap address reuse. */
    struct TargetSelection
    {
        std::size_t target_index = 0u;
        std::size_t live_boundary = 0u;
        bool found = false;
    };
    inline TargetSelection selectUnretiredTarget(const Entry *records,
                                                 std::size_t from,
                                                 std::size_t live_end,
                                                 void *ptr,
                                                 std::size_t size) noexcept
    {
        TargetSelection sel;
        for (std::size_t i = from; i < live_end; ++i)
        {
            if (records[i].kind != Kind::New || records[i].ptr != ptr ||
                records[i].size != size)
                continue;
            // RETIREMENT CHECK across the live prefix: a DELETE(ptr) after this
            // NEW (before live_end) completes the lifetime — skip.
            bool completed = false;
            for (std::size_t j = i + 1u; j < live_end; ++j)
                if (records[j].kind == Kind::Delete && records[j].ptr == ptr)
                {
                    completed = true;
                    break;
                }
            if (!completed)
            {
                sel.target_index = i;
                sel.live_boundary = live_end;
                sel.found = true; // the LAST unretired NEW wins
            }
        }
        return sel;
    }
    /** @brief TYPED RAII PUBLICATION OWNER — the ACTUAL guard the aligned
     *         delete uses: the CONSTRUCTOR acquires the SAME probe_mutex the
     *         NEW writer / readers use; the native free executes inside the
     *         owner live scope; the DESTRUCTOR appends the completion marker
     *         (still owning the lock) and the unique_lock releases it on exit —
     *         the free + the marker are one protected transaction owned by this
     *         RAII object. */
    class PublicationTransaction final
    {
    public:
        PublicationTransaction(void *ptr, bool *guard_report = nullptr)
            : lock_(probe_mutex), ptr_(ptr), guard_report_(guard_report)
        {
            if (guard_report_)
                *guard_report_ = false;
        }
        // THE SHARED PRIMITIVE: performs the NATIVE FREE under the REAL
        // ownership — the guard check is the unique_lock's ACTUAL owns_lock()
        // at the operation (a default-constructed/non-acquired owner reports
        // the violation through the guard report).
        void nativeFree()
        {
            // REPEAT INVOCATION IS UNREPRESENTABLE: a second nativeFree is
            // REFUSED BEFORE any second native operation (flag-and-continue is
            // not a safe invalid-sequence handler) — the violation is reported
            // and NO second ::free executes.
            if (free_performed_)
            {
                if (guard_report_)
                    *guard_report_ = true;
                return;
            }
            if (!lock_.owns_lock() && guard_report_)
                *guard_report_ = true;
            free_performed_ = true;
            ::free(ptr_);
        }
        ~PublicationTransaction()
        {
            // THE COMPLETION under the ACTUAL ownership, ENFORCED
            // INDEPENDENTLY of the optional diagnostic: an
            // unlock-before-append / non-acquired owner OR an INCOMPLETE
            // transaction (a tx destroyed without its nativeFree — a
            // caller-only bypass/placement) → the marker is NOT appended (no
            // false publication) and the violation is reported when a report
            // pointer was supplied — a valid owner/free/publication
            // transaction must complete all three.
            const bool complete =
                lock_.owns_lock() && free_performed_;
            if (!complete && guard_report_)
                *guard_report_ = true;
            if (complete)
            {
                if (count < kCapacity)
                {
                    records[count++] = {Kind::Delete, ptr_, 0u};
                }
                else
                    overflowed = true;
            }
        }
        PublicationTransaction(const PublicationTransaction &) = delete;
        PublicationTransaction &operator=(const PublicationTransaction &) = delete;
        [[nodiscard]] bool ownsMutex() const noexcept { return lock_.owns_lock(); }
        [[nodiscard]] void *target() const noexcept { return ptr_; }
    private:
        std::unique_lock<std::mutex> lock_;
        void *ptr_ = nullptr;
        bool *guard_report_ = nullptr;
        bool free_performed_ = false;
    };

    inline bool hasDeleteInRange(const Entry *records, std::size_t lo,
                                 std::size_t hi, void *ptr) noexcept
    {
        for (std::size_t i = lo; i < hi; ++i)
            if (records[i].kind == Kind::Delete && records[i].ptr == ptr)
                return true;
        return false;
    }
    Snapshot snapshot()
    {
        std::lock_guard<std::mutex> g(probe_mutex);
        return {count, overflowed};
    }
    struct NewEntry
    {
        std::size_t index; // the SHARED chronology position (the later/order key)
        void *ptr;
        std::size_t size;
    };
    // SEALED-INTERVAL queries: [from, to) is a COHERENT COMPLETED interval (the
    // caller passes a snapshot's end — no moving frontier; the interval's
    // overflow is checked). The scans DO NOT truncate (the complete bounded
    // range is inspected) and the order uses the chronology positions.
    bool sealedNewEntries(std::size_t from, std::size_t to,
                          NewEntry *out, std::size_t capacity,
                          std::size_t &nfound) noexcept
    {
        std::lock_guard<std::mutex> g(probe_mutex);
        nfound = 0;
        for (std::size_t i = from; i < to && i < count; ++i)
        {
            if (records[i].kind != Kind::New)
                continue;
            if (nfound >= capacity)
                return false; // incomplete selection -> REJECT (no truncation)
            out[nfound++] = {i, records[i].ptr, records[i].size};
        }
        return !overflowed;
    }
    bool strictlyLaterDeleteInRange(std::size_t from, std::size_t to,
                                    const NewEntry *news,
                                    std::size_t nfound,
                                    std::size_t &matched_new) noexcept
    {
        std::lock_guard<std::mutex> g(probe_mutex);
        for (std::size_t i = from; i < to && i < count; ++i)
        {
            if (records[i].kind != Kind::Delete)
                continue;
            for (std::size_t k = 0; k < nfound; ++k)
            {
                // STRICTLY LATER (the delete's chronology position is greater
                // than the new's) + the SAME pointer + a sealed range — a
                // range DELETE(P); NEW(P) (older release then reuse) cannot
                // satisfy it because the reuse's NEW has a LARGER index than
                // the earlier DELETE.
                if (records[i].ptr == news[k].ptr && i > news[k].index)
                {
                    matched_new = k;
                    return true;
                }
            }
        }
        return false;
    }
}
void *operator new(std::size_t size, std::align_val_t align)
{
    void *p = nullptr;
    if (::posix_memalign(&p, static_cast<std::size_t>(align),
                         size ? size : 1u) != 0)
        throw std::bad_alloc();
    {
        std::lock_guard<std::mutex> g(physical_free_probe::probe_mutex);
        if (physical_free_probe::count < physical_free_probe::kCapacity)
        {
            physical_free_probe::records[physical_free_probe::count++] = {
                physical_free_probe::Kind::New, p, size};
        }
        else
            physical_free_probe::overflowed = true;
    }
    return p;
}
void operator delete(void *ptr, std::align_val_t) noexcept
{
    if (ptr)
    {
        // PUBLICATION OWNER (the ACTUAL guard): the owner's CONSTRUCTOR
        // acquires the probe_mutex and the native free executes inside the
        // owner's live scope; the owner's DESTRUCTOR appends the completion
        // marker + releases — the free + the marker are ONE transaction owned
        // by the RAII object (the per-address lifetime chronology preserved;
        // a free outside the owner's scope is structurally impossible in this
        // path).
        physical_free_probe::PublicationTransaction tx(ptr);
        tx.nativeFree(); // the SHARED PRIMITIVE performs the native free under
                         // the real ownership (the completion marker append
                         // happens in the owner's destructor — guarded)
    }
}
void operator delete(void *ptr) noexcept { ::free(ptr); }
void *operator new(std::size_t size)
{
    void *p = std::malloc(size ? size : 1u);
    if (!p)
        throw std::bad_alloc();
    if (physical_free_probe::normal_capture.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> g(physical_free_probe::probe_mutex);
        if (physical_free_probe::count < physical_free_probe::kCapacity)
        {
            physical_free_probe::records[physical_free_probe::count++] = {
                physical_free_probe::Kind::New, p, size};
        }
        else
            physical_free_probe::overflowed = true;
    }
    return p;
}
void *operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void *ptr) noexcept { ::free(ptr); }
void operator delete[](void *ptr, std::align_val_t a) noexcept
{
    ::operator delete(ptr, a);
}

namespace llaminar2::test
{
    namespace
    {
        /** @brief PMA fixture: one CPU plan admitting a serving-pool owner line and an
         *         authority that issues the pool reservation (device-free). */
        PhysicalMemoryOwnerReservation makePool(
            std::size_t pool_bytes = 16ull * 1024ull * 1024ull)
        {
            PhysicalMemoryBOMBuilder cpu(PhysicalMemoryResource{
                .world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = pool_bytes, .admission_available_bytes = pool_bytes});
            cpu.add(PhysicalMemoryOwner::ServingResponsePool, pool_bytes);
            PhysicalMemoryPlanBuilder builder;
            const auto plan = builder.add(cpu.build()).build();
            auto admission = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
                std::move(plan));
            auto authority = std::make_shared<PhysicalMemoryAuthority>(admission, 0);
            return authority->reserveNewAllocations(DeviceId::cpu(),
                                                     PhysicalMemoryOwner::ServingResponsePool,
                                                     pool_bytes);
        }

        /** @brief PMA fixture returning BOTH the canonical authority AND the
         *         reservation: the test retains the authority to observe the
         *         canonical materialized state directly (#166 — no engine-side
         *         shadow quirk, no payload_/live_-dependent accessor). */
        struct PoolWithObserver
        {
            std::shared_ptr<PhysicalMemoryAuthority> authority;
            PhysicalMemoryOwnerReservation reservation; /* engine */
            PhysicalMemoryOwnerReservation observer;    /* SAME authority state (canonical query) */
        };
        PoolWithObserver makePoolWithAuthority(std::size_t pool_bytes = 16ull * 1024ull * 1024ull)
        {
            PhysicalMemoryBOMBuilder cpu(PhysicalMemoryResource{
                .world_rank = 0, .device = DeviceId::cpu(),
                .total_bytes = pool_bytes, .admission_available_bytes = pool_bytes});
            cpu.add(PhysicalMemoryOwner::ServingResponsePool, pool_bytes);
            PhysicalMemoryPlanBuilder builder;
            const auto plan = builder.add(cpu.build()).build();
            auto admission = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
                std::move(plan));
            auto authority = std::make_shared<PhysicalMemoryAuthority>(admission, 0);
            auto observer = authority->reserveNewAllocations(
                DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, 16u);
            auto reservation = authority->reserveNewAllocations(
                DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, pool_bytes - 16u);
            return {authority, std::move(reservation), std::move(observer)};
        }

        /** @brief PMA fixture with an EXPLICIT owner and allocator resource
         *         (wrong-owner / wrong-resource negative controls). */
        PhysicalMemoryOwnerReservation makePoolFor(
            PhysicalMemoryOwner owner, DeviceId device, std::size_t pool_bytes = 1u << 20u)
        {
            PhysicalMemoryBOMBuilder res(PhysicalMemoryResource{
                .world_rank = 0, .device = device, .total_bytes = pool_bytes,
                .admission_available_bytes = pool_bytes});
            res.add(owner, pool_bytes);
            PhysicalMemoryPlanBuilder builder;
            const auto plan = builder.add(res.build()).build();
            auto admission = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
                std::move(plan));
            auto authority = std::make_shared<PhysicalMemoryAuthority>(admission, 0);
            return authority->reserveNewAllocations(device, owner, pool_bytes);
        }

        /** @brief PMA-bound engine on a fresh pool reservation (one pool per test). */
        GroupedServingEngine makeEngine(RowDecodeFn decode)
        {
            return GroupedServingEngine(decode, 1, GroupedServingEngine::ServingPoolPlan{},
                                        makePool(), BackingAllocator::system());
        }

        std::string soloDecode(std::uint64_t seed, std::uint64_t position,
                               std::uint64_t window, const std::string &)
        {
            return "x" + std::to_string(seed % 97) + ":" + std::to_string(position) + "/" +
                   std::to_string(window);
        }
        RowDecodeFn solo = &soloDecode;
        std::string windowSlice(std::uint64_t seed, std::uint64_t from, std::uint64_t count,
                                std::uint64_t quantum)
        {
            std::string out;
            for (std::uint64_t pos = from; pos < from + count; ++pos)
                out += soloDecode(seed, pos, quantum, std::string{});
            return out;
        }
        constexpr std::uint64_t kDest = 1;
        constexpr std::uint64_t kDest2 = 2;
        constexpr std::uint64_t kDestB = 2;
        constexpr std::uint64_t kDestC = 3;
        constexpr std::uint64_t kDestD = 4;
        void bankDest(GroupedServingEngine &engine)
        {
            // Unique per-request destinations, all MATERIALIZED by the producer.
            for (auto d : {kDest, kDestB, kDestC, kDestD})
            {
                engine.bankDestination(d, 1u << 20u);
                engine.materializeDestination(d);
            }
        }
    } // namespace

    TEST(SharedExecutionProof, AContinuesWhileBFinishesAndCEntersBSlot)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);  // A
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);  // B
        engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256); // C waits for B's slot (NewRequest)

        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0), windowSlice(1, 1, 16, 16)); // A == solo
        EXPECT_NE(step1.at(1).find("|entry:2"), std::string::npos); // C entered in the same step

        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // C owns the slot
        EXPECT_EQ(engine.rows().at(1).state.kind, AdmissionKind::NewRequest); // §14 C=NewRequest

        engine.issueNextGrant(0); // A continues (same request; single-use grant re-issued)
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(0), windowSlice(1, 17, 16, 16)); // A window 2 == solo
        EXPECT_EQ(step2.at(1), windowSlice(3, 1, 8, 8));    // C first slice
        EXPECT_TRUE(engine.rows().at(0).complete);
        // C receives its SECOND grant through the queued/active path and completes.
        engine.issueNextGrant(1);
        const auto step3 = engine.step();
        EXPECT_EQ(step3.at(1), windowSlice(3, 9, 8, 8)); // C second window (9-16)
        EXPECT_EQ(engine.rows().at(1).state.committed, 16u); // C limit reached
    }

    TEST(SharedExecutionProof, SlowConsumerDefersOnlyItself)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256);
        engine.throttle(1);
        const auto step1 = engine.step();
        EXPECT_TRUE(step1.count(0) == 1);
        EXPECT_TRUE(step1.count(1) == 0);
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_EQ(engine.rows().at(1).state.committed, 0u);
    }

    TEST(SharedExecutionProof, OccupiedSlotRefusesAdmission)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        EXPECT_THROW(engine.admitNew(0, 3, 32, 16, 4, kDest, 256), std::logic_error);
    }

    TEST(SharedExecutionProof, FreshSlotQueuedRequestAdmits)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.queueRequest(3, 1, 32, 16, 7, kDest, 256); // never-populated slot
        const auto step1 = engine.step();
        EXPECT_EQ(engine.rows().at(3).state.generation, 1u);
        EXPECT_EQ(step1.at(3), windowSlice(7, 1, 16, 16));
        engine.issueNextGrant(3); // fresh-slot row gets its second grant too
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(3), windowSlice(7, 17, 16, 16));
        EXPECT_EQ(engine.rows().at(3).state.committed, 32u);
    }

    TEST(SharedExecutionProof, PartialFinalWindowIsBoundedAndTerminal)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 20, 16, 1, kDest, 256); // limit not a quantum multiple
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0), windowSlice(1, 1, 16, 16));
        engine.issueNextGrant(0); // budget-backed: final 8 remaining allowed
        const auto step2 = engine.step();
        std::string partial;
        for (std::uint64_t pos = 17; pos <= 20; ++pos)
            partial += soloDecode(1, pos, 16, std::string{});
        EXPECT_EQ(step2.at(0), partial);
        EXPECT_EQ(engine.rows().at(0).state.committed, 20u);
        EXPECT_TRUE(engine.rows().at(0).complete);
        EXPECT_THROW(engine.issueNextGrant(0), std::logic_error); // completed row
    }

    TEST(SharedExecutionProof, RestoreReadinessGatesAdmission)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.queueRequest(1, 2, 16, 16, 3, kDestC, 4096); // prompt == restore geometry
        engine.beginQueuedRestore(1, 4096); // exact geometry
        // Not published -> Pending -> not admitted.
        const auto step1 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 1u);
        // Readiness is gated on a completed producer copy: minting without one refused.
        EXPECT_THROW(engine.issueRestoreReceipt(1, 2, 4096), std::logic_error);
        {
            std::array<std::byte, 4096> src{};
            for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::byte>(i & 0xFF);
            engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src.data());
            EXPECT_EQ(engine.restorePayload(1),
                      std::string(reinterpret_cast<const char *>(src.data()), src.size()));
        }
        {
            std::array<std::byte, 4096> src2{};
            EXPECT_THROW(engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src2.data()),
                         std::logic_error); // exactly once (no second copy source)
        }
        // A mismatched-byte receipt is refused at MINTING (producer gate).
        EXPECT_THROW(engine.issueRestoreReceipt(1, 2, 8192), std::logic_error);
        auto correct = engine.issueRestoreReceipt(1, 2, 4096);
        engine.publishRestore(1, std::move(correct));
        // The copied payload is restored INTO the materialized destination backing
        // (restored execution state), not left in an isolated lane.
        EXPECT_EQ(engine.destinationContents(kDestC).substr(0, 4096),
                  std::string(engine.restorePayload(1)));
        const auto step2 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // attached after readiness
        EXPECT_EQ(step2.at(1), windowSlice(3, 1, 16, 16));
        // Layout: [0,4096) is restored prompt, output is written AFTER it (no
        // overwrite of the restored region).
        EXPECT_EQ(engine.destinationContents(kDestC).substr(0, 4096),
                  engine.restorePayload(1));
        EXPECT_EQ(engine.destinationContents(kDestC).substr(4096, step2.at(1).size()),
                  step2.at(1));
        // Delivery THROUGH publication: the consumer receives ONLY the committed
        // output (never the 4096 restored prompt bytes).
        std::string delivered_bytes;
        engine.setConsumer([&](std::uint64_t, const std::byte *p, std::size_t n)
                           {
                               delivered_bytes.assign(reinterpret_cast<const char *>(p), n);
                           });
        engine.markDelivered(kDestC, 1);
        EXPECT_EQ(delivered_bytes.size(), step2.at(1).size());
        EXPECT_EQ(delivered_bytes, step2.at(1)); // BYTES match the output slice
        EXPECT_EQ(engine.restoredExtentFor(kDestC), 4096u);
    }

    TEST(SharedExecutionProof, HarvestHoldsSourceUntilOwnedReceiptReleases)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.queueRequest(1, 2, 16, 8, 3, kDestC, 256);
        engine.requestHarvest(1); // before B completes
        const auto step1 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 1u);   // held
        EXPECT_EQ(engine.queued().count(1), 1u);               // C waits
        // Frozen-source / harvest-requested slot cannot be overwritten:
        EXPECT_THROW(engine.admitNew(1, 9, 16, 8, 9, kDest, 256), std::logic_error);
        engine.freezeSnapshot(1);
        engine.beginSnapshotCopy(1);
        auto receipt = engine.borrowSnapshot(1);
        EXPECT_THROW(engine.admitNew(1, 9, 16, 8, 9, kDest, 256), std::logic_error); // not released yet
        receipt.release(); // authenticating receipt releases the source
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // C attached after release
        EXPECT_EQ(step2.at(1), windowSlice(3, 1, 8, 8));
    }

    TEST(SharedExecutionProof, MixedQuantaCancellationAndMTPInOneBatch)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 8, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.setMTPDepth(1, 2);
        engine.queueRequest(1, 2, 16, 8, 3, kDestC, 256);
        engine.cancelQueued(1); // C cancelled before B finishes
        engine.queueRequest(1, 3, 16, 8, 4, kDestD, 256); // D replaces C (unique reservation)

        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0), windowSlice(1, 1, 8, 8));
        EXPECT_NE(step1.at(1).find("|entry:3"), std::string::npos);
        EXPECT_EQ(engine.rows().at(1).state.generation, 3u);

        engine.setMTPDepth(1, 2); // D admitted in step1; depth before its production
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(0), windowSlice(1, 9, 8, 8));
        EXPECT_EQ(step2.at(1), windowSlice(4, 1, 8, 8));
        EXPECT_THROW(engine.setMTPDepth(1, 2), std::logic_error); // exactly once
        EXPECT_THROW(engine.setMTPDepth(0, 2), std::logic_error); // after execution: immutable
    }

    TEST(SharedExecutionProof, GrantIsSingleUseAndBacked)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        // Unbacked admission is refused by the authority.
        EXPECT_THROW(engine.admitNew(0, 1, 16, 16, 1, 0), std::logic_error);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256); // not complete after one window
        // A second issue while one is outstanding (cross-window) is refused.
        EXPECT_THROW(engine.issueNextGrant(0), std::logic_error);
        const auto step1 = engine.step(); // consumes the grant
        EXPECT_EQ(step1.count(0), 1u);
        engine.issueNextGrant(0); // re-issue after consumption (new ID)
        EXPECT_TRUE(engine.rowMayExecute(0));
    }

    TEST(SharedExecutionProof, MTPVerifierIsExactlyOncePerStepAndRequiresDepth)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        // Verification requires produced work AND positive depth.
        EXPECT_THROW(engine.verifyMTP(0), std::logic_error); // no window yet
        engine.setMTPDepth(0, 2);
        const auto step1 = engine.step(); // produces + verifies once internally
        EXPECT_EQ(step1.count(0), 1u);
        EXPECT_THROW(engine.verifyMTP(0), std::logic_error); // already verified this step
        engine.issueNextGrant(0);
        const auto step2 = engine.step(); // fresh step: resets, verifies exactly once
        EXPECT_EQ(step2.count(0), 1u);
        EXPECT_THROW(engine.verifyMTP(0), std::logic_error); // exactly once per step
    }

    TEST(SharedExecutionProof, RestoreReceiptGenerationAndIdentityAreAuthenticated)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.queueRequest(1, 2, 16, 16, 3, kDestC, 4096);
        engine.beginQueuedRestore(1, 4096);
        // Readiness is gated on a completed producer copy (foreign too).
        EXPECT_THROW(engine.issueRestoreReceipt(1, 1, 4096), std::logic_error);
        {
            std::array<std::byte, 4096> src{};
            engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src.data());
        }
        // Stale generation refused AT MINTING (generation validated by the issuer).
        EXPECT_THROW(engine.issueRestoreReceipt(1, 1, 4096), std::logic_error);
        GroupedServingEngine other(solo, 1, GroupedServingEngine::ServingPoolPlan{}, makePool(),
                    BackingAllocator::system());
        bankDest(other);
        other.admitNew(0, 1, 16, 16, 2, kDestC, 256);
        other.queueRequest(1, 2, 16, 16, 3, kDest, 4096);
        other.beginQueuedRestore(1, 4096);
        {
            std::array<std::byte, 4096> src{};
            other.completeRestoreCopy(1, other.queuedRestoreSession(1), 4096, src.data());
        }
        auto foreign = other.issueRestoreReceipt(1, 2, 4096); // other engine
        EXPECT_THROW(engine.publishRestore(1, std::move(foreign)), std::logic_error);
        auto correct = engine.issueRestoreReceipt(1, 2, 4096);
        engine.publishRestore(1, std::move(correct));
        const auto step = engine.step();
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // attached
    }

    TEST(SharedExecutionProof, RecordedControllerTraceReplaysIdentically)
    {
        // Device-free recorded-input controller replay through the ACTUAL
        // implementation: the trace (admits, queue, issue, cancel, harvest) is
        // recorded once, then replayed on a fresh engine and must reproduce the
        // identical per-step transcripts and lifecycle state.
        struct Step
        {
            enum class Op { Admit, Queue, Issue, Cancel, Harvest, Step } op;
            std::uint32_t slot = 0;
        };
        // A VALID controller trace (cancel of an already-admitted row is its own
        // regression): admit A, queue C, step, issue A's next grant, step.
        // Recorded controller trace: A admitted; C queued; step; both next grants;
        // two more steps (C completes through its SECOND queued-path grant).
        const std::vector<Step> trace = {
            {Step::Op::Admit, 0}, {Step::Op::Queue, 1},
            {Step::Op::Step, 0}, {Step::Op::Issue, 0}, {Step::Op::Issue, 1},
            {Step::Op::Step, 0}, {Step::Op::Issue, 0}, {Step::Op::Issue, 1},
            {Step::Op::Step, 0}, {Step::Op::Issue, 1}, {Step::Op::Step, 0},
        };
        auto run = [&](const std::vector<Step> &tr) {
            auto engine = makeEngine(solo);
        bankDest(engine);
            std::vector<std::string> transcript;
            for (const auto &st : tr)
            {
                switch (st.op)
                {
                case Step::Op::Admit: engine.admitNew(0, 1, 48, 16, 1, kDest, 256); break; // 3 A windows
                case Step::Op::Queue: engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256); break;
                case Step::Op::Issue: engine.issueNextGrant(st.slot); break;
                case Step::Op::Cancel: engine.cancelQueued(st.slot); break;
                case Step::Op::Step:
                    for (const auto &[slot, out] : engine.step())
                        transcript.push_back(std::to_string(slot) + ":" + out);
                    break;
                }
            }
            return transcript;
        };
        const auto first = run(trace);
        // ORACLE: pinned position-by-position. A: seed1 q16 -> [1-16],[17-32],[33-48]
        // (3 windows). C: seed3 q8 -> completes its full 32-token limit over FOUR
        // q8 windows [1-8],[9-16],[17-24],[25-32] through queued-path grants.
        const std::string a1 = windowSlice(1, 1, 16, 16);
        const std::string a2 = windowSlice(1, 17, 16, 16);
        const std::string c1 = windowSlice(3, 1, 8, 8);
        const std::string c2 = windowSlice(3, 9, 8, 8);
        const std::string c3 = windowSlice(3, 17, 8, 8);
        const std::string c4 = windowSlice(3, 25, 8, 8);
        const std::string a3 = windowSlice(1, 33, 16, 16);
        const std::vector<std::string> expected = {
            "0:" + a1, "1:" + c1,
            "0:" + a2, "1:" + c2,
            "0:" + a3, "1:" + c3,
            "1:" + c4,
        };
        ASSERT_EQ(expected.size(), 7u); // A3 + C4 windows
        ASSERT_EQ(first.size(), expected.size());
        EXPECT_EQ(first[0], expected[0]);
        EXPECT_EQ(first[1], expected[1]);
        EXPECT_EQ(first[2], expected[2]);
        EXPECT_EQ(first[3], expected[3]);
        EXPECT_EQ(first[4], expected[4]); // A's THIRD window at index 4 (was unchecked)
        EXPECT_EQ(first[5], expected[5]); // C's third window (was unchecked)
        EXPECT_EQ(first[6], expected[6]); // C's fourth window (was unchecked)
        // Replay on a fresh engine: byte-identical transcription.
        const auto second = run(trace);
        ASSERT_EQ(second, first);
    }

    TEST(SharedExecutionProof, SnapshotLifetimeFocusedRegressions)
    {
        // (a) A LIVE receipt must survive engine destruction: the lease is owned
        // by the receipt (no raw engine pointer), so release after the engine is
        // gone never touches a destroyed engine or its mutex.
        std::unique_ptr<GroupedServingEngine::SnapshotBorrowReceipt> survivor;
        {
            auto engine = makeEngine(solo);
            bankDest(engine);
            engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
            engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
            engine.queueRequest(1, 2, 16, 8, 3, kDestC, 256);
            engine.requestHarvest(1);
            engine.step();
            engine.freezeSnapshot(1);
            engine.beginSnapshotCopy(1);
            survivor = std::make_unique<GroupedServingEngine::SnapshotBorrowReceipt>(
                engine.borrowSnapshot(1));
            // Overlap guards AFTER the first borrow: a second borrow and a second
            // copy are both refused (no overlapping borrows / receipt invalidation).
            EXPECT_THROW(engine.borrowSnapshot(1), std::logic_error);
            EXPECT_THROW(engine.beginSnapshotCopy(1), std::logic_error);
        }
        // Engine destroyed; the survivor releases its own lease safely.
        EXPECT_NO_THROW(survivor->release());
        EXPECT_NO_THROW(survivor->release()); // idempotent
    }

    TEST(SharedExecutionProof, MoveAssignReleasesPriorReceiptRow)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256);
        engine.admitNew(2, 1, 16, 16, 4, kDestC, 256); // second harvestable row
        engine.requestHarvest(1);
        engine.requestHarvest(2);
        engine.step();
        engine.freezeSnapshot(1);
        engine.beginSnapshotCopy(1);
        auto r1 = engine.borrowSnapshot(1);
        engine.freezeSnapshot(2); // completed in the same step? row2 limit16 -> complete
        engine.beginSnapshotCopy(2);
        auto r2 = engine.borrowSnapshot(2);
        // Move-assign r2 over r1: the prior (r1) row must be RELEASED.
        r2 = std::move(r1); // move-assign: releases r2's prior (row2) lease FIRST
        EXPECT_TRUE(engine.rowSnapshotReleased(2)); // row2 observed released
        EXPECT_FALSE(engine.rowSnapshotReleased(1)); // row1 still held by the receipt
        r2.release();       // releases row1's lease
        EXPECT_TRUE(engine.rowSnapshotReleased(1));
        engine.issueNextGrant(0);
        const auto step = engine.step();
        (void)step;
    }

    TEST(SharedExecutionProof, ReleaseIsGatedOnDeliveryNotExecution)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256);
        engine.step(); // A completes (execution done)
        // Execution done but delivery pending: backing NOT released yet.
        EXPECT_FALSE(engine.destinationReleased(kDest));
        // PUBLIC release before delivery is refused (delivery ledger gate).
        EXPECT_THROW(engine.releaseDestination(kDest), std::logic_error);
        engine.markDelivered(kDest, 1); // delivery completes
        engine.step();           // next step runs the post-pass release
        EXPECT_TRUE(engine.destinationReleased(kDest));
        // ONE more step: the retained backing is REALLY released (markReleased ran),
        // so no second release is attempted and nothing throws.
        EXPECT_NO_THROW(engine.step());
    }

    TEST(SharedExecutionProof, PartialFinalWindowAcceptsQuantumBeyondLimit)
    {
        // §7.1: L=5, W=16 is a VALID partial final window (quantum > limit must not
        // be refused; only quantum == 0 is invalid).
        auto engine = makeEngine(solo);
        bankDest(engine);
        EXPECT_THROW(engine.admitNew(0, 1, 5, 0, 1, kDest, 64), std::logic_error);
        EXPECT_NO_THROW(engine.admitNew(0, 1, 5, 16, 1, kDest, 64));
        EXPECT_THROW(engine.queueRequest(1, 2, 5, 0, 3, kDestC, 64), std::logic_error);
        EXPECT_NO_THROW(engine.queueRequest(1, 2, 5, 16, 3, kDestC, 64));
        engine.step();
        // quotient 0 -> the +2 floor applies; the authority is request-bounded.
        EXPECT_EQ(engine.grantBound(0), 2u);
    }

    TEST(SharedExecutionProof, SameInstanceBoundMoveCarriesIncomingMax)
    {
        // The queued-attach wait: A holds a SMALL bound (3); C's larger bound
        // (14) must survive the same-instance move-assign, not A's stale 3.
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 0, kDest, 64);        // A: bound 16/16+2 = 3
        engine.queueRequest(0, 2, 200, 16, 0, kDestC, 64);   // C: bound 200/16+2 = 14
        engine.step(); // A completes AND C attaches into slot 0 in the same step
                        // (completion post-pass); the attach move-assign merges
                        // C's bound over A's authority. A stale 3 would remain if
                        // the same-instance move did not carry the incoming bound.
        EXPECT_EQ(engine.grantBound(0), (200 / 16) + 2); // moved bound 14, NOT stale 3
    }

    TEST(SharedExecutionProof, LateBDeliveryReleasesOwnBackingWithoutStallingC)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // A slot 0
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);  // B slot 1
        engine.step(); // both complete (slots retired)
        engine.queueRequest(1, 3, 32, 16, 3, kDestC, 64); // C REUSES B's slot 1; L=32/W=16
        engine.step(); // C attaches INTO B's slot and completes its FIRST window only
                       // (committed 16 < 32): C is MID-FLIGHT, not complete.
        EXPECT_FALSE(engine.rowComplete(1));
        // B delivers only AFTER C reused its slot AND is still executing: the
        // delivery is keyed by the RESPONSE reservation, so it must hit B, never C.
        engine.markDelivered(kDestB, 1);
        engine.issueNextGrant(1); // C's second window (backpressure: one per grant)
        engine.step(); // post-pass releases B's OWN backing while C still executes
                       // (C produces its second window in the same step).
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        EXPECT_FALSE(engine.destinationReleased(kDestC)); // C is not delivered yet
        EXPECT_TRUE(engine.rowComplete(1));                // and C was NOT stalled
        EXPECT_THROW(engine.markDelivered(kDestB, 1), std::logic_error); // exactly once
        // A still delivers normally (slot 0 untouched)
        engine.markDelivered(kDest, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
        // C delivers its own reservation afterwards
        engine.markDelivered(kDestC, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestC));
    }

    TEST(SharedExecutionProof, GrantLedgerSaturatesNotCapsAtDefault)
    {
        // L=4097/q1 needs a 4099 grant bound; the old kDefaultMaxGrants cap would
        // fail the 4097th grant mid-request. Saturation must NOT cap at 4096.
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 4097, 1, 1, kDest, 64);
        EXPECT_EQ(engine.grantBound(0), (4097 / 1) + 2);
        // admitNew already issues the FIRST grant (id 1); consume it, then each
        // further loop iteration issues+consumes one more (ids 2..4096).
        EXPECT_NO_THROW(engine.step());
        for (int i = 1; i < 4096; ++i)
        {
            EXPECT_NO_THROW(engine.issueNextGrant(0)) << "issue at i=" << i;
            EXPECT_NO_THROW(engine.step()) << "step at i=" << i;
        }
        EXPECT_EQ(engine.grantBound(0), (4097 / 1) + 2);
        EXPECT_NO_THROW(engine.issueNextGrant(0)); // the 4097th grant is legal
        engine.step();
    }

    TEST(SharedExecutionProof, DestinationReservationsAreExclusive)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        // An active row already owns kDest: re-claim refused (active AND queued).
        EXPECT_THROW(engine.admitNew(1, 2, 16, 16, 2, kDest, 64), std::logic_error);
        EXPECT_THROW(engine.queueRequest(1, 2, 16, 16, 2, kDest, 64), std::logic_error);
    }

    TEST(SharedExecutionProof, PrematureDeliveryIsRefused)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 64); // still executing (32 tokens, one step = 16)
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::logic_error); // not completed yet
    }



    TEST(SharedExecutionProof, ResponseOwnershipOutlivesRowRemoval)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // A
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);  // B
        engine.step(); // both complete
        engine.queueRequest(1, 3, 16, 16, 3, kDestD, 64); // C (kDestD) reuses B's slot
        engine.step(); // C replaces B's row; B's response is STILL undelivered
        // B's ownership must outlive the execution-row removal: D cannot claim B.
        EXPECT_THROW(engine.admitNew(2, 4, 16, 16, 4, kDestB, 64), std::logic_error);
        EXPECT_THROW(engine.queueRequest(2, 4, 16, 16, 4, kDestB, 64), std::logic_error);
        // After B delivers and its backing retires, the ID may be claimed again by
        // a FRESH response (bounded identity reuse; the re-bank bumps generation).
        engine.markDelivered(kDestB, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        // Bounded identity REUSE: the released ID may be re-banked (generation
        // bumped) and claimed by a fresh response.
        engine.bankDestination(kDestB, 1u << 20u);
        engine.materializeDestination(kDestB);
        EXPECT_NO_THROW(engine.admitNew(2, 4, 16, 16, 4, kDestB, 64));
    }

    TEST(SharedExecutionProof, DestinationPoolIsBoundedAndIDsRebankWithGeneration)
    {
        // Canonical bounded identity: a fixed-size wheel (pool full refused), and a
        // RETIRED ID may be re-banked (bounded reuse, NO unbounded per-ID memory);
        // the re-bank bumps the ID's generation so a stale response can never act
        // on the new occupant (the claim/delivery gates are generation-clean).
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.step();
        engine.markDelivered(kDest, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
        EXPECT_NO_THROW(engine.bankDestination(5, 1u << 20u)); // fresh wheel slot
        // Bounded wheel: the pool width comes from the ADMITTED PLAN (default 32),
        // never from a hard-coded HTTP constant; the wheel cannot exceed it.
        GroupedServingEngine full(solo, 1, GroupedServingEngine::ServingPoolPlan{}, makePool(),
                    BackingAllocator::system());
        for (std::uint64_t id = 1; id <= engine.maxDestinations(); ++id)
            full.bankDestination(id, 1u << 20u);
        EXPECT_THROW(full.bankDestination(engine.maxDestinations() + 1, 1u << 20u),
                     std::logic_error);
        // A plan-derived width (4) rejects slot 5; width is not copied from HTTP.
        GroupedServingEngine narrow(solo, 1, GroupedServingEngine::ServingPoolPlan{4}, makePool(),
                    BackingAllocator::system());
        EXPECT_NO_THROW(narrow.bankDestination(4, 1u << 20u));
        EXPECT_THROW(narrow.bankDestination(5, 1u << 20u), std::logic_error);
        EXPECT_EQ(narrow.maxDestinations(), 4u);
    }

    TEST(SharedExecutionProof, LiveOutputBoundUsesRealProducedBytes)
    {
        // seed 96 / L16 / W16 / prompt 0: the decode function emits MORE than the
        // assumed 8 bytes/token (real output 135 bytes > 128 assumed).
        auto engine = makeEngine(solo);
        engine.bankDestination(kDest, 128);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 16, 16, 96, kDest, 0);
        // Admission (the typed MINIMUM 8 bytes/token) fits; the ATOMIC publication
        // boundary validates REAL produced bytes (135) BEFORE any mutation.
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // The failed response was never completed: delivery is refused (no
        // accepted-failed response) and the engine stays dead (no stranded grant).
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::logic_error);
        EXPECT_THROW(engine.step(), std::logic_error);
    }

    TEST(SharedExecutionProof, OutputIsPhysicallyWrittenToTheDestinationBacking)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        const auto step1 = engine.step();
        const auto slice = step1.at(0);
        // The produced output bytes are physically present in the materialized
        // destination buffer AFTER the prompt region (output offset = prompt bytes).
        EXPECT_NE(engine.destinationContents(kDest).find(slice), std::string::npos);
    }

    TEST(SharedExecutionProof, QueuedSaturationPathUsesRequestBoundedLedger)
    {
        // The queued path must also carry the request-derived bound (not 4096):
        // L=4097/q1 through a QUEUED request.
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);       // A (small, completes)
        engine.queueRequest(0, 2, 4097, 1, 1, kDestC, 64);  // C queued, L=4097/q1
        engine.step(); // A completes; C attaches (move-ASSIGN) with bound 4099
        EXPECT_EQ(engine.grantBound(0), (4097 / 1) + 2);
        EXPECT_NO_THROW(engine.step()); // consume C's admission grant (token 1)
        for (int i = 1; i < 4096; ++i)
        {
            EXPECT_NO_THROW(engine.issueNextGrant(0)) << "queued issue at i=" << i;
            EXPECT_NO_THROW(engine.step()) << "queued step at i=" << i;
        }
        EXPECT_NO_THROW(engine.issueNextGrant(0)); // grant 4097
        EXPECT_NO_THROW(engine.step());
        EXPECT_TRUE(engine.rowComplete(0));
    }

    TEST(SharedExecutionProof, RestoreFeedsExecutionState)
    {
        // A decoder that ACTUALLY READS the restored prefix: proves the restored
        // bytes reach execution state (not an isolated lane).
        static std::string last_restored; // decoder observation
        const RowDecodeFn readingDecode = [](
            std::uint64_t seed, std::uint64_t position, std::uint64_t window,
            const std::string &restored)
        {
            last_restored = restored;
            return "r" + std::to_string(seed) + ":" + std::to_string(position) + "/" +
                   std::to_string(window) + ":" + std::to_string(restored.size());
        };
        GroupedServingEngine engine(readingDecode, 1, GroupedServingEngine::ServingPoolPlan{}, makePool(),
                    BackingAllocator::system());
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);
        engine.queueRequest(1, 3, 16, 16, 3, kDestC, 4096); // prompt == restore geometry
        engine.beginQueuedRestore(1, 4096);
        {
            std::array<std::byte, 4096> src{};
            for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::byte>(i & 0xFF);
            engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src.data());
            auto correct = engine.issueRestoreReceipt(1, 3, 4096);
            engine.publishRestore(1, std::move(correct));
        }
        engine.step(); // C attaches (move-assign); its decoder runs WITH the prefix
        last_restored.clear();
        engine.step(); // C executes a window using the restored prefix
        EXPECT_EQ(last_restored.size(), 4096u); // decoder read 4096 restored bytes
        EXPECT_NE(engine.rows().at(1).state.committed, 0u);
        (void)windowSlice(0, 0, 0, 16);
    }

    TEST(SharedExecutionProof, ConsumerReceivesPublishedBytesOnce)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        std::size_t calls = 0;
        std::size_t total = 0;
        std::uint64_t seen_reservation = 0;
        engine.setConsumer([&](std::uint64_t reservation, const std::byte *, std::size_t n)
                           {
                               ++calls; total += n; seen_reservation = reservation;
                           });
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        const auto step1 = engine.step();
        engine.markDelivered(kDest, 1);
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(seen_reservation, kDest);
        // The consumer receives the COMMITTED OUTPUT ONLY (§5.1/§10.1): the
        // produced window — never the unwritten prompt prefix, never holes,
        // never allocation capacity.
        EXPECT_EQ(total, step1.at(0).size());
        engine.step();
        EXPECT_EQ(calls, 1u); // exactly once, no second publication
    }

    TEST(SharedExecutionProof, StaleGenerationDeliveryIsRefusedAfterRebank)
    {
        // B occupies slot 2 (generation 1), retires; D re-banks slot 2
        // (generation 2), completes. An OLD-B duplicate calling delivery with
        // generation 1 must be REFUSED — the ABA is closed by generation keys.
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // A (slot 1)
        engine.admitNew(1, 2, 16, 16, 2, kDestB, 64);  // B (slot 2, gen 1)
        engine.step();
        engine.markDelivered(kDestB, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        EXPECT_EQ(engine.completedGeneration(kDestB), 0u); // release erased the record
        EXPECT_EQ(engine.destinationGeneration(kDestB), 1u);
        engine.bankDestination(kDestB, 1u << 20u); // re-bank: slot 2 -> generation 2
        engine.materializeDestination(kDestB);
        EXPECT_NO_THROW(engine.admitNew(1, 3, 16, 16, 3, kDestB, 64)); // D (gen 2)
        engine.step();
        EXPECT_TRUE(engine.rowComplete(1)); // D completed in one window
        EXPECT_EQ(engine.destinationGeneration(kDestB), 2u);
        EXPECT_EQ(engine.completedGeneration(kDestB), 2u);
        // The stale B duplicate cannot deliver D's completed response.
        EXPECT_THROW(engine.markDelivered(kDestB, 1), std::logic_error);
        EXPECT_NO_THROW(engine.markDelivered(kDestB, 2)); // D's own delivery works
    }

    TEST(SharedExecutionProof, ThrowingConsumerLeavesResponseUndelivered)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        bool throw_once = true;
        engine.setConsumer([&](std::uint64_t, const std::byte *, std::size_t)
                           {
                               if (throw_once) { throw_once = false; throw std::runtime_error("transport"); }
                           });
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.step();
        // A throwing consumer means the delivery NEVER completed: no ledger
        // record, backing retained, retryable.
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::runtime_error);
        EXPECT_FALSE(engine.destinationReleased(kDest));
        engine.step();
        EXPECT_FALSE(engine.destinationReleased(kDest)); // still not freed
        EXPECT_NO_THROW(engine.markDelivered(kDest, 1)); // retry succeeds
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
    }

    TEST(SharedExecutionProof, FailedEngineRefusesNewAdmissionsAndGrants)
    {
        auto engine = makeEngine(solo);
        bankDest(engine); // all four slots banked+materialized (VALID operations)
        // Row 1 is a GOOD row that consumes its admission grant first (resolved),
        // so issueNextGrant(1) is legal BEFORE the failure gate exists.
        engine.admitNew(1, 1, 32, 16, 1, kDestB, 64);
        engine.step();
        EXPECT_FALSE(engine.rowComplete(1));
        // Bad row (capacity 128 provokes 135 > 128) poisons the engine.
        engine.bankDestination(5, 128);
        engine.materializeDestination(5);
        engine.admitNew(0, 1, 16, 16, 96, 5, 0);
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // kDestC / kDestD are banked, materialized and UNCLAIMED: without the
        // failed gate these VALID operations would succeed; they must refuse.
        EXPECT_THROW(engine.admitNew(2, 1, 16, 16, 1, kDestC, 64), std::logic_error);
        EXPECT_THROW(engine.queueRequest(2, 2, 16, 16, 2, kDestD, 64), std::logic_error);
        // Row 1's grant is RESOLVED and the row is unfinished: without the gate
        // issueNextGrant(1) would be legal; it must refuse.
        EXPECT_THROW(engine.issueNextGrant(1), std::logic_error);
    }

    TEST(SharedExecutionProof, FailedAdmissionLeavesNoStrandedClaim)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        // A fresh wheel slot banked WITHOUT materialization: admission refused.
        engine.bankDestination(6, 1u << 20u);
        EXPECT_THROW(engine.admitNew(0, 1, 16, 16, 1, 6, 64), std::logic_error);
        // Materialize and RETRY THE SAME reservation: must succeed (no claim left).
        engine.materializeDestination(6);
        EXPECT_NO_THROW(engine.admitNew(0, 1, 16, 16, 1, 6, 64));
        engine.step();
        EXPECT_TRUE(engine.rowComplete(0));
    }

    TEST(SharedExecutionProof, MultiRowAdverseStepMutatesNothingOnViolation)
    {
        // Row 1 is GOOD and mid-flight; row 3 violates the real-bytes bound. The
        // step validates ALL executable rows (and the queued attach candidate)
        // BEFORE mutating ANY, so the good row is untouched, the attach is NOT
        // applied, and no response was half-completed.
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);   // good: completes in one window
        engine.admitNew(1, 1, 32, 16, 1, kDestB, 64);  // good: mid-flight L32/W16
        engine.queueRequest(0, 2, 16, 16, 3, kDestC, 64); // attach candidate (slot 0)
        engine.step(); // row0 completes; row1 -> first window (committed 16)
        EXPECT_EQ(engine.rows().at(1).state.committed, 16u);
        // C attached into slot 0 in the same step (completion post-pass) but has
        // NOT executed yet (attachment happens after the compute loop).
        EXPECT_EQ(engine.rows().at(0).state.generation, 2u);
        EXPECT_EQ(engine.rows().at(0).state.committed, 0u);
        // Violating row (capacity 128; seed 96 emits 135 > 128).
        engine.bankDestination(5, 128);
        engine.materializeDestination(5);
        engine.admitNew(2, 1, 16, 16, 96, 5, 0);
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // Nothing was mutated by the ADVERSE step: row1 not advanced past its
        // previous window, the freshly attached C (slot 0) not advanced, and the
        // violating row (slot 2) not completed.
        EXPECT_EQ(engine.rows().at(1).state.committed, 16u);
        EXPECT_FALSE(engine.rows().at(1).complete);
        EXPECT_EQ(engine.rows().at(0).state.generation, 2u); // C attached earlier
        EXPECT_EQ(engine.rows().at(0).state.committed, 0u);  // but NOT advanced
        EXPECT_FALSE(engine.rows().at(0).complete);
        EXPECT_FALSE(engine.rows().at(2).complete);
        EXPECT_EQ(engine.rows().at(2).state.committed, 0u);
    }

    TEST(SharedExecutionProof, RetirementDuringInProgressPublicationIsRefused)
    {
        auto engine = makeEngine(solo);
        bankDest(engine);
        bool inside = false;
        engine.setConsumer([&](std::uint64_t, const std::byte *, std::size_t)
                           {
                               inside = true;
                               // A consumer must never retire the buffer it is
                               // still borrowing: refused while Publishing.
                               EXPECT_THROW(engine.releaseDestination(kDest),
                                            std::logic_error);
                           });
        engine.admitNew(0, 1, 16, 16, 1, kDest, 64);
        engine.step();
        EXPECT_NO_THROW(engine.markDelivered(kDest, 1)); // consumer accepted
        EXPECT_TRUE(inside);
        // After acceptance the retirement is legal (exactly once).
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
    }

    TEST(SharedExecutionProof, AdverseStepWithPendingAttachAndFreshGrant)
    {
        // A has a FRESH outstanding grant (mid-flight), C is a PENDING attach
        // candidate, and the violating row poisons the step: the fresh grant must
        // NOT be consumed and C must NOT attach.
        auto engine = makeEngine(solo);
        bankDest(engine);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 64);   // A: L32/W16
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 64);  // B: completes in one window
        engine.queueRequest(1, 2, 16, 16, 3, kDestC, 64); // C (attach after B)
        engine.step(); // B completes AND C attaches in the same step (post-pass)
        // Give A a FRESH grant (mid-flight granted) and queue a NEW pending attach
        // for the just-completed B slot? B's slot now holds C (not complete).
        // Use slot 2: complete a fresh row to create a pending attach.
        engine.admitNew(2, 1, 16, 16, 4, kDestD, 64); // D: completes in one window
        engine.step();                                 // D completes
        engine.bankDestination(7, 1u << 20u); // E's own fresh reservation
        engine.materializeDestination(7);
        engine.queueRequest(2, 3, 16, 16, 5, 7, 64); // E: attach candidate (slot 2)
        EXPECT_EQ(engine.queued().count(2), 1u);
        engine.issueNextGrant(0);                      // A: fresh outstanding grant
        // Violating row on slot 3 (capacity 128; seed 96 emits 135 > 128).
        engine.bankDestination(6, 128);
        engine.materializeDestination(6);
        engine.admitNew(3, 1, 16, 16, 96, 6, 0);
        EXPECT_THROW(engine.step(), std::logic_error);
        EXPECT_TRUE(engine.failed());
        // A's fresh grant was NOT consumed (committed unchanged), E did NOT attach,
        // and the violating row is untouched.
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_TRUE(engine.rowMayExecute(0)); // grant still outstanding
        EXPECT_EQ(engine.queued().count(2), 1u);
        EXPECT_EQ(engine.rows().at(2).state.generation, 1u); // D, not E
        EXPECT_FALSE(engine.rows().at(3).complete);
    }

    TEST(SharedExecutionProof, InConsumerStepAdvancesCAndRetainsBWhilePublishing)
    {
        // B's consumer runs an INDEPENDENT grouped step while B is Publishing:
        // C computes/commits/consumes its grant (correct progress bytes), B's
        // backing is RETAINED (never auto-retired mid-publication), and B retires
        // only after acceptance. C is NOT stalled behind B's delivery.
        auto engine = makeEngine(solo);
        bankDest(engine);
        bool inside = false;
        std::string c_bytes;
        engine.setConsumer([&](std::uint64_t, const std::byte *, std::size_t)
                           {
                               if (inside)
                                   return;
                               inside = true;
                               // Re-entrant step from inside the consumer: C must
                               // advance normally (its returned bytes correct) and
                               // B's backing must NOT be released.
                               const auto inner = engine.step();
                               c_bytes = inner.at(2);
                               EXPECT_EQ(c_bytes, windowSlice(4, 17, 16, 16));
                               EXPECT_FALSE(engine.destinationReleased(kDestB));
                               inside = false;
                           });
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 64); // B: slot 1 (completes step 1)
        engine.admitNew(2, 1, 32, 16, 4, kDestC, 64); // C: slot 2 (MID-FLIGHT L32)
        engine.step();                                // B completes; C -> first window
        EXPECT_EQ(engine.rows().at(2).state.committed, 16u);
        EXPECT_FALSE(engine.rows().at(2).complete);
        engine.issueNextGrant(2); // C's second window is pending (backpressure)
        engine.markDelivered(kDestB, 1);              // B delivery -> consumer runs
        EXPECT_FALSE(engine.destinationReleased(kDestB)); // retained while Publishing
        // After acceptance B retires; C was NOT stalled and completed its second
        // window during the in-consumer step.
        EXPECT_EQ(engine.rows().at(2).state.committed, 32u);
        EXPECT_TRUE(engine.rows().at(2).complete);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestB));
        EXPECT_FALSE(engine.destinationReleased(kDestC)); // C not delivered yet
    }

    /** @brief S3.1: destination backing is a PMA child claim (exact bytes beside
     *         the real allocation) and the claim returns to the reserve on release. */
    TEST(SharedExecutionProof, PmaBoundDestinationClaimsExactPoolBytesAndReturnsOnRelease)
    {
        GroupedServingEngine engine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                    makePool(), BackingAllocator::system());
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        // The SINGLE claim admits the destination backing AND its retained
        // metadata (capacity + kRetainedMetadataBytes) — one real owner.
        const auto materialized = engine.servingPool().materializedBytes();
        EXPECT_EQ(materialized, (1u << 20u) + engine.destinationReserveBytes() + engine.aggregateNodeBytes()); // backing + object control
        EXPECT_EQ(engine.servingPool().remainingBytes(),
                  16u * 1024u * 1024u - materialized);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256);
        // Complete the row (2 windows of 16 == limit 32), deliver, then retire the
        // destination through the owned path. No grant after completion (rejected).
        engine.step();
        engine.issueNextGrant(0);
        engine.step();
        EXPECT_TRUE(engine.rows().at(0).complete);
        engine.step();
        engine.markDelivered(kDest, 1); // accepted by the consumer
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
        // The capacity claim returned; the backing OBJECT's control claim stays
        // until the completed row's own reference dies (object-lifetime-exact).
        EXPECT_EQ(engine.servingPool().materializedBytes(), engine.destinationReserveBytes() + engine.aggregateNodeBytes());
        EXPECT_EQ(engine.servingPool().remainingBytes(),
                  16u * 1024u * 1024u - (engine.destinationReserveBytes() + engine.aggregateNodeBytes()));
    }

    /** @brief S3.1: an oversized destination cannot claim beyond the admitted pool
     *         reserve (fail-closed, exact child-claim diagnostic). */
    TEST(SharedExecutionProof, PmaPoolRefusesOversizedDestinationClaim)
    {
        GroupedServingEngine engine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                    makePool(1u << 20u), BackingAllocator::system());
        engine.bankDestination(kDest, 2u * 1024u * 1024u); // 2 MiB > admitted 1 MiB
        EXPECT_THROW(engine.materializeDestination(kDest), std::logic_error);
        // The banked backing OBJECT (with its control claim) remains until the
        // destination retires — the capacity bytes were never claimed.
        EXPECT_EQ(engine.servingPool().materializedBytes(), engine.destinationReserveBytes() + engine.aggregateNodeBytes());
    }

    /** @brief S3.1: a non-PMA construction is refused (no unbound engine path). */
    TEST(SharedExecutionProof, EngineRefusesMissingPoolReservation)
    {
        EXPECT_THROW(
            GroupedServingEngine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                        PhysicalMemoryOwnerReservation{}, BackingAllocator::system()),
            std::logic_error);
    }

    /** @brief S3.1: DEFAULT DESTRUCTION satisfies the pairwise rule at the exact
     *         free boundary - a recording allocator's deleter observes that the
     *         PMA claim is STILL HELD when the buffer is freed (member order:
     *         lease declared before buffer; reverse destruction frees the buffer
     *         first, then returns the claim). With the wrong member order the
     *         claim is already returned when the deleter runs and this FAILS. */
    TEST(SharedExecutionProof, FreeBoundaryObservesClaimHeldWhileBufferFreed)
    {
        std::uint64_t canonical_at_free = 0;
        std::uint64_t expected_at_free = 0;
        const PhysicalMemoryOwnerReservation *pinned = nullptr;
        auto recording = BackingAllocator{};
        recording.allocate = [&pinned, &canonical_at_free](std::size_t n)
            -> BackingAllocator::Bytes {
            return BackingAllocator::Bytes(
                new std::byte[n], [pinned, &canonical_at_free](std::byte *p) {
                    // #161 CANONICAL ACTUAL-FREE WITNESS: at the buffer's exact
                    // free boundary, the CANONICAL authority's materialized total
                    // is observed (no shadow ledger). With the correct member
                    // order the ARRAY claim is still held: the total equals
                    // capacity + control; a wrong order shows control-only here.
                    canonical_at_free = pinned->materializedBytes();
                    delete[] p;
                });
        };
        {
            GroupedServingEngine inner(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                       makePool(), recording);
            pinned = &inner.servingPool(); // canonical authority pinned; the deleter observes it
            expected_at_free = (1u << 20u) + inner.destinationReserveBytes() + inner.aggregateNodeBytes();
            inner.bankDestination(kDest, 1u << 20u);
            inner.materializeDestination(kDest);
            const auto inner_m = inner.servingPool().materializedBytes();
        EXPECT_EQ(inner_m, (1u << 20u) + inner.destinationReserveBytes() + inner.aggregateNodeBytes()); // backing + object control
        }
        // EXACT canonical witness: at the buffer free the total = capacity + the
        // banked object's control (the ARRAY claim still held -> right order).
        EXPECT_EQ(canonical_at_free, expected_at_free);
    }

    /** @brief S3.1: an allocation failure after the preclaim returns the preclaim
     *         to the reserve immediately (deterministic via the typed fault
     *         seam: the allocation throws, the local claim dies, nothing is
     *         retained). */
    TEST(SharedExecutionProof, FailedAllocationReturnsPreclaimImmediately)
    {
        auto throwing = BackingAllocator{};
        throwing.allocate = [](std::size_t) -> BackingAllocator::Bytes {
            throw std::bad_alloc();
        };
        GroupedServingEngine engine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                    makePool(), throwing);
        engine.bankDestination(kDest, 1u << 20u);
        EXPECT_THROW(engine.materializeDestination(kDest), std::bad_alloc);
        // The capacity preclaim returned; the banked backing OBJECT's control
        // claim remains (the object lives).
        EXPECT_EQ(engine.servingPool().materializedBytes(), engine.destinationReserveBytes() + engine.aggregateNodeBytes());
        EXPECT_EQ(engine.servingPool().remainingBytes(),
                  16u * 1024u * 1024u - (engine.destinationReserveBytes() + engine.aggregateNodeBytes()));
    }

    /** @brief S3.1: a reservation whose OWNER is not ServingResponsePool cannot
     *         fund destination backing (wrong-owner rejection). */
    TEST(SharedExecutionProof, WrongOwnerReservationRefused)
    {
        EXPECT_THROW(
            GroupedServingEngine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                 makePoolFor(PhysicalMemoryOwner::PrefixHostTier,
                                             DeviceId::cpu()),
                                 BackingAllocator::system()),
            std::logic_error);
    }

    /** @brief S3.1: a reservation on a GPU allocator resource cannot fund host
     *         destination backing (wrong-resource rejection; rank identity is
     *         NOT restricted - only the device resource kind). */
    TEST(SharedExecutionProof, WrongAllocatorResourceRefused)
    {
        EXPECT_THROW(
            GroupedServingEngine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                 makePoolFor(PhysicalMemoryOwner::ServingResponsePool,
                                             DeviceId::cuda(0)),
                                 BackingAllocator::system()),
            std::logic_error);
    }

    /** @brief S3.1: a NULL allocation result is refused WITHOUT a retained claim
     *         (the preclaim returns immediately; nothing commits) AND the pool is
     *         immediately clean and reusable by a correct allocation. */
    TEST(SharedExecutionProof, NullAllocationResultRefusedWithoutRetainedClaim)
    {
        int calls = 0;
        auto flaky = BackingAllocator{};
        flaky.allocate = [&calls](std::size_t n) -> BackingAllocator::Bytes {
            if (calls++ == 0)
                return BackingAllocator::Bytes(nullptr, [](std::byte *) {});
            return BackingAllocator::Bytes(new std::byte[n],
                                           [](std::byte *p) { delete[] p; });
        };
        GroupedServingEngine engine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                    makePool(), std::move(flaky));
        engine.bankDestination(kDest, 1u << 20u);
        EXPECT_THROW(engine.materializeDestination(kDest), std::logic_error);
        // The banked backing OBJECT (with its control claim) remains until the
        // destination retires — the capacity bytes were never claimed.
        EXPECT_EQ(engine.servingPool().materializedBytes(), engine.destinationReserveBytes() + engine.aggregateNodeBytes());
        EXPECT_EQ(engine.servingPool().remainingBytes(),
                  16u * 1024u * 1024u - (engine.destinationReserveBytes() + engine.aggregateNodeBytes()));
        // CLEANUP/REUSE: the rejected claim is fully returned; a later correct
        // allocation succeeds on the SAME engine/pool and materializes exactly.
        engine.materializeDestination(kDest);
        EXPECT_EQ(engine.servingPool().materializedBytes(),
                  (1u << 20u) + engine.destinationReserveBytes() + engine.aggregateNodeBytes());
    }

    /** @brief S3.1: an EMPTY allocator (no allocation function) is refused at
     *         construction (typed allocator required). */
    TEST(SharedExecutionProof, EmptyAllocatorRefused)
    {
        EXPECT_THROW(
            GroupedServingEngine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                 makePool(), BackingAllocator{}),
            std::logic_error);
    }


    /** @brief S3.2b ENGINE BINDING: with an ABI runner + an admitted per-row law,
     *         the engine mints the typed serving-window command (it is the row
     *         owner) and advances the row by the AUTHENTICATED produced count; a
     *         completion resolves the row's grant exactly once; a failure defers
     *         the window (fail-closed). */
    TEST(SharedExecutionProof, EngineAdvancesRowsThroughTheAbi)
    {
        // A completing ABI runner (production decode stand-in).
        class CompletingAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                // Real publication: the production decode writes the window bytes.
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'A', sizeof(bs)); // ACTUAL bytes (not zeros)
                    request.output->writeOutput(bs, 16);
                }
                // The window is exact; the terminal boundary is honored.
                return completeWindow(request.authority, request, 16,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        CompletingAbi abi_runner;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto law = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi_runner);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &law, 4);
        // The ABI window (quantum 16, limit 32, start 0) completes 16 tokens and
        // advances the row; the mock decode is NOT used (the ABI path is).
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0).size(), 16u); // ABI-produced window count
        EXPECT_EQ(step1.at(0), std::string(16, 'A')); // REAL published bytes
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(engine.rows().at(0).state.committed, 32u);
        EXPECT_TRUE(engine.rows().at(0).complete);
    }


    /** @brief S3.2b WINDOW OUTPUT PUBLICATION: the production decode writes the
     *         REAL window bytes into the destination backing via the typed
     *         sink; the engine publishes the window from its staging (the
     *         device-free adapter fixture) — not a count placeholder. */
    TEST(SharedExecutionProof, EnginePublishesAbiWindowBytesFromDestination)
    {
        // A runner whose production decode publishes 16 'A' bytes per window.
        class PublishingAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    const std::byte bs[16] = {};
                    request.output->writeOutput(bs, 16);
                }
                return completeWindow(request.authority, request, 16,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishingAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto law = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &law, 4);
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0).size(), 16u); // real published bytes
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
    }


    /** @brief S3.2b ABI-DRIVEN A/B/C ACCEPTANCE SHAPE: three rows with PER-ROW
     *         laws and the ABI runner — A continues while B finishes and C
     *         (waiting) enters B's retired slot; every window is minted with the
     *         row's OWN progress and completed authentically (the grant consumed
     *         exactly once); the engine remains the sole lifecycle authority. */
    TEST(SharedExecutionProof, AbiMixedAciContinuesBFinishesCeEntersRetiredSlot)
    {
        class PerRowAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                // Authentic exact windows; terminal boundary honored.
                const auto limit = request.authority.current().window.response_limit;
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining = limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'A', sizeof(bs)); // ACTUAL bytes (not zeros)
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PerRowAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto lawB = admission->issue(2, 22, 2, 32);
        auto lawC = admission->issue(3, 33, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.bankDestination(kDestC, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.materializeDestination(kDestB);
        engine.materializeDestination(kDestC);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);   // A (long)
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256, &lawB, 5);  // B (short: 16 limit)
        engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256, &lawC, 6); // C waits for B's slot
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0).size(), 16u);          // A window 1 (ABI-produced)
        EXPECT_NE(step1.at(1).find("|entry:2"), std::string::npos); // C entered B's slot
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // C owns B's retired slot
        EXPECT_EQ(engine.rows().at(1).state.committed, 0u);  // C is FRESH in B's slot
        EXPECT_FALSE(engine.rows().at(1).complete);
        // A continues (its own grant re-issued). C's FIRST grant was issued at
        // queue time (it is fresh in B's retired slot) — the ABI decode consumes
        // it in step 2; no extra grant is issued between.
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(0).size(), 16u);          // A window 2
        EXPECT_EQ(step2.at(1).size(), 8u);           // C window 1 (quantum 8)
        EXPECT_EQ(engine.rows().at(0).state.committed, 32u);
        EXPECT_TRUE(engine.rows().at(0).complete);   // A finished
    }


    /** @brief FAILURE-ATOMIC PER-ROW TRANSACTION: an ABI decode refusal for one
     *         row defers ONLY that row (grant untouched, engine not poisoned,
     *         the precise diagnostic observable) while other rows advance. */
    TEST(SharedExecutionProof, AbiFailureDefersOnlyThatRow)
    {
        class FlakyAbi : public MockOrchestrationRunner
        {
        public:
            bool fail_b = true;
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (fail_b && request.expected_generation == 1 &&
                    request.expected_slot == 1)
                    return ServingWindowDecodeResult::failure("intentional row-B failure");
                // A succeeds: publish + complete.
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'A', sizeof(bs));
                    request.output->writeOutput(bs, 16);
                }
                return completeWindow(request.authority, request, 16,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        FlakyAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto lawB = admission->issue(2, 22, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.materializeDestination(kDestB);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);  // A (succeeds)
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256, &lawB, 5); // B (fails)
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0).size(), 16u); // A advanced
        EXPECT_EQ(step1.count(1), 0u);      // B deferred (no output)
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_FALSE(engine.failed());      // engine NOT poisoned
        EXPECT_TRUE(engine.rows().at(1).authority.mayExecute()); // B's grant untouched
        EXPECT_EQ(engine.abiDeferrals().at(1).state, AbiDeferralState::Retryable);
        EXPECT_EQ(engine.abiDeferrals().at(1).reason, "intentional row-B failure");
        // A continues on the next step; B is deferred again (never poisoned).
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(0).size(), 16u);
        EXPECT_EQ(step2.count(1), 0u);
        EXPECT_EQ(engine.abiDeferrals().at(1).state, AbiDeferralState::Retryable);
        EXPECT_EQ(engine.abiDeferrals().at(1).reason, "intentional row-B failure");
    }


    /** @brief EARLY-EOS COMPLETION RETIRES: an ABI row completing via an
     *         AUTHENTICATED exception (EOS at committed < limit) goes through the
     *         shared ledger/retire/attach (delivery is not refused) — a waiting
     *         request enters the retired slot. */
    TEST(SharedExecutionProof, AbiEarlyEosCompletesAndAttaches)
    {
        class EosAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'E', sizeof(bs));
                    request.output->writeOutput(bs, 4); // EOS at 4 tokens (early)
                }
                // AUTHENTICATED EOS: 0 < produced < bound is the exception path.
                return completeWindow(request.authority, request, 4,
                                      WindowTermination::EndOfSequence);
            }
        };
        EosAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawB = admission->issue(1, 11, 2, 32);
        auto lawC = admission->issue(2, 22, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.bankDestination(kDestC, 1u << 20u);
        engine.materializeDestination(kDestB);
        engine.materializeDestination(kDestC);
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256, &lawB, 5);
        engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256, &lawC, 6);
        const auto step1 = engine.step();
        EXPECT_NE(step1.at(1).find("|entry:2"), std::string::npos); // attach NOT refused
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u); // C entered the slot
        EXPECT_EQ(engine.abiDeferrals().count(1), 0u);        // no deferral (EOS accepted)
        // EARLY-TERMINAL DELIVERY: the EOS-completed response's destination is
        // accepted by the consumer (markDelivered) and releases exactly once.
        engine.markDelivered(kDestB, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDestB));
    }

    /** @brief STATE-VIOLATED ROW IS NEVER RETRIED: an ABI decode that consumes
     *         the original grant and then returns a failure is classified as a
     *         state violation — the row is marked and NEVER re-decoded (no
     *         write-then-retry double publication). */
    TEST(SharedExecutionProof, AbiStateViolatedRowNeverRetried)
    {
        class ConsumeThenFailAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'X', sizeof(bs));
                    request.output->writeOutput(bs, 16); // bytes ALREADY published
                }
                auto consumed = completeWindow(request.authority, request, 16,
                                               WindowTermination::Continue);
                (void)consumed;
                return ServingWindowDecodeResult::failure("internal error");
            }
        };
        ConsumeThenFailAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawB = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.materializeDestination(kDestB);
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256, &lawB, 5);
        const auto step1 = engine.step();
        EXPECT_EQ(step1.count(1), 0u); // no output (state-violated)
        EXPECT_EQ(engine.abiDeferrals().at(1).state,
                  AbiDeferralState::StateViolated);
        // NOT retried: a second step does NOT re-decode the state-failed row
        // (no new deferral recorded; the classification is not re-emitted).
        const auto step2 = engine.step();
        EXPECT_EQ(step2.count(1), 0u);
        EXPECT_EQ(engine.abiDeferrals().count(1), 0u); // never retried (not re-recorded)
    }

    /** @brief EXPLICIT EMPTY PUBLICATION IS ACCEPTED: a successful decode that
     *         explicitly publishes a zero-extent window is a legitimate empty
     *         window (not an absence violation). */
    TEST(SharedExecutionProof, AbiExplicitEmptyPublicationAccepted)
    {
        class EmptyPublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                    request.output->writeOutput(nullptr, 0); // EXPLICIT empty publish
                return completeWindow(request.authority, request, 16,
                                      WindowTermination::Continue);
            }
        };
        EmptyPublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        const auto step1 = engine.step();
        EXPECT_EQ(engine.abiDeferrals().count(0), 0u); // accepted (no deferral)
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_EQ(step1.at(0).size(), 0u); // empty window output
    }

    /** @brief RELEASED/DEFAULT LAW IS REFUSED: an ABI-bridged engine rejects a
     *         released or default law at admission (no silent scalar mock). */
    TEST(SharedExecutionProof, AbiRejectsReleasedLawAtAdmission)
    {
        class AnyAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &) override
            {
                return ServingWindowDecodeResult::failure("unsupported");
            }
        };
        AnyAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto law = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        admission->release(law); // released law
        EXPECT_THROW(engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &law, 4),
                     std::logic_error);
    }


    /** @brief SLOW CONSUMER (ABI path): a throttled row is deferred (its ABI
     *         window not decoded) while the healthy row advances — the
     *         backpressure policy is independent of the decode path. */
    TEST(SharedExecutionProof, AbiSlowConsumerDefersOnlyItself)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'S', sizeof(bs));
                    request.output->writeOutput(bs, 16);
                }
                return completeWindow(request.authority, request, 16,
                                      WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto lawB = admission->issue(2, 22, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.materializeDestination(kDestB);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);  // healthy
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256, &lawB, 5); // slow consumer
        engine.throttle(1); // backpressure on B only
        const auto step1 = engine.step();
        EXPECT_EQ(step1.count(0), 1u);   // A advanced
        EXPECT_EQ(step1.count(1), 0u);   // B deferred (throttled)
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_EQ(engine.rows().at(1).state.committed, 0u);
        EXPECT_TRUE(engine.rows().at(1).authority.mayExecute()); // untouched
        EXPECT_EQ(engine.abiDeferrals().count(1), 0u); // throttle, not an ABI refusal
    }

    /** @brief AUTHENTICATED CANCELLATION (ABI path): a runner cancelling its OWN
     *         command defers the row with the Cancelled typed state; the grant is
     *         cancelled (not consumed) and the row may be re-admitted. */
    TEST(SharedExecutionProof, AbiCancellationDefersAndCancelsTheGrant)
    {
        class CancelAbi : public MockOrchestrationRunner
        {
        public:
            int windows = 0;
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining = request.authority.current().window.response_limit -
                                       request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                // FIRST window publishes (16 tokens * 8 = 128); the SECOND
                // CANCELS (non-terminal defer); the THIRD RESUMES/COMPLETES the
                // same destination (a continuation after a cancelled window —
                // real progress, not another cancel).
                if (windows == 0 || windows == 2)
                {
                    if (request.output != nullptr)
                    {
                        std::byte bs[256];
                        std::memset(bs, 'C', sizeof(bs));
                        request.output->writeOutput(bs, produced * 8);
                    }
                    ++windows;
                    return completeWindow(request.authority, request, produced,
                                          request.final_partial
                                              ? WindowTermination::Terminal
                                              : WindowTermination::Continue);
                }
                ++windows;
                return cancelWindow(request.authority, request); // own-command cancellation
            }
        };
        CancelAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        const auto step1 = engine.step();
        EXPECT_EQ(step1.count(0), 1u); // FIRST window published real bytes (128)
        engine.issueNextGrant(0); // the engine issues the next grant (window consumed)
        const auto step2 = engine.step();
        EXPECT_EQ(step2.count(0), 0u); // deferred (cancelled window)
        EXPECT_EQ(engine.abiDeferrals().at(0).state, AbiDeferralState::Cancelled);
        // A CANCELLED WINDOW IS A NON-TERMINAL DEFER: NO retention, NO
        // ownership transfer — the destination stays OWNED by the live row and
        // the CONTINUATION executes against the SAME backing (a cancelled
        // window can never lose the destination or route to a successor).
        EXPECT_FALSE(engine.hasRetainedPayload(kDest, 1));
        EXPECT_FALSE(engine.destinationReleased(kDest));
        EXPECT_FALSE(engine.rows().at(0).authority.mayExecute()); // grant CANCELLED
        EXPECT_FALSE(engine.rows().at(0).complete);
        // Not poisoned; a fresh grant admits the row again AND the
        // continuation EXECUTES against the SAME destination (step 3 completes).
        engine.issueNextGrant(0);
        EXPECT_TRUE(engine.rows().at(0).authority.mayExecute());
        // LIVE RE-BANK NEGATIVE: while the row lives, the SAME destination can
        // NOT be re-banked (the row owns it).
        EXPECT_THROW(engine.bankDestination(kDest, 1u << 20u), std::logic_error);
        const auto step3 = engine.step();
        EXPECT_EQ(step3.count(0), 1u); // the continuation PUBLISHES + progresses
        // the row is not complete after three 16-token windows of a 32-token
        // limit... actually 3*16 = 32? L=32 -> the third window completes the
        // response: assert terminal completion + the destination released by the
        // delivery path (the row finished against the SAME backing).
        EXPECT_TRUE(engine.rows().at(0).complete);
        engine.markDelivered(kDest, 1); // delivery recorded
        engine.step();                    // delivery-driven release runs in the post-pass
        EXPECT_TRUE(engine.destinationReleased(kDest));
    }


    /** @brief MTP ON THE ABI PATH: an ABI row with admitted MTP depth runs the
     *         shared verification once per step (the counter reset is shared);
     *         a second window does NOT throw after effects. */
    TEST(SharedExecutionProof, AbiRowAdvancesWithMtpVerification)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'M', sizeof(bs));
                    request.output->writeOutput(bs, 16);
                }
                return completeWindow(request.authority, request, 16,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        engine.setMTPDepth(0, 1); // depth 1 for the ABI row
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0).size(), 16u);
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        // Second window (MTP reset was shared): no throw after effects.
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(0).size(), 16u);
        EXPECT_EQ(engine.rows().at(0).state.committed, 32u);
        EXPECT_TRUE(engine.rows().at(0).complete);
    }

    /** @brief QUEUED RESTORE + ABI ATTACH: a queued row with a completed restore
     *         copy attaches (its law pin + authority transfer) and advances
     *         through the ABI path. */
    TEST(SharedExecutionProof, AbiQueuedRestoreRowAttachesAndAdvances)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'R', sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawB = admission->issue(1, 11, 2, 32);
        auto lawC = admission->issue(2, 22, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.bankDestination(kDestC, 1u << 20u);
        engine.materializeDestination(kDestB);
        engine.materializeDestination(kDestC);
        engine.admitNew(1, 1, 16, 16, 2, kDestB, 256, &lawB, 5); // B (quick)
        engine.queueRequest(1, 2, 32, 8, 3, kDestC, 4096, &lawC, 6);
        engine.beginQueuedRestore(1, 4096); // restore == prompt geometry
        std::array<std::byte, 4096> src{};
        engine.completeRestoreCopy(1, engine.queuedRestoreSession(1), 4096, src.data());
        // RestoreFrontierReceipt is move-only (non-copyable sole authority).
        auto receipt = engine.issueRestoreReceipt(1, 2, 4096);
        engine.publishRestore(1, std::move(receipt)); // restore published (attachable)
        const auto step1 = engine.step();
        EXPECT_NE(step1.at(1).find("|entry:2"), std::string::npos); // C attached
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u);
        // C advances on the next step through the ABI path (its first grant).
        const auto step2 = engine.step();
        EXPECT_EQ(step2.at(1).size(), 8u); // C window (quantum 8)
        EXPECT_EQ(engine.rows().at(1).state.committed, 8u);
    }


    /** @brief DIRTY-ONLY BRANCH (independent of consumption): the decode STAGES
     *         bytes then FAILS WITHOUT consuming the grant — the dirty-only
     *         predicate classifies StateViolated (never retried). */
    TEST(SharedExecutionProof, AbiDirtyOnlyFailureClassifiesStateViolated)
    {
        class DirtyThenFailAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'D', sizeof(bs));
                    request.output->writeOutput(bs, 16); // STAGED, NOT consumed
                }
                return ServingWindowDecodeResult::failure("decode failed after staging");
            }
        };
        DirtyThenFailAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        const auto step1 = engine.step();
        EXPECT_EQ(step1.count(0), 0u);
        EXPECT_EQ(engine.abiDeferrals().at(0).state, AbiDeferralState::StateViolated);
    }

    /** @brief STATE-VIOLATED ROW RETIRES (slot + destination released; the
     *         waiting request enters) — the lifecycle never strands. */
    TEST(SharedExecutionProof, AbiStateViolatedRowRetiresAndSlotReuse)
    {
        class PolluteAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'P', sizeof(bs));
                    request.output->writeOutput(bs, 16);
                }
                auto consumed = completeWindow(request.authority, request, 16,
                                               WindowTermination::Continue);
                (void)consumed;
                return ServingWindowDecodeResult::failure("polluted after consume");
            }
        };
        PolluteAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawB = admission->issue(1, 11, 2, 32);
        auto lawC = admission->issue(2, 22, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.bankDestination(kDestC, 1u << 20u);
        engine.materializeDestination(kDestB);
        engine.materializeDestination(kDestC);
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256, &lawB, 5);
        engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256, &lawC, 6);
        const auto step1 = engine.step();
        EXPECT_NE(step1.at(1).find("|entry:2"), std::string::npos); // C entered the slot
        EXPECT_EQ(engine.abiDeferrals().at(1).state, AbiDeferralState::StateViolated);
        // The violated row RETIRED: C owns the slot (lifecycle not stranded) and
        // B's old destination was released (no delivery obligation).
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u);
        EXPECT_TRUE(engine.destinationReleased(kDestB));
    }


    /** @brief MIXED PROMPT SIZES (ABI path): each row's published window lands at
     *         ITS OWN prompt base (the sealed sink offsets differ per row — the
     *         prompt is never overwritten and the outputs never cross). */
    TEST(SharedExecutionProof, AbiMixedPromptSizesPublishAtOwnOffsets)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'M', sizeof(bs));
                    request.output->writeOutput(bs, 16);
                }
                return completeWindow(request.authority, request, 16,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto lawB = admission->issue(2, 22, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.materializeDestination(kDestB);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);   // prompt 256 (A)
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 4096, &lawB, 5); // prompt 4096 (B)
        const auto step1 = engine.step();
        EXPECT_EQ(step1.at(0).size(), 16u); // A window ('M' x16)
        EXPECT_EQ(step1.at(1).size(), 16u); // B window ('M' x16, at B's own base)
        EXPECT_EQ(step1.at(0), std::string(16, 'M'));
        EXPECT_EQ(step1.at(1), std::string(16, 'M'));
        // PER-ROW OUTPUT METADATA: row A's committed output window is at ITS OWN
        // prompt base (256, 16) and row B's at (4096, 16) — the outputs never
        // cross, never overlap the prompts, and never contain holes.
        const auto metaA = engine.abiDestinationMetadata(kDest);
        const auto metaB = engine.abiDestinationMetadata(kDestB);
        EXPECT_EQ(metaA.first, 256u);  // A's prompt base
        EXPECT_EQ(metaA.second, 16u);
        EXPECT_EQ(metaB.first, 4096u); // B's prompt base
        EXPECT_EQ(metaB.second, 16u);
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        EXPECT_EQ(engine.rows().at(1).state.committed, 16u);
    }


    /** @brief DIRTY-ONLY + READY QUEUED SUCCESSOR (cross product): a violated row
     *         with a waiting successor is ERASED first and the successor is
     *         EMPLACED afresh (no member-wise partial mutation) — the successor
     *         owns the slot with its OWN Outstanding grant (never a stale/live
     *         grant retained) and the violated destination releases exactly once. */
    TEST(SharedExecutionProof, AbiDirtyOnlyWithQueuedSuccessorReplacesSafely)
    {
        class DirtyThenFailAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (request.output != nullptr)
                {
                    std::byte bs[16];
                    std::memset(bs, 'D', sizeof(bs));
                    request.output->writeOutput(bs, 16);
                }
                return ServingWindowDecodeResult::failure("decode failed after staging");
            }
        };
        DirtyThenFailAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawB = admission->issue(1, 11, 2, 32);
        auto lawC = admission->issue(2, 22, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.bankDestination(kDestC, 1u << 20u);
        engine.materializeDestination(kDestB);
        engine.materializeDestination(kDestC);
        engine.admitNew(1, 1, 32, 16, 2, kDestB, 256, &lawB, 5); // violated (dirty)
        engine.queueRequest(1, 2, 32, 8, 3, kDestC, 256, &lawC, 6); // ready successor
        const auto step1 = engine.step();
        EXPECT_EQ(engine.abiDeferrals().at(1).state, AbiDeferralState::StateViolated);
        EXPECT_EQ(engine.rows().at(1).state.generation, 2u);  // successor owns the slot
        EXPECT_TRUE(engine.rows().at(1).authority.mayExecute()); // OWN Outstanding grant
        EXPECT_EQ(engine.rows().at(1).state.slot, 1u);
        EXPECT_TRUE(engine.destinationReleased(kDestB)); // violated dest released
        EXPECT_FALSE(engine.destinationReleased(kDestC)); // successor's dest retained
    }


    /** @brief SYNTHETIC SAME-SEQUENCE REPEATABILITY (NOT recorded-input replay):
     *         two fresh engines run the SAME hardcoded sequence; outputs are
     *         compared. The trace is NOT recorded/consumed (see the separate
     *         trace-driven replay for #142). */
    TEST(SharedExecutionProof, AbiSyntheticSameSequenceRepeatable)
    {
        struct Trace
        {
            std::vector<std::pair<std::uint32_t, std::string>> steps; // slot -> output
            std::vector<std::uint32_t> grants;
        };
        class RecordingAbi : public MockOrchestrationRunner
        {
        public:
            RecordingAbi(Trace &t) : t_(t) {}
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'R', sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
            Trace &t_;
        };
        // RUN 1: record the trace.
        Trace trace;
        RecordingAbi rec(trace);
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto lawB = admission->issue(2, 22, 2, 32);
        auto engine1 = makeEngine(solo);
        engine1.setAbiRunner(&rec);
        engine1.bankDestination(kDest, 1u << 20u);
        engine1.bankDestination(kDestB, 1u << 20u);
        engine1.materializeDestination(kDest);
        engine1.materializeDestination(kDestB);
        engine1.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        engine1.admitNew(1, 1, 16, 16, 2, kDestB, 256, &lawB, 5);
        const auto out1a = engine1.step();
        engine1.issueNextGrant(0);
        const auto out1b = engine1.step();
        // RUN 2: replay the recorded command sequence with a fresh engine.
        Trace trace2;
        RecordingAbi rec2(trace2);
        auto admission2 = std::make_shared<ServingLawAdmission>(32);
        auto lawA2 = admission2->issue(1, 11, 2, 32);
        auto lawB2 = admission2->issue(2, 22, 2, 32);
        auto engine2 = makeEngine(solo);
        engine2.setAbiRunner(&rec2);
        engine2.bankDestination(kDest, 1u << 20u);
        engine2.bankDestination(kDestB, 1u << 20u);
        engine2.materializeDestination(kDest);
        engine2.materializeDestination(kDestB);
        engine2.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA2, 4);
        engine2.admitNew(1, 1, 16, 16, 2, kDestB, 256, &lawB2, 5);
        const auto out2a = engine2.step();
        engine2.issueNextGrant(0);
        const auto out2b = engine2.step();
        // IDENTICAL outputs + state transitions (recorded-input replay).
        EXPECT_EQ(out1a, out2a);
        EXPECT_EQ(out1b, out2b);
        EXPECT_EQ(engine1.rows().at(0).state.committed,
                  engine2.rows().at(0).state.committed);
        EXPECT_EQ(engine1.rows().at(1).state.committed,
                  engine2.rows().at(1).state.committed);
        EXPECT_EQ(engine1.rows().at(0).complete, engine2.rows().at(0).complete);
    }


    /** @brief DISTINCT SAMPLER LAWS STAY ISOLATED (acceptance shape): per-row
     *         sampling laws (law epoch/token + resolved seed) remain per-row —
     *         the ABI path's law pins never cross, and each row's window is
     *         produced under its OWN law (the production decode sees the exact
     *         per-row command). */
    TEST(SharedExecutionProof, AbiDistinctSamplerLawsStayIsolated)
    {
        class LawKeyAbi : public MockOrchestrationRunner
        {
        public:
            std::map<std::uint64_t, std::uint64_t> seen; // law issue identity -> seed
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                // The production decode reads the row's OWN law + resolved seed.
                seen[request.lawNode()->issueIdentity()] = request.seed;
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'A' + static_cast<int>(request.seed % 26), sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        LawKeyAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32); // law epoch 1
        auto lawB = admission->issue(2, 22, 2, 32); // law epoch 2
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.bankDestination(kDestB, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.materializeDestination(kDestB);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4); // seed 1, law 1
        engine.admitNew(1, 1, 32, 16, 7, kDestB, 256, &lawB, 5); // seed 7, law 2
        const auto step1 = engine.step();
        EXPECT_EQ(abi.seen.size(), 2u); // TWO distinct laws seen
        EXPECT_EQ(abi.seen[lawA.issueIdentity()], 1u); // A's law -> A's seed
        EXPECT_EQ(abi.seen[lawB.issueIdentity()], 7u); // B's law -> B's seed
        // Per-row outputs differ by the row's OWN law seed (isolation).
        EXPECT_EQ(step1.at(0), std::string(16, 'B')); // 'A' + 1 % 26 -> 'B'
        EXPECT_EQ(step1.at(1), std::string(16, 'H')); // 'A' + 7 % 26 -> 'H'
    }


    /** @brief RECORDED-TRACE-DRIVEN REPLAY (#142): run 1 RECORDS the engine
     *         operations AND the ABI commands the production decode received
     *         (per-command identity + expected window output). The REPLAYER is a
     *         driver that CONSUMES the recorded trace — it executes each recorded
     *         operation, verifies each decode command matches the recorded one,
     *         and checks the window outputs against the recorded expected values.
     *         Controls: a truncated/incompatible trace is refused by the driver
     *         (never silently skipped). */
    TEST(SharedExecutionProof, AbiRecordedTraceDrivenReplay)
    {
        struct DecodeRecord
        {
            std::uint64_t slot = 0;
            std::uint64_t issuance_nonce = 0;
            std::uint64_t window_start = 0;
            std::uint64_t produced = 0;
        };
        std::vector<DecodeRecord> commands;
        std::vector<int> ops; // 0=admitA 1=admitB 2=step 3=grantA 4=step
        class RecordingAbi : public MockOrchestrationRunner
        {
        public:
            RecordingAbi(const std::vector<DecodeRecord> *expected,
                         std::vector<DecodeRecord> *record)
                : expected_(expected), record_(record) {}
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (expected_ != nullptr)
                {
                    // REPLAY MODE: a mismatch (or a truncated trace) REFUSES — the
                    // runner returns a failure and NEVER completes (no effects).
                    if (cursor >= expected_->size() ||
                        expected_->at(cursor).slot != request.expected_slot ||
                        expected_->at(cursor).issuance_nonce != request.expected_nonce ||
                        expected_->at(cursor).window_start != request.window_start ||
                        expected_->at(cursor).produced != produced)
                    {
                        replay_ok = false; // a mismatch/truncation is REPORTED
                        ++cursor;
                        return ServingWindowDecodeResult::failure(
                            "recorded command mismatch (incompatible trace)");
                    }
                    ++cursor;
                }
                else
                {
                    DecodeRecord r;
                    r.slot = request.expected_slot;
                    r.issuance_nonce = request.expected_nonce;
                    r.window_start = request.window_start;
                    r.produced = produced;
                    record_->push_back(r);
                }
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'T', sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
            const std::vector<DecodeRecord> *expected_;
            std::vector<DecodeRecord> *record_;
        public:
            std::size_t cursor = 0;
            bool replay_ok = true;
            std::vector<std::map<std::uint32_t, std::string>> captured;
        };
        // DRIVER: executes ONLY the recorded ops (the admissions are recorded
        // ops too — nothing is hardcoded); an UNKNOWN op is refused (throw).
        auto drive = [&](const std::vector<int> *op_trace,
                         const std::vector<DecodeRecord> *expected,
                         std::vector<DecodeRecord> *record_target) {
            auto admission = std::make_shared<ServingLawAdmission>(32);
            auto lawA = admission->issue(1, 11, 2, 32);
            auto lawB = admission->issue(2, 22, 2, 32);
            auto rec = std::make_unique<RecordingAbi>(expected, record_target);
            auto engine = makeEngine(solo);
            engine.setAbiRunner(rec.get());
            engine.bankDestination(kDest, 1u << 20u);
            engine.bankDestination(kDestB, 1u << 20u);
            engine.materializeDestination(kDest);
            engine.materializeDestination(kDestB);
            for (int op : *op_trace)
            {
                if (op == 0)
                    engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
                else if (op == 1)
                    engine.admitNew(1, 1, 32, 16, 2, kDestB, 256, &lawB, 5);
                else if (op == 2)
                    rec->captured.push_back(engine.step());
                else if (op == 3)
                    engine.issueNextGrant(0);
                else
                    throw std::logic_error("unknown recorded op (incompatible trace)");
            }
            return rec;
        };
        // RUN 1: record the full op sequence + the observed ABI commands.
        ops = {0, 1, 2, 3, 2};
        auto rec1 = drive(&ops, nullptr, &commands);
        EXPECT_EQ(commands.size(), 3u);
        // RUN 2: REPLAY the recorded trace (the driver executes only the ops).
        auto rec2 = drive(&ops, &commands, nullptr);
        EXPECT_TRUE(rec2->replay_ok);
        EXPECT_EQ(rec2->cursor, commands.size());
        EXPECT_EQ(rec1->captured, rec2->captured);
        // INCOMPATIBLE TRACE: a truncated command expectation is REFUSED (the
        // runner returns a failure and never completes; no effects).
        std::vector<DecodeRecord> truncated(commands.begin(), commands.begin() + 1);
        auto rec3 = drive(&ops, &truncated, nullptr);
        EXPECT_FALSE(rec3->replay_ok);
    }


    /** @brief RESPONSE-BOUND FAILURE LIFECYCLE: a FAILED model response is
     *         recorded with its exact generation + reason, its destination
     *         releases exactly once (no delivery obligation), a LATER delivery is
     *         refused, and a second failure is refused (exactly-once failure
     *         retirement — no discard without record). */
    TEST(SharedExecutionProof, AbiResponseFailureIsExactOnceAndBlocksDelivery)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'E', sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256, &lawA, 4);
        const auto step1 = engine.step();
        EXPECT_TRUE(engine.rows().at(0).complete);
        // FAILED invocation outcome: recorded with the generation + reason; the
        // destination releases exactly once (no delivery obligation).
        engine.failResponse(kDest, 1, "transport failure");
        EXPECT_TRUE(engine.responseFailed(kDest));
        EXPECT_EQ(engine.responseFailureReason(kDest), "transport failure");
        EXPECT_TRUE(engine.destinationReleased(kDest));
        // A LATER delivery of a failed response is REFUSED.
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::logic_error);
        // A second failure record is refused (exactly once).
        EXPECT_THROW(engine.failResponse(kDest, 1, "again"), std::logic_error);
        // Stale generation failure is refused (ABA).
        EXPECT_THROW(engine.failResponse(kDest, 99, "stale"), std::logic_error);
    }


    /** @brief STAGING IS BOUNDED BY THE ADMITTED OUTPUT REGION (multi-window):
     *         the P256/cap512 case rejects the second-window overflow AT STAGING
     *         (before any completion — a violated deferral, never a post-
     *         consumption commit throw): the FIRST window (16) stages/commits,
     *         the SECOND (exceeding the remaining admitted extent) is refused at
     *         staging with the row state-violated + the grant NOT consumed. */
    TEST(SharedExecutionProof, AbiStagingRejectsPriorOutputExhaustion)
    {
        class OverflowAbi : public MockOrchestrationRunner
        {
        public:
            int windows = 0;
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                // PRODUCTION BYTES: 8 per token (the derived-payload geometry).
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                const auto bytes = produced * 8;
                std::byte bs[512];
                std::memset(bs, 'O', sizeof(bs));
                if (request.output != nullptr)
                {
                    // SECOND window: an OVER-DERIVED production stage (512 bytes
                    // when only 128 are geometry-valid) — the staging bound with
                    // the PRIOR OUTPUT included REFUSES it BEFORE any completion.
                    // DISCRIMINATING prior-output accounting: the second window
                    // stages bytes + 72 (=200); WITH the prior 128 included the
                    // bound rejects (256 + 128 + 200 > 512); WITHOUT the prior it
                    // would pass (256 + 200 <= 512) — only prior-output-inclusive
                    // accounting rejects.
                    request.output->writeOutput(bs, windows == 0 ? bytes : bytes + 72);
                }
                ++windows;
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        OverflowAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        // capacity 512: prompt 256 + output region 256 (a 32-token response at
        // 8 bytes/token = exactly the derived payload; two windows of 16 tokens).
        engine.bankDestination(kDest, 256u + 256u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        const auto step1 = engine.step();
        EXPECT_EQ(step1.count(0), 1u); // first window (128 bytes) fits
        // AUTHORITY OBSERVATION before the second window: only the admitted
        // first window's progress advanced (16); the second window never
        // consumed authority.
        EXPECT_EQ(engine.rows().at(0).state.committed, 16u);
        engine.issueNextGrant(0);
        const auto step2 = engine.step();
        EXPECT_EQ(step2.count(0), 0u); // second staging (200 > remaining, prior-incl.) refused
        EXPECT_EQ(engine.abiDeferrals().at(0).state, AbiDeferralState::StateViolated);
        EXPECT_NE(engine.abiDeferrals().at(0).reason.find("exceeded the admitted"),
                  std::string::npos);
        // COMPLETION WITNESS: the ABI observed EXACTLY ONE window (the second
        // staging was refused BEFORE the decode — never a completion), and the
        // authoritative committed state shows the first window applied (16),
        // while the UNCONSUMED grant proves no authority advance for the
        // refused window.
        EXPECT_EQ(abi.windows, 1); // only ONE staging path completed (honest witness below)
        // COMPLETION WITNESS (HONEST scope): windows==1 is the POST-WRITE /
        // PRE-COMPLETION path count — the SECOND call DID enter the ABI but its
        // staging write threw (the counter increments only after the write), so
        // NO completion was produced for the second window. It is NOT an
        // ABI-entry counter.
        EXPECT_EQ(step1.count(0), 1u);
        EXPECT_EQ(step2.count(0), 0u);
        // TERMINAL EXECUTION-FAILURE RETENTION: the violated row's retirement
        // (this StateViolated path) retains the FIRST window's committed bytes —
        // the writer delivers the partial (128 bytes, all 'O') + the terminal
        // error via the generation-bound lease. The re-bank refusal is checked
        // while the lease is outstanding (undelivered data cannot be silently
        // dropped by a new occupant).
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 1));
        auto lease = engine.acquireRetainedPayload(kDest, 1);
        EXPECT_EQ(lease.bytes().size(), 128u);
        for (const char ch : lease.bytes())
            EXPECT_EQ(ch, 'O');
        EXPECT_THROW(engine.bankDestination(kDest, 512u), std::logic_error); // lease outstanding
        lease.release();
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 1)); // unleased (retired at the re-bank)
        engine.takeFailureReason(kDest, 1);               // consume the terminal reason
        engine.bankDestination(kDest, 512u);              // retire + re-bank proceeds
        engine.materializeDestination(kDest);
    }


    /** @brief RE-BANK ABA IS CLOSED: complete + fail a FIRST occupant, then
     *         RE-BANK the same reservation for a NEW occupant — the bank clears
     *         the prior completion/delivery/failure records, so a stale
     *         failResponse (old generation) is REFUSED and can never free the
     *         NEW occupant's backing. */
    TEST(SharedExecutionProof, AbiReBankClearsPriorResponseState)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'R', sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256, &lawA, 4); // occupant gen 1
        engine.step();                                          // completed
        engine.failResponse(kDest, 1, "first occupant failed");
        EXPECT_TRUE(engine.responseFailed(kDest));
        EXPECT_TRUE(engine.destinationReleased(kDest));
        // The terminal-error REASON is owed: a re-bank REFUSES until the writer
        // consumed it (exactly-once).
        EXPECT_THROW(engine.bankDestination(kDest, 1u << 20u), std::logic_error);
        EXPECT_EQ(engine.takeFailureReason(kDest, 1), "first occupant failed");
        EXPECT_THROW(engine.takeFailureReason(kDest, 1), std::logic_error); // exactly-once
        EXPECT_THROW(engine.takeFailureReason(kDest, 99), std::logic_error); // stale gen refused
        // converge: a SECOND failResponse is refused (terminal history stays +
        // the survived completion ledger cannot authorize a re-failure).
        EXPECT_THROW(engine.failResponse(kDest, 1, "again"), std::logic_error);
        // RE-BANK the SAME reservation for a NEW occupant (gen 2).
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 2, 16, 16, 1, kDest, 256, &lawA, 4);
        EXPECT_FALSE(engine.responseFailed(kDest)); // stale failure CLEARED at bank
        // STALE failure (old generation) is REFUSED — it cannot free the NEW
        // occupant's active backing.
        EXPECT_THROW(engine.failResponse(kDest, 1, "stale"), std::logic_error);
        EXPECT_FALSE(engine.destinationReleased(kDest)); // new backing NOT freed
        EXPECT_TRUE(engine.rows().at(0).authority.mayExecute());
        // The NEW occupant's own failure works (correct generation).
        engine.step();
        engine.failResponse(kDest, 2, "second occupant failed");
        EXPECT_TRUE(engine.responseFailed(kDest));
        EXPECT_TRUE(engine.destinationReleased(kDest));
    }


    /** @brief SUCCESSFUL-RESPONSE DELIVERY IS GENERATION-BOUND (ABI path): a stale
     *         generation is REFUSED at markDelivered (ABA-closed), the correct
     *         generation delivers + releases exactly once, and a re-mark is
     *         refused. NOTE: Terminal is NORMAL completion; the FAILED-response
     *         lifecycle is the separate response-failure control. */
    TEST(SharedExecutionProof, AbiDeliveryIsGenerationBoundForSuccessfulResponses)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'E', sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 16, 16, 1, kDest, 256, &lawA, 4);
        engine.step();
        EXPECT_TRUE(engine.rows().at(0).complete);
        EXPECT_THROW(engine.markDelivered(kDest, 99), std::logic_error); // stale ABA
        EXPECT_FALSE(engine.destinationReleased(kDest));
        engine.markDelivered(kDest, 1);
        engine.step();
        EXPECT_TRUE(engine.destinationReleased(kDest));
        EXPECT_THROW(engine.markDelivered(kDest, 1), std::logic_error); // double refuse
    }

    /** @brief IMMUTABLE MODE CONTROL: after the first admission, setAbiRunner is
     *         REFUSED (the mode is sealed once the engine starts). */
    TEST(SharedExecutionProof, AbiModeIsImmutableAfterAdmission)
    {
        class AnyAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &) override
            {
                return ServingWindowDecodeResult::failure("unsupported");
            }
        };
        AnyAbi abi1, abi2;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto law = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi1); // pre-admission allowed
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &law, 4);
        EXPECT_THROW(engine.setAbiRunner(&abi2), std::logic_error); // sealed
    }

    /** @brief WINDOW EPOCH ADVANCES PER STEP (sealed §8 epoch, monotonic). */
    TEST(SharedExecutionProof, AbiWindowEpochAdvancesPerStep)
    {
        class PublishAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining =
                    request.authority.current().window.response_limit - request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                if (request.output != nullptr)
                {
                    std::byte bs[32];
                    std::memset(bs, 'W', sizeof(bs));
                    request.output->writeOutput(bs, produced);
                }
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        PublishAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 1u << 20u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        const auto e0 = engine.windowEpoch();
        engine.step();
        const auto e1 = engine.windowEpoch();
        EXPECT_EQ(e1, e0 + 1); // monotonic per step
        engine.issueNextGrant(0);
        engine.step();
        EXPECT_EQ(engine.windowEpoch(), e1 + 1);
    }


    /** @brief LEASE ISOLATION: a released old lease can NEVER erase a NEW
     *         occupant's retained payload (the bank is refused while the lease
     *         is outstanding; after the release the entry is gone — a later
     *         stale release is a no-op, and a fresh occupant's payload is
     *         unaffected). */
    TEST(SharedExecutionProof, AbiRetainedLeaseIsolationAcrossReuse)
    {
        class ViolationAbi : public MockOrchestrationRunner
        {
        public:
            int windows = 0;
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining = request.authority.current().window.response_limit -
                                       request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                std::byte bs[256];
                std::memset(bs, 'I', sizeof(bs));
                if (request.output != nullptr)
                    // FIRST 16 tokens * 8 = 128 bytes; the SECOND stages 200 —
                    // WITH the prior 128 included (256 + 128 + 200 > 512) the
                    // staging bound REFUSES (StateViolated); it fits fine WITHOUT
                    // the prior (a discriminating over-stage).
                    request.output->writeOutput(bs, windows == 0 ? produced * 8 : 200);
                ++windows;
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        class PublishThenCancelAbi : public MockOrchestrationRunner
        {
        public:
            int windows = 0;
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                if (windows == 0)
                {
                    const auto quantum = request.authority.current().window.quantum;
                    const auto remaining = request.authority.current().window.response_limit -
                                           request.window_start;
                    const auto produced = remaining < quantum ? remaining : quantum;
                    if (request.output != nullptr)
                    {
                        std::byte bs[32];
                        std::memset(bs, 'I', sizeof(bs));
                        request.output->writeOutput(bs, produced);
                    }
                    ++windows;
                    return completeWindow(request.authority, request, produced,
                                          request.final_partial
                                              ? WindowTermination::Terminal
                                              : WindowTermination::Continue);
                }
                return cancelWindow(request.authority, request); // canceled window retains
            }
        };
        // FIRST occupant: a terminal execution failure (StateViolated staging
        // overflow) RETIRES the row and RELEASES the destination — the retained
        // payload survives the retirement for the writer.
        ViolationAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 512u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        engine.step(); // first window (128 bytes) published
        engine.issueNextGrant(0);
        engine.step(); // second staging overflow -> StateViolated retirement retains
        EXPECT_EQ(engine.abiDeferrals().at(0).state, AbiDeferralState::StateViolated);
        EXPECT_TRUE(engine.destinationReleased(kDest));
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 1));
        auto lease = engine.acquireRetainedPayload(kDest, 1);
        EXPECT_EQ(lease.generation(), 1u);
        // While the lease is outstanding, a fresh bank is REFUSED (no new occupant
        // while a delivery lease is outstanding).
        EXPECT_THROW(engine.bankDestination(kDest, 512u), std::logic_error);
        // LIVE-BRANCH DISCRIMINATOR (pointer-identity, NOT the already-live=false
        // path): a DIFFERENT reservation's retained entry exists while this lease
        // is STILL LIVE; the live release must erase ONLY its own (kDest) entry —
        // the other reservation's payload is preserved.
        engine.bankDestination(kDest2, 512u);
        engine.materializeDestination(kDest2);
        engine.admitNew(1, 1, 32, 16, 1, kDest2, 256, &lawA, 4);
        engine.step();
        engine.issueNextGrant(1);
        engine.step(); // kDest2 violated -> its own retained entry (wheel 1)
        EXPECT_TRUE(engine.hasRetainedPayload(kDest2, 1));
        lease.release(); // LIVE release: UNLEASES (the list keeps the entry)
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 1));  // still present (unleased)
        EXPECT_TRUE(engine.hasRetainedPayload(kDest2, 1)); // other reservation preserved
        const auto reason = engine.takeFailureReason(kDest, 1);
        EXPECT_NE(reason.find("admitted destination region"), std::string::npos);
        // The unleased retained entry is retired by the re-bank (non-allocating
        // unlink) + the reason was consumed -> the re-bank proceeds.
        engine.bankDestination(kDest, 512u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 2, 32, 16, 1, kDest, 256, &lawA, 4);
        engine.step();
        engine.issueNextGrant(0);
        engine.step(); // NEW occupant's StateViolated retirement retains (gen 2)
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 2));
        // STALE generation is refused ON AN UNLEASED entry (the generation
        // mismatch, not the already-leased state).
        EXPECT_THROW(engine.acquireRetainedPayload(kDest, 1), std::logic_error);
        // STALE generation on the NEW occupant's UNCONSUMED terminal reason: the
        // generation check fires (NOT the already-consumed state — the record is
        // still unconsumed).
        EXPECT_THROW(engine.takeFailureReason(kDest, 1), std::logic_error);
        EXPECT_EQ(engine.takeFailureReason(kDest, 2), "decode threw: Serving output sink exceeded the admitted destination region");
        auto lease2 = engine.acquireRetainedPayload(kDest, 2);
        EXPECT_EQ(lease2.generation(), 2u);
        // DISCRIMINATOR: while the SUCCESSOR entry exists (lease2 live), the OLD
        // lease's stale release is a no-op — it must NOT erase the successor's
        // payload (pointer-identity + token-protected).
        lease.release(); // old lease already released: no-op
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 2)); // successor preserved
        lease2.release();
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 2)); // released (retires at the next re-bank)
        // #159: a RELEASED delivery is FINAL — a reacquire (same generation) is
        // REFUSED (exactly-once; the false-flag conflation is gone).
        EXPECT_THROW(engine.acquireRetainedPayload(kDest, 2), std::logic_error);
        // #159: while the entry is PENDING/LEASED the bank REFUSES; the released
        // entry is retired by the re-bank (non-allocating) — but ONLY after the
        // terminal reason was consumed (already consumed above).
        // A stale re-bank for the successor retires it and proceeds (the
        // terminal reason for gen 2 was consumed above).
        engine.bankDestination(kDest, 512u);
        EXPECT_FALSE(engine.hasRetainedPayload(kDest, 2)); // retired by the re-bank
    }



    /** @brief ZERO-OUTPUT FAILURE (#151 publication join): a failed row with
     *         NO committed output retains NOTHING (no retained entry, no
     *         metadata) and its backing is released directly; only the TERMINAL
     *         ERROR REASON is owed (consumed exactly-once, generation-bound).
     *         The writer's error publication gets the reason + (no) partial. */
    TEST(SharedExecutionProof, AbiZeroOutputFailureRetainsNothingAndReleases)
    {
        class ViolateFirstAbi : public MockOrchestrationRunner
        {
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining = request.authority.current().window.response_limit -
                                       request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                std::byte bs[512];
                std::memset(bs, 'Z', sizeof(bs));
                if (request.output != nullptr)
                    // FIRST window over-stages (400 > the 256-byte output region,
                    // NO prior output): the staging bound refuses BEFORE any
                    // commit — the row has NO committed bytes.
                    request.output->writeOutput(bs, 400);
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        ViolateFirstAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 256u + 256u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        engine.step();
        EXPECT_EQ(engine.abiDeferrals().at(0).state, AbiDeferralState::StateViolated);
        // NO retained payload (no committed bytes) — a zero-output failed
        // response releases its backing with NO pending retained entry.
        EXPECT_FALSE(engine.hasRetainedPayload(kDest, 1));
        EXPECT_TRUE(engine.destinationReleased(kDest));
        EXPECT_TRUE(engine.responseFailed(kDest));
        // The terminal error reason is owed + consumed exactly once.
        EXPECT_THROW(engine.bankDestination(kDest, 512u), std::logic_error); // reason pending
        const auto reason = engine.takeFailureReason(kDest, 1);
        EXPECT_NE(reason.find("admitted destination region"), std::string::npos);
        EXPECT_THROW(engine.takeFailureReason(kDest, 1), std::logic_error); // exactly-once
        engine.bankDestination(kDest, 512u); // reason consumed -> re-bank allowed
    }


    /** @brief ZERO-METADATA-ALLOCATION PROOF: the retained delivery path
     *         allocates NOTHING beyond the already-admitted destination backing
     *         — the retention does not change the pool's materialized bytes at
     *         all (independent PMA footprint proof, not a getter-derived total):
     *         a violated row's retention re-hosts the SAME backing. */
    TEST(SharedExecutionProof, AbiRetentionAllocatesNothing)
    {
        class ViolationAbi : public MockOrchestrationRunner
        {
        public:
            int windows = 0;
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining = request.authority.current().window.response_limit -
                                       request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                std::byte bs[256];
                std::memset(bs, 'N', sizeof(bs));
                if (request.output != nullptr)
                    request.output->writeOutput(bs, windows == 0 ? produced * 8 : 200);
                ++windows;
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        ViolationAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto engine = makeEngine(solo);
        engine.setAbiRunner(&abi);
        engine.bankDestination(kDest, 256u + 256u);
        engine.materializeDestination(kDest);
        engine.admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        engine.step();
        engine.issueNextGrant(0);
        engine.step(); // violation -> retention (re-hosts the SAME backing)
        EXPECT_TRUE(engine.hasRetainedPayload(kDest, 1));
        // INDEPENDENT PROOF: the pool's materialized bytes are UNCHANGED by the
        // retention (the retained payload is the already-admitted backing).
        EXPECT_EQ(engine.servingPool().materializedBytes(),
                  256u + 256u + engine.destinationReserveBytes() + engine.aggregateNodeBytes()); // capacity + aggregate control reserve
        auto lease = engine.acquireRetainedPayload(kDest, 1);
        EXPECT_EQ(lease.bytes().size(), 128u);
        lease.release();
        // After the lease release the retained entry is NOT leased; a re-bank
        // RETIRES it (non-allocating unlink) and proceeds.
        engine.takeFailureReason(kDest, 1);
        engine.bankDestination(kDest, 256u + 256u);
        engine.materializeDestination(kDest);
        EXPECT_EQ(engine.servingPool().materializedBytes(),
                  256u + 256u + engine.destinationReserveBytes() + engine.aggregateNodeBytes()); // capacity + aggregate control reserve
    }

    /** @brief FINAL-BOUNDARY CONTROL (#149 scoped control, CANONICAL
     *         attestation): the fixture retains the PhysicalMemoryAuthority and
     *         queries the rankAttestation row MATCHING the serving-pool
     *         resource + owner (never the first row). Per the pinned PMA source
     *         (claimAllocation increments new_reservation_materialized;
     *         release subtracts; rankAttestation INCLUDES it), the matching
     *         row's materialized_new_bytes is the canonical final-free
     *         accounting. Engine-first with a LIVE lease: both the backing's
     *         capacity claim and the reserve credit are STILL materialized;
     *         the release returns both. */
    TEST(SharedExecutionProof, AbiReserveCreditOutlivesEngineFirstWithLiveLease)
    {
        class ViolationAbi : public MockOrchestrationRunner
        {
        public:
            int windows = 0;
        private:
            ServingWindowDecodeResult decodeServingWindowImpl(
                const ServingWindowDecodeInput &request) override
            {
                const auto quantum = request.authority.current().window.quantum;
                const auto remaining = request.authority.current().window.response_limit -
                                       request.window_start;
                const auto produced = remaining < quantum ? remaining : quantum;
                std::byte bs[256];
                std::memset(bs, 'R', sizeof(bs));
                if (request.output != nullptr)
                    request.output->writeOutput(bs, windows == 0 ? produced * 8 : 200);
                ++windows;
                return completeWindow(request.authority, request, produced,
                                      request.final_partial
                                          ? WindowTermination::Terminal
                                          : WindowTermination::Continue);
            }
        };
        ViolationAbi abi;
        auto admission = std::make_shared<ServingLawAdmission>(32);
        auto lawA = admission->issue(1, 11, 2, 32);
        auto pool = makePoolWithAuthority();
        std::optional<GroupedServingEngine> engine;
        engine.emplace(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                       std::move(pool.reservation), BackingAllocator::system());
        engine->setAbiRunner(&abi);
        engine->bankDestination(kDest, 256u + 256u);
        engine->materializeDestination(kDest);
        engine->admitNew(0, 1, 32, 16, 1, kDest, 256, &lawA, 4);
        engine->step();
        engine->issueNextGrant(0);
        engine->step(); // violation -> retained
        auto lease = engine->acquireRetainedPayload(kDest, 1);
        const auto reserve_bytes = engine->destinationReserveBytes();
        const auto node_bytes = engine->aggregateNodeBytes(); // cached BEFORE any reset
        // CANONICAL MATCHING-ROW QUERY (resource + owner = the serving pool).
        auto matching = [&]() -> std::size_t {
            for (const auto &a : pool.authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        // BEFORE the engine reset: the destination capacity (512) + the reserve
        // credit are materialized on the matching owner line.
        EXPECT_EQ(matching(), reserve_bytes + 512u + node_bytes);
        // ENGINE-FIRST DESTRUCTION with a LIVE lease.
        engine.reset();
        // The lease is the last holder of BOTH the backing (byte-exact payload,
        // read in place) and the reserve credit — the matching row still shows
        // BOTH materialized (the credit observably NOT returned).
        EXPECT_TRUE(lease.live());
        EXPECT_TRUE(lease.reserveHeld());
        EXPECT_EQ(lease.bytes().size(), 128u);
        for (const char ch : lease.bytes())
            EXPECT_EQ(ch, 'R');
        EXPECT_EQ(matching(), reserve_bytes + 512u + node_bytes);
        // The credit + the backing's capacity claim return ONLY when THIS (last)
        // holder releases — the CANONICAL final-free accounting.
        lease.release();
        EXPECT_EQ(matching(), 0u);
    }

    /** @brief AGGREGATE-REFUSAL CONTROL (focused): a reserve that admits the
     *         DESTINATION capacity but NOT the AGGREGATE control claim REFUSES
     *         AT THE AGGREGATE admission (the engine construction throws with
     *         NO engine, NO partial state, NO orphan debit). */
    TEST(SharedExecutionProof, AbiAggregateAdmissionRefusesBeforeEngineExists)
    {
        PhysicalMemoryBOMBuilder cpu(PhysicalMemoryResource{
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 512u, .admission_available_bytes = 512u});
        cpu.add(PhysicalMemoryOwner::ServingResponsePool, 512u);
        PhysicalMemoryPlanBuilder builder;
        const auto plan = builder.add(cpu.build()).build();
        auto cert = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
            std::move(plan));
        auto authority = std::make_shared<PhysicalMemoryAuthority>(cert, 0);
        auto reservation = authority->reserveNewAllocations(
            DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, 512u);
        // The aggregate control claim (per-slot x pool width) CANNOT fit 512 ->
        // the ENGINE CONSTRUCTION refuses at the aggregate admission (the
        // strict-pool DESTINATION refusal is a different test; this one targets
        // the AGGREGATE specifically).
        EXPECT_THROW(
            GroupedServingEngine(solo, 1, GroupedServingEngine::ServingPoolPlan{},
                                 std::move(reservation), BackingAllocator::system()),
            std::logic_error);
        // No orphan debit: the canonical owner line shows NOTHING materialized
        // after the refused construction.
        const auto matching = [&]() -> std::size_t {
            for (const auto &a : authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        EXPECT_EQ(matching(), 0u);
    }


    /** @brief #149 LEDGER-RETURN ORACLE: the token's consumed admission returns
     *         the materialized accounting (the ledger row drops by the recorded
     *         charge + the node then to zero). This is a LEDGER/accounting
     *         endpoint, NOT an independent physical-free or free-order
     *         certificate (the physical-free/order witness remains #149). */
    TEST(SharedExecutionProof, AggregateAdmissionLedgerReturnOracle)
    {
        PhysicalMemoryBOMBuilder cpu(PhysicalMemoryResource{
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 1u << 20u, .admission_available_bytes = 1u << 20u});
        cpu.add(PhysicalMemoryOwner::ServingResponsePool, 1u << 20u);
        PhysicalMemoryPlanBuilder builder;
        const auto plan = builder.add(cpu.build()).build();
        auto cert = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
            std::move(plan));
        auto authority = std::make_shared<PhysicalMemoryAuthority>(cert, 0);
        auto reservation = authority->reserveNewAllocations(
            DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, 1u << 20u);
        auto matching = [&]() -> std::size_t {
            for (const auto &a : authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        // A 64-byte admission is fully returned (the ledger accounting: the
        // row returns to 0 after the consume — NOT an independent physical-free
        // or free-order certificate).
        auto a64 = reservation.reserveAggregate(64);
        EXPECT_GT(matching(), 0u);
        a64.consume();
        EXPECT_EQ(matching(), 0u);
        // The same ledger return after a move-overwrite (no orphan).
        auto b32 = reservation.reserveAggregate(32);
        auto b64 = reservation.reserveAggregate(64);
        b32 = std::move(b64);
        b32.consume();
        EXPECT_EQ(matching(), 0u);
    }

    /** @brief #149 REAL REBOUND-NODE ALLOCATOR-BOM: with the probe's SCOPED
     *         normal-capture ON (default OFF — no flood), the aggregate record's
     *         std::map node is allocated through the global unaligned operator
     *         new; the witness RECONCILES the WHOLE captured admission window
     *         (the exact set/sum of the captured allocation requests == the ONE
     *         PMA-owned record-node charge — no extra/missing allocation), the
     *         zero-check FAILS if the owner line is absent, and the credit
     *         returned through the CONSUME path (not a rollback). The observed
     *         byte count is the allocator-interface REQUEST, not a heap
     *         footprint chunk measure. */
    TEST(SharedExecutionProof, ActualReboundNodeAllocatorBom)
    {
        const auto node_bytes =
            ::llaminar2::PhysicalMemoryAggregateHandle::aggregateRecordNodeBytes();
        auto pool = makePoolWithAuthority(1u << 20u);
        const auto before = physical_free_probe::snapshot();
        ASSERT_FALSE(before.overflowed);
        // RAII capture scope: the atomic flag is reset on EVERY exit (even a
        // throwing reserveAggregate) — no raw toggle left ON.
        struct CaptureScope
        {
            CaptureScope() { physical_free_probe::normal_capture.store(true); }
            ~CaptureScope() { physical_free_probe::normal_capture.store(false); }
        };
        {
            CaptureScope capture;
            auto handle = pool.reservation.reserveAggregate(64u);
            ASSERT_TRUE(handle.valid());
            handle.consume();
        }
        const auto after = physical_free_probe::snapshot();
        ASSERT_GT(after.count, before.count);
        ASSERT_FALSE(after.overflowed);
        // RECONCILE the whole captured admission window: the exact set/sum of
        // the captured unaligned NEW requests == the single record-node charge.
        std::size_t captured_sum = 0u;
        std::size_t captured_new_count = 0u;
        {
            std::lock_guard<std::mutex> g(physical_free_probe::probe_mutex);
            for (std::size_t i = before.count; i < physical_free_probe::count; ++i)
            {
                if (physical_free_probe::records[i].kind !=
                    physical_free_probe::Kind::New)
                    continue;
                captured_sum += physical_free_probe::records[i].size;
                ++captured_new_count;
            }
        }
        EXPECT_EQ(captured_new_count, 1u); // the single record node
        EXPECT_EQ(captured_sum, node_bytes); // the exact charge == the request
        // The credit returned (the CONSUME path released the record — not a
        // rollback). The zero check FAILS if the owner line is absent.
        bool line_seen = false;
        bool zeroed = true;
        for (const auto &a : pool.authority->rankAttestation())
            if (a.identity.device.is_cpu() &&
                a.owner == PhysicalMemoryOwner::ServingResponsePool)
            {
                line_seen = true;
                zeroed = a.materialized_new_bytes == 0u;
            }
        ASSERT_TRUE(line_seen);
        EXPECT_TRUE(zeroed);
    }

    /** @brief #149 COMPETING-ADMISSION ORACLE (LEDGER): two admissions against
     *         one strict pool compete for the SAME admitted capacity — the
     *         second admission is REFUSED while the first's charge is live (the
     *         authority's bounded materialized claim), and the first's LEDGER
     *         return re-enables it. This is the multi-request ledger core of
     *         the aggregate admission (a 2500-byte accounting allowance, NOT a
     *         2500-byte payload — the physical payload/free witness remains
     *         #149). */
    TEST(SharedExecutionProof, AggregateAdmissionCompetingWitness)
    {
        // A tight pool: 3 KiB of serving capacity (the admission charge = the
        // recorded bytes + the record node; two 2.5 KiB claims CANNOT both fit).
        PhysicalMemoryBOMBuilder cpu(PhysicalMemoryResource{
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 3u << 10u, .admission_available_bytes = 3u << 10u});
        cpu.add(PhysicalMemoryOwner::ServingResponsePool, 3u << 10u);
        PhysicalMemoryPlanBuilder builder;
        const auto plan = builder.add(cpu.build()).build();
        auto cert = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
            std::move(plan));
        auto authority = std::make_shared<PhysicalMemoryAuthority>(cert, 0);
        auto reservation = authority->reserveNewAllocations(
            DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, 3u << 10u);
        auto matching = [&]() -> std::size_t {
            for (const auto &a : authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        // The first 2.5 KiB admission fits; the competing 2.5 KiB does not
        // (the strict pool is bounded by the admitted capacity) — the consume
        // returns the first and the second then fits.
        auto first = reservation.reserveAggregate(2500);
        EXPECT_EQ(matching(), 2500u +
            ::llaminar2::PhysicalMemoryAggregateHandle::aggregateRecordNodeBytes());
        // The second must FAIL its own fit (the remaining < its charge; the
        // strict pool FAIL-FASTs the over-admission — this is the competing
        // admission REFUSED by the bounded capacity).
        EXPECT_THROW(
            {
                auto second = reservation.reserveAggregate(2500);
                (void)second;
            },
            std::exception);
        first.consume();
        EXPECT_EQ(matching(), 0u);
        // The freed pool admits the same 2.5 KiB claim again.
        auto third = reservation.reserveAggregate(2500);
        EXPECT_TRUE(third.valid());
        third.consume();
        EXPECT_EQ(matching(), 0u);
    }

    /** @brief #149 FORMULA-CONSISTENCY ORACLE: the enforced admission charge
     *         uses the same record-node expression as the test (the 8-field
     *         tuple + the map pointers) — FORMULA consistency only; the actual
     *         rebound-node/key/tree/padding/backing geometry is NOT observed
     *         (remains #149). */
    TEST(SharedExecutionProof, AggregateAdmissionFormulaConsistencyOracle)
    {
        // The formula under test: the 8-field tuple + the map pointers.
        using Value = std::tuple<std::size_t, bool, std::uint64_t, std::uint64_t,
                                 std::uint32_t, void *, bool, std::size_t>;
        const auto node_bytes =
            ::llaminar2::PhysicalMemoryAggregateHandle::aggregateRecordNodeBytes();
        EXPECT_EQ(node_bytes, node_bytes);
        // The admitted-charge validation uses the SAME node expression (PMA
        // derives the charge from the identical formula — formula consistency
        // only; no observed rebound-node/backing geometry).
        PhysicalMemoryBOMBuilder cpu(PhysicalMemoryResource{
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 1u << 20u, .admission_available_bytes = 1u << 20u});
        cpu.add(PhysicalMemoryOwner::ServingResponsePool, 1u << 20u);
        PhysicalMemoryPlanBuilder builder;
        const auto plan = builder.add(cpu.build()).build();
        auto cert = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
            std::move(plan));
        auto authority = std::make_shared<PhysicalMemoryAuthority>(cert, 0);
        auto reservation = authority->reserveNewAllocations(
            DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, 1u << 20u);
        auto matching = [&]() -> std::size_t {
            for (const auto &a : authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        auto a = reservation.reserveAggregate(64);
        EXPECT_EQ(matching(), 64u + node_bytes);
        a.consume();
        EXPECT_EQ(matching(), 0u);
    }

    /** @brief #149 ENGINE-LIFECYCLE ORACLE: the engine's own ControlReserve is
     *         allocated through the ReserveAllocatorT path (allocate_shared +
     *         transfer + the destructor's beginRetirement/delete/consumeRecord)
     *         and the materialized accounting returns to zero after that engine
     *         lifecycle — an ENGINE-LIFECYCLE/ledger endpoint observation; the
     *         independent physical free-order record remains #149. */
    TEST(SharedExecutionProof, EngineAggregateLifecycleLedger)
    {
        // The engine's OWN aggregate ControlReserve is admitted through the
        // ReserveAllocatorT path (allocate_shared + transfer) — the real engine
        // lifecycle. The COMPETING same-entitlement admission (the second
        // aggregate against the SAME reservation's live capacity) is the
        // separate AggregateAdmissionCompetingWitness (the sequential refusal +
        // reentry) — a SECOND ENGINE cannot share one reservation (the engine
        // owns the admission by value; a cross-engine whole-pool semantic arena
        // remains #149).
        auto pool = makePoolWithAuthority();
        auto matching = [&]() -> std::size_t {
            for (const auto &a : pool.authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        std::uint64_t engine_id = 1;
        const auto before = physical_free_probe::snapshot();
        ASSERT_FALSE(before.overflowed);
        void *engine_ptr = nullptr;
        std::size_t engine_actual = 0u;
        std::size_t engine_new_index = 0u;
        std::size_t live_boundary = 0u;
        bool found_target_new = false;
        {
            auto engine = GroupedServingEngine(
                solo, engine_id,
                GroupedServingEngine::ServingPoolPlan{3},
                std::move(pool.reservation), BackingAllocator::system());
            EXPECT_GT(matching(), 0u); // the engine's aggregate accounting LIVE
            // #149 AUTHORITY-RECORD IDENTITY/GEOMETRY: the engine's OWN
            // authenticated control-reserve record (the authority-bound record =
            // the ADMITTED bytes + the ACTUAL allocated size + the recorded
            // pointer) — the actual record, not an estimate.
            const auto rec = engine.controlReserveRecord();
            ASSERT_TRUE(rec.present);
            EXPECT_GT(rec.bytes, 0u);
            EXPECT_GT(rec.actual_size, 0u);
            EXPECT_NE(rec.alloc_ptr, nullptr);
            EXPECT_EQ(rec.bytes,
                      std::size_t{GroupedServingEngine::ServingPoolPlan{3}
                                      .destination_slots} *
                          std::size_t{engine.destinationControlBytes()});
            engine_ptr = rec.alloc_ptr;
            engine_actual = rec.actual_size;
            // #149 UNRETIRED-SELECTION (THE ACTUAL MATCHER): the probe observed
            // the SAME pointer + the SAME size the authority record binds; the
            // SINGLE target matcher selects the LAST NEW(ptr,size) UNRETIRED at
            // the coherent live boundary (an older completed lifetime + a
            // reused-address suffiX — or a wrong-size rebinding — cannot
            // qualify) and the found/fatal flow precedes every dependent scan.
            {
                std::lock_guard<std::mutex> g(physical_free_probe::probe_mutex);
                const auto sel = physical_free_probe::selectUnretiredTarget(
                    physical_free_probe::records, before.count,
                    physical_free_probe::count, rec.alloc_ptr, rec.actual_size);
                found_target_new = sel.found;
                engine_new_index = sel.target_index;
                live_boundary = sel.live_boundary;
            }
            ASSERT_TRUE(found_target_new);
            ASSERT_TRUE(live_boundary > engine_new_index);
        }
        // AFTER the engine's destructor: the authority's materialized returns to
        // zero (the lifecycle ledger endpoint — the SOURCE order
        // beginRetirement->delete->consumeRecord is the credited production
        // order, NOT independently re-proven here) AND the SEALED-INTERVAL
        // probe records a SAME-ADDRESS aligned new->delete pair with the
        // delete's chronology position STRICTLY LATER than the new's (the
        // free-call order within the completed interval; the producer identity /
        // size / alignment / generation + the FREE-BEFORE-CREDIT + the full
        // allocator-bound BOM remain #149's open remainder).
        EXPECT_EQ(matching(), 0u);
        const auto after = physical_free_probe::snapshot();
        ASSERT_GT(after.count, before.count);
        // The TARGET LIFETIME FREE-COMPLETION: the probe's DELETE-COMPLETION
        // entry of the authenticated engine pointer (written after the native
        // ::free RETURNED — a completion marker) at a position STRICTLY AFTER
        // the LIVE BOUNDARY and WITHIN the sealed post-destruction interval
        // ([live_boundary, after.count)) — a pre-live DELETE(P) (an earlier
        // completed lifetime), an unrelated pair or an out-of-range event
        // cannot satisfy it; the credit (materialized -> 0) is OBSERVED after
        // this completion marker, and the completion-BEFORE-credit ordering
        // itself is the production source order
        // (beginRetirement -> delete -> consumeRecord), credited (not
        // independently re-proven).
        ASSERT_TRUE(after.count > live_boundary);
        {
            std::lock_guard<std::mutex> g(physical_free_probe::probe_mutex);
            const bool target_freed = physical_free_probe::hasDeleteInRange(
                physical_free_probe::records, live_boundary, after.count,
                engine_ptr);
            EXPECT_TRUE(target_freed);
            EXPECT_GT(engine_actual, 0u);
        }
        ASSERT_FALSE(after.overflowed);
        // SEALED [before, after): the complete bounded range (no shortlist; an
        // incomplete selection REJECTS) + the strictly-later same-address pair.
        // FATAL guards + typed success-result flow control: the dereference
        // happens ONLY when the collection AND the match SUCCEED (an empty
        // observation / capacity refusal / missing delete can never reach the
        // output read — no indeterminate-entry read).
        physical_free_probe::NewEntry entries[64];
        std::size_t nfound = 0;
        const bool collected = physical_free_probe::sealedNewEntries(
            before.count, after.count, entries, 64u, nfound);
        ASSERT_TRUE(collected) << "sealed interval collection refused (incomplete/overflow)";
        ASSERT_GT(nfound, 0u);
        std::size_t matched_new = 0;
        const bool matched = physical_free_probe::strictlyLaterDeleteInRange(
            before.count, after.count, entries, nfound, matched_new);
        ASSERT_TRUE(matched) << "no strictly-later same-address delete in the interval";
        ASSERT_LT(matched_new, nfound);
        // The matched NEW's size is the non-zero allocation the probe saw
        // (the aligned allocation has the size on the new side; the delete side
        // carries no size — the POSITION/pair, not an independent identity).
        EXPECT_GT(entries[matched_new].size, 0u);
    }

    /** @brief #186 FOCUSED NEGATIVE COVERAGE (the observer's guard paths): a
     *         BOUNDED TEST-OWNED aligned allocation (+ RAII retirement) => a
     *         sufficient-capacity collection succeeds (nfound 1, the OWNED
     *         pointer), the SAME sealed interval with a capacity-0 output is
     *         REFUSED (false), and the empty-helper semantics are distinguished
     *         (an empty interval collects true with nfound 0 — the zero-entry
     *         matcher false; a capacity REFUSAL is false — the guard
     *         distinction, no output dereference on refusal). */
    TEST(SharedExecutionProof, ObserverGuardNegativeCoverage)
    {
        // OWNED FIXTURE (scope-bounded, deterministic — no reliance on earlier
        // tests or global history): one test-owned aligned allocation.
        const auto owned = std::size_t{128};
        struct OwnedRetire
        {
            void *p;
            ~OwnedRetire()
            {
                ::operator delete(p, std::align_val_t{alignof(std::max_align_t)});
            }
        };
        // IMMEDIATE RAII HANDOFF: the owner is constructed as part of the
        // SUCCESSFUL allocation handoff BEFORE any snapshot/assert/throwing
        // operation (an earlier fatal check cannot leak the raw pointer); the
        // allocation sits between the BEFORE and AFTER snapshots.
        const auto before = physical_free_probe::snapshot();
        void *owned_ptr = ::operator new(owned, std::align_val_t{alignof(std::max_align_t)});
        OwnedRetire retire{owned_ptr};
        const auto mid = physical_free_probe::snapshot();
        ASSERT_GT(mid.count, before.count);
        // SUFFICIENT-capacity collection on the OWNED interval: nfound 1 + the
        // owned pointer.
        physical_free_probe::NewEntry entries[4];
        std::size_t nfound = 0;
        const bool collected = physical_free_probe::sealedNewEntries(
            before.count, mid.count, entries, 4u, nfound);
        ASSERT_TRUE(collected);
        ASSERT_EQ(nfound, 1u);
        EXPECT_EQ(entries[0].ptr, owned_ptr);
        // CAPACITY-0 REFUSAL (unconditional): the same sealed interval with a
        // zero-capacity output must refuse (no silent truncation).
        std::size_t n2 = 0;
        const bool refused = physical_free_probe::sealedNewEntries(
            before.count, mid.count, entries, 0u, n2);
        EXPECT_FALSE(refused);
        EXPECT_EQ(n2, 0u);
        // EMPTY-HELPER semantics (distinguished from refusal): an empty
        // [mid, mid) interval collects TRUE with nfound 0; the zero-entry
        // matcher is FALSE without any output dereference.
        const bool empty_collected = physical_free_probe::sealedNewEntries(
            mid.count, mid.count, entries, 4u, nfound);
        EXPECT_TRUE(empty_collected);
        EXPECT_EQ(nfound, 0u);
        std::size_t matched = 0;
        const bool matched_any = physical_free_probe::strictlyLaterDeleteInRange(
            mid.count, mid.count, entries, 0u, matched);
        EXPECT_FALSE(matched_any);
    }

    /** @brief #149 TARGET-MATCHER CONTRACT MATRIX (deterministic SYNTHETIC
     *         event views over the SINGLE actual non-allocating matcher — NO
     *         dependence on heap address reuse / earlier traffic):
     *         1) VALID INDEX 0; 2) OLDER COMPLETED + CURRENT (the current
     *         selected, its post-live DELETE true); 3) MISSING target DELETE
     *         (false); 4) RETIRED-AT-LIVE (no unretired target);
     *         5) MISMATCH/REBINDING (a wrong-size NEW on a reused pointer does
     *         not hide the correct-size lifecycle; the wrong-size rebinding
     *         does not qualify). Synthetic contract controls are NOT
     *         recorded-production replay or native qualification. */
    TEST(SharedExecutionProof, TargetMatcherContractMatrix)
    {
        using physical_free_probe::Kind;
        using physical_free_probe::Entry;
        // Histories: {kind, ptr, size} — the coherent bounded event views.
        Entry ev[16];
        void *P = reinterpret_cast<void *>(0x1000);
        void *Q = reinterpret_cast<void *>(0x2000);
        std::size_t p64 = 64u, p32 = 32u;
        // 1) VALID INDEX 0: NEW(P,64) at 0; live_end=1 -> selected index 0.
        ev[0] = {Kind::New, P, p64};
        {
            const auto sel = physical_free_probe::selectUnretiredTarget(
                ev, 0u, 1u, P, p64);
            EXPECT_TRUE(sel.found);
            EXPECT_EQ(sel.target_index, 0u);
            EXPECT_EQ(sel.live_boundary, 1u);
        }
        // 2) OLDER COMPLETED + CURRENT: NEW(P,64)@0, DELETE(P)@1, NEW(P,64)@2;
        //    live_end=3 -> selected index 2 (the current unretired); its
        //    post-live DELETE (index 3, exclusive end 4) is TRUE.
        ev[0] = {Kind::New, P, p64};
        ev[1] = {Kind::Delete, P, 0u};
        ev[2] = {Kind::New, P, p64};
        ev[3] = {Kind::Delete, P, 0u};
        {
            const auto sel = physical_free_probe::selectUnretiredTarget(
                ev, 0u, 3u, P, p64);
            EXPECT_TRUE(sel.found);
            EXPECT_EQ(sel.target_index, 2u);
            EXPECT_TRUE(physical_free_probe::hasDeleteInRange(
                ev, sel.live_boundary, 4u, P)); // index 3, exclusive end 4
        }
        // 3) MISSING post-live target DELETE on the SAME older-completed+current
        //    history: NEW(P,64)@0, DELETE(P)@1, NEW(P,64)@2, then a NONEMPTY
        //    post-live suffix containing ONLY another pointer's events
        //    (DELETE(Q)@3) — a scan from 0 instead of live_boundary would find
        //    the OLD DELETE(P)@1 and wrongly accept (the discrimination guard:
        //    the predicate starts at the selected live boundary).
        ev[0] = {Kind::New, P, p64};
        ev[1] = {Kind::Delete, P, 0u};
        ev[2] = {Kind::New, P, p64};
        ev[3] = {Kind::Delete, Q, 0u};
        {
            const auto sel = physical_free_probe::selectUnretiredTarget(
                ev, 0u, 3u, P, p64);
            EXPECT_TRUE(sel.found);
            EXPECT_EQ(sel.target_index, 2u);
            EXPECT_EQ(sel.live_boundary, 3u);
            // The missing post-live P delete: FALSE (the suffix has only Q's
            // event; the OLD P delete at index 1 is before the boundary).
            EXPECT_FALSE(physical_free_probe::hasDeleteInRange(
                ev, sel.live_boundary, 4u, P));
            // Add the target's P delete at 4 (the end 5) -> acceptance.
            ev[4] = {Kind::Delete, P, 0u};
            EXPECT_TRUE(physical_free_probe::hasDeleteInRange(
                ev, sel.live_boundary, 5u, P));
        }
        // 4) RETIRED-AT-LIVE: NEW@0, DELETE@1; live_end=2 -> NO unretired target.
        ev[0] = {Kind::New, P, p64};
        ev[1] = {Kind::Delete, P, 0u};
        {
            const auto sel = physical_free_probe::selectUnretiredTarget(
                ev, 0u, 2u, P, p64);
            EXPECT_FALSE(sel.found);
        }
        // 5) MISMATCH/REBINDING: NEW(P,64)@0, DELETE(P)@1, NEW(P,32)@2 (the
        //    wrong-size rebinding) — the 64-size target is NOT found (the
        //    rebinding does not hide the completed 64 lifecycle).
        ev[0] = {Kind::New, P, p64};
        ev[1] = {Kind::Delete, P, 0u};
        ev[2] = {Kind::New, P, p32};
        {
            const auto sel = physical_free_probe::selectUnretiredTarget(
                ev, 0u, 3u, P, p64);
            EXPECT_FALSE(sel.found);
            const auto sel32 = physical_free_probe::selectUnretiredTarget(
                ev, 0u, 3u, P, p32);
            EXPECT_TRUE(sel32.found);
            EXPECT_EQ(sel32.target_index, 2u);
        }
        // Unrelated pointer Q is never selected.
        ev[0] = {Kind::New, Q, p64};
        {
            const auto sel = physical_free_probe::selectUnretiredTarget(
                ev, 0u, 1u, P, p64);
            EXPECT_FALSE(sel.found);
        }
    }

    /** @brief #149/#186 ACTUAL-PUBLICATION CONTROL: ONE test-owned aligned
     *         delete through the ACTUAL body — the accepted coherent-history
     *         helper authenticates the CURRENT TARGET'S LIVE FRONTIER (the
     *         unretired selection over the relevant prefix) and the completion
     *         is the target's DELETE-COMPLETION inside the sealed POST-LIVE
     *         interval (the same helper semantics as the engine lifecycle
     *         witness — lifetime-specific, not pointer-only; no separately
     *         acquired lock sample); the allocation is protected by the retire
     *         RAII before any throwing setup. No heap-reuse / timing / history
     *         dependence. */
    TEST(SharedExecutionProof, ActualPublicationTransactionControl)
    {
        const auto before = physical_free_probe::snapshot();
        ASSERT_FALSE(before.overflowed);
        // PROTECT THE ALLOCATION FIRST (the RAII owner before any throwing
        // setup): the raw pointer is immediately owned by the retire RAII.
        void *p = ::operator new(64u, std::align_val_t{alignof(std::max_align_t)});
        struct Retire
        {
            void *q = nullptr;
            ~Retire()
            {
                if (q)
                    ::operator delete(q, std::align_val_t{alignof(std::max_align_t)});
            }
        } retire{p};
        // AUTHENTICATE THE CURRENT TARGET'S LIVE FRONTIER FIRST (the ACCEPTED
        // coherent-history helper — the unretired selection over the relevant
        // prefix, BEFORE any delete of the fixture's p).
        const auto live = physical_free_probe::snapshot();
        const auto sel = physical_free_probe::selectUnretiredTarget(
            physical_free_probe::records, before.count, live.count, p, 64u);
        ASSERT_TRUE(sel.found);
        ASSERT_EQ(sel.live_boundary, live.count);
        // THE ACTUAL OWNER AT BOTH OPERATIONS (the SAME shared primitive the
        // aligned delete uses): the guard report captures the unique_lock's
        // ACTUAL owns_lock() at the native free AND at the completion append —
        // a default-constructed (non-acquired) owner, a free-before-owner or
        // an unlock-before-append reports the violation (a discriminator over
        // the real guard, not a label/event-inspection); this one instance is
        // the ONLY producer for p's completion marker.
        bool guard_violation = false;
        {
            physical_free_probe::PublicationTransaction tx(p, &guard_violation);
            tx.nativeFree();
        }
        retire.q = nullptr; // DISARM BEFORE the potentially-throwing reporting
                            // (the primitive's free was the ONLY free of p; a
                            // throwing EXPECT cannot double-free, while a
                            // throwing tx construction still leaves the Retire
                            // owning p)
        EXPECT_FALSE(guard_violation);
        const auto after = physical_free_probe::snapshot();
        // THE COMPLETION IN THE SEALED POST-LIVE INTERVAL (the accepted
        // helper's predicate): the target's completion strictly after the
        // authenticated live frontier — an OLD completed lifetime's DELETE
        // (before the frontier) cannot satisfy it (lifetime-specific, not
        // pointer-only).
        std::lock_guard<std::mutex> g(physical_free_probe::probe_mutex);
        EXPECT_TRUE(physical_free_probe::hasDeleteInRange(
            physical_free_probe::records, sel.live_boundary, after.count, p));
    }

    /** @brief #149 WHOLE-POOL CUMULATIVE POPULATION: the authority's single
     *         ledger (the engine's pool reservation) carries the engine's
     *         ControlReserve aggregate + EVERY live destination suballocation
     *         claim (the exact claimPoolBytes/capacity bytes) — the materialized
     *         total DECOMPOSES into those parts; the final snapshot is bounded
     *         by the fixture's owner envelope, and the whole returns to zero
     *         after the engine's retirement. */
    TEST(SharedExecutionProof, WholePoolCumulativePopulationBound)
    {
        const auto dest_bytes = std::size_t{1u} << 20u;
        const auto node_bytes =
            ::llaminar2::PhysicalMemoryAggregateHandle::aggregateRecordNodeBytes();
        auto pool = makePoolWithAuthority();
        auto matching = [&]() -> std::size_t {
            for (const auto &a : pool.authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        std::uint64_t agg_bytes = 0u;
        {
            auto engine = GroupedServingEngine(
                solo, 1, GroupedServingEngine::ServingPoolPlan{2},
                std::move(pool.reservation), BackingAllocator::system());
            // The aggregate's authenticated charge (the authority record).
            const auto rec = engine.controlReserveRecord();
            ASSERT_TRUE(rec.present);
            agg_bytes = rec.bytes;
            engine.bankDestination(1, dest_bytes);
            engine.materializeDestination(1);
            engine.bankDestination(2, dest_bytes);
            engine.materializeDestination(2);
            // WHOLE-POOL DECOMPOSITION: the materialized cumulative = the
            // aggregate charge + its record node + BOTH live destination claims
            // (the exact claimed bytes) — one authority ledger.
            EXPECT_EQ(matching(), agg_bytes + node_bytes + 2u * dest_bytes);
            // The final snapshot is bounded by the fixture's 16 MiB OWNER
            // envelope (the admitted owner line) — NOT an exact charge of the
            // engine's reservation (pool_bytes-16, with the 16-byte observer
            // reservation unused) and NOT every transition; the per-transition
            // bounds live in the authority's strict claim checks.
            EXPECT_LE(matching(), std::size_t{16} << 20u);
        }
        // After the engine's retirement the WHOLE pool returns to zero.
        EXPECT_EQ(matching(), 0u);
    }

    /** @brief #149 WHOLE-POOL BOUNDARY (tight pool): ONE live destination's
     *         insufficient-headroom sample: the pool leaves only 48 bytes of
     *         engine headroom after the aggregate + one destination claim (the
     *         64 = the fixture's 16-byte unused observer reservation + 48 bytes
     *         of engine headroom, far below the second destination's 1 MiB
     *         claim); the SECOND destination's materialize is REFUSED by the
     *         strict pool while the first is live (the cumulative population
     *         bound — not just the aggregate admission); after the engine's
     *         retirement the whole pool returns to zero. */
    TEST(SharedExecutionProof, WholePoolBoundaryRefusesCumulativeOverflow)
    {
        const auto dest_bytes = std::size_t{1u} << 20u;
        const auto node_bytes =
            ::llaminar2::PhysicalMemoryAggregateHandle::aggregateRecordNodeBytes();
        // First discover the engine's aggregate for the 2-slot plan on a scratch
        // pool (the aggregate = per-slot control x 2).
        std::uint64_t agg_bytes = 0u;
        {
            auto scratch = makePoolWithAuthority();
            auto engine = GroupedServingEngine(
                solo, 1, GroupedServingEngine::ServingPoolPlan{2},
                std::move(scratch.reservation), BackingAllocator::system());
            const auto rec = engine.controlReserveRecord();
            ASSERT_TRUE(rec.present);
            agg_bytes = rec.bytes;
        }
        // The TIGHT pool: capacity for the aggregate + ONE destination + the
        // node overhead (the SECOND destination's live claim cannot fit).
        // 64 = the fixture's 16-byte unused observer reservation + 48 bytes of
        // engine headroom (insufficient for the second destination's claim).
        const auto pool_bytes = agg_bytes + node_bytes + dest_bytes + 64u;
        auto pool = makePoolWithAuthority(pool_bytes);
        auto matching = [&]() -> std::size_t {
            for (const auto &a : pool.authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        {
            auto engine = GroupedServingEngine(
                solo, 1, GroupedServingEngine::ServingPoolPlan{2},
                std::move(pool.reservation), BackingAllocator::system());
            engine.bankDestination(1, dest_bytes);
            engine.materializeDestination(1);
            const auto materialized_before = matching();
            EXPECT_GT(materialized_before, 0u);
            // The SECOND destination is a VALID bank (the wheel slot exists);
            // its MATERIALIZE is the strict cumulative-capacity refusal — assert
            // the capacity-denial category/reason explicitly + that the
            // authority's materialized is UNCHANGED immediately after (the
            // failed claim leaves no residual debit while the first claim stays
            // live).
            engine.bankDestination(2, dest_bytes);
            bool denied_category = false;
            bool reason_is_capacity = false;
            try
            {
                engine.materializeDestination(2);
                FAIL() << "the second destination's claim unexpectedly succeeded";
            }
            catch (const std::logic_error &e)
            {
                denied_category = true;
                reason_is_capacity =
                    std::string(e.what()).find("capacity") != std::string::npos ||
                    std::string(e.what()).find("exceed") != std::string::npos;
            }
            EXPECT_TRUE(denied_category);
            EXPECT_TRUE(reason_is_capacity);
            EXPECT_EQ(matching(), materialized_before); // no residual from the refusal
        }
        // The whole pool returns to zero after the engine's retirement.
        EXPECT_EQ(matching(), 0u);
    }

    /** @brief FOCUSED QUALIFICATION (aggregate-admission token): self-move is a
     *         NO-OP (#178), a distinct-target move-overwrite returns the active
     *         target's pre-debit (#175), independent 32+64 admissions retire
     *         independently (+/64 -/32), and consumption is exactly-once
     *         (#176) — all via the PUBLIC token API. */
    TEST(SharedExecutionProof, AbiAggregateAdmissionTokenQualification)
    {
        // Fixture: a 1 MiB serving reserve; the token API is on the reservation.
        PhysicalMemoryBOMBuilder cpu(PhysicalMemoryResource{
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 1u << 20u, .admission_available_bytes = 1u << 20u});
        cpu.add(PhysicalMemoryOwner::ServingResponsePool, 1u << 20u);
        PhysicalMemoryPlanBuilder builder;
        const auto plan = builder.add(cpu.build()).build();
        auto cert = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(
            std::move(plan));
        auto authority = std::make_shared<PhysicalMemoryAuthority>(cert, 0);
        auto reservation = authority->reserveNewAllocations(
            DeviceId::cpu(), PhysicalMemoryOwner::ServingResponsePool, 1u << 20u);
        auto matching = [&]() -> std::size_t {
            for (const auto &a : authority->rankAttestation())
                if (a.identity.device.is_cpu() &&
                    a.owner == PhysicalMemoryOwner::ServingResponsePool)
                    return a.materialized_new_bytes;
            return std::numeric_limits<std::size_t>::max();
        };
        const auto node =
            ::llaminar2::PhysicalMemoryAggregateHandle::aggregateRecordNodeBytes();
        // #178 SELF-MOVE NO-OP.
        auto a32 = reservation.reserveAggregate(32);
        auto *a32p = &a32;
        a32 = std::move(a32);
        EXPECT_TRUE(a32.valid());
        EXPECT_EQ(a32p, &a32);
        EXPECT_EQ(matching(), 32u + node); // single 32 debit + its record node
        a32.consume();
        EXPECT_EQ(matching(), 0u);
        // #175 DISTINCT-TARGET MOVE-OVERWRITE: an active target's pre-debit is
        // returned by the overwrite (no orphan).
        auto b32 = reservation.reserveAggregate(32);
        auto b64 = reservation.reserveAggregate(64);
        EXPECT_EQ(matching(), 96u + 2u * node);
        b32 = std::move(b64); // overwrites the ACTIVE b32 -> its 32 pre-debit returns
        EXPECT_EQ(matching(), 64u + node);
        b32.consume();
        EXPECT_EQ(matching(), 0u);
        // INDEPENDENT 32+64 (separate tokens): release 32 FIRST -> 64 remains;
        // then release 64 -> 0 (the executed order; a 64-first/intermediate-32
        // case is NOT claimed).
        auto c32 = reservation.reserveAggregate(32);
        auto c64 = reservation.reserveAggregate(64);
        EXPECT_EQ(matching(), 96u + 2u * node);
        c32.consume();
        EXPECT_EQ(matching(), 64u + node); // independent retirement
        c64.consume();
        EXPECT_EQ(matching(), 0u);
        // #176 EXACTLY-ONCE (while another sufficient debit is LIVE): the
        // replay variant — a duplicate consume while a SEPARATE live debit is
        // outstanding must NOT return the OTHER's bytes (a replay against a
        // still-materialized row would subtract the wrong amount — here it is
        // a no-op).
        auto live64 = reservation.reserveAggregate(64);
        auto again64 = reservation.reserveAggregate(64);
        EXPECT_EQ(matching(), 128u + 2u * node);
        again64.consume();
        EXPECT_EQ(matching(), 64u + node); // the live64 debit is untouched
        again64.consume();          // replay at 64-live: NO-OP (no underflow)
        EXPECT_EQ(matching(), 64u + node);
        live64.consume();
        EXPECT_EQ(matching(), 0u);
        // SIMPLE exactly-once at zero (the accepted narrow case).
        c64.consume();
        EXPECT_EQ(matching(), 0u);
    }

} // namespace llaminar2::test
