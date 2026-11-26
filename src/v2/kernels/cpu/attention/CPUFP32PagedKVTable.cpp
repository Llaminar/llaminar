/**
 * @file CPUFP32PagedKVTable.cpp
 * @brief Stable PMA table metadata and atomic nonempty CPU snapshot publication.
 */
#include "CPUFP32PagedKVTable.h"
#include <atomic>
#include <limits>
#include <new>
#include <utility>

namespace llaminar2
{
void CPUFP32PagedKVTableGeometry::validate() const
{
    if (!page_capacity || !bank_count || !readers_per_bank || page_tokens <= 0)
        throw std::invalid_argument("CPU page table requires positive exact metadata/concurrency geometry");
    if (bank_count > std::numeric_limits<size_t>::max() / page_capacity ||
        bank_count * page_capacity > std::numeric_limits<size_t>::max() / sizeof(CPUFP32KVPage))
        throw std::overflow_error("CPU page-table object storage overflows");
    if (!issuance_budget || issuance_headroom >= issuance_budget)
        throw std::invalid_argument("CPU page-table issuance budget must exceed its reserved headroom");
}
size_t CPUFP32PagedKVTable::physicalBytes(CPUFP32PagedKVTableGeometry geometry, DeviceId device, CPUStoragePlacement placement)
{
    geometry.validate();
    const auto cells = CPUStorageRoot<std::byte>(geometry.bank_count * geometry.page_capacity * sizeof(CPUFP32KVPage),
        PhysicalMemoryOwner::KVCache, device, placement).physicalBytes();
    const auto banks = CPUStorageRoot<Bank>(geometry.bank_count, PhysicalMemoryOwner::KVCache, device, placement).physicalBytes();
    // The effective-use journal is admitted metadata of THIS instance: one
    // slot per live incarnation, priced by the same authority so the instance
    // plan BOM stays exact (no hidden heap allocation outside PMA).
    const auto lease_cells = CPUStorageRoot<EffectiveUseJournal::LeaseCell>(
        geometry.bank_count * geometry.readers_per_bank,
        PhysicalMemoryOwner::KVCache, device, placement).physicalBytes();
    const auto replica_records = CPUStorageRoot<EffectiveUseJournal::ReplicaRecord>(
        geometry.bank_count,
        PhysicalMemoryOwner::KVCache, device, placement).physicalBytes();
    if (banks > std::numeric_limits<size_t>::max() - cells)
        throw std::overflow_error("CPU page-table physical BOM overflows");
    const auto cells_and_banks = cells + banks;
    if (lease_cells > std::numeric_limits<std::size_t>::max() - cells_and_banks)
        throw std::overflow_error("CPU page-table physical BOM overflows");
    const auto with_cells = cells_and_banks + lease_cells;
    if (replica_records > std::numeric_limits<std::size_t>::max() - with_cells)
        throw std::overflow_error("CPU page-table physical BOM overflows");
    return with_cells + replica_records;
}
std::shared_ptr<CPUFP32PagedKVTable> CPUFP32PagedKVTable::create(std::shared_ptr<PhysicalMemoryAuthority> authority,
    DeviceId device, CPUFP32PagedKVTableGeometry geometry, CPUStoragePlacement placement)
{
    if (!authority || !device.is_cpu()) throw std::invalid_argument("CPU page table requires its original PMA CPU allocator");
    return std::shared_ptr<CPUFP32PagedKVTable>(new CPUFP32PagedKVTable(std::move(authority), device, geometry, placement));
}
namespace
{
    /** @brief Process-wide atomic table incarnation counter (K1: unique, non-zero). */
    std::atomic<std::uint64_t> &nextTableIncarnation()
    {
        static std::atomic<std::uint64_t> counter{0};
        return counter;
    }
} // namespace

CPUFP32PagedKVTable::CPUFP32PagedKVTable(std::shared_ptr<PhysicalMemoryAuthority> authority, DeviceId device,
    CPUFP32PagedKVTableGeometry geometry, CPUStoragePlacement placement)
    : authority_(authority), geometry_(geometry), authority_identity_(
          nextTableIncarnation().fetch_add(1, std::memory_order_relaxed) + 1)
{
    (void)physicalBytes(geometry, device, placement);
    static_assert(alignof(CPUFP32KVPage) <= AlignedVector<std::byte>::ALIGNMENT);
    cells_ = CPUStorageRoot<std::byte>(geometry.bank_count * geometry.page_capacity * sizeof(CPUFP32KVPage),
        PhysicalMemoryOwner::KVCache, device, placement, CPUStorageInitialization::Uninitialized).allocate(authority);
    banks_ = CPUStorageRoot<Bank>(geometry.bank_count, PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
    // Lease cells (concurrency) and replica records (retained incarnations)
    // are both admitted by the same authority and priced in physicalBytes, so
    // no heap allocation exists outside PMA and the instance BOM is exact.
    use_cells_ = CPUStorageRoot<EffectiveUseJournal::LeaseCell>(
        geometry.bank_count * geometry.readers_per_bank,
        PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
    use_records_ = CPUStorageRoot<EffectiveUseJournal::ReplicaRecord>(
        geometry.bank_count,
        PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
    if (!effective_use_.bind(use_cells_, use_records_))
        throw std::logic_error("CPU page-table effective-use journal bind failed");
    // Placement array construction starts real object lifetimes in the exact
    // admitted root; inactive cells own no pages and allocate no payload.
    pages_ = new (cells_.data()) CPUFP32KVPage[geometry.bank_count * geometry.page_capacity];
}
CPUFP32PagedKVTable::~CPUFP32PagedKVTable()
{
    for (const auto &bank : banks_) if (bank.phase != Phase::Free) std::terminate();
    std::destroy_n(pages_, geometry_.bank_count * geometry_.page_capacity);
}
CPUFP32PagedKVTable::Bank &CPUFP32PagedKVTable::require(size_t bank, std::uint64_t version, Phase phase)
{
    if (bank >= banks_.size() || !version || banks_[bank].version != version || banks_[bank].phase != phase)
        throw std::logic_error("CPU page-table access has a foreign, stale or wrong-phase bank");
    return banks_[bank];
}
void CPUFP32PagedKVTable::clear(size_t bank) noexcept
{
    auto &state = banks_[bank];
    for (size_t page = 0; page < state.pages; ++page)
        pages_[bank * geometry_.page_capacity + page] = CPUFP32KVPage{};
    state.pages = state.readers = 0;
    state.accepting_readers = false;
    state.phase = Phase::Free;
    // M5: the request binding belongs to the publication and dies with it.
    // A bank returning to Free is entirely unbound; a reused bank can never
    // inherit another request's binding.
    state.request_id = 0;
    state.slot = -1;
    // The bank's publication retires; live touch LEASES remain owned by their
    // receipts (the snapshot keeps the exact identity) until each is
    // acknowledged. Clearing never drops an obligation (R10).
}
std::optional<CPUFP32PagedKVTable::Update> CPUFP32PagedKVTable::tryBeginUpdate()
{
    for (size_t bank = 0; bank < banks_.size(); ++bank)
    {
        auto &state = banks_[bank];
        if (state.phase != Phase::Free) continue;
        // FINITE issuance admission, not bank exhaustion: normal updates never
        // consume the reserved headroom reserved for accepted work. The finite
        // budget and the outstanding reservation capacity are both respected
        // so accepted work can always complete within its own grant.
        if (next_version_ >= geometry_.issuance_budget - geometry_.issuance_headroom)
            return std::nullopt;
        auto self = shared_from_this();
        state.version = ++next_version_;
        state.phase = Phase::Preparing;
        return Update(std::move(self), bank, state.version);
    }
    return std::nullopt;
}

std::optional<CPUFP32PagedKVTable::IssuanceReservation>
CPUFP32PagedKVTable::reserveAcceptedWork(std::size_t capacity)
{
    // Checked finite arithmetic with the COMPLETE reservation conservation
    // rule: issued versions (next_version_) PLUS every unconsumed outstanding
    // reservation PLUS the new capacity may never exceed the finite budget.
    // The headroom bound alone is insufficient (issued + outstanding can still
    // overbook the absolute budget after partial consumption).
    if (capacity > geometry_.issuance_headroom - reserved_outstanding_)
        return std::nullopt;
    if (reserved_outstanding_ > geometry_.issuance_budget - next_version_)
        return std::nullopt; // Invariant violation would already overbook.
    const auto absolute_remaining =
        geometry_.issuance_budget - next_version_ - reserved_outstanding_;
    if (capacity > absolute_remaining)
        return std::nullopt;
    auto self = shared_from_this();
    if (reservation_serial_ == std::numeric_limits<std::uint64_t>::max())
        return std::nullopt; // Nonce EXHAUSTION terminal: the counter stays at
                             // max, so no wrap can re-issue serial1 and an
                             // old handle can never replay. New reservations
                             // are permanently refused (sealed).
    ++reservation_serial_;
    reserved_outstanding_ += capacity;
    return IssuanceReservation(std::move(self), capacity, reservation_serial_);
}

std::optional<CPUFP32PagedKVTable::Update>
CPUFP32PagedKVTable::tryBeginReserved(IssuanceReservation &reservation)
{
    if (!reservation.table_ || reservation.table_.get() != this || !reservation.remaining_)
        return std::nullopt; // Foreign, moved-from or exhausted handle: refuse.
    for (size_t bank = 0; bank < banks_.size(); ++bank)
    {
        auto &state = banks_[bank];
        if (state.phase != Phase::Free) continue;
        if (next_version_ >= geometry_.issuance_budget)
            return std::nullopt; // Absolute finite bound.
        auto self = shared_from_this();
        state.version = ++next_version_;
        state.phase = Phase::Preparing;
        --reservation.remaining_;
        // Consumed capacity is no longer outstanding (issued already counts
        // it against the budget), so normal admission remains protected by
        // the remaining headroom only.
        --reserved_outstanding_;
        return Update(std::move(self), bank, state.version);
    }
    return std::nullopt;
}
CPUFP32PagedKVTable::IssuanceReservation::IssuanceReservation(
    std::shared_ptr<CPUFP32PagedKVTable> table, std::size_t capacity,
    std::uint64_t serial)
    : table_(std::move(table)), remaining_(capacity), serial_(serial) {}
CPUFP32PagedKVTable::IssuanceReservation::~IssuanceReservation() { release(); }
CPUFP32PagedKVTable::IssuanceReservation::IssuanceReservation(IssuanceReservation &&other) noexcept
    : table_(std::move(other.table_)), remaining_(std::exchange(other.remaining_, 0)),
      serial_(std::exchange(other.serial_, 0)) {}
CPUFP32PagedKVTable::IssuanceReservation &
CPUFP32PagedKVTable::IssuanceReservation::operator=(IssuanceReservation &&other) noexcept
{
    if (this != &other)
    {
        release();
        table_ = std::move(other.table_);
        remaining_ = std::exchange(other.remaining_, 0);
        serial_ = std::exchange(other.serial_, 0);
    }
    return *this;
}
void CPUFP32PagedKVTable::IssuanceReservation::release() noexcept
{
    if (!table_) return;
    auto &self = *table_;
    // Return any unconsumed capacity to the outstanding pool (no allocation).
    if (remaining_)
        self.reserved_outstanding_ -= remaining_;
    table_.reset();
}
std::optional<EffectiveUseLease> CPUFP32PagedKVTable::touchEffectiveUse(
    const Snapshot &snapshot)
{
    // AUTHENTICATE the touch against the actual published bank BEFORE issuing
    // a lease: a burned or unpublished version is not a successful attachment,
    // and a foreign table never admits anything. The lease (not the snapshot)
    // carries the full identity, so acknowledgement stays authoritative even
    // after the bank clears and no snapshot mutation exists.
    if (snapshot.table_.get() != this || snapshot.bank_ >= banks_.size())
        return std::nullopt;
    auto &state = banks_[snapshot.bank_];
    if (state.phase != Phase::Published ||
        state.version != snapshot.version_ || !state.pages)
        return std::nullopt;
    // Ordinal exhaustion is TERMINAL and rejects BEFORE incrementing (no wrap
    // re-issue; the counter is never mutated on the sealed path).
    if (effective_use_sequence_ >= std::numeric_limits<std::uint64_t>::max() - 1)
        return std::nullopt;
    // The ordinal is minted ONLY when the journal admits the touch: a
    // refused attempt (journal full, foreign, obsolete) consumes NO ordinal,
    // so the identity space is never burned by failed proposals (R14 note).
    // The candidate is the strictly next value; the assignment below happens
    // only on admission, and the serving session is the single table owner.
    const auto sequence = effective_use_sequence_ + 1;
    // The replica incarnation is the PUBLISHED VERSION (monotonic, never
    // recycled): the bank INDEX is reused after retirement and can never
    // authenticate an incarnation. Journal refusal leaves no lease.
    // The lease retains THIS TABLE as its lifetime token, so journal storage
    // can never be freed while a lease is outstanding.
    auto lease = effective_use_.touch(EffectiveUseReceipt{
        .content_version=state.version, .table_version=state.version,
        .replica_incarnation=state.version, .sequence=sequence},
        shared_from_this());
    if (lease)
        effective_use_sequence_ = sequence; // Admitted: consume the ordinal.
    return lease;
}



void CPUFP32PagedKVTable::closeBorrowers(const Snapshot &snapshot)
{
    if (snapshot.table_.get() != this)
        throw std::logic_error("CPU page-table closure has a foreign publication owner");
    auto &state = require(snapshot.bank_, snapshot.version_, Phase::Published);
    state.accepting_readers = false;
}
void CPUFP32PagedKVTable::releaseSnapshot(size_t bank, std::uint64_t version) noexcept
{
    try
    {
        auto &state = require(bank, version, Phase::Published);
        if (!state.readers) std::terminate();
        if (--state.readers == 0) clear(bank);
    }
    catch (...) { std::terminate(); }
}
CPUFP32PagedKVTable::Update::Update(std::shared_ptr<CPUFP32PagedKVTable> table, size_t bank, std::uint64_t version)
    : table_(std::move(table)), bank_(bank), version_(version) {}
CPUFP32PagedKVTable::Update::~Update() { release(); }
CPUFP32PagedKVTable::Update::Update(Update &&other) noexcept
    : table_(std::move(other.table_)), bank_(other.bank_), version_(other.version_) {}
CPUFP32PagedKVTable::Update &CPUFP32PagedKVTable::Update::operator=(Update &&other) noexcept
{
    if (this != &other) { release(); table_ = std::move(other.table_); bank_ = other.bank_; version_ = other.version_; }
    return *this;
}
void CPUFP32PagedKVTable::Update::attachArchivedPrefix(const PagedKVArchive &archive,
    PhysicalPagePool::Grant &grant, int page_tokens, std::uint64_t trust_domain,
    std::uint64_t ancestry_id)
{
    if (!table_) throw std::logic_error("CPU page-table prefix attach uses an empty update");
    // Provenance gate before any mutation: FP32 attention only, exact
    // heads/dimension must match the declared table geometry, and the
    // authenticated trust domain / ancestry must agree with the caller.
    if (archive.empty() || archive.meta().format != KVStorageFormat::FP32 ||
        archive.meta().family != KVCacheFamily::AttentionOnly)
        throw std::invalid_argument("Archived prefix attach requires a nonempty FP32 attention section");
    if (trust_domain != archive.meta().trust_domain)
        throw std::invalid_argument("Archived prefix trust domain disagrees with the authenticated isolation policy");
    if (ancestry_id != archive.meta().ancestry_id)
        throw std::invalid_argument("Archived prefix ancestry disagrees with the required exact provenance");
    auto &state = table_->require(bank_, version_, Phase::Preparing);
    // Provenance must agree with any page geometry already staged in this
    // update; the view constructor on commit rejects mixed geometry anyway.
    if (state.pages)
    {
        const auto &first = table_->pages_[bank_ * table_->geometry_.page_capacity];
        if (first.heads() != archive.meta().heads || first.dimension() != archive.meta().dimension)
            throw std::invalid_argument("Archived prefix provenance disagrees with the staged destination geometry");
    }
    // COMPLETE preflight before a single byte is staged (atomicity across the
    // table and the caller's grant): the archive's exact page count must fit
    // the remaining metadata cells and the grant's protected growth, and the
    // caller's physical authority must be this table's original authority.
    // materialize() itself rechecks its own grant growth before assignment.
    const std::size_t pages_needed = archive.pageCount(page_tokens);
    if (pages_needed > table_->geometry_.page_capacity - state.pages)
        throw std::invalid_argument("Archived prefix exceeds the admitted table metadata cells");
    if (grant.issuer() != table_->authority_.get())
        throw std::invalid_argument("Archived prefix materialization has a foreign physical authority");
    if (pages_needed > grant.remainingCommitment())
        throw std::invalid_argument("Archived prefix materialization grant lacks protected growth");
    // Private materialization under the caller grant; the returned pages are
    // moved into this update. No sampler/RNG/history/placement state is
    // restored by this attachment — only native K/V bytes enter the bank.
    auto pages = archive.materialize(grant, page_tokens);
    // Infallible insertion: cells were prevalidated and the authority was
    // checked before materialize, so a page cannot exceed the admitted
    // capacity or carry a foreign owner here. A throw inside this loop is
    // still caught to preserve the grant/table refund (defense in depth).
    try
    {
        for (auto &page : pages)
        {
            if (state.pages == table_->geometry_.page_capacity)
                throw std::logic_error("CPU page-table prefix attach exceeds its admitted metadata cells");
            if (page.issuer() != table_->authority_.get())
                throw std::invalid_argument("CPU page-table prefix attach has a foreign physical authority");
            table_->pages_[bank_ * table_->geometry_.page_capacity + state.pages] = std::move(page);
            ++state.pages;
        }
    }
    catch (...)
    {
        // Roll back both owners: retire any page already inserted back to the
        // table cells, then drop the materialized pages (each ref leaves the
        // grant through its retained publication) and report the refusal.
        auto &clean = table_->require(bank_, version_, Phase::Preparing);
        // Retire every page that entered this bank during the attachment. The
        // state.pages counter is the precise inserted extent; each cell above
        // it was already empty, and assigning an empty page vend the owner.
        while (clean.pages)
        {
            auto &cell = table_->pages_[bank_ * table_->geometry_.page_capacity + clean.pages - 1];
            cell = CPUFP32KVPage();
            --clean.pages;
        }
        throw;
    }
}

void CPUFP32PagedKVTable::Update::appendShared(PhysicalPagePool::SharedPage &&shared, int heads, int dimension)
{
    if (!table_) throw std::logic_error("CPU page-table shared append uses an empty update");
    auto &state = table_->require(bank_, version_, Phase::Preparing);
    if (state.pages == table_->geometry_.page_capacity)
        throw std::logic_error("CPU page-table shared append exceeds its admitted metadata cells");
    CPUFP32KVPage entry(std::move(shared), heads, dimension);
    if (entry.issuer() != table_->authority_.get())
        throw std::invalid_argument("CPU page-table shared attachment has a foreign physical authority");
    table_->pages_[bank_ * table_->geometry_.page_capacity + state.pages] = std::move(entry);
    ++state.pages;
}

void CPUFP32PagedKVTable::Update::append(CPUFP32KVPage &&page)
{
    if (!table_) throw std::logic_error("CPU page-table append uses an empty update");
    auto &state = table_->require(bank_, version_, Phase::Preparing);
    if (page.issuer() != table_->authority_.get())
        throw std::invalid_argument("CPU page-table attachment has a foreign physical authority");
    if (state.pages == table_->geometry_.page_capacity)
        throw std::logic_error("CPU page-table append exceeds its admitted metadata cells");
    table_->pages_[bank_ * table_->geometry_.page_capacity + state.pages] = std::move(page);
    ++state.pages;
}
CPUFP32PagedKVTable::Snapshot CPUFP32PagedKVTable::Update::commit()
{
    if (!table_) throw std::logic_error("CPU page-table commit uses an empty update");
    auto &state = table_->require(bank_, version_, Phase::Preparing);
    const auto pages = std::span<const CPUFP32KVPage>(table_->pages_ + bank_ * table_->geometry_.page_capacity, state.pages);
    (void)CPUFP32PagedKVView(pages, table_->geometry_.page_tokens);
    state.phase = Phase::Published;
    state.accepting_readers = true;
    state.readers = 1;
    return Snapshot(std::move(table_), bank_, version_);
}
void CPUFP32PagedKVTable::Update::release() noexcept
{
    if (!table_) return;
    try { (void)table_->require(bank_, version_, Phase::Preparing); table_->clear(bank_); }
    catch (...) { std::terminate(); }
    table_.reset();
}
CPUFP32PagedKVTable::Snapshot::Snapshot(std::shared_ptr<CPUFP32PagedKVTable> table, size_t bank, std::uint64_t version)
    : table_(std::move(table)), bank_(bank), version_(version) {}
CPUFP32PagedKVTable::Snapshot::~Snapshot() { release(); }
CPUFP32PagedKVTable::Snapshot::Snapshot(Snapshot &&other) noexcept
    : table_(std::move(other.table_)), bank_(other.bank_), version_(other.version_) {}
CPUFP32PagedKVTable::Snapshot &CPUFP32PagedKVTable::Snapshot::operator=(Snapshot &&other) noexcept
{
    if (this != &other)
    {
        release();
        table_ = std::move(other.table_);
        bank_ = other.bank_;
        version_ = other.version_;
    }
    return *this;
}
CPUFP32PagedKVView CPUFP32PagedKVTable::Snapshot::view() const
{
    if (!table_) throw std::logic_error("CPU page-table view uses an empty snapshot");
    const auto &state = table_->require(bank_, version_, Phase::Published);
    return CPUFP32PagedKVView({table_->pages_ + bank_ * table_->geometry_.page_capacity, state.pages}, table_->geometry_.page_tokens);
}
bool CPUFP32PagedKVTable::bindPublication(Snapshot &snapshot,
    std::uint64_t request_id, int slot)
{
    // L2: only THIS table's live published snapshot can be bound. A foreign
    // table's snapshot (or an unpublished version) refuses.
    if (snapshot.table_.get() != this)
        return false;
    auto &state = require(snapshot.bank_, snapshot.version_, Phase::Published);
    // M5: bind exactly once. A published bank is owned by one request
    // publication; rebinding would reassign shared history under a new
    // identity while existing readers still hold the old refs. clear()
    // resets the binding when the bank returns to Free. Unbound means
    // request_id == 0 (request identities are always non-zero; K1 refuses
    // request_id 0 at resolve time), so the slot sentinel is irrelevant here.
    if (state.request_id != 0)
        return false;
    if (request_id == 0)
        return false; // A zero request can never own a publication.
    state.request_id = request_id;
    state.slot = slot;
    return true;
}

std::optional<CPUFP32PagedKVTable::Snapshot> CPUFP32PagedKVTable::resolveSnapshotRef(
    const PagedKVSnapshotRef &ref)
{
    if (!ref.valid() || ref.authority_id != authority_identity_)
        return std::nullopt; // Foreign incarnation: refuse (never a different table).
    if (ref.request_id == 0)
        return std::nullopt; // K1: request binding is mandatory (never a bare ref).
    // O(1) resolution: the bank hint is VERIFIED against the exact version
    // before any borrow (it is a hint, never identity). A hint pointing at a
    // reused bank with a different version refuses rather than scanning on
    // (the caller may only fall back to a scan when the hint is invalid).
    if (ref.bank_hint >= 0 &&
        static_cast<std::size_t>(ref.bank_hint) < banks_.size())
    {
        auto &hinted = banks_[static_cast<std::size_t>(ref.bank_hint)];
        if (hinted.phase == Phase::Published && hinted.pages &&
            hinted.version == ref.publication_version)
        {
            // L2: exact request and slot binding match (an unbound, or a
            // differently-bound, publication never resolves).
            if (hinted.request_id != ref.request_id || hinted.slot != ref.slot)
                return std::nullopt;
            if (hinted.accepting_readers &&
                hinted.readers < geometry_.readers_per_bank)
            {
                auto self = shared_from_this();
                ++hinted.readers;
                return Snapshot(std::move(self), static_cast<std::size_t>(ref.bank_hint),
                    hinted.version);
            }
            return std::nullopt; // Exact hint, reader envelope exhausted: refuse.
        }
        // Hint mismatch: a reused/superseded bank can never serve this version.
        return std::nullopt; // O(1): never scan a bank that wrongly hints.
    }
    // No hint: fall back to the authenticated scan (J3 path, used when the
    // minted ref predates the hint field or came from a legacy owner).
    for (std::size_t bank = 0; bank < banks_.size(); ++bank)
    {
        auto &state = banks_[bank];
        // EXACT version authentication: a retired/reused bank has a different
        // monotonic version, so a stale ref can never borrow another request's
        // history or a superseded publication (J3).
        if (state.phase != Phase::Published || !state.pages ||
            state.version != ref.publication_version)
            continue;
        if (state.request_id != ref.request_id || state.slot != ref.slot)
            continue; // L2: a differently-bound publication never resolves.
        if (!state.accepting_readers || state.readers == geometry_.readers_per_bank)
            return std::nullopt;
        auto self = shared_from_this();
        ++state.readers;
        return Snapshot(std::move(self), bank, state.version);
    }
    return std::nullopt; // No live publication with this exact version.
}

std::optional<CPUFP32PagedKVTable::Snapshot> CPUFP32PagedKVTable::Snapshot::tryBorrow() const
{
    if (!table_) throw std::logic_error("CPU page-table borrow uses an empty snapshot");
    auto &state = table_->require(bank_, version_, Phase::Published);
    if (!state.accepting_readers || state.readers == table_->geometry_.readers_per_bank) return std::nullopt;
    auto self = table_;
    ++state.readers;
    // A borrow CONTAINS NO touch capability: the borrower must touch and own
    // its own lease to acknowledge anything (R11 independent consumers).
    return Snapshot(std::move(self), bank_, version_);
}
void CPUFP32PagedKVTable::Snapshot::release() noexcept
{
    if (table_) { table_->releaseSnapshot(bank_, version_); table_.reset(); }
}
}
