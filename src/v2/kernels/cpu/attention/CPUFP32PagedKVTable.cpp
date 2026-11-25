/**
 * @file CPUFP32PagedKVTable.cpp
 * @brief Stable PMA table metadata and atomic nonempty CPU snapshot publication.
 */
#include "CPUFP32PagedKVTable.h"
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
}
size_t CPUFP32PagedKVTable::physicalBytes(CPUFP32PagedKVTableGeometry geometry, DeviceId device, CPUStoragePlacement placement)
{
    geometry.validate();
    const auto cells = CPUStorageRoot<std::byte>(geometry.bank_count * geometry.page_capacity * sizeof(CPUFP32KVPage),
        PhysicalMemoryOwner::KVCache, device, placement).physicalBytes();
    const auto banks = CPUStorageRoot<Bank>(geometry.bank_count, PhysicalMemoryOwner::KVCache, device, placement).physicalBytes();
    if (banks > std::numeric_limits<size_t>::max() - cells)
        throw std::overflow_error("CPU page-table physical BOM overflows");
    return cells + banks;
}
std::shared_ptr<CPUFP32PagedKVTable> CPUFP32PagedKVTable::create(std::shared_ptr<PhysicalMemoryAuthority> authority,
    DeviceId device, CPUFP32PagedKVTableGeometry geometry, CPUStoragePlacement placement)
{
    if (!authority || !device.is_cpu()) throw std::invalid_argument("CPU page table requires its original PMA CPU allocator");
    return std::shared_ptr<CPUFP32PagedKVTable>(new CPUFP32PagedKVTable(std::move(authority), device, geometry, placement));
}
CPUFP32PagedKVTable::CPUFP32PagedKVTable(std::shared_ptr<PhysicalMemoryAuthority> authority, DeviceId device,
    CPUFP32PagedKVTableGeometry geometry, CPUStoragePlacement placement) : authority_(authority), geometry_(geometry)
{
    (void)physicalBytes(geometry, device, placement);
    static_assert(alignof(CPUFP32KVPage) <= AlignedVector<std::byte>::ALIGNMENT);
    cells_ = CPUStorageRoot<std::byte>(geometry.bank_count * geometry.page_capacity * sizeof(CPUFP32KVPage),
        PhysicalMemoryOwner::KVCache, device, placement, CPUStorageInitialization::Uninitialized).allocate(authority);
    banks_ = CPUStorageRoot<Bank>(geometry.bank_count, PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
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
}
std::optional<CPUFP32PagedKVTable::Update> CPUFP32PagedKVTable::tryBeginUpdate()
{
    for (size_t bank = 0; bank < banks_.size(); ++bank)
    {
        auto &state = banks_[bank];
        if (state.phase != Phase::Free) continue;
        if (next_version_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("CPU page-table publication generation exhausted");
        auto self = shared_from_this();
        state.version = ++next_version_;
        state.phase = Phase::Preparing;
        return Update(std::move(self), bank, state.version);
    }
    return std::nullopt;
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
    // Private materialization under the caller grant; the returned pages are
    // moved into this update. No sampler/RNG/history/placement state is
    // restored by this attachment — only native K/V bytes enter the bank.
    auto pages = archive.materialize(grant, page_tokens);
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
    if (this != &other) { release(); table_ = std::move(other.table_); bank_ = other.bank_; version_ = other.version_; }
    return *this;
}
CPUFP32PagedKVView CPUFP32PagedKVTable::Snapshot::view() const
{
    if (!table_) throw std::logic_error("CPU page-table view uses an empty snapshot");
    const auto &state = table_->require(bank_, version_, Phase::Published);
    return CPUFP32PagedKVView({table_->pages_ + bank_ * table_->geometry_.page_capacity, state.pages}, table_->geometry_.page_tokens);
}
std::optional<CPUFP32PagedKVTable::Snapshot> CPUFP32PagedKVTable::Snapshot::tryBorrow() const
{
    if (!table_) throw std::logic_error("CPU page-table borrow uses an empty snapshot");
    auto &state = table_->require(bank_, version_, Phase::Published);
    if (!state.accepting_readers || state.readers == table_->geometry_.readers_per_bank) return std::nullopt;
    auto self = table_;
    ++state.readers;
    return Snapshot(std::move(self), bank_, version_);
}
void CPUFP32PagedKVTable::Snapshot::release() noexcept
{
    if (table_) { table_->releaseSnapshot(bank_, version_); table_.reset(); }
}
}
