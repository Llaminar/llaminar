/**
 * @file ImmutableRowTable.cpp
 * @brief Bounded native row-table mappings and RAM materialization.
 *
 * File descriptors are opened once. RAM mode owns one anonymous, native-format
 * payload using explicit task first-touch or certified exact-node placement; mmap mode
 * retains only a fixed LRU window directory. Shared file-cache page placement
 * follows Linux task/system policy and is not a per-VMA NUMA locality promise.
 * Eviction retires a window's mapping before reusing its slot, bounding process RSS
 * independently of the full table extent. Linux's reclaimable file page cache
 * remains an OS cache, distinct from the process-owned mapping budget. There is
 * no model-wide prefault, checksum, float expansion or accelerator allocation.
 */
#include "ImmutableRowTable.h"
#include "memory/NUMAAllocator.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace llaminar2
{
namespace
{
/** @brief Checked physical/file arithmetic shared by planning and materialization. */
std::size_t multiply(std::uint64_t a, std::uint64_t b)
{
    if (a == 0 || b == 0 || a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("immutable row table has zero or overflowing extent");
    return static_cast<std::size_t>(a * b);
}

/**
 * @brief Validate the complete native gather metadata/response spans before reading or writing them.
 * @param metadata Caller-owned IDs or compact scatter targets.
 * @param count Exact number of metadata records, excluding allocation capacity.
 * @param element_bytes Native metadata record width.
 * @param response Exact caller-owned response span for this operation.
 * @throws std::invalid_argument For byte/address overflow, null nonempty spans or overlap.
 * @details Gather rereads IDs while copying. Even valid row bytes can overwrite a
 * later ID if spans alias. Both contiguous and scattered operations must reject
 * that frame before source authentication, resolver work or output mutation.
 */
void requireDisjointNativeResponse(const void *metadata, std::size_t count,
    std::size_t element_bytes, std::span<std::byte> response)
{
    if (!element_bytes || count > std::numeric_limits<std::size_t>::max() / element_bytes)
        throw std::invalid_argument("immutable row gather metadata extent overflows");
    const auto bytes = count * element_bytes;
    const auto a = reinterpret_cast<std::uintptr_t>(metadata);
    const auto b = reinterpret_cast<std::uintptr_t>(response.data());
    if ((bytes && !metadata) || (!response.empty() && !response.data()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - a ||
        response.size() > std::numeric_limits<std::uintptr_t>::max() - b ||
        (bytes && !response.empty() && a < b + response.size() && b < a + bytes))
        throw std::invalid_argument("immutable row gather metadata aliases or exceeds its response address range");
}

/** @brief Report the failing syscall without erasing its native error. */
[[noreturn]] void ioError(const char *operation, const std::string &path)
{
    throw std::runtime_error(std::string("immutable row table ") + operation +
                             " failed for " + path + ": " + std::strerror(errno));
}

/** @brief Authenticate regular-file identity using metadata only. */
ImmutableRowFileIdentity fileIdentity(const struct stat &value)
{
    if (!S_ISREG(value.st_mode) || value.st_size < 0)
        throw std::invalid_argument("immutable row table requires a regular file");
    return {static_cast<std::uint64_t>(value.st_dev),
            static_cast<std::uint64_t>(value.st_ino),
            static_cast<std::uint64_t>(value.st_size),
            value.st_mtim.tv_sec, value.st_mtim.tv_nsec,
            value.st_ctim.tv_sec, value.st_ctim.tv_nsec};
}

/** @brief Sentinel shared by the fixed hash chains and intrusive recency list. */
constexpr auto NoWindow = std::numeric_limits<std::size_t>::max();

/** @brief Fixed descriptor; indices keep hash and LRU maintenance allocation-free. */
struct MappingWindow
{
    void *base = MAP_FAILED;
    std::uint64_t file_offset = 0;
    std::size_t length = 0;
    std::size_t hash_previous = NoWindow;
    std::size_t hash_next = NoWindow;
    std::size_t older = NoWindow;
    std::size_t newer = NoWindow;
};

/** @brief Round a mapping extent with an already authenticated native page size. */
std::size_t pageAligned(std::size_t bytes, std::size_t page)
{
    if (page == 0 || bytes > std::numeric_limits<std::size_t>::max() - (page - 1))
        throw std::invalid_argument("immutable row table page-rounded extent overflow");
    return (bytes + page - 1) / page * page;
}

/** @brief Resolve the native page authority during setup, retaining existing RAM pricing. */
std::size_t pageAligned(std::size_t bytes)
{
    const auto page = ::sysconf(_SC_PAGESIZE);
    if (page <= 0)
        throw std::invalid_argument("immutable row table page-rounded extent overflow");
    return pageAligned(bytes, static_cast<std::size_t>(page));
}
}

std::size_t ImmutableRowTableSource::rowBytes() const
{
    GGUFTensorInfo tensor{};
    tensor.type = format;
    const auto block = tensor.isQuantized() ? tensor.getBlockSize() : 1;
    const auto bytes = tensor.getTypeSize();
    if (block == 0 || bytes == 0 || row_elements == 0 || row_elements % block != 0)
        throw std::invalid_argument("immutable row table requires complete native quantization blocks");
    return multiply(row_elements / block, bytes);
}

std::size_t ImmutableRowTableSource::tensorBytes() const
{
    return multiply(rows, rowBytes());
}

std::size_t ImmutableRowTable::physicalBytes(
    const ImmutableRowTableSource &source, const ImmutableRowTableStoragePolicy &policy)
{
    const auto table_bytes = source.tensorBytes();
    if (source.path.empty() || source.tensor_name.empty())
        throw std::invalid_argument("immutable row table requires file and tensor identities");
    return physicalBytesForNativeExtent(table_bytes, policy);
}

std::size_t ImmutableRowTable::physicalBytesForNativeExtent(
    std::size_t table_bytes, const ImmutableRowTableStoragePolicy &policy)
{
    if (table_bytes == 0)
        throw std::invalid_argument("immutable row table requires a positive exact native extent");
    if (policy.storage() == ImmutableRowStorage::VRAM)
        return table_bytes;
    if (policy.storage() == ImmutableRowStorage::RAM)
        return pageAligned(table_bytes);
    if (policy.windowBytes() > std::numeric_limits<std::size_t>::max() - sizeof(MappingWindow) - sizeof(std::size_t))
        throw std::invalid_argument("immutable row mmap window directory overflow");
    return multiply(policy.windowBytes() + sizeof(MappingWindow) + sizeof(std::size_t), policy.windowCount());
}

struct ImmutableRowTable::Impl
{
    ImmutableRowTableSource source;
    ImmutableRowTableStoragePolicy policy = ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::RAM);
    DeviceId host_device;
    ImmutableRowTableHostPlacement host_placement = ImmutableRowTableHostPlacement::taskPolicy();
    ImmutableRowFileIdentity identity;
    // Authority/lease outlive every mapping and the anonymous RAM owner.
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    PhysicalMemoryAllocationLease lease;
    int fd = -1;
    void *ram = MAP_FAILED;
    std::vector<MappingWindow> windows;
    std::vector<std::size_t> buckets;
    std::size_t oldest = NoWindow;
    std::size_t newest = NoWindow;
    mutable std::mutex mutex;
    ImmutableRowTableStats counters;
    std::uint64_t clock = 0;
    // The native page size is immutable for this table. Resolver accounting
    // performs pure integer rounding, never a syscall per lookup or miss.
    const std::size_t page_bytes = pageAligned(1);

    /** @brief Retire physical storage even when construction fails partway through. */
    ~Impl()
    {
        for (const auto &window : windows)
            if (window.base != MAP_FAILED)
                ::munmap(window.base, window.length);
        if (ram != MAP_FAILED)
            ::munmap(ram, counters.physical_budget_bytes);
        if (fd >= 0)
            ::close(fd);
    }

    /** @brief Reject external replacement/mutation before accessing a mapped page. */
    void requireUnchanged()
    {
        struct stat opened{}, named{};
        if (::fstat(fd, &opened) != 0)
            ioError("fstat", source.path);
        if (::stat(source.path.c_str(), &named) != 0)
            ioError("stat", source.path);
        if (counters.source_metadata_checks == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("immutable source metadata observation counter exhausted");
        ++counters.source_metadata_checks;
        if (fileIdentity(opened) != identity || fileIdentity(named) != identity)
            throw std::runtime_error("immutable row table source changed: " + source.path);
    }

    /** @brief Copy one exact original native row; contiguous/scattered transactions share this byte authority. */
    void copyRow(std::uint64_t id, std::size_t row_bytes, std::byte *out)
    {
        auto offset = id * row_bytes;
        if (policy.storage() == ImmutableRowStorage::RAM)
            std::memcpy(out, static_cast<const std::byte *>(ram) + offset, row_bytes);
        else
        {
            std::size_t copied = 0;
            offset += source.file_offset;
            while (copied < row_bytes)
            {
                const auto [data, available] = resolve(offset + copied);
                const auto count = std::min(available, row_bytes - copied);
                std::memcpy(out + copied, data, count);
                copied += count;
            }
        }
    }

    /**
     * @brief Select a fixed bucket from the window number, never its aligned byte offset.
     * @details A power-of-two byte alignment must not collapse all keys into bucket zero.
     */
    std::size_t bucketFor(std::uint64_t aligned) const
    {
        return (aligned / policy.windowBytes()) % buckets.size();
    }

    /** @brief Unlink one retained mapping before its slot is reused; no tombstones accumulate. */
    void eraseIndex(std::size_t index)
    {
        auto &slot = windows[index];
        if (slot.hash_previous == NoWindow)
            buckets[bucketFor(slot.file_offset)] = slot.hash_next;
        else
            windows[slot.hash_previous].hash_next = slot.hash_next;
        if (slot.hash_next != NoWindow)
            windows[slot.hash_next].hash_previous = slot.hash_previous;
        slot.hash_previous = slot.hash_next = NoWindow;
    }

    /** @brief Publish one successfully mapped slot into its fixed collision chain. */
    void insertIndex(std::size_t index)
    {
        auto &slot = windows[index];
        auto &head = buckets[bucketFor(slot.file_offset)];
        slot.hash_next = head;
        if (head != NoWindow)
            windows[head].hash_previous = index;
        head = index;
    }

    /**
     * @brief Move one resolved slot to most recent without scanning other windows.
     * @details Initially all slots are unused in ascending order, matching the old
     * min-timestamp tie break. Each successful access removes precisely that slot
     * and appends it, preserving exact replacement order without timestamp sorting.
     */
    void touch(std::size_t index)
    {
        if (index == newest)
            return;
        auto &slot = windows[index];
        if (slot.older == NoWindow)
            oldest = slot.newer;
        else
            windows[slot.older].newer = slot.newer;
        windows[slot.newer].older = slot.older;
        windows[newest].newer = index;
        slot.older = newest;
        slot.newer = NoWindow;
        newest = index;
    }

    /** @brief Resolve one byte through the indexed bounded LRU; return readable contiguous extent. */
    std::pair<const std::byte *, std::size_t> resolve(std::uint64_t offset)
    {
        const auto aligned = offset - offset % policy.windowBytes();
        auto index = buckets[bucketFor(aligned)];
        while (index != NoWindow && windows[index].file_offset != aligned)
            index = windows[index].hash_next;
        if (index != NoWindow)
            ++counters.window_hits;
        else
        {
            ++counters.window_misses;
            index = oldest;
            auto &slot = windows[index];
            if (slot.base != MAP_FAILED)
            {
                if (::munmap(slot.base, slot.length) != 0)
                    ioError("munmap", source.path);
                counters.mapped_virtual_bytes -= pageAligned(slot.length, page_bytes);
                eraseIndex(index);
                slot.base = MAP_FAILED;
                ++counters.window_evictions;
            }
            const auto length = static_cast<std::size_t>(std::min<std::uint64_t>(
                policy.windowBytes(), identity.file_bytes - aligned));
            void *base = ::mmap(nullptr, length, PROT_READ, MAP_SHARED, fd,
                                static_cast<off_t>(aligned));
            if (base == MAP_FAILED)
                ioError("mmap", source.path);
            slot.base = base;
            slot.file_offset = aligned;
            slot.length = length;
            insertIndex(index);
            // mmap/munmap cover complete native pages even at EOF. Keep
            // length as readable file bytes, excluding padding from row access.
            counters.mapped_virtual_bytes += pageAligned(length, page_bytes);
            if (::madvise(base, length, MADV_RANDOM) != 0)
                ioError("madvise", source.path);
        }
        // Retain the existing lifecycle limit even though recency uses links.
        if (clock == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("immutable row table LRU generation exhausted");
        ++clock;
        touch(index);
        const auto &slot = windows[index];
        const auto within = static_cast<std::size_t>(offset - slot.file_offset);
        return {static_cast<const std::byte *>(slot.base) + within, slot.length - within};
    }
};

ImmutableRowTable::ImmutableRowTable(ImmutableRowTableSource source,
                                   ImmutableRowTableStoragePolicy policy,
                                   DeviceId host_device,
                                   std::shared_ptr<PhysicalMemoryAuthority> memory,
                                   ImmutableRowTableHostPlacement host_placement)
    : impl_(std::make_unique<Impl>())
{
    if (policy.storage() == ImmutableRowStorage::VRAM)
        throw std::invalid_argument("Host row service cannot materialize a VRAM placement");
    const auto physical = physicalBytes(source, policy);
    if (!host_device.is_valid() || !host_device.is_cpu())
        throw std::invalid_argument("immutable row table requires an explicit CPU/NUMA host resource");
    if (!memory)
        throw std::invalid_argument("immutable row table requires canonical physical admission");
    auto &state = *impl_;
    state.source = std::move(source);
    state.policy = policy;
    state.host_device = host_device;
    state.host_placement = host_placement;
    state.memory = std::move(memory);
    state.counters.storage = policy.storage();
    state.counters.table_file_bytes = state.source.tensorBytes();
    state.counters.physical_budget_bytes = physical;
    state.fd = ::open(state.source.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (state.fd < 0)
        ioError("open", state.source.path);
    struct stat info{};
    if (::fstat(state.fd, &info) != 0)
        ioError("fstat", state.source.path);
    state.identity = fileIdentity(info);
    if (state.source.file_offset > state.identity.file_bytes ||
        state.counters.table_file_bytes > state.identity.file_bytes - state.source.file_offset)
        throw std::invalid_argument("immutable row table tensor exceeds source file extent");
    state.lease = state.memory->claimNewAllocation(host_device,
        PhysicalMemoryOwner::ImmutableTableStorage, physical);
    if (policy.storage() == ImmutableRowStorage::MemoryMapped)
    {
        state.windows.resize(policy.windowCount());
        state.buckets.resize(policy.windowCount(), NoWindow);
        state.oldest = 0;
        state.newest = policy.windowCount() - 1;
        for (std::size_t index = 0; index < state.windows.size(); ++index)
        {
            state.windows[index].older = index ? index - 1 : NoWindow;
            state.windows[index].newer = index + 1 < state.windows.size() ? index + 1 : NoWindow;
        }
        return;
    }
    state.ram = ::mmap(nullptr, physical, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (state.ram == MAP_FAILED)
        ioError("anonymous mmap", state.source.path);
    // Exact-node locality uses the existing first-touch/certification authority
    // before any source read. Task policy deliberately leaves fresh pages
    // untouched: pread faults them under inherited Linux task/system policy.
    // CPU allocator ordinals never select physical memory placement.
    if (const auto node = host_placement.numaNode(); node &&
        !NUMAAllocator::instance().prepareExternalReceiveRangeOnNode(state.ram, physical, *node))
        throw std::runtime_error("immutable row table RAM NUMA placement failed for node " +
                                 std::to_string(*node));
    std::size_t done = 0;
    while (done < state.counters.table_file_bytes)
    {
        // Large sequential reads are setup work, not captured execution.
        const auto wanted = std::min<std::size_t>(state.counters.table_file_bytes - done,
                                                16u * 1024u * 1024u);
        const auto result = ::pread(state.fd, static_cast<std::byte *>(state.ram) + done,
            wanted, static_cast<off_t>(state.source.file_offset + done));
        if (result < 0 && errno == EINTR)
            continue;
        if (result < 0)
            ioError("pread", state.source.path);
        if (result == 0)
            throw std::runtime_error("immutable row table truncated during RAM load");
        done += static_cast<std::size_t>(result);
    }
    state.requireUnchanged();
    if (::mprotect(state.ram, physical, PROT_READ) != 0)
        ioError("mprotect", state.source.path);
}

ImmutableRowTable::~ImmutableRowTable() = default;
std::size_t ImmutableRowTable::physicalBytes() const noexcept { return impl_->lease.bytes(); }
HostImmutableRowOwnerObservation ImmutableRowTable::retainedOwnerObservation() const
{
    std::lock_guard lock(impl_->mutex);
    impl_->requireUnchanged();
    if (!impl_->memory || !impl_->host_device.is_cpu() || !impl_->lease.valid() ||
        impl_->lease.owner() != PhysicalMemoryOwner::ImmutableTableStorage)
        throw std::logic_error("Immutable row table has no exact live PMA storage lease");
    return {impl_->memory, {impl_->memory->worldRank(), impl_->host_device},
        impl_->host_placement, impl_->lease.bytes(), 0u};
}

void ImmutableRowTable::authenticateSource() const
{
    std::lock_guard lock(impl_->mutex);
    impl_->requireUnchanged();
}


DeviceId ImmutableRowTable::hostDevice() const noexcept { return impl_->host_device; }
ImmutableRowTableHostPlacement ImmutableRowTable::hostPlacement() const noexcept { return impl_->host_placement; }
ImmutableRowStorage ImmutableRowTable::storage() const noexcept { return impl_->policy.storage(); }

ImmutableRowGatherObservation ImmutableRowTable::gather(std::span<const std::int64_t> ids,
                               std::span<std::byte> destination)
{
    auto &state = *impl_;
    const auto row_bytes = state.source.rowBytes();
    if (ids.size() > std::numeric_limits<std::size_t>::max() / row_bytes ||
        destination.size() != ids.size() * row_bytes)
        throw std::invalid_argument("immutable row gather requires exact destination extent");
    requireDisjointNativeResponse(ids.data(), ids.size(), sizeof(std::int64_t), destination);
    for (const auto id : ids)
        if (id < 0 || static_cast<std::uint64_t>(id) >= state.source.rows)
            throw std::out_of_range("immutable row gather ID exceeds table rows");
    std::lock_guard lock(state.mutex);
    state.requireUnchanged();
    // One mutex interval owns both snapshots and every resolver/copy between
    // them. Sampling stats() around a call in an independent service would
    // race with other clients and attribute their cache work more than once.
    const ImmutableRowGatherObservation before{state.counters.window_hits,
        state.counters.window_misses,state.counters.window_evictions};
    auto *out = destination.data();
    for (const auto id : ids)
    {
        state.copyRow(static_cast<std::uint64_t>(id), row_bytes, out);
        out += row_bytes;
    }
    ++state.counters.requests;
    state.counters.rows += ids.size();
    state.counters.copied_bytes += destination.size();
    return {state.counters.window_hits-before.windowHits,
        state.counters.window_misses-before.windowMisses,
        state.counters.window_evictions-before.windowEvictions};
}

ImmutableRowGatherObservation ImmutableRowTable::gatherScattered(
    std::span<const ImmutableRowTableScatterRow> rows, std::span<std::byte> destination)
{
    auto &state = *impl_;
    const auto row_bytes = state.source.rowBytes();
    if (rows.size() > std::numeric_limits<std::size_t>::max() / sizeof(ImmutableRowTableScatterRow) ||
        destination.size() % row_bytes != 0 || rows.size() > destination.size() / row_bytes)
        throw std::invalid_argument("native scatter requires complete caller response rows");
    requireDisjointNativeResponse(rows.data(), rows.size(), sizeof(ImmutableRowTableScatterRow), destination);
    std::size_t previous = 0;
    bool first = true;
    for (const auto &row : rows)
    {
        if (row.source_row < 0 || static_cast<std::uint64_t>(row.source_row) >= state.source.rows)
            throw std::out_of_range("native scatter source row exceeds original table");
        if (row.response_row >= destination.size() / row_bytes || (!first && row.response_row <= previous))
            throw std::invalid_argument("native scatter response positions must be increasing and nonoverlapping");
        previous = row.response_row;
        first = false;
    }
    std::lock_guard lock(state.mutex);
    if (state.counters.requests == std::numeric_limits<std::uint64_t>::max() ||
        rows.size() > std::numeric_limits<std::uint64_t>::max() - state.counters.rows ||
        rows.size() * row_bytes > std::numeric_limits<std::uint64_t>::max() - state.counters.copied_bytes)
        throw std::overflow_error("native scatter observation counter exhausted");
    state.requireUnchanged();
    const ImmutableRowGatherObservation before{state.counters.window_hits,
        state.counters.window_misses, state.counters.window_evictions};
    for (const auto &row : rows)
        state.copyRow(static_cast<std::uint64_t>(row.source_row), row_bytes,
            destination.data() + row.response_row * row_bytes);
    state.requireUnchanged();
    ++state.counters.requests;
    state.counters.rows += rows.size();
    state.counters.copied_bytes += rows.size() * row_bytes;
    return {state.counters.window_hits - before.windowHits,
        state.counters.window_misses - before.windowMisses,
        state.counters.window_evictions - before.windowEvictions};
}

const ImmutableRowTableSource &ImmutableRowTable::source() const noexcept
{
    return impl_->source;
}

const ImmutableRowFileIdentity &ImmutableRowTable::identity() const noexcept
{
    return impl_->identity;
}

ImmutableRowTableStats ImmutableRowTable::stats() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->counters;
}
} // namespace llaminar2
