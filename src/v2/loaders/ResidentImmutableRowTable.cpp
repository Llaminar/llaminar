/**
 * @file ResidentImmutableRowTable.cpp
 * @brief Bounded setup loading of exact immutable native GGUF row bytes.
 *
 * Source reads fill one already-admitted ordinary-loader pinned slot. Every
 * chunk contains complete source rows and drains its exact upload event before
 * reuse. Final descriptor/path authentication precedes sealing/publication,
 * so partial or mixed-file bytes never become an executable model source.
 */
#include "loaders/ResidentImmutableRowTable.h"
#include "transfer/TransferEngine.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
namespace llaminar2
{
namespace
{
/** @brief Preserve the same immutable open-file/path metadata authority as the host service. */
ImmutableRowFileIdentity nativeFileIdentity(const struct stat &info)
{
    return {static_cast<std::uint64_t>(info.st_dev), static_cast<std::uint64_t>(info.st_ino),
        static_cast<std::uint64_t>(info.st_size), info.st_mtim.tv_sec, info.st_mtim.tv_nsec,
        info.st_ctim.tv_sec, info.st_ctim.tv_nsec};
}
/** @brief Setup I/O errors name the exact original source; no alternate file/tier is attempted. */
[[noreturn]] void sourceError(const char *operation, const std::string &path)
{ throw std::runtime_error(std::string("Resident native table ") + operation + ": " + path + ": " + std::strerror(errno)); }
}
struct ResidentImmutableRowTable::Impl
{
    ImmutableRowTableSource source;
    ImmutableRowFileIdentity identity;
    std::uint64_t first_row=0,owned_rows=0; ///< Source-relative geometry, not a second source or byte ledger.
    std::shared_ptr<const ImmutableNativeDeviceRegion> storage;
    int fd = -1;
    /** @brief Last captured borrower retires actual bytes before the PMA claim. */
    ~Impl() { storage.reset(); if (fd >= 0) ::close(fd); }
    /** @brief Check both retained descriptor and pathname identity without reading payload. */
    void authenticate() const
    {
        struct stat open_info{}, path_info{};
        if (::fstat(fd, &open_info) != 0) sourceError("fstat", source.path);
        if (::stat(source.path.c_str(), &path_info) != 0) sourceError("stat", source.path);
        if (nativeFileIdentity(open_info) != identity || nativeFileIdentity(path_info) != identity)
            throw std::runtime_error("Resident native table source changed during setup/reuse: " + source.path);
    }
};
ResidentImmutableRowTable::ResidentImmutableRowTable(std::unique_ptr<Impl> state) : impl_(std::move(state)) {}
ResidentImmutableRowTable::~ResidentImmutableRowTable() = default;
std::shared_ptr<const ResidentImmutableRowTable> ResidentImmutableRowTable::prepare(
    ImmutableRowTableSource source, const ResolvedImmutableRowTablePlacement &placement,
    std::shared_ptr<PhysicalMemoryAuthority> memory, DeviceId device,
    const ReleasedWeightLoadStaging &staging, void *setup_stream, const TransferEngine &transfers)
{
    if(!memory || placement.isHostAdaptive() || placement.isFixedHeadShards() ||
        placement.storage()!=ImmutableRowStorage::VRAM || placement.worldRank()!=memory->worldRank() ||
        !placement.containsAllocator({memory->worldRank(),device}))
        throw std::invalid_argument("Whole resident table requires its exact whole-source VRAM/PMA placement");
    const auto full=ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes());
    if(placement.extent()!=full)throw std::invalid_argument("Whole resident source differs from its declared original extent");
    return prepareOwnedRange(std::move(source),ImmutableRowTableSourceRange::whole(full),std::move(memory),device,staging,setup_stream,transfers);
}
std::shared_ptr<const ResidentImmutableRowTable> ResidentImmutableRowTable::prepareRange(
    ImmutableRowTableSource source,const ResolvedImmutableRowTablePlacement &placement,
    std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId device,const ReleasedWeightLoadStaging &staging,
    void *setup_stream,const TransferEngine &transfers)
{
    if(!memory || !placement.isFixedHeadShards() || placement.worldRank()!=memory->worldRank() ||
        !placement.containsAllocator({memory->worldRank(),device}))
        throw std::invalid_argument("Resident range upload requires its checked fixed original-source owner/PMA");
    const auto range=placement.fixedHeadSourceRangeFor({memory->worldRank(),device});
    return prepareOwnedRange(std::move(source),range,std::move(memory),device,staging,setup_stream,transfers);
}
std::shared_ptr<const ResidentImmutableRowTable> ResidentImmutableRowTable::prepareOwnedRange(
    ImmutableRowTableSource source,ImmutableRowTableSourceRange range,std::shared_ptr<PhysicalMemoryAuthority> memory,
    DeviceId device,const ReleasedWeightLoadStaging &staging,void *setup_stream,const TransferEngine &transfers)
{
    if(isGraphCaptureActive() || !memory || !setup_stream || !device.is_gpu() || source.path.empty() || source.tensor_name.empty() ||
        source.row_elements>static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        range.sourceExtent()!=ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()))
        throw std::invalid_argument("Resident native upload requires exact original source, explicit GPU/stream and checked range");
    const auto row_bytes = source.rowBytes();
    // Reject unsupported native layouts before any source read or GPU upload.
    qwen4exp::lookup_detail::validateGeometry(ggufToTensorType(source.format),
        static_cast<int>(source.row_elements), row_bytes);
    staging.requireFor(memory, device, row_bytes);
    auto state = std::make_unique<Impl>(); state->source = std::move(source);
    state->first_row=range.firstRow();state->owned_rows=range.rows();
    state->fd = ::open(state->source.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (state->fd < 0) sourceError("open", state->source.path);
    struct stat info{};
    if (::fstat(state->fd, &info) != 0) sourceError("fstat", state->source.path);
    if (info.st_size < 0 || !S_ISREG(info.st_mode))
        throw std::invalid_argument("Resident native source must be an immutable regular file");
    state->identity = nativeFileIdentity(info); state->authenticate();
    // Authenticate the COMPLETE source, including bytes outside this range.
    const auto full_bytes = state->source.tensorBytes();
    const auto bytes=range.nativeBytes();
    if (state->source.file_offset > state->identity.file_bytes ||
        full_bytes > state->identity.file_bytes - state->source.file_offset ||
        state->source.file_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        full_bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) - state->source.file_offset)
        throw std::invalid_argument("Resident native source extent exceeds the authenticated file/address range");
    // The receipt carries original loader geometry; table size cannot enlarge it.
    auto slot = transfers.prepareImmutableNativeUploadStaging(staging, memory, device, row_bytes);
    const auto chunk_capacity = slot->sizeBytes() / row_bytes * row_bytes;
    auto upload = transfers.beginImmutableNativeDeviceUpload(memory, device, bytes, row_bytes, setup_stream);
    while (upload.copiedBytes() < bytes)
    {
        const auto offset = upload.copiedBytes();
        const auto live = std::min(chunk_capacity, bytes - offset);
        std::size_t read_bytes = 0;
        while (read_bytes < live)
        {
            const auto result = ::pread(state->fd, slot->data() + read_bytes, live - read_bytes,
                static_cast<off_t>(state->source.file_offset + range.byteOffset() + offset + read_bytes));
            if (result < 0 && errno == EINTR) continue;
            if (result < 0) sourceError("pread", state->source.path);
            if (result == 0) throw std::runtime_error("Resident native source truncated during setup upload");
            read_bytes += static_cast<std::size_t>(result);
        }
        transfers.uploadImmutableNativeChunk(upload, *slot, live, setup_stream);
    }
    state->authenticate();
    state->storage = transfers.sealImmutableNativeDeviceUpload(std::move(upload));
    // Validate the actual source codec/pointer before any model-visible return.
    auto result = std::shared_ptr<const ResidentImmutableRowTable>(new ResidentImmutableRowTable(std::move(state)));
    (void)result->rangeView();
    return result;
}
const ImmutableRowTableSource &ResidentImmutableRowTable::source() const noexcept { return impl_->source; }
const ImmutableRowFileIdentity &ResidentImmutableRowTable::identity() const noexcept { return impl_->identity; }
DeviceId ResidentImmutableRowTable::device() const noexcept { return impl_->storage->device(); }
std::size_t ResidentImmutableRowTable::nativeBytes() const noexcept { return impl_->storage->sizeBytes(); }
ImmutableRowTableSourceRange ResidentImmutableRowTable::sourceRange() const
{
    return ImmutableRowTableSourceRange::checked(ImmutableRowTableExtent::fromNative(source().format,source().rows,source().rowBytes()),
        impl_->first_row,impl_->owned_rows);
}
qwen4exp::ImmutableResidentTableRangeView ResidentImmutableRowTable::rangeView() const
{
    return qwen4exp::ImmutableResidentTableRangeView::bind(device(),ggufToTensorType(source().format),impl_->storage->data(),
        source().rows,impl_->first_row,impl_->owned_rows,static_cast<int>(source().row_elements),source().rowBytes(),
        nativeBytes(),impl_->storage->generation());
}
qwen4exp::ImmutableResidentTableView ResidentImmutableRowTable::view() const
{
    if(!sourceRange().isWhole())throw std::logic_error("Partial native source range cannot expose a whole-table lookup view");
    return qwen4exp::ImmutableResidentTableView::bind(device(), ggufToTensorType(source().format),
        impl_->storage->data(), source().rows, static_cast<int>(source().row_elements),
        source().rowBytes(), nativeBytes(), impl_->storage->generation());
}
void ResidentImmutableRowTable::authenticateSource() const
{
    if (isGraphCaptureActive()) throw std::logic_error("Resident source authentication is setup-only");
    impl_->authenticate();
}
ImmutableNativeDeviceRead ResidentImmutableRowTable::acquire(void *consumer_stream, const TransferEngine &transfers) const
{ return transfers.acquireImmutableNativeDeviceInput(impl_->storage, device(), consumer_stream); }
}
