/**
 * @file ImmutableNativeDeviceStorage.cpp
 * @brief Exact admitted immutable-device storage and bounded setup-only uploads.
 *
 * The setup writer drains each submitted copy's exact event before its host
 * slot may be reused. Sealing moves the sole allocation and PMA lease into a
 * const owner; no mutable pointer is ever published. Consumer read tokens join
 * that producer during preparation and only validate immutable identity during
 * capture. This is an intentional initial-I/O boundary, never a hot transfer.
 */
#include "transfer/ImmutableNativeDeviceStorage.h"
#include "transfer/TransferEngine.h"
#include "loaders/gpu_pipeline/ReleasedWeightLoadStaging.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "backends/IBackend.h"
#include "utils/Logger.h"
#include <atomic>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
namespace llaminar2
{
namespace
{
/** @brief Allocation identity is immutable setup metadata, not an execution ledger. */
std::uint64_t nextImmutableStorageGeneration()
{
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;)
    {
        if (!value || value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Immutable native allocation identity exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value;
    }
}
/** @brief No setup operation may enter a captured/hot execution interval. */
void requireImmutableSetup()
{
    if (isGraphCaptureActive())
        throw std::logic_error("Immutable native storage preparation is forbidden during capture");
}
}
/** @brief One actual native allocation and its sole PMA claim; no host table. */
struct ImmutableNativeDeviceStorageState
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PhysicalMemoryAllocationLease lease;
    IBackend *backend = nullptr;
    DeviceId device = DeviceId::invalid();
    void *allocation = nullptr;
    void *event = nullptr;
    void *stream = nullptr;
    std::size_t bytes = 0, row_bytes = 0, copied_bytes = 0;
    std::uint64_t generation = 0, runtime_generation = 0;
    /** @brief Every submitted operation already drained; free backing before claim. */
    ~ImmutableNativeDeviceStorageState()
    {
        if (backend && (allocation || event))
        {
            if (backend->deviceRuntimeGeneration(device.gpu_ordinal()) != runtime_generation)
            {
                LOG_ERROR("Immutable native backing outlived its authenticated runtime generation");
                std::terminate();
            }
            if (event) backend->destroyEvent(event, device.gpu_ordinal());
            if (allocation) backend->free(allocation, device.gpu_ordinal());
        }
    }
};
/** @brief Exactly one reused ordinary-loader host-slot claim and pinned owner. */
struct ImmutableNativeUploadStagingState
{
    std::shared_ptr<PhysicalMemoryAuthority> authority;
    PhysicalMemoryAllocationLease lease;
    std::shared_ptr<PinnedHostTransferBuffer> buffer;
    DeviceId device = DeviceId::invalid();
    /** @brief Pinned bytes retire before the matching temporary PMA claim. */
    ~ImmutableNativeUploadStagingState() { buffer.reset(); }
};
ImmutableNativeDeviceUpload::ImmutableNativeDeviceUpload(std::unique_ptr<ImmutableNativeDeviceStorageState> s)
    : state_(std::move(s)) {}
ImmutableNativeDeviceUpload::~ImmutableNativeDeviceUpload() = default;
ImmutableNativeDeviceUpload::ImmutableNativeDeviceUpload(ImmutableNativeDeviceUpload &&) noexcept = default;
ImmutableNativeDeviceUpload &ImmutableNativeDeviceUpload::operator=(ImmutableNativeDeviceUpload &&) noexcept = default;
std::size_t ImmutableNativeDeviceUpload::copiedBytes() const noexcept { return state_ ? state_->copied_bytes : 0; }
ImmutableNativeUploadStaging::ImmutableNativeUploadStaging(std::unique_ptr<ImmutableNativeUploadStagingState> s)
    : state_(std::move(s)) {}
ImmutableNativeUploadStaging::~ImmutableNativeUploadStaging() = default;
std::size_t ImmutableNativeUploadStaging::sizeBytes() const noexcept { return state_->buffer->sizeBytes(); }
std::byte *ImmutableNativeUploadStaging::data() const noexcept
{ return static_cast<std::byte *>(state_->buffer->mutableData()); }
ImmutableNativeDeviceRegion::ImmutableNativeDeviceRegion(std::unique_ptr<ImmutableNativeDeviceStorageState> s)
    : state_(std::move(s)) {}
ImmutableNativeDeviceRegion::~ImmutableNativeDeviceRegion() = default;
DeviceId ImmutableNativeDeviceRegion::device() const noexcept { return state_->device; }
std::size_t ImmutableNativeDeviceRegion::sizeBytes() const noexcept { return state_->bytes; }
const std::byte *ImmutableNativeDeviceRegion::data() const noexcept
{ return static_cast<const std::byte *>(state_->allocation); }
std::uint64_t ImmutableNativeDeviceRegion::generation() const noexcept { return state_->generation; }
std::uint64_t ImmutableNativeDeviceRegion::runtimeGeneration() const noexcept { return state_->runtime_generation; }

ImmutableNativeDeviceUpload TransferEngine::beginImmutableNativeDeviceUpload(
    std::shared_ptr<PhysicalMemoryAuthority> memory, DeviceId device,
    size_t bytes, size_t row_bytes, void *stream) const
{
    requireImmutableSetup();
    IBackend *backend = device.is_gpu() ? resolveBackend(device) : nullptr;
    if (!memory || !device.is_valid() || !device.is_gpu() || !stream || !bytes || !row_bytes || bytes % row_bytes ||
        !backend || backend->backendDeviceType() != device.type ||
        !backend->deviceRuntimeGeneration(device.gpu_ordinal()))
        throw std::invalid_argument("Immutable native upload requires exact GPU/PMA/source/stream ownership");
    auto state = std::make_unique<ImmutableNativeDeviceStorageState>();
    state->authority = std::move(memory);
    // PMA authenticates the exact admitted allocator/owner before any allocation.
    state->lease = state->authority->claimNewAllocation(device,
        PhysicalMemoryOwner::ImmutableTableStorage, bytes);
    state->device = device; state->bytes = bytes; state->row_bytes = row_bytes;
    state->backend = backend; state->stream = stream;
    state->runtime_generation = backend->deviceRuntimeGeneration(device.gpu_ordinal());
    state->generation = nextImmutableStorageGeneration();
    state->allocation = backend->allocate(bytes, device.gpu_ordinal());
    if (!state->allocation) throw std::runtime_error("Immutable native device allocation failed");
    state->event = backend->createEvent(device.gpu_ordinal());
    if (!state->event) throw std::runtime_error("Immutable native setup event allocation failed");
    return ImmutableNativeDeviceUpload(std::move(state));
}
std::unique_ptr<ImmutableNativeUploadStaging> TransferEngine::prepareImmutableNativeUploadStaging(
    const ReleasedWeightLoadStaging &receipt, std::shared_ptr<PhysicalMemoryAuthority> memory,
    DeviceId device, size_t row_bytes) const
{
    requireImmutableSetup(); receipt.requireFor(memory, device, row_bytes);
    auto state = std::make_unique<ImmutableNativeUploadStagingState>();
    state->authority = std::move(memory); state->device = device;
    state->lease = state->authority->claimNewAllocation(receipt.hostDevice(),
        PhysicalMemoryOwner::WeightLoadStaging, receipt.slotBytes());
    state->buffer = allocatePinnedHostBuffer(receipt.slotBytes(), device);
    return std::unique_ptr<ImmutableNativeUploadStaging>(new ImmutableNativeUploadStaging(std::move(state)));
}
void TransferEngine::uploadImmutableNativeChunk(ImmutableNativeDeviceUpload &upload,
    const ImmutableNativeUploadStaging &staging, size_t bytes, void *stream) const
{
    requireImmutableSetup();
    auto *s = upload.state_.get();
    if (!s || !stream || stream != s->stream || !bytes || bytes % s->row_bytes ||
        bytes > staging.sizeBytes() || bytes > s->bytes - s->copied_bytes ||
        staging.state_->device != s->device || staging.state_->authority.get() != s->authority.get() ||
        resolveBackend(s->device) != s->backend ||
        s->backend->deviceRuntimeGeneration(s->device.gpu_ordinal()) != s->runtime_generation)
        throw std::invalid_argument("Immutable native upload chunk has foreign ownership or incomplete row bounds");
    bool copied = false;
    std::exception_ptr failure;
    try
    {
        copied = s->backend->hostToDeviceOnStream(
            static_cast<std::byte *>(s->allocation) + s->copied_bytes,
            staging.data(), bytes, s->device.gpu_ordinal(), stream);
    }
    catch (...) { failure = std::current_exception(); }
    // Even an unsuccessful enqueue may follow previously submitted native work.
    // Never recycle host bytes or release backing without its exact setup edge.
    bool drained=false;
    try
    {
        drained=s->backend->recordEvent(s->event,s->device.gpu_ordinal(),stream) &&
            waitForEventWithProxy(s->backend,s->event,s->device.gpu_ordinal(),s->device);
    }
    catch (...)
    {
        LOG_ERROR("Immutable native setup event retirement threw before safe backing release");
        std::terminate();
    }
    if (!drained)
    {
        LOG_ERROR("Immutable native setup upload could not retire its exact copy event");
        std::terminate();
    }
    if (failure) std::rethrow_exception(failure);
    if (!copied) throw std::runtime_error("Immutable native setup chunk upload failed");
    s->copied_bytes += bytes;
}
std::shared_ptr<const ImmutableNativeDeviceRegion> TransferEngine::sealImmutableNativeDeviceUpload(
    ImmutableNativeDeviceUpload &&upload) const
{
    requireImmutableSetup();
    auto *s = upload.state_.get();
    if (!s || s->copied_bytes != s->bytes || resolveBackend(s->device) != s->backend ||
        s->backend->deviceRuntimeGeneration(s->device.gpu_ordinal()) != s->runtime_generation)
        throw std::logic_error("Immutable native storage cannot publish missing or stale source bytes");
    return std::shared_ptr<const ImmutableNativeDeviceRegion>(
        new ImmutableNativeDeviceRegion(std::move(upload.state_)));
}
ImmutableNativeDeviceRead TransferEngine::acquireImmutableNativeDeviceInput(
    std::shared_ptr<const ImmutableNativeDeviceRegion> owner, DeviceId device, void *stream) const
{
    requireImmutableSetup();
    if (!owner || !stream || !device.is_valid() || !device.is_gpu() || owner->device() != device)
        throw std::invalid_argument("Immutable native read requires its exact GPU/consumer stream");
    auto &s = *owner->state_;
    if (resolveBackend(device) != s.backend ||
        s.backend->deviceRuntimeGeneration(device.gpu_ordinal()) != s.runtime_generation ||
        !s.backend->streamWaitEvent(stream, s.event, device.gpu_ordinal()))
        throw std::runtime_error("Immutable native consumer could not acquire its setup producer event");
    return ImmutableNativeDeviceRead(std::move(owner), stream);
}
void TransferEngine::requireImmutableNativeDeviceInput(const ImmutableNativeDeviceRead &read,
    DeviceId device, void *stream) const
{
    const auto &owner = read.owner();
    if (!owner || !stream || read.consumerStream() != stream || owner->device() != device ||
        resolveBackend(device) != owner->state_->backend ||
        owner->state_->backend->deviceRuntimeGeneration(device.gpu_ordinal()) != owner->runtimeGeneration())
        throw std::logic_error("Immutable native read lost exact captured device/stream/runtime identity");
}
}
