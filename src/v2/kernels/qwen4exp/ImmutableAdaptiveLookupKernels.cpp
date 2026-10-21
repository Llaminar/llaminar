/**
 * @file ImmutableAdaptiveLookupKernels.cpp
 * @brief Checked host enqueue boundary for canonical adaptive cache/frame/source ownership.
 *
 * All storage is prepared before capture. The existing root seal and canonical
 * controller reject released/rebound/foreign storage. Setup writes publish their
 * exact stream through TransferEngine; captured writes use actual enclosing
 * stage contracts/publication. No allocation, readback or private
 * byte ledger is introduced by this component.
 */
#include "AdaptiveLookupPreparedStorage.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp
{
namespace
{
/** @brief Reject null/default streams before any capture/native operation. */
void requireStream(void *stream)
{
    if (!stream || reinterpret_cast<std::uintptr_t>(stream) <= 2)
        throw std::invalid_argument("Adaptive native lookup requires its exact non-default stream");
}
/** @brief One checked operand span; no physical accounting or payload observation. */
struct Span
{
    std::uintptr_t begin, end;
};
/** @return Checked pointer bounds, rejecting overflow before native access. */
Span span(const void *p, std::size_t bytes, std::size_t alignment)
{
    const auto n = reinterpret_cast<std::uintptr_t>(p);
    if (!p || n % alignment || bytes > std::numeric_limits<std::uintptr_t>::max() - n)
        throw std::invalid_argument("Adaptive native operand is missing, misaligned or overflowing");
    return {n, n + bytes};
}
/** @brief Reject distinct mutable/control authorities sharing a physical span. */
void disjoint(Span a, Span b)
{
    if (a.begin < b.end && b.begin < a.end)
        throw std::invalid_argument("Adaptive native operands alias independent authorities");
}
/** @brief Reject missing, misaligned or overlapping actual component roots before any setup enqueue. */
void validateRoots(const AdaptiveNativeParameters &p, AdaptiveLookupDeviceStorage s)
{
    const auto frame = span(s.frame, p.frame.bytes, 64);
    const auto protocol = span(s.protocol, sizeof(ImmutableLookupDeviceState), 64);
    disjoint(frame, protocol);
    if (p.slots)
    {
        const auto cache = span(s.cache, p.cache.bytes, 64);
        disjoint(cache, frame);
        disjoint(cache, protocol);
    }
    else if (s.cache)
        throw std::invalid_argument("Zero-slot adaptive owner cannot bind an optional cache root");
}
} // namespace
AdaptiveLookupPreparedStorage::AdaptiveLookupPreparedStorage(AdaptiveImmutableRowCacheNativePlan plan,
                                                             std::shared_ptr<TensorBase> cache,
                                                             std::shared_ptr<TensorBase> frame,
                                                             std::shared_ptr<TensorBase> protocol)
    : plan_(plan), cache_(std::move(cache)), frame_(std::move(frame)), protocol_(std::move(protocol))
{
    declarations();
    std::size_t n = 0;
    if (cache_)
        roots_[n++] = cache_.get();
    roots_[n++] = frame_.get();
    roots_[n++] = protocol_.get();
    prepared_.emplace(plan_.parameters().device, std::span<TensorBase *const>(roots_.data(), n));
}
void AdaptiveLookupPreparedStorage::declarations() const
{
    const auto &p = plan_.parameters();
    const auto root = [](const std::shared_ptr<TensorBase> &t, TensorType type, std::size_t bytes)
    {
        return t && !t->is_view() && t->transferStorageOwner() == t.get() && t->native_type() == type &&
               t->size_bytes() == bytes;
    };
    if (static_cast<bool>(cache_) != (p.slots > 0) ||
        (cache_ && !root(cache_, TensorType::INT8, p.cache.bytes)) ||
        !root(frame_, TensorType::INT8, p.frame.bytes) ||
        !root(protocol_, TensorType::INT32, sizeof(ImmutableLookupDeviceState)) ||
        frame_.get() == protocol_.get() ||
        (cache_ && (cache_.get() == frame_.get() || cache_.get() == protocol_.get())))
        throw std::invalid_argument(
            "Adaptive native cache/frame/protocol require exact distinct canonical roots");
}
void AdaptiveLookupPreparedStorage::seal()
{
    std::lock_guard lock(setup_mutex_);
    declarations();
    validateRoots(plan_.parameters(), native());
    prepared_->seal();
}
void AdaptiveLookupPreparedStorage::requireFor(const AdaptiveImmutableRowCacheNativePlan &plan) const
{
    std::lock_guard lock(setup_mutex_);
    requireForUnlocked(plan);
}
void AdaptiveLookupPreparedStorage::requireForUnlocked(const AdaptiveImmutableRowCacheNativePlan &plan) const
{
    if (plan != plan_)
        throw std::invalid_argument("Adaptive native source/geometry changed");
    declarations();
    prepared_->validate();
    validateRoots(plan_.parameters(), native());
}
AdaptiveLookupDeviceStorage AdaptiveLookupPreparedStorage::native() const
{
    return {cache_ ? static_cast<std::byte *>(cache_->gpu_data_ptr()) : nullptr,
            static_cast<std::byte *>(frame_->gpu_data_ptr()),
            static_cast<ImmutableLookupDeviceState *>(protocol_->gpu_data_ptr())};
}
std::uint64_t AdaptiveLookupPreparedStorage::captureIdentity() const
{
    std::lock_guard lock(setup_mutex_);
    declarations();
    validateRoots(plan_.parameters(), native());
    const auto physical = prepared_->preparedIdentity();
    std::uint64_t h = plan_.captureIdentity();
    h ^= physical;
    h *= 1099511628211ULL;
    return h ? h : 1;
}
std::uint64_t
IImmutableAdaptiveLookupKernels::captureIdentity(const AdaptiveLookupPreparedStorage &storage) const
{
    storage.requireFor(plan());
    source().requireFor(plan().parameters());
    source().requireStorage(storage);
    std::uint64_t h = storage.captureIdentity();
    h ^= source().captureIdentity();
    h *= 1099511628211ULL;
    h ^= nativeCaptureIdentity();
    h *= 1099511628211ULL;
    return h ? h : 1;
}
void IImmutableAdaptiveLookupKernels::initialize(AdaptiveLookupPreparedStorage &storage, void *stream) const
{
    requireStream(stream);
    std::lock_guard lock(storage.setup_mutex_);
    storage.requireForUnlocked(plan());
    source().requireFor(plan().parameters());
    source().requireStorage(storage);
    if (storage.setup_.load(std::memory_order_acquire) != AdaptiveLookupPreparedStorage::Setup::Declared ||
        isGraphCaptureActive())
        throw std::logic_error("Adaptive cache initialization is a once-only ordinary setup operation");
    // Mark failed before the first possible native enqueue: a partial submission
    // or failed coherence publication may never reset/retry warmed metadata.
    storage.setup_ = AdaptiveLookupPreparedStorage::Setup::Failed;
    initializeNative(storage.native(), stream);
    auto &t = TransferEngine::instance();
    if (storage.cache_)
        t.publishDeviceWrite(storage.cache_.get(), plan().parameters().device, stream);
    t.publishDeviceWrite(storage.frame_.get(), plan().parameters().device, stream);
    storage.setup_ = AdaptiveLookupPreparedStorage::Setup::MetadataSubmitted;
}
void IImmutableAdaptiveLookupKernels::admit(
    AdaptiveLookupPreparedStorage &storage, std::uint64_t generation,
    ImmutableLookupAdmissionArgs::ProtocolInitialization initialization, void *stream) const
{
    requireStream(stream);
    std::lock_guard lock(storage.setup_mutex_);
    storage.requireForUnlocked(plan());
    source().requireFor(plan().parameters());
    source().requireStorage(storage);
    const auto setup = storage.setup_.load(std::memory_order_acquire);
    const bool first = setup == AdaptiveLookupPreparedStorage::Setup::MetadataSubmitted;
    const bool reused = setup == AdaptiveLookupPreparedStorage::Setup::Admitted;
    if ((!first && !reused) || isGraphCaptureActive() || !generation ||
        (first && initialization != ImmutableLookupAdmissionArgs::ProtocolInitialization::InitializeOnce) ||
        (reused && initialization != ImmutableLookupAdmissionArgs::ProtocolInitialization::Preserve))
        throw std::invalid_argument(
            "Adaptive native admission requires initialized original roots; reset preserves warmed state");
    storage.setup_ = AdaptiveLookupPreparedStorage::Setup::Failed;
    admitNative(storage.native(), generation, initialization, stream);
    TransferEngine::instance().publishDeviceWrite(storage.protocol_.get(), plan().parameters().device,
                                                  stream);
    storage.setup_ = AdaptiveLookupPreparedStorage::Setup::Admitted;
}
void IImmutableAdaptiveLookupKernels::gather(AdaptiveLookupPreparedStorage &storage,
                                             const AdaptiveLookupQuery &q,
                                             const DeviceGenerationControlBinding &control,
                                             void *stream) const
{
    requireStream(stream);
    storage.requireFor(plan());
    const auto setup = storage.setup_.load(std::memory_order_acquire);
    if (setup != AdaptiveLookupPreparedStorage::Setup::Admitted)
    {
        // Cold materialization records ONLY the original permanent addresses.
        // The actual executor ledger must name this exact protocol as an
        // address-only external of its current stage/device/stream. Native
        // capture alone, an unrelated ledger or an ordinary ready-byte graph
        // cannot manufacture request admission. Initialization still publishes
        // cache/frame metadata; only real admit produces protocol bytes.
        auto *ledger = currentGraphCaptureDependencyLedger();
        if (setup != AdaptiveLookupPreparedStorage::Setup::MetadataSubmitted ||
            !isGraphCaptureActive() || !ledger ||
            ledger->classifyInput(storage.protocol_.get(), plan().parameters().device, stream) !=
                GraphCaptureDependencyLedger::InputDisposition::SetupAddressOnlyExternal)
            throw std::logic_error(
                "Adaptive native gather requires admission or its exact setup-only protocol frontier");
    }
    const auto &p = plan().parameters();
    source().requireFor(p);
    source().requireStorage(storage);
    control.requireFor(p.device);
    const auto identity = q.token_rows.captureIdentity();
    if (identity.first_physical_row != 0 || identity.repeats_per_row != 1 ||
        identity.physical_rows != identity.count_capacity || identity.count_capacity <= 0 ||
        static_cast<std::uint64_t>(identity.physical_rows) > p.maximum_ids / 16)
        throw std::invalid_argument(
            "Adaptive native query requires the complete original nonrepeated counted prefix");
    const auto ids = static_cast<std::size_t>(identity.physical_rows) * 16;
    if (q.id_capacity < ids || q.output_elements < ids * static_cast<unsigned>(p.row_elements))
        throw std::invalid_argument("Adaptive native query output/input capacity is too small");
    const auto s = storage.native();
    const Span ranges[] = {span(q.ids, ids * 8, 8),
                           span(q.output, ids * static_cast<unsigned>(p.row_elements) * 4, 4),
                           span(s.frame, p.frame.bytes, 64),
                           span(s.protocol, sizeof(ImmutableLookupDeviceState), 64),
                           span(control.healthy(), sizeof(int), 4),
                           span(control.error(), sizeof(int), 4)};
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < i; ++j)
            disjoint(ranges[i], ranges[j]);
    if (s.cache)
    {
        const auto cache = span(s.cache, p.cache.bytes, 64);
        for (const auto &r : ranges)
            disjoint(cache, r);
    }
    if (q.token_rows.countOwner())
    {
        const auto count = span(q.token_rows.countOwner(), sizeof(std::int32_t), 4);
        for (const auto &r : ranges)
            disjoint(count, r);
        if (s.cache)
            disjoint(count, span(s.cache, p.cache.bytes, 64));
    }
    const auto controller = span(control.owner()->gpu_data_ptr(), control.owner()->size_bytes(), 8);
    for (unsigned i = 0; i < 4; ++i)
        disjoint(controller, ranges[i]);
    if (s.cache)
        disjoint(controller, span(s.cache, p.cache.bytes, 64));
    const auto &c = source().deviceView();
    const Span mapped[] = {
        span(c.base.request, sizeof(ImmutableLookupRequest), 64),
        span(c.base.completion, sizeof(ImmutableLookupCompletion), 64),
        span(c.base.acknowledgement, sizeof(ImmutableLookupAcknowledgement), 64),
        span(c.base.ids, p.maximum_ids * std::size_t{8}, 8),
        span(c.intents, p.maximum_ids * sizeof(ImmutableRowSubordinateIntent), 8),
        span(c.acknowledgements, p.maximum_ids * sizeof(ImmutableRowInvalidationAcknowledgement), 8),
        span(c.base.nativeRows, p.maximum_ids * p.row_bytes, 4)};
    for (const auto &m : mapped)
    {
        for (const auto &r : ranges)
            disjoint(m, r);
        disjoint(m, controller);
        if (s.cache)
            disjoint(m, span(s.cache, p.cache.bytes, 64));
    }
    gatherNative({p, s, q, c, control.healthy(), control.error(), control.row()}, stream);
}
void IImmutableAdaptiveLookupKernels::close(AdaptiveLookupPreparedStorage &storage, void *stream) const
{
    requireStream(stream);
    storage.requireFor(plan());
    source().requireFor(plan().parameters());
    source().requireStorage(storage);
    if (storage.setup_.load(std::memory_order_acquire) != AdaptiveLookupPreparedStorage::Setup::Admitted)
        throw std::logic_error("Adaptive terminal has no initialized original protocol admission");
    closeNative(storage.native(), nullptr, nullptr, stream);
}
void IImmutableAdaptiveLookupKernels::closeWhenTerminal(AdaptiveLookupPreparedStorage &storage,
                                                        const DeviceGenerationControlBinding &control,
                                                        void *stream) const
{
    requireStream(stream);
    storage.requireFor(plan());
    source().requireFor(plan().parameters());
    source().requireStorage(storage);
    control.requireFor(plan().parameters().device);
    if (storage.setup_.load(std::memory_order_acquire) != AdaptiveLookupPreparedStorage::Setup::Admitted)
        throw std::logic_error("Adaptive terminal has no initialized original protocol admission");
    const auto s = storage.native();
    const auto controller = span(control.owner()->gpu_data_ptr(), control.owner()->size_bytes(), 8);
    disjoint(controller, span(s.frame, plan().parameters().frame.bytes, 64));
    disjoint(controller, span(s.protocol, sizeof(ImmutableLookupDeviceState), 64));
    if (s.cache)
        disjoint(controller, span(s.cache, plan().parameters().cache.bytes, 64));
    closeNative(s, control.healthy(), control.row() + sampling_math::kDeviceGenerationControlRequestComplete,
                stream);
}
} // namespace llaminar2::qwen4exp
