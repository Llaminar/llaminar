/**
 * @file PhysicalPagePool.cpp
 * @brief Materialize exact PMA page roots and convert fungible growth into private frames.
 *
 * Refusal precedes mutation. All arrays are bounded and materialized at setup;
 * assignment and rollback allocate nothing. No frame is published to a native
 * consumer in this initial slice, so rollback cannot race a submitted reader.
 */
#include "PhysicalPagePool.h"
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{
/** @brief Issue immutable setup identity; failed construction never recycles an incarnation. */
std::uint64_t issuePoolIdentity()
{
    static std::atomic<std::uint64_t> next{1};
    auto value = next.load(std::memory_order_relaxed);
    for (;;)
    {
        if (value == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Physical page pool identity exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed))
            return value;
    }
}
}

void PhysicalPagePoolGeometry::validate() const
{
    if (!frames || !frame_bytes || !request_records || !max_pages_per_request ||
        max_pages_per_request > frames)
        throw std::invalid_argument("Page pool requires positive exact geometry within frame capacity");
    if (frames > std::numeric_limits<std::size_t>::max() / frame_bytes ||
        request_records > std::numeric_limits<std::size_t>::max() / max_pages_per_request)
        throw std::overflow_error("Page pool physical or table geometry overflows");
}

std::size_t PhysicalPagePool::physicalBytes(PhysicalPagePoolGeometry g, DeviceId device, CPUStoragePlacement p)
{
    g.validate();
    std::size_t result = 0;
    const auto add = [&](std::size_t bytes) {
        if (bytes > std::numeric_limits<std::size_t>::max() - result)
            throw std::overflow_error("Page pool physical BOM overflows");
        result += bytes;
    };
    add(CPUStorageRoot<std::byte>(g.frames * g.frame_bytes, PhysicalMemoryOwner::KVCache, device, p).physicalBytes());
    add(CPUStorageRoot<Frame>(g.frames, PhysicalMemoryOwner::KVCache, device, p).physicalBytes());
    add(CPUStorageRoot<Request>(g.request_records, PhysicalMemoryOwner::KVCache, device, p).physicalBytes());
    add(CPUStorageRoot<FrameHandle>(g.request_records * g.max_pages_per_request, PhysicalMemoryOwner::KVCache, device, p).physicalBytes());
    // Bounded witness workspace: fixed request/step/refusal cells, priced once
    // in the authority BOM rather than allocated per admission.
    add(CPUStorageRoot<std::byte>(planning::witnessWorkspaceBytes(g.request_records),
        PhysicalMemoryOwner::KVCache, device, p).physicalBytes());
    return result;
}

PhysicalPagePool::PhysicalPagePool(std::shared_ptr<PhysicalMemoryAuthority> authority, DeviceId device,
    PhysicalPagePoolGeometry geometry, CPUStoragePlacement placement, std::uint64_t pool_identity)
    : authority_(authority), pool_identity_(pool_identity), geometry_(geometry)
{
    (void)physicalBytes(geometry, device, placement);
    backing_ = CPUStorageRoot<std::byte>(geometry.frames * geometry.frame_bytes,
        PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
    frames_ = CPUStorageRoot<Frame>(geometry.frames, PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
    requests_ = CPUStorageRoot<Request>(geometry.request_records, PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
    request_frames_ = CPUStorageRoot<FrameHandle>(geometry.request_records * geometry.max_pages_per_request,
        PhysicalMemoryOwner::KVCache, device, placement).allocate(authority);
    witness_cells_ = CPUStorageRoot<std::byte>(planning::witnessWorkspaceBytes(geometry.request_records),
        PhysicalMemoryOwner::KVCache, device, placement, CPUStorageInitialization::Uninitialized).allocate(authority);
    // Bind the non-owning workspace over the exact cell layout:
    // request cells, then step cells, then the refusal buffer.
    const std::size_t request_cells = geometry.request_records + 1;
    std::byte *cursor = witness_cells_.data();
    auto *requests = reinterpret_cast<planning::WitnessRequest *>(cursor);
    cursor += request_cells * sizeof(planning::WitnessRequest);
    auto *steps = reinterpret_cast<planning::WitnessStep *>(cursor);
    cursor += (request_cells * 2 + 1) * sizeof(planning::WitnessStep);
    auto *refusal = reinterpret_cast<char *>(cursor);
    witness_workspace_ = planning::WitnessWorkspace(
        {requests, request_cells}, {steps, request_cells * 2 + 1},
        {refusal, planning::kWitnessRefusalBytes});
}

std::shared_ptr<PhysicalPagePool> PhysicalMemoryAuthority::createCPUPagePool(DeviceId device,
    PhysicalPagePoolGeometry geometry, const CPUStoragePlacement &placement)
{
    if (!device.is_cpu() || !contains(device))
        throw std::invalid_argument("CPU page pool requires an original admitted local CPU allocator");
    return std::shared_ptr<PhysicalPagePool>(new PhysicalPagePool(shared_from_this(), device, geometry, placement, issuePoolIdentity()));
}

bool PhysicalPagePool::tryAdmitTogether(std::span<const Demand> demands,
    std::span<std::optional<Grant>> results)
{
    if (demands.empty() || demands.size() != results.size())
        throw std::invalid_argument("Page transaction requires exact nonempty demand/output geometry");
    // Validate every original input before issuing a grant or touching output.
    for (std::size_t i = 0; i < demands.size(); ++i)
    {
        const auto &demand = demands[i];
        if (results[i] || !demand.pool || !demand.pages ||
            demand.pages > demand.pool->geometry_.max_pages_per_request ||
            demand.pool->authority_ != demands.front().pool->authority_)
            throw std::invalid_argument("Page transaction has nonempty output, invalid demand or foreign PMA");
        for (std::size_t prior = 0; prior < i; ++prior)
            if (demand.pool == demands[prior].pool)
                throw std::invalid_argument("Page transaction repeats a storage owner instead of a combined demand");
    }
    // CPU authority is single runner-thread owned. No published views or
    // native consumers exist in this interface; partial grants are invisible
    // until this call returns, and their RAII rollback preserves old owners.
    try
    {
        for (std::size_t i = 0; i < demands.size(); ++i)
        {
            results[i] = demands[i].pool->tryAdmit(demands[i].pages);
            if (!results[i])
            {
                for (auto &result : results) result.reset();
                return false;
            }
        }
    }
    catch (...)
    {
        for (auto &result : results) result.reset();
        throw;
    }
    return true;
}

void PhysicalPagePool::fillWitnessRequests(std::span<planning::WitnessRequest> out) const
{
    // Conservatively treat every outstanding finite grant (G = remaining) as
    // already occupied: admission must prove the worst case where all grants
    // convert to frames, never only the currently assigned O subset. Fixed
    // output span: no per-admission heap allocation. The span is zeroed first
    // because the workspace is reused across admissions; stale entries must
    // never be interpreted as live requests.
    for (auto &cell : out) cell = {};
    std::size_t count = 0;
    for (std::size_t record = 0; record < requests_.size() && count < out.size(); ++record)
    {
        const auto &request = requests_[record];
        if (!request.live) continue;
        out[count++] = {.generation=request.generation, .record=record,
            .peak_frames=request.declared_peak,
            .occupied_frames=request.assigned + request.remaining,
            .preserved=false};
    }
}

std::vector<planning::WitnessRequest> PhysicalPagePool::liveWitnessRequests() const
{
    std::vector<planning::WitnessRequest> result;
    result.reserve(requests_.size());
    for (std::size_t record = 0; record < requests_.size(); ++record)
    {
        const auto &request = requests_[record];
        if (!request.live) continue;
        result.push_back({.generation=request.generation, .record=record,
            .peak_frames=request.declared_peak,
            .occupied_frames=request.assigned + request.remaining,
            .preserved=false});
    }
    return result;
}

planning::WitnessBaseOccupancy PhysicalPagePool::physicalBaseOccupancy() const
{
    // Canonical physical count, one frame exactly once: every frame that is
    // occupied outside a live request record is either shared (immutable,
    // retained by its reference group) or retiring (pinned, awaiting the last
    // borrower). Aliases are logical, never a second frame.
    std::size_t shared = 0, retiring = 0;
    for (const auto &frame : frames_)
    {
        if (frame.shared) { ++shared; continue; }
        if (frame.retiring && (frame.pins || frame.readers_closed)) ++retiring;
    }
    return {.shared_frames=shared, .retiring_frames=retiring};
}

std::optional<PhysicalPagePool::Grant> PhysicalPagePool::tryAdmitWithWitness(
    std::size_t declared_peak, std::size_t initial_grant)
{
    if (!declared_peak || declared_peak > geometry_.max_pages_per_request)
        throw std::invalid_argument("Witness peak is outside the declared per-request bound");
    if (!initial_grant || initial_grant > declared_peak)
        throw std::invalid_argument("Initial witness grant must be positive and within the declared peak");
    // Proof before mutation using the fixed workspace: the proposed live set
    // must have a completion order. The new request enters with its finite
    // initial grant as occupied because that grant converts to O immediately.
    auto &workspace = witness_workspace_;
    fillWitnessRequests(workspace.requests());
    std::size_t live = workspace.requests().size();
    for (std::size_t i = 0; i < workspace.requests().size(); ++i)
        if (!workspace.requests()[i].peak_frames) { live = i; break; }
    workspace.requests()[live] =
        {.generation=0, .peak_frames=declared_peak, .occupied_frames=initial_grant, .preserved=false};
    if (!planning::proveCompletionOrderInto(workspace, geometry_.frames,
        physicalBaseOccupancy(), workspace.requests().first(live + 1)))
        return std::nullopt;
    // Conservation under the INFINITE-GRANT model is the finite-operation
    // sum; the witness bounds the peak promise, never the converted grant.
    if (initial_grant > geometry_.frames - protected_pages_)
        return std::nullopt;
    for (std::size_t i = 0; i < requests_.size(); ++i)
    {
        auto &request = requests_[i];
        if (request.live) continue;
        if (request.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Page grant generation exhausted");
        auto self = shared_from_this();
        ++request.generation;
        request.declared_peak = declared_peak;
        request.remaining = initial_grant;
        request.assigned = 0;
        request.live = true;
        protected_pages_ += initial_grant;
        return Grant(std::move(self), i, request.generation);
    }
    return std::nullopt;
}

bool PhysicalPagePool::extendWithWitness(Grant &grant, std::size_t pages)
{
    (void)require(grant);
    auto &request = requests_[grant.record_];
    if (!pages || request.remaining > geometry_.max_pages_per_request - pages ||
        request.assigned + request.remaining + pages > request.declared_peak)
        throw std::logic_error("Witness extension exceeds the declared peak");
    if (pages > geometry_.frames - protected_pages_)
        return false;
    // Revalidate the PROPOSED state in the fixed workspace: this request's
    // outstanding grant grows by `pages`; every other live request keeps its
    // current committed occupancy.
    auto &workspace = witness_workspace_;
    fillWitnessRequests(workspace.requests());
    std::size_t live = workspace.requests().size();
    for (std::size_t i = 0; i < workspace.requests().size(); ++i)
        if (!workspace.requests()[i].peak_frames) { live = i; break; }
    for (std::size_t i = 0; i < live; ++i)
        if (workspace.requests()[i].record == grant.record_)
            workspace.requests()[i].occupied_frames += pages;
    if (!planning::proveCompletionOrderInto(workspace, geometry_.frames,
        physicalBaseOccupancy(), workspace.requests().first(live)))
        return false;
    request.remaining += pages;
    protected_pages_ += pages;
    return true;
}

std::optional<PhysicalPagePool::Grant> PhysicalPagePool::tryAdmit(std::size_t pages)
{
    if (!pages || pages > geometry_.max_pages_per_request)
        throw std::invalid_argument("Page commitment is outside the declared per-request bound");
    if (pages > geometry_.frames - protected_pages_)
        return std::nullopt;
    for (std::size_t i = 0; i < requests_.size(); ++i)
    {
        auto &request = requests_[i];
        if (request.live) continue;
        if (request.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Page grant generation exhausted");
        // Acquire retained ownership before mutating the request record.
        auto self = shared_from_this();
        ++request.generation;
        request.declared_peak = pages;
        request.remaining = pages;
        request.assigned = 0;
        request.live = true;
        protected_pages_ += pages;
        return Grant(std::move(self), i, request.generation);
    }
    return std::nullopt;
}

const PhysicalPagePool::Request &PhysicalPagePool::require(const Grant &grant) const
{
    if (grant.pool_.get() != this || grant.record_ >= requests_.size())
        throw std::logic_error("Page grant belongs to another authority");
    const auto &request = requests_[grant.record_];
    if (!request.live || request.generation != grant.generation_)
        throw std::logic_error("Page grant is stale or retired");
    return request;
}

PhysicalPagePool::FrameHandle PhysicalPagePool::assign(const Grant &grant)
{
    const auto &original = require(grant);
    if (!original.remaining)
        throw std::logic_error("Page assignment exceeds the protected completion commitment");
    for (std::size_t i = 0; i < frames_.size(); ++i)
    {
        auto &frame = frames_[i];
        if (!assignable(frame)) continue;
        if (frame.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Physical page incarnation exhausted");
        auto &request = requests_[grant.record_];
        ++frame.generation;
        frame.record = grant.record_;
        frame.request_generation = grant.generation_;
        frame.return_commitment = false;
        frame.valid_bytes = 0;
        frame.committed_version = 0;
        frame.readers_closed = false;
        frame.owned = true;
        frame.shared = false;
        frame.shared_references = 0;
        const FrameHandle handle{pool_identity_, i, frame.generation};
        request_frames_[grant.record_ * geometry_.max_pages_per_request + request.assigned] = handle;
        ++request.assigned;
        --request.remaining;
        return handle;
    }
    throw std::logic_error("Protected page growth has no free physical frame");
}

void PhysicalPagePool::release(const Grant &grant) noexcept
{
    try
    {
        const auto &request = require(grant);
        for (const auto &handle : grant.assignedFrames())
        {
            auto &frame = frames_[handle.index];
            if (handle.pool_identity != pool_identity_ || (!frame.owned && !frame.shared) ||
                frame.record != grant.record_ || frame.request_generation != grant.generation_ || frame.generation != handle.generation)
                std::terminate();
            if (frame.shared) continue; // Shared frames retire only with their last shared reference.
            if (frame.pins)
                frame.retiring = true;
            else
            {
                frame.owned = false;
                --protected_pages_;
            }
        }
        protected_pages_ -= request.remaining;
        requests_[grant.record_].live = false;
    }
    catch (...) { std::terminate(); }
}

PhysicalPagePool::FramePin PhysicalPagePool::pin(const Grant &grant, FrameHandle handle)
{
    (void)require(grant);
    if (handle.pool_identity != pool_identity_ || handle.index >= frames_.size())
        throw std::logic_error("Private frame pin belongs to a foreign pool");
    auto &frame = frames_[handle.index];
    if (!frame.owned || frame.retiring || frame.writer || frame.record != grant.record_ ||
        frame.request_generation != grant.generation_ || frame.generation != handle.generation)
        throw std::logic_error("Private frame pin is stale, retiring or owned by another request");
    if (frame.pins == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Private frame pin count exhausted");
    auto self = shared_from_this();
    ++frame.pins;
    return FramePin(std::move(self), handle);
}

void PhysicalPagePool::unpin(FrameHandle handle) noexcept
{
    if (handle.pool_identity != pool_identity_ || handle.index >= frames_.size())
        std::terminate();
    auto &frame = frames_[handle.index];
    // A finally retiring shared frame is neither owned nor shared while its
    // last publication pin remains; identity and pin count still authenticate.
    if (frame.generation != handle.generation || !frame.pins ||
        (frame.retiring ? false : (!frame.owned && !frame.shared)))
        std::terminate();
    --frame.pins;
    if (!frame.pins && frame.retiring && frame.owned) reclaim(frame);
    else if (!frame.pins && frame.retiring && !frame.owned && !frame.shared)
    {
        if (!protected_pages_) std::terminate();
        --protected_pages_; // Final shared-frame retirement after the last pin drops.
        frame.retiring = false;
    }
}

void PhysicalPagePool::reclaim(Frame &frame) noexcept
{
    if (!frame.owned || frame.shared || !frame.retiring || frame.pins || frame.writer || !protected_pages_)
        std::terminate();
    auto &original = requests_[frame.record];
    if (frame.return_commitment && original.live && original.generation == frame.request_generation)
    {
        if (original.remaining >= geometry_.max_pages_per_request) std::terminate();
        ++original.remaining; // Occupied -> unconverted promise: protected_pages_ is unchanged.
    }
    else
        --protected_pages_; // The original request retired; no later incarnation receives its promise.
    frame.owned = false;
    frame.retiring = false;
    frame.return_commitment = false;
}

void PhysicalPagePool::discard(const Grant &grant, FrameHandle handle)
{
    (void)require(grant);
    auto &frame = requireFrame(handle);
    if (frame.shared)
        throw std::logic_error("Page discard cannot retire a shared immutable frame");
    if (frame.record != grant.record_ || frame.request_generation != grant.generation_ || frame.retiring || frame.writer)
        throw std::logic_error("Page discard lacks its original private request owner");
    auto &request = requests_[grant.record_];
    const size_t base = grant.record_ * geometry_.max_pages_per_request;
    size_t ordinal = 0;
    while (ordinal < request.assigned && request_frames_[base + ordinal] != handle) ++ordinal;
    if (ordinal == request.assigned) throw std::logic_error("Page discard has no original frame grant");
    // Remove only allocation ownership metadata. A versioned native KV table
    // separately owns logical order and its readers still pin the old referent.
    for (size_t next = ordinal + 1; next < request.assigned; ++next)
        request_frames_[base + next - 1] = request_frames_[base + next];
    --request.assigned;
    frame.retiring = true;
    frame.return_commitment = true;
    frame.readers_closed = true;
    if (!frame.pins) reclaim(frame);
}

PhysicalPagePool::Frame &PhysicalPagePool::requireFrame(FrameHandle handle)
{
    if (handle.pool_identity != pool_identity_ || handle.index >= frames_.size())
        throw std::logic_error("Page access has a foreign pool or frame");
    auto &frame = frames_[handle.index];
    if ((!frame.owned && !frame.shared) || frame.generation != handle.generation)
        throw std::logic_error("Page access has a stale frame incarnation");
    return frame;
}

std::optional<PhysicalPagePool::AppendPlan> PhysicalPagePool::prepareAppend(
    const Grant &grant, FrameHandle handle, std::size_t bytes)
{
    (void)require(grant);
    auto &frame = requireFrame(handle);
    if (frame.shared || frame.record != grant.record_ || frame.request_generation != grant.generation_ ||
        frame.retiring || !bytes || bytes > geometry_.frame_bytes - frame.valid_bytes)
        throw std::invalid_argument("Page append lacks private ownership or exceeds its exact writable extent");
    if (frame.writer || frame.pins) return std::nullopt;
    if (frame.next_content_version == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Page content version exhausted");
    auto self = shared_from_this();
    ++frame.next_content_version;
    frame.writer = true;
    frame.readers_closed = true;
    frame.pins = 1;
    return AppendPlan(std::move(self), handle, frame.next_content_version, frame.valid_bytes, bytes);
}

PhysicalPagePool::AppendPlan::AppendPlan(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle,
    std::uint64_t version, std::size_t start, std::size_t count)
    : pool_(std::move(pool)), handle_(handle), version_(version), start_(start), count_(count) {}
PhysicalPagePool::AppendPlan::~AppendPlan() { release(); }
PhysicalPagePool::AppendPlan::AppendPlan(AppendPlan &&other) noexcept
    : pool_(std::move(other.pool_)), handle_(other.handle_), version_(other.version_),
      start_(other.start_), count_(other.count_), copied_(other.copied_) {}
PhysicalPagePool::AppendPlan &PhysicalPagePool::AppendPlan::operator=(AppendPlan &&other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = std::move(other.pool_); handle_ = other.handle_; version_ = other.version_;
        start_ = other.start_; count_ = other.count_; copied_ = other.copied_;
    }
    return *this;
}
void PhysicalPagePool::AppendPlan::copyFrom(std::span<const std::byte> source)
{
    if (!pool_ || copied_ || source.size() != count_)
        throw std::logic_error("Page append copy requires one complete original write extent");
    copyNext(source);
}
bool PhysicalPagePool::AppendPlan::overlaps(std::span<const std::byte> source) const
{
    if (!pool_) throw std::logic_error("Page append alias query uses an empty write plan");
    const auto &frame = pool_->requireFrame(handle_);
    if (!frame.writer || frame.retiring || frame.next_content_version != version_)
        throw std::logic_error("Page append copy has revoked or obsolete write ownership");
    const auto target = reinterpret_cast<std::uintptr_t>(pool_->backing_.data() +
        handle_.index * pool_->geometry_.frame_bytes + start_);
    const auto input = reinterpret_cast<std::uintptr_t>(source.data());
    if (source.empty()) return false;
    return input <= target ? target - input < source.size() : input - target < count_;
}
bool PhysicalPagePool::AppendPlan::destinationAligned(std::size_t alignment) const
{
    if (!alignment) throw std::invalid_argument("CPU page destination alignment must be positive");
    (void)overlaps({}); // Authenticate the original exclusive write before observing its address.
    return reinterpret_cast<std::uintptr_t>(pool_->backing_.data() +
        handle_.index * pool_->geometry_.frame_bytes + start_) % alignment == 0;
}
void PhysicalPagePool::AppendPlan::copyNext(std::span<const std::byte> source)
{
    if (!pool_ || !source.data() || source.empty() || source.size() > count_ - copied_)
        throw std::logic_error("Page append fragment exceeds its remaining original write extent");
    if (overlaps(source)) throw std::invalid_argument("Page append source aliases its exclusive destination");
    std::memcpy(pool_->backing_.data() + handle_.index * pool_->geometry_.frame_bytes + start_ + copied_,
        source.data(), source.size());
    copied_ += source.size();
}
PhysicalPagePool::CommittedPage PhysicalPagePool::AppendPlan::commit()
{
    if (!pool_ || copied_ != count_)
        throw std::logic_error("Page publication requires a completed original CPU write");
    auto &frame = pool_->requireFrame(handle_);
    if (!frame.writer || frame.retiring || frame.next_content_version != version_ || frame.valid_bytes != start_)
        throw std::logic_error("Page publication has revoked or obsolete write ownership");
    frame.valid_bytes = start_ + count_;
    frame.committed_version = version_;
    frame.writer = false;
    frame.readers_closed = false;
    // Transfer the plan's pin, without an unowned interval, to the publication.
    return CommittedPage(std::move(pool_), handle_, version_, start_ + count_);
}
void PhysicalPagePool::AppendPlan::release() noexcept
{
    if (!pool_) return;
    auto &frame = pool_->frames_[handle_.index];
    if (!frame.writer || frame.generation != handle_.generation || frame.next_content_version != version_)
        std::terminate();
    frame.writer = false;
    pool_->unpin(handle_);
    pool_.reset();
}

PhysicalPagePool::ReadAccess PhysicalPagePool::registerRead(const CommittedPage &page)
{
    auto &frame = requireFrame(page.handle_);
    if (frame.retiring || frame.readers_closed || frame.writer ||
        frame.committed_version != page.version_ || frame.valid_bytes != page.bytes_)
        throw std::logic_error("Page reader registration uses a closed, retiring or obsolete publication");
    if (frame.pins == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Page reader registration count exhausted");
    auto self = shared_from_this();
    ++frame.pins;
    return ReadAccess(std::move(self), page.handle_, page.version_, page.bytes_);
}

PhysicalPagePool::ReadAccess::ReadAccess(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle,
    std::uint64_t version, std::size_t bytes)
    : pool_(std::move(pool)), handle_(handle), version_(version), bytes_(bytes) {}
PhysicalPagePool::ReadAccess::~ReadAccess() { release(); }
PhysicalPagePool::ReadAccess::ReadAccess(ReadAccess &&other) noexcept
    : pool_(std::move(other.pool_)), handle_(other.handle_), version_(other.version_),
      bytes_(other.bytes_), phase_(other.phase_) {}
PhysicalPagePool::ReadAccess &PhysicalPagePool::ReadAccess::operator=(ReadAccess &&other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = std::move(other.pool_); handle_ = other.handle_; version_ = other.version_;
        bytes_ = other.bytes_; phase_ = other.phase_;
    }
    return *this;
}
std::span<const std::byte> PhysicalPagePool::ReadAccess::beginCPU()
{
    if (!pool_ || phase_ != Phase::Registered)
        throw std::logic_error("CPU page access must begin from its original registered lease");
    const auto &frame = pool_->requireFrame(handle_);
    if (frame.writer || frame.committed_version != version_ || frame.valid_bytes != bytes_)
        throw std::logic_error("Registered CPU page reader lost its immutable content");
    // Closure/retirement rejects new readers, never an already registered one.
    phase_ = Phase::Executing;
    return {pool_->backing_.data() + handle_.index * pool_->geometry_.frame_bytes, bytes_};
}
void PhysicalPagePool::ReadAccess::completeCPU()
{
    if (!pool_ || phase_ != Phase::Executing)
        throw std::logic_error("CPU page completion has no original executing consumer");
    phase_ = Phase::Completed;
    pool_->unpin(handle_);
    pool_.reset();
}
void PhysicalPagePool::ReadAccess::release() noexcept
{
    if (!pool_) return;
    // Destruction may cancel a not-yet-submitted registration. It is never
    // evidence that executing work completed: losing that owner fails fatally.
    if (phase_ == Phase::Executing) std::terminate();
    pool_->unpin(handle_);
    pool_.reset();
}

PhysicalPagePool::CommittedPage::CommittedPage(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle,
    std::uint64_t version, std::size_t bytes)
    : pool_(std::move(pool)), handle_(handle), version_(version), bytes_(bytes) {}
PhysicalPagePool::CommittedPage::~CommittedPage() { release(); }
PhysicalPagePool::CommittedPage::CommittedPage(CommittedPage &&other) noexcept
    : pool_(std::move(other.pool_)), handle_(other.handle_), version_(other.version_), bytes_(other.bytes_) {}
PhysicalPagePool::CommittedPage &PhysicalPagePool::CommittedPage::operator=(CommittedPage &&other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = std::move(other.pool_); handle_ = other.handle_; version_ = other.version_; bytes_ = other.bytes_;
    }
    return *this;
}
std::span<const std::byte> PhysicalPagePool::CommittedPage::read() const
{
    if (!pool_) throw std::logic_error("Page read uses an empty publication");
    const auto &frame = pool_->requireFrame(handle_);
    if (frame.writer || frame.committed_version != version_ || frame.valid_bytes != bytes_)
        throw std::logic_error("Page read uses an incomplete or obsolete content publication");
    return {pool_->backing_.data() + handle_.index * pool_->geometry_.frame_bytes, bytes_};
}
const PhysicalMemoryAuthority *PhysicalPagePool::CommittedPage::issuer() const noexcept
{
    return pool_ ? pool_->authority_.get() : nullptr;
}

PhysicalPagePool::ReadAccess PhysicalPagePool::CommittedPage::registerRead() const
{
    if (!pool_) throw std::logic_error("Page registration uses an empty publication");
    return pool_->registerRead(*this);
}

void PhysicalPagePool::CommittedPage::release() noexcept
{
    if (pool_) { pool_->unpin(handle_); pool_.reset(); }
}

PhysicalPagePool::SharedPage PhysicalPagePool::CommittedPage::share() const
{
    if (!pool_) throw std::logic_error("Page sharing uses an empty publication");
    return pool_->share(*this);
}

PhysicalPagePool::SharedPage PhysicalPagePool::share(const CommittedPage &page)
{
    if (!page.pool_) throw std::logic_error("Page sharing uses an empty publication");
    auto &frame = requireFrame(page.handle_);
    if (frame.writer || frame.retiring || frame.shared ||
        frame.committed_version != page.version_ || frame.valid_bytes != page.bytes_)
        throw std::logic_error("Page sharing requires an original private committed publication");
    auto &request = requests_[frame.record];
    const size_t base = frame.record * geometry_.max_pages_per_request;
    size_t ordinal = 0;
    while (ordinal < request.assigned && request_frames_[base + ordinal] != page.handle_) ++ordinal;
    if (ordinal == request.assigned)
        throw std::logic_error("Page sharing lost its original private frame grant");
    for (size_t next = ordinal + 1; next < request.assigned; ++next)
        request_frames_[base + next - 1] = request_frames_[base + next];
    --request.assigned;
    // O unchanged: the frame stays occupied, owned by the shared reference group
    // instead of one private request. Its grant no longer observes it.
    frame.owned = false;
    frame.shared = true;
    frame.shared_references = 1;
    return SharedPage(page.pool_, page.handle_, page.version_, page.bytes_);
}

PhysicalPagePool::SharedPage::SharedPage(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle,
    std::uint64_t version, std::size_t bytes)
    : pool_(std::move(pool)), handle_(handle), version_(version), bytes_(bytes) {}
PhysicalPagePool::SharedPage::~SharedPage() { release(); }
PhysicalPagePool::SharedPage::SharedPage(SharedPage &&other) noexcept
    : pool_(std::move(other.pool_)), handle_(other.handle_), version_(other.version_), bytes_(other.bytes_) {}
PhysicalPagePool::SharedPage &PhysicalPagePool::SharedPage::operator=(SharedPage &&other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = std::move(other.pool_); handle_ = other.handle_;
        version_ = other.version_; bytes_ = other.bytes_;
    }
    return *this;
}
std::span<const std::byte> PhysicalPagePool::SharedPage::read() const
{
    if (!pool_) throw std::logic_error("Shared page read uses an empty reference");
    const auto &frame = pool_->requireFrame(handle_);
    if (!frame.shared || frame.writer || frame.committed_version != version_ || frame.valid_bytes != bytes_)
        throw std::logic_error("Shared page read lost its immutable content");
    return {pool_->backing_.data() + handle_.index * pool_->geometry_.frame_bytes, bytes_};
}
const PhysicalMemoryAuthority *PhysicalPagePool::SharedPage::issuer() const noexcept
{
    return pool_ ? pool_->authority_.get() : nullptr;
}
PhysicalPagePool::ReadAccess PhysicalPagePool::SharedPage::registerRead() const
{
    if (!pool_) throw std::logic_error("Shared page registration uses an empty reference");
    auto &frame = pool_->requireFrame(handle_);
    if (!frame.shared || frame.retiring || frame.readers_closed || frame.writer ||
        frame.committed_version != version_ || frame.valid_bytes != bytes_)
        throw std::logic_error("Shared page registration uses a closed or obsolete publication");
    if (frame.pins == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Shared page reader registration count exhausted");
    auto self = pool_->shared_from_this();
    ++frame.pins;
    return ReadAccess(std::move(self), handle_, version_, bytes_);
}
PhysicalPagePool::SharedPage PhysicalPagePool::SharedPage::cloneForBorrow() const
{
    if (!pool_) throw std::logic_error("Shared page clone uses an empty reference");
    auto &frame = pool_->requireFrame(handle_);
    if (!frame.shared || frame.retiring ||
        frame.committed_version != version_ || frame.valid_bytes != bytes_)
        throw std::logic_error("Shared page clone uses a retired or obsolete publication");
    if (frame.shared_references == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Shared page reference count exhausted");
    ++frame.shared_references;
    return SharedPage(std::shared_ptr<PhysicalPagePool>(pool_->shared_from_this()),
        handle_, version_, bytes_);
}
void PhysicalPagePool::SharedPage::release() noexcept
{
    if (pool_) { pool_->unshare(handle_); pool_.reset(); }
}

void PhysicalPagePool::unshare(FrameHandle handle) noexcept
{
    if (handle.pool_identity != pool_identity_ || handle.index >= frames_.size())
        std::terminate();
    auto &frame = frames_[handle.index];
    if (!frame.shared || frame.generation != handle.generation || !frame.shared_references)
        std::terminate();
    --frame.shared_references;
    if (!frame.shared_references)
    {
        // Last shared reference: retire only after its last borrower pin also
        // drops; otherwise the frame stays occupied and unassignable.
        frame.shared = false;
        if (frame.pins)
            frame.retiring = true;
        else
        {
            if (!protected_pages_) std::terminate();
            --protected_pages_;
        }
    }
}

std::optional<PhysicalPagePool::SharedPage> PhysicalPagePool::adoptShared(const Grant &grant, const SharedPage &source)
{
    (void)require(grant);
    if (!source.pool_ || source.pool_.get() != this)
        throw std::logic_error("Shared page adoption requires its original pool authority");
    auto &frame = requireFrame(source.handle_);
    if (!frame.shared || frame.retiring ||
        frame.committed_version != source.version_ || frame.valid_bytes != source.bytes_)
        throw std::logic_error("Shared page adoption uses a retired or obsolete reference");
    if (frame.shared_references == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Shared page reference count exhausted");
    ++frame.shared_references;
    return SharedPage(std::shared_ptr<PhysicalPagePool>(shared_from_this()),
        source.handle_, source.version_, source.bytes_);
}

std::optional<PhysicalPagePool::COWPlan> PhysicalPagePool::prepareCOW(
    const Grant &grant, const SharedPage &source, std::size_t fresh_bytes)
{
    (void)require(grant);
    if (!source.pool_ || source.pool_.get() != this)
        throw std::logic_error("Copy-on-write requires the original pool authority");
    auto &source_frame = requireFrame(source.handle_);
    if (!source_frame.shared || source_frame.retiring ||
        source_frame.committed_version != source.version_ || source_frame.valid_bytes != source.bytes_)
        throw std::logic_error("Copy-on-write uses a retired or obsolete shared source");
    if (!fresh_bytes || source.bytes_ + fresh_bytes > geometry_.frame_bytes)
        throw std::invalid_argument("Copy-on-write fresh extent exceeds the destination frame");
    auto &request = requests_[grant.record_];
    if (!request.remaining)
        return std::nullopt; // No protected destination growth; refuse before any mutation.
    std::size_t frame_index = frames_.size();
    for (std::size_t i = 0; i < frames_.size(); ++i)
        if (assignable(frames_[i])) { frame_index = i; break; }
    if (frame_index == frames_.size())
        return std::nullopt; // No free physical frame; refuse before any mutation.
    auto &frame = frames_[frame_index];
    if (frame.generation == std::numeric_limits<std::uint64_t>::max() ||
        frame.next_content_version == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Copy-on-write identity issuance exhausted");
    // All checks passed: convert G -> O exactly once and open the write plan.
    ++frame.generation;
    frame.record = grant.record_;
    frame.request_generation = grant.generation_;
    frame.return_commitment = false;
    frame.valid_bytes = 0;
    frame.committed_version = 0;
    frame.readers_closed = true;
    frame.owned = true;
    frame.shared = false;
    frame.shared_references = 0;
    frame.retiring = false;
    ++frame.next_content_version;
    frame.writer = true;
    frame.pins = 1;
    const FrameHandle destination{pool_identity_, frame_index, frame.generation};
    request_frames_[grant.record_ * geometry_.max_pages_per_request + request.assigned] = destination;
    ++request.assigned;
    --request.remaining;
    // Keep the shared source alive through the copy, independent of the caller's reference.
    ++source_frame.shared_references;
    return COWPlan(std::shared_ptr<PhysicalPagePool>(shared_from_this()), destination,
        source.handle_, frame.next_content_version, source.version_, source.bytes_, fresh_bytes);
}

PhysicalPagePool::COWPlan::COWPlan(std::shared_ptr<PhysicalPagePool> pool, FrameHandle destination,
    FrameHandle source, std::uint64_t destination_version, std::uint64_t source_version,
    std::size_t source_bytes, std::size_t fresh_bytes)
    : pool_(std::move(pool)), destination_(destination), source_(source),
      destination_version_(destination_version), source_version_(source_version),
      source_bytes_(source_bytes), fresh_bytes_(fresh_bytes) {}
PhysicalPagePool::COWPlan::~COWPlan() { release(); }
PhysicalPagePool::COWPlan::COWPlan(COWPlan &&other) noexcept
    : pool_(std::move(other.pool_)), destination_(other.destination_), source_(other.source_),
      destination_version_(other.destination_version_), source_version_(other.source_version_),
      source_bytes_(other.source_bytes_), fresh_bytes_(other.fresh_bytes_), copied_(other.copied_),
      source_copied_(other.source_copied_) {}
PhysicalPagePool::COWPlan &PhysicalPagePool::COWPlan::operator=(COWPlan &&other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = std::move(other.pool_); destination_ = other.destination_;
        source_ = other.source_; destination_version_ = other.destination_version_;
        source_version_ = other.source_version_; source_bytes_ = other.source_bytes_;
        fresh_bytes_ = other.fresh_bytes_; copied_ = other.copied_;
        source_copied_ = other.source_copied_;
    }
    return *this;
}
void PhysicalPagePool::COWPlan::copySource()
{
    if (!pool_ || copied_ || source_copied_)
        throw std::logic_error("Copy-on-write source copy must start once from its original empty destination");
    auto &destination = pool_->requireFrame(destination_);
    auto &source = pool_->requireFrame(source_);
    if (!destination.writer || destination.retiring || destination.next_content_version != destination_version_)
        throw std::logic_error("Copy-on-write destination write ownership was revoked");
    if (!source.shared || source.committed_version != source_version_ || source.valid_bytes != source_bytes_)
        throw std::logic_error("Copy-on-write source lost its immutable shared content");
    std::memcpy(pool_->backing_.data() + destination_.index * pool_->geometry_.frame_bytes,
        pool_->backing_.data() + source_.index * pool_->geometry_.frame_bytes, source_bytes_);
    copied_ = source_bytes_;
    source_copied_ = true;
}
bool PhysicalPagePool::COWPlan::overlaps(std::span<const std::byte> source) const
{
    if (!pool_) throw std::logic_error("Copy-on-write alias query uses an empty plan");
    const auto &destination = pool_->requireFrame(destination_);
    if (!destination.writer || destination.retiring || destination.next_content_version != destination_version_)
        throw std::logic_error("Copy-on-write destination write ownership was revoked");
    const auto target = reinterpret_cast<std::uintptr_t>(pool_->backing_.data() +
        destination_.index * pool_->geometry_.frame_bytes + source_bytes_);
    const auto input = reinterpret_cast<std::uintptr_t>(source.data());
    if (source.empty()) return false;
    return input <= target ? target - input < source.size() : input - target < fresh_bytes_;
}
void PhysicalPagePool::COWPlan::copyNext(std::span<const std::byte> source)
{
    if (!pool_ || !source.data() || source.empty())
        throw std::logic_error("Copy-on-write fresh fragment has no original bytes");
    // Typed phase: fresh rows may only follow the authenticated shared source copy.
    if (!source_copied_ || !source.data() || source.size() > fresh_bytes_ - (copied_ - source_bytes_))
        throw std::logic_error("Copy-on-write fresh fragment precedes its source copy or exceeds the remaining extent");
    if (overlaps(source)) throw std::invalid_argument("Copy-on-write fresh rows alias the destination frame");
    std::memcpy(pool_->backing_.data() + destination_.index * pool_->geometry_.frame_bytes +
        source_bytes_ + (copied_ - source_bytes_), source.data(), source.size());
    copied_ += source.size();
}
PhysicalPagePool::CommittedPage PhysicalPagePool::COWPlan::commit()
{
    if (!pool_ || copied_ != source_bytes_ + fresh_bytes_)
        throw std::logic_error("Copy-on-write publication requires the complete shared and fresh bytes");
    auto &destination = pool_->requireFrame(destination_);
    auto &source = pool_->requireFrame(source_);
    if (!destination.writer || destination.retiring || destination.next_content_version != destination_version_ ||
        destination.valid_bytes != 0)
        throw std::logic_error("Copy-on-write publication has revoked or reused destination ownership");
    if (!source.shared || source.committed_version != source_version_ || source.valid_bytes != source_bytes_)
        throw std::logic_error("Copy-on-write publication lost its shared source");
    destination.valid_bytes = source_bytes_ + fresh_bytes_;
    destination.committed_version = destination_version_;
    destination.writer = false;
    destination.readers_closed = false;
    // Release the plan's extra shared-source reference, then transfer the
    // destination pin to the new private publication.
    pool_->unshare(source_);
    return CommittedPage(std::move(pool_), destination_, destination_version_, source_bytes_ + fresh_bytes_);
}
void PhysicalPagePool::COWPlan::release() noexcept
{
    if (!pool_) return;
    // Drop the plan's extra shared-source reference first, matching prepareCOW.
    if (source_.index < pool_->frames_.size())
    {
        auto &source = pool_->frames_[source_.index];
        if (source.shared && source.shared_references)
            pool_->unshare(source_);
    }
    if (destination_.index < pool_->frames_.size())
    {
        auto &destination = pool_->frames_[destination_.index];
        if (destination.writer && destination.generation == destination_.generation &&
            destination.next_content_version == destination_version_)
        {
            destination.writer = false;
            pool_->unpin(destination_);
        }
    }
    pool_.reset();
}

PhysicalPagePool::FramePin::FramePin(std::shared_ptr<PhysicalPagePool> pool, FrameHandle handle)
    : pool_(std::move(pool)), handle_(handle) {}
PhysicalPagePool::FramePin::~FramePin() { release(); }
PhysicalPagePool::FramePin::FramePin(FramePin &&other) noexcept
    : pool_(std::move(other.pool_)), handle_(other.handle_) {}
PhysicalPagePool::FramePin &PhysicalPagePool::FramePin::operator=(FramePin &&other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = std::move(other.pool_);
        handle_ = other.handle_;
    }
    return *this;
}
void PhysicalPagePool::FramePin::release() noexcept
{
    if (pool_) { pool_->unpin(handle_); pool_.reset(); }
}

PhysicalPagePool::Grant::Grant(std::shared_ptr<PhysicalPagePool> pool, std::size_t record, std::uint64_t generation)
    : pool_(std::move(pool)), record_(record), generation_(generation) {}
PhysicalPagePool::Grant::~Grant() { release(); }
PhysicalPagePool::Grant::Grant(Grant &&other) noexcept
    : pool_(std::move(other.pool_)), record_(other.record_), generation_(other.generation_) {}
PhysicalPagePool::Grant &PhysicalPagePool::Grant::operator=(Grant &&other) noexcept
{
    if (this != &other)
    {
        release();
        pool_ = std::move(other.pool_);
        record_ = other.record_;
        generation_ = other.generation_;
    }
    return *this;
}
void PhysicalPagePool::Grant::release() noexcept
{
    if (pool_) { pool_->release(*this); pool_.reset(); }
}
PhysicalPagePool::FrameHandle PhysicalPagePool::Grant::assignFrame()
{
    if (!pool_) throw std::logic_error("Page assignment uses an empty grant");
    return pool_->assign(*this);
}
void PhysicalPagePool::Grant::discardFrame(FrameHandle handle)
{
    if (!pool_) throw std::logic_error("Page discard uses an empty grant");
    pool_->discard(*this, handle);
}

PhysicalPagePool::FramePin PhysicalPagePool::Grant::pinFrame(FrameHandle handle)
{
    if (!pool_) throw std::logic_error("Private frame pin uses an empty grant");
    return pool_->pin(*this, handle);
}

std::optional<PhysicalPagePool::AppendPlan> PhysicalPagePool::Grant::prepareAppend(FrameHandle handle, std::size_t bytes)
{
    if (!pool_) throw std::logic_error("Page append uses an empty grant");
    return pool_->prepareAppend(*this, handle, bytes);
}

std::optional<PhysicalPagePool::AppendPlan> PhysicalPagePool::Grant::prepareFreshWrite(FrameHandle handle, std::size_t bytes)
{
    if (!pool_) throw std::logic_error("Fresh write uses an empty page grant");
    (void)pool_->require(*this);
    if (pool_->requireFrame(handle).valid_bytes)
        throw std::logic_error("Fresh native page write cannot reinterpret an existing byte prefix");
    return pool_->prepareAppend(*this, handle, bytes);
}
std::optional<PhysicalPagePool::SharedPage> PhysicalPagePool::Grant::adoptShared(const SharedPage &source)
{
    if (!pool_) throw std::logic_error("Shared adoption uses an empty page grant");
    return pool_->adoptShared(*this, source);
}
std::optional<PhysicalPagePool::COWPlan> PhysicalPagePool::Grant::prepareCOW(const SharedPage &source, std::size_t fresh_bytes)
{
    if (!pool_) throw std::logic_error("Copy-on-write uses an empty page grant");
    return pool_->prepareCOW(*this, source, fresh_bytes);
}
const PhysicalMemoryAuthority *PhysicalPagePool::Grant::issuer() const noexcept
{
    return pool_ ? pool_->authority_.get() : nullptr;
}

void PhysicalPagePool::Grant::closeReaders(FrameHandle handle)
{
    if (!pool_) throw std::logic_error("Page closure uses an empty grant");
    (void)pool_->require(*this);
    auto &frame = pool_->requireFrame(handle);
    if (frame.shared || frame.record != record_ || frame.request_generation != generation_ || frame.retiring || frame.writer)
        throw std::logic_error("Page closure has no original private owner");
    frame.readers_closed = true;
}

std::span<const PhysicalPagePool::FrameHandle> PhysicalPagePool::Grant::assignedFrames() const
{
    if (!pool_) throw std::logic_error("Page view uses an empty grant");
    const auto &request = pool_->require(*this);
    return {pool_->request_frames_.data() + record_ * pool_->geometry_.max_pages_per_request, request.assigned};
}
std::size_t PhysicalPagePool::Grant::remainingCommitment() const
{
    if (!pool_) throw std::logic_error("Page commitment uses an empty grant");
    return pool_->require(*this).remaining;
}
}
