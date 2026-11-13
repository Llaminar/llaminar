/**
 * @file PipelineMaintenanceAdmission.cpp
 * @brief Exact PMA-backed scalar reduction recordings for original local native pipeline admission.
 *
 * Recordings contain only an original controller read, native minimum, and
 * tail-domain broadcast. They have no standalone executable or host count
 * observation. Original worker/context revocation and source storage lifetimes
 * match the enclosing captured pipeline family.
 */
#include "PipelineMaintenanceAdmission.h"
#include "backends/BackendManager.h"
#include "backends/NativeContextPreparation.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "transfer/TransferEngine.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include <array>
#include <atomic>
#include "utils/PerfStatsCollector.h"
#include "utils/Logger.h"
#include <algorithm>
#include <barrier>
#include <chrono>
#include <future>
#include <thread>
namespace llaminar2 {
namespace {
/** @brief Publish original absorbing abort before any worker can unwind a native capture phase. */
void abortAdmission(ILocalTPContext &context) noexcept {
    try { if (!context.signalNativeCompletionAbort()) std::terminate(); } catch (...) { std::terminate(); }
}
}
/** @brief Source definition retires before its exact scalar allocation claim. */
struct PipelineMaintenanceAdmission::Participant final {
    PreparedGPUContextUsePermit original;
    PipelineMaintenanceSource source;
    IBackend *backend;
    void *stream;
    std::array<uint32_t, 4> initial{0, uint32_t(std::numeric_limits<int32_t>::max()), 0, 0};
    bool initialization_pending = false;
    PhysicalMemoryAllocationLease claim;
    std::shared_ptr<DeviceTransferBuffer> storage;
    std::unique_ptr<IGPUGraphCapture> recording;
    /** @brief Bind original identities only; materialization follows full cohort validation. */
    Participant(PreparedGPUContextUsePermit permit, PipelineMaintenanceSource input, void *setup)
        : original(std::move(permit)), source(input), backend(getBackendFor(original.device())), stream(setup) {}
    /** @brief A partial initialization must retire before its inline H2D source or physical roots are released. */
    void finishInitialization() {
        if (!initialization_pending) return;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
        for (;;) {
            GPUStreamExecutionState state = GPUStreamExecutionState::Pending;
            original.worker()->submitAndWait([&] { state = original.worker()->queryStreamExecutionState(stream, "pipeline maintenance admission initialization"); });
            if (state == GPUStreamExecutionState::Complete) break;
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("Pipeline admission initialization did not retire");
            std::this_thread::yield();
        }
        initialization_pending = false;
    }
    /** @brief Cold failure cleanup retains the exact source words until native initialization completes. */
    ~Participant() { try { finishInitialization(); } catch (...) { std::terminate(); } }
};
PipelineMaintenanceAdmission::PipelineMaintenanceAdmission(ILocalTPContext &native, std::size_t tail_first)
    : original_(native.borrowOriginalNativeContext()), tail_first_(tail_first), members_(native.devices()) {}
PipelineMaintenanceAdmission::~PipelineMaintenanceAdmission() noexcept
{
    try {
        // The outer cohort retirement has already retired every stage graph
        // while retaining its original context. Release actual source and frame
        // roots before publishing these immutable teardown observations.
        participants_.clear();
        if (phase_ == Phase::Recorded)
            for (const auto &member : members_)
                PerfStatsCollector::addCounter("generation", "pipeline_maintenance_budget_retirements", 1.0,
                    "graph_teardown", member.toLocalDeviceId().toString(), {{"frame_bytes", std::to_string(bytes())}});
    } catch (const std::exception &error) {
        LOG_ERROR("Original pipeline maintenance source retirement failed: " << error.what());
        std::terminate();
    } catch (...) { std::terminate(); }
}
std::shared_ptr<PipelineMaintenanceAdmission> PipelineMaintenanceAdmission::create(ILocalTPContext &native, int tail_first,
    ILocalTPContext *tail_context, std::span<const PipelineMaintenanceSource> sources,
    const std::shared_ptr<PhysicalMemoryAuthority> &memory, std::span<void *const> streams)
{
    if (isGraphCaptureActive() || !memory || !native.isLocal() || native.degree() < 2 || tail_first <= 0 ||
        tail_first >= native.degree() || sources.size() != std::size_t(native.degree()) ||
        native.devices().size() != sources.size() || streams.size() != sources.size() ||
        (native.backend() != CollectiveBackendType::NCCL && native.backend() != CollectiveBackendType::RCCL))
        throw std::invalid_argument("Pipeline maintenance admission requires its complete original local native cohort");
    const auto tail_count = std::size_t(native.degree() - tail_first);
    if ((tail_count > 1) != bool(tail_context) || (tail_context && (tail_context->degree() != int(tail_count) || !tail_context->isLocal() ||
         tail_context->backend() != native.backend() || tail_context->devices().size() != tail_count)))
        throw std::invalid_argument("Pipeline admission changed its actual terminal domain membership");
    for (std::size_t member = 0; member < sources.size(); ++member) {
        const auto identity = " at member " + std::to_string(member);
        if (!sources[member].valid())
            throw std::invalid_argument("Pipeline admission lost its original controller extent or positive cadence" + identity);
        if (sources[member].device() && sources[member].device() != native.devices()[member].toLocalDeviceId())
            throw std::invalid_argument("Pipeline admission changed its original controller device" + identity);
        if (reinterpret_cast<std::uintptr_t>(streams[member]) <= 2)
            throw std::invalid_argument("Pipeline admission lost its exact explicit stream" + identity);
        if (member >= std::size_t(tail_first) && tail_context &&
            tail_context->devices()[member - tail_first] != native.devices()[member])
            throw std::invalid_argument("Pipeline admission changed its original terminal-domain address" + identity);
    }
    if (std::none_of(sources.begin(), sources.end(), [](const auto &source) { return source.remaining() != nullptr; }))
        throw std::invalid_argument("Pipeline maintenance admission has no active original stage authority");
    auto result = std::shared_ptr<PipelineMaintenanceAdmission>(new PipelineMaintenanceAdmission(native, tail_first));
    const auto original = result->original_.acquire();
    for (std::size_t member = 0; member < sources.size(); ++member) {
        auto permit = GPUDeviceContextPool::instance().retainExistingPreparedContext(result->device(member), memory);
        if (permit.worker()->ownsCurrentThread())
            throw std::invalid_argument("Pipeline admission setup must not occupy an original GPU worker");
        auto participant = std::make_shared<Participant>(std::move(permit), sources[member], streams[member]);
        participant->claim = memory->claimNewAllocation(result->device(member), PhysicalMemoryOwner::ActivationTransportStaging, bytes());
        participant->original.worker()->submitAndWait([&] {
            participant->storage = TransferEngine::instance().allocateDeviceTransferBuffer(bytes(), participant->original.device());
            participant->initialization_pending = true;
            if (!participant->backend || !participant->backend->hostToDeviceOnStream(participant->storage->mutableDeviceData(),
                    participant->initial.data(), bytes(), participant->original.device().ordinal, participant->stream))
                throw std::runtime_error("Pipeline admission original scalar initialization failed");
        });
        result->participants_.push_back(std::move(participant));
    }
    // This is cold initialization only. Keep each exact source object and its
    // initial host words alive until the original setup stream retires.
    for (const auto &participant : result->participants_) participant->finishInitialization();
    std::barrier begun(std::ptrdiff_t(sources.size())), recorded(std::ptrdiff_t(sources.size())), ended(std::ptrdiff_t(sources.size()));
    std::vector<uint8_t> began(sources.size()); std::vector<std::exception_ptr> failures(sources.size());
    std::vector<std::atomic<bool>> entered(sources.size());
    for (auto &entry : entered) entry.store(false, std::memory_order_relaxed);
    std::vector<std::future<void>> captures; captures.reserve(sources.size());
    std::exception_ptr submission_error;
    try {
    for (std::size_t member = 0; member < sources.size(); ++member) captures.push_back(result->participants_[member]->original.worker()->submitAsync([&, member] {
        entered[member].store(true, std::memory_order_release);
        auto &participant = *result->participants_[member];
        native.prepareOriginalNativeParticipant([&] {
            std::unique_ptr<ScopedBackendGraphCapture> capture;
            try {
                participant.recording = participant.original.worker()->createGraphCapture(participant.stream);
                if (!participant.recording) throw std::runtime_error("Pipeline maintenance minimum capture owner is missing");
                capture = std::make_unique<ScopedBackendGraphCapture>(*participant.original.worker(), *participant.recording,
                    "pipeline original maintenance minimum");
                if (!capture->begin()) throw std::runtime_error("Pipeline maintenance minimum capture begin failed");
                began[member] = 1;
            } catch (...) { failures[member] = std::current_exception(); abortAdmission(native); }
            begun.arrive_and_wait();
            if (std::all_of(began.begin(), began.end(), [](auto value) { return value != 0; })) try {
                void *const value = participant.storage->mutableDeviceData();
                const void *const source = participant.source.remaining() ? participant.source.remaining() :
                    static_cast<const uint32_t *>(participant.storage->deviceData()) + 1;
                if (!enqueueOriginalMinimum(native, tail_context, tail_first, int(member), source, value, participant.stream))
                    throw std::runtime_error("Pipeline admission original native scalar plan failed");
            } catch (...) { failures[member] = std::current_exception(); abortAdmission(native); }
            recorded.arrive_and_wait();
            if (capture) try { capture->finish(); } catch (...) {
                if (!failures[member]) failures[member] = std::current_exception(); abortAdmission(native);
            }
            ended.arrive_and_wait();
        });
    }));
    } catch (...) {
        submission_error = std::current_exception(); abortAdmission(native);
        for (std::size_t absent = captures.size(); absent < sources.size(); ++absent) {
            begun.arrive_and_drop(); recorded.arrive_and_drop(); ended.arrive_and_drop();
        }
    }
    joinOriginalRecordings(native, captures, entered, begun, recorded, ended, submission_error);
    for (const auto &failure : failures) if (failure) std::rethrow_exception(failure);
    for (std::size_t member = 0; member < sources.size(); ++member)
        PerfStatsCollector::addCounter("generation", "pipeline_maintenance_budget_recordings", 1.0,
            "graph_setup", result->device(member).toString(), {{"native_scalar_bytes", "4"},
                {"frame_bytes", std::to_string(bytes())}, {"tail_width", std::to_string(tail_count)},
                {"source", sources[member].remaining() ? "original_stage_controller" : "inactive_identity"}});
    result->phase_ = Phase::Recorded;
    return result;
}
void PipelineMaintenanceAdmission::joinOriginalRecordings(ILocalTPContext &native,
    std::span<std::future<void>> captures, std::span<std::atomic<bool>> entered,
    std::barrier<> &begun, std::barrier<> &recorded, std::barrier<> &ended,
    std::exception_ptr submission_error)
{
    if (entered.size() < captures.size()) throw std::invalid_argument("Original capture entry evidence is incomplete");
    // A worker wrapper can fail before entering our callback. Observe ready
    // failures without joining an earlier blocked future, then withdraw only
    // that never-entered participant from the three cold capture phases.
    std::vector<bool> joined(captures.size(), false);
    std::size_t retired = 0;
    const auto retirement_deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs);
    while (retired != captures.size()) {
        for (std::size_t member = 0; member < captures.size(); ++member) {
            if (joined[member] || captures[member].wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
            try { captures[member].get(); } catch (...) {
                if (!submission_error) submission_error = std::current_exception();
                abortAdmission(native);
                if (!entered[member].load(std::memory_order_acquire)) {
                    begun.arrive_and_drop(); recorded.arrive_and_drop(); ended.arrive_and_drop();
                }
            }
            joined[member] = true; ++retired;
        }
        if (retired == captures.size()) break;
        if (std::chrono::steady_clock::now() >= retirement_deadline) {
            abortAdmission(native);
            std::terminate(); // Pending callbacks still borrow this factory's barriers and roots.
        }
        std::this_thread::yield();
    }
    if (submission_error) std::rethrow_exception(submission_error);
}
bool PipelineMaintenanceAdmission::enqueueOriginalMinimum(ILocalTPContext &native, ILocalTPContext *tail,
    int tail_first, int member, const void *source, void *result, void *stream)
{
    if (!source || !result || reinterpret_cast<std::uintptr_t>(stream) <= 2 || !native.isLocal() || native.degree() < 2 ||
        native.devices().size() != std::size_t(native.degree()) || tail_first <= 0 || tail_first >= native.degree() ||
        member < 0 || member >= native.degree() || ((native.degree() - tail_first > 1) != bool(tail)) ||
        (tail && (tail->degree() != native.degree() - tail_first || !tail->isLocal() ||
         tail->backend() != native.backend() || tail->devices().size() != std::size_t(tail->degree())))) return false;
    if (tail) for (int local = 0; local < tail->degree(); ++local)
        if (tail->devices()[local] != native.devices()[tail_first + local]) return false;
    if (!native.reduceRawOnStream(source, result, 1, CollectiveDataType::INT32, CollectiveOp::ALLREDUCE_MIN,
            tail_first, member, stream, "pipeline_original_maintenance_budget")) return false;
    return member < tail_first || !tail || tail->broadcastRawOnStream(result, result, 1, CollectiveDataType::INT32,
        0, member - tail_first, stream, "pipeline_terminal_maintenance_bound");
}
DeviceId PipelineMaintenanceAdmission::device(std::size_t participant) const { return members_.at(participant).toLocalDeviceId(); }
const IGPUGraphCapture *PipelineMaintenanceAdmission::recording(std::size_t participant) const {
    (void)original_.acquire();
    return participants_.at(participant)->recording.get();
}
const uint32_t *PipelineMaintenanceAdmission::remaining(std::size_t participant) const {
    if (participant < tail_first_) throw std::logic_error("Only an original tail member consumes the derived pipeline admission bound");
    return static_cast<const uint32_t *>(participants_.at(participant)->storage->deviceData());
}
const uint32_t *PipelineMaintenanceAdmission::inactiveStatus(std::size_t participant) const {
    if (participant < tail_first_ || participants_.at(participant)->source.remaining())
        throw std::logic_error("Only an actually inactive original tail controller may bind neutral status");
    return static_cast<const uint32_t *>(participants_.at(participant)->storage->deviceData()) + 2;
}
uint32_t *PipelineMaintenanceAdmission::inactiveMarker(std::size_t participant) const {
    (void)inactiveStatus(participant);
    // The preparation kernel reads status and writes acknowledgement through
    // restrict-qualified arguments. Keep these exact physical words distinct.
    return static_cast<uint32_t *>(participants_.at(participant)->storage->mutableDeviceData()) + 3;
}
} // namespace llaminar2
