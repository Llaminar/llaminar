/**
 * @file ImmutableLookupWorkerGateReplayAccess.h
 * @brief Closed causal gate on the SAME original immutable I/O worker.
 *
 * This fixture delays only the original host task entry. It grants no source,
 * native admission, response, graph policy or retirement capability. The exact
 * existing processor runs once after release, or after original cancellation.
 */
#pragma once
#include "execution/qwen4exp/ImmutableLookupExecutionBoundary.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <optional>
#include <thread>

namespace llaminar2::test {
/** @brief Narrow existing-friend-style access, never compiled into production callers. */
struct ImmutableLookupWorkerGateReplayAccess final {
    /** @brief Borrowed gate context, retained until the actual worker has left it. */
    class Gate final {
    public:
        /** @brief Retain the exact existing source only while its task borrows this gate.
         * @param source Same original source service retained by the production boundary.
         * @throws std::invalid_argument For absent or independently owned aliases.
         */
        explicit Gate(std::shared_ptr<qwen4exp::ImmutableLookupService> source):source_(std::move(source)) {
            if(!source_)throw std::invalid_argument("Delayed original worker requires its real source");
            const auto original=source_->shared_from_this();
            if(source_.owner_before(original)||original.owner_before(source_))
                throw std::invalid_argument("Delayed original worker changed its source control block");
        }
        /** @brief Release on every assertion/exception, then wait only until entry no longer borrows this gate. */
        ~Gate() noexcept {
            release();
            const bool borrowed_dispatched = ImmutableLookupWorkerGateReplayAccess::withdraw(*this);
            if(armed_ || borrowed_dispatched || entered_.load(std::memory_order_acquire)){
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                while(!departed_.load(std::memory_order_acquire)){
                    if(std::chrono::steady_clock::now()>=deadline)std::terminate();
                    std::this_thread::yield();
                }
            }
        }
        Gate(const Gate&)=delete;Gate& operator=(const Gate&)=delete;
        /** @brief Release the one original worker task without changing any protocol value. */
        void release()noexcept{released_.store(true,std::memory_order_release);}
        /** @return Actual worker entry has reached the fixture delay. */
        bool entered()const noexcept{return entered_.load(std::memory_order_acquire);}
        /** @return The diagnostic gate still delays the actual source task. */
        bool held()const noexcept{return entered()&&!released_.load(std::memory_order_acquire)&&!departed_.load(std::memory_order_acquire);}
        /** @return A source delay exceeded its original standard deadline; never a successful witness. */
        bool timedOut()const noexcept{return timed_out_.load(std::memory_order_acquire);}
    private:
        friend struct ImmutableLookupWorkerGateReplayAccess;
        std::shared_ptr<qwen4exp::ImmutableLookupService> source_;
        std::atomic<bool> released_{false},entered_{false},departed_{false},timed_out_{false};
        bool armed_=false;
        RetainedParentTicketServiceWorker::BorrowedDispatchEntry next_entry_;

    };
    /** @return The actual source owner, borrowed only by this closed diagnostic fixture. */
    static std::shared_ptr<qwen4exp::ImmutableLookupService> originalService(
        const qwen4exp::ImmutableLookupExecutionBoundary& boundary)
    {
        return boundary.service_;
    }
    /**
     * @brief Snapshot the retired mapped sequence before installing a new gate.
     * @param gate Same original service; no active source interval may remain.
     * @param device Original CPU source channel, not a foreign compute-reader alias.
     * @return Immutable previous publication, never an admission or response proof.
     */
    static std::uint64_t retiredSequence(const Gate& gate, DeviceId device)
    {
        auto& source = *gate.source_;
        std::lock_guard lock(source.lifecycle_);
        if (source.state_ != qwen4exp::ImmutableLookupService::Lifecycle::Idle || source.outstanding_.valid())
            throw std::logic_error("Retired source snapshot requires its original completed interval");
        return std::atomic_ref(request(source, device).publication).load(std::memory_order_acquire);
    }
    /**
     * @brief Acquire a genuinely new request while the same CPU worker is held.
     * @param gate Consumed original dispatch wrapper, still delaying its actual task.
     * @param device Original source channel.
     * @param previous Retired sequence observed before this original admission.
     * @return Only a complete request matching the actual current group generation,
     * original table, exact live extent and strictly next retained channel sequence.
     * Source Publish follows all current compact requests/Probes. This frontier
     * rejects stale warm Ready2 without a new GPU event or readback. The request
     * has no worker-invocation field; that identity remains the original service's.
     */
    static std::optional<qwen4exp::ImmutableLookupRequest> currentRequest(
        const Gate& gate, DeviceId device, std::uint64_t previous)
    {
        auto& source = *gate.source_;
        std::lock_guard lock(source.lifecycle_);
        if (!dispatched(gate) || !gate.held() ||
            source.state_ != qwen4exp::ImmutableLookupService::Lifecycle::Serving || !source.outstanding_.valid())
            return std::nullopt;
        auto& original = request(source, device);
        const auto acquired = std::atomic_ref(original.publication).load(std::memory_order_acquire);
        if (!acquired || previous == ~std::uint64_t{0} || acquired != previous + 1)
            return std::nullopt;
        qwen4exp::ImmutableLookupRequest snapshot;
        std::memcpy(&snapshot, &original, sizeof(snapshot));
        const auto& layout = source.backing_->layout;
        if (std::atomic_ref(original.publication).load(std::memory_order_acquire) != acquired ||
            qwen4exp::ImmutableLookupProtocol::validateRequest(snapshot, acquired,
                source.outstanding_.generation, source.backing_->tableId, previous,
                layout.maximumIds(), layout.rowBytes(), layout.protocol()) !=
                qwen4exp::ImmutableLookupStatus::Success)
            return std::nullopt;
        return snapshot;
    }
    /**
     * @brief Delay the next SAME original run/this dispatch without issuing a ticket.
     * @param gate Original source/context retained through actual worker departure.
     * @throws std::logic_error For active source/worker, duplicate or reused installation.
     * The original Rank armSourceGroup/armGroup remains the only admission issuer.
     */
    static void installNextOriginalDispatch(Gate& gate)
    {
        auto& source = *gate.source_;
        std::lock_guard source_lock(source.lifecycle_);
        auto& worker = source.worker_;
        std::lock_guard worker_lock(worker.mutex_);
        using Worker = RetainedParentTicketServiceWorker;
        if (source.state_ != qwen4exp::ImmutableLookupService::Lifecycle::Idle ||
            source.outstanding_.valid() || worker.state_ != Worker::State::Idle ||
            worker.next_borrowed_entry_ ||
            gate.next_entry_.outcome.load(std::memory_order_acquire) != Worker::BorrowedEntryOutcome::Uninstalled)
            throw std::logic_error("Closed source gate requires its original idle, unclaimed dispatch");
        gate.next_entry_.expected_task = &qwen4exp::ImmutableLookupService::run;
        gate.next_entry_.expected_context = &source;
        gate.next_entry_.wrapper_task = &run;
        gate.next_entry_.wrapper_context = &gate;
        gate.next_entry_.outcome.store(Worker::BorrowedEntryOutcome::Installed, std::memory_order_release);
        worker.next_borrowed_entry_ = &gate.next_entry_;
    }
    /** @return Whether the exact original dispatch consumed this fixture borrow. */
    static bool dispatched(const Gate& gate) noexcept
    {
        return gate.next_entry_.outcome.load(std::memory_order_acquire) ==
            RetainedParentTicketServiceWorker::BorrowedEntryOutcome::Dispatched;
    }
    /** @return Actual dispatch rejected and withdrew this borrowed descriptor. */
    static bool rejected(const Gate& gate) noexcept
    {
        return gate.next_entry_.outcome.load(std::memory_order_acquire) ==
            RetainedParentTicketServiceWorker::BorrowedEntryOutcome::Rejected;
    }
    /**
     * @brief Exercise original dispatch rejection without creating any invocation.
     * @param gate Same original worker/source, previously idle with installed borrow.
     * @throws std::invalid_argument Through the actual incomplete-descriptor guard.
     */
    static void rejectIncompleteDispatch(Gate& gate)
    {
        (void)gate.source_->worker_.dispatch(nullptr, gate.source_.get());
    }
    /**
     * @brief Exercise exact-owner mismatch through the actual dispatch guard.
     * @param gate Same original worker with a fixture-owned context mismatch.
     * @throws std::logic_error Before any invocation or source work is published.
     */
    static void rejectForeignDispatch(Gate& gate)
    {
        (void)gate.source_->worker_.dispatch(&qwen4exp::ImmutableLookupService::run, &gate);
    }
    /** @brief Enter the original boundary admission with its SAME task behind the closed gate.
     * @param boundary Actual original single-channel source boundary.
     * @param request_epoch Original API request epoch. @param stream Exact native source stream.
     * @param gate Borrowed diagnostic context, which must survive native/service retirement.
     * @throws std::exception Through unchanged original admission validation.
     */
    static void begin(qwen4exp::ImmutableLookupExecutionBoundary& boundary,std::uint64_t request_epoch,void* stream,Gate& gate){
        if(boundary.service_!=gate.source_ || boundary.service_.owner_before(gate.source_) || gate.source_.owner_before(boundary.service_))throw std::invalid_argument("Worker delay replaced its original service");
        boundary.beginAdmissionImpl(request_epoch,stream,qwen4exp::ImmutableLookupService::WorkerEntry{&run,&gate});
        gate.armed_=true;
    }
    /** @brief Exercise the same original single-channel arm body in a device-free worker contract.
     * @param gate Real source/task context. @param generation Original monotonically increasing source epoch.
     * @return Actual original worker ticket; callers must close/cancel and await it.
     */
    static auto arm(Gate& gate,std::uint64_t generation){
        auto ticket=gate.source_->armSingleChannel(generation,{&run,&gate});gate.armed_=true;return ticket;
    }
private:
    /** @return The original mapped CPU request, with no new binding or pointer authority. */
    static qwen4exp::ImmutableLookupRequest& request(qwen4exp::ImmutableLookupService& source, DeviceId device)
    {
        const auto index = source.channelIndex(device);
        auto* base = static_cast<std::byte*>(source.backing_->mapped->mutableHostData());
        return *reinterpret_cast<qwen4exp::ImmutableLookupRequest*>(base + source.geometry_.offset(index));
    }
    /**
     * @brief Withdraw an unsent borrow or identify its actual original dispatch.
     * @param gate Lexical fixture owner; no lock is held across worker departure.
     * @return True only for an accepted original invocation requiring wrapper drain.
     */
    static bool withdraw(Gate& gate) noexcept
    {
        auto& worker = gate.source_->worker_;
        std::lock_guard lock(worker.mutex_);
        if (worker.next_borrowed_entry_ == &gate.next_entry_)
        {
            worker.next_borrowed_entry_ = nullptr;
            gate.next_entry_.outcome.store(
                RetainedParentTicketServiceWorker::BorrowedEntryOutcome::Withdrawn,
                std::memory_order_release);
        }
        return dispatched(gate);
    }
    /** @brief Delay the original CPU entry, borrowing its original destructor-protected context.
     * No worker-owned shared_ptr may keep the service alive forever when a
     * failed fixture drops its last external owner: original cancel/await must run.
     * The last gate edge is published only after its source pointer is copied.
     * @param context Original Gate retained by the native fixture.
     * @return Exact original service result, never a synthesized completion.
     */
    static bool run(void* context){
        auto& gate=*static_cast<Gate*>(context);auto* source=gate.source_.get();
        gate.entered_.store(true,std::memory_order_release);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(!gate.released_.load(std::memory_order_acquire)&&!source->cancelled_.load(std::memory_order_acquire)){
            if(std::chrono::steady_clock::now()>=deadline){gate.timed_out_.store(true,std::memory_order_release);break;}
            std::this_thread::yield();
        }
        gate.departed_.store(true,std::memory_order_release);
        return qwen4exp::ImmutableLookupService::run(source);
    }
};
} // namespace llaminar2::test
