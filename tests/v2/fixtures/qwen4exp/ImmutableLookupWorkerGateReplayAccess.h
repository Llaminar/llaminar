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
            if(armed_ || entered_.load(std::memory_order_acquire)){
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
    };
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
