/**
 * @file OriginalNativeGraphCaptureGroup.h
 * @brief Matched cold recording on the exact original native participant workers.
 *
 * Setup and input acquisition precede this owner. Every original native member
 * enters capture before any collective is recorded, and every member exits
 * before an executable may be instantiated. Existing LocalTP rendezvous and
 * abort authority own progress; this group creates no communicator, GPU fence,
 * allocation budget or execution state. A failed submission signals native
 * abort before joining any already-dispatched CPU job.
 */
#pragma once
#include "collective/ILocalTPContext.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <exception>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

namespace llaminar2 {
/**
 * @brief Retire ready CPU submissions without hiding a later failure behind a blocked peer.
 * @param pending Exact original CPU enqueue futures; get invalidates each consumed slot.
 * @param failure Publish the first original error and abort before peers are drained.
 * @param deadline Optional cold-retirement deadline; recurring enqueue has no whole-forward timeout.
 * @note Polls CPU task readiness only. It performs no GPU query, model readback,
 * event/stream wait or allocation. Callbacks and their roots stay alive until
 * every submitted future has been consumed; unproven cold retirement is fatal.
 */
template<class Failure>
void joinOriginalNativeWorkerSubmissions(std::span<std::future<void>> pending,Failure&& failure,
    std::optional<std::chrono::steady_clock::time_point> deadline={})
{
    auto remaining=std::count_if(pending.begin(),pending.end(),[](const auto& future){return future.valid();});
    while(remaining) {
        for(auto& future:pending) {
            if(!future.valid() || future.wait_for(std::chrono::seconds(0))!=std::future_status::ready)continue;
            try{future.get();}catch(...){failure(std::current_exception());}
            --remaining;
        }
        if(!remaining)break;
        if(deadline && std::chrono::steady_clock::now()>=*deadline) {
            failure(std::make_exception_ptr(std::runtime_error("Original native cold capture workers did not retire after abort")));
            std::terminate(); // Callback captures and original roots still have outstanding readers.
        }
        std::this_thread::yield();
    }
}
/** @brief One original prepared recording owner and its already-planned body. */
struct OriginalNativeGraphRecording final {
    std::shared_ptr<IWorkerGPUContext> worker;
    DeviceId device;
    IGPUGraphCapture* capture;
    GraphCaptureDependencyLedger* dependencies;
    std::function<void()> record;
};
/** @brief Cold symmetric capture; only the existing native context owns membership. */
class OriginalNativeGraphCaptureGroup final {
public:
    /**
     * @brief Record every exact graph after cold preparation has succeeded.
     * @param jobs Original graph/worker/body declarations, each device once.
     * @param native Revocable authority projected from the actual bound transfer.
     * @param label Exact original operation identity used by LocalTP phase checks.
     * @throws First recording/submission error after abort and all CPU jobs join.
     * @note Mapped participants may coexist with the native roster; they have no
     * native rendezvous. No executable is instantiated or launched here.
     */
    static void record(std::span<const OriginalNativeGraphRecording> jobs,
        const std::optional<NativeLocalTPContextBorrow>& native,const std::string& label)
    {
        if(jobs.empty() || label.empty() || isGraphCaptureActive())
            throw std::invalid_argument("Original capture group requires a nonempty cold declaration");
        std::vector<int> coordinates(jobs.size(),-1);
        std::vector<bool> represented;
        if(native) {
            const auto use=native->acquire();const auto& context=use.context();
            if(context.degree()<2 || context.devices().size()!=std::size_t(context.degree()) ||
               (context.backend()!=CollectiveBackendType::NCCL && context.backend()!=CollectiveBackendType::RCCL))
                throw std::invalid_argument("Original capture group requires its exact native LocalTP context");
            represented.resize(context.devices().size());
            for(std::size_t i=0;i<jobs.size();++i)
                for(std::size_t p=0;p<context.devices().size();++p)
                    if(context.devices()[p].toLocalDeviceId()==jobs[i].device) {
                        if(represented[p])throw std::invalid_argument("Original capture group duplicated a native participant");
                        represented[p]=true;coordinates[i]=static_cast<int>(p);
                    }
            if(std::find(represented.begin(),represented.end(),false)!=represented.end())
                throw std::invalid_argument("Original capture group omitted an actual native participant");
        }
        for(std::size_t i=0;i<jobs.size();++i) {
            const auto& job=jobs[i];
            if(!job.worker || !job.device.is_gpu() || job.worker->deviceOrdinal()!=job.device.ordinal ||
               job.worker->ownsCurrentThread() || !job.capture || !job.record ||
               reinterpret_cast<std::uintptr_t>(job.capture->executionStream())<=2)
                throw std::invalid_argument("Original capture group requires each exact off-worker graph owner and stream");
            for(std::size_t previous=0;previous<i;++previous)
                if(jobs[previous].device==job.device || jobs[previous].worker==job.worker || jobs[previous].capture==job.capture)
                    throw std::invalid_argument("Original capture group reused a participant worker or definition");
        }
        std::mutex errorMutex;std::exception_ptr firstError;
        const auto fail=[&](std::exception_ptr error) noexcept {
            {std::lock_guard lock(errorMutex);if(!firstError)firstError=std::move(error);}
            if(native)try {
                const auto use=native->acquire();
                if(!use.context().signalNativeCompletionAbort())std::terminate();
            }catch(...){std::terminate();}
        };
        const auto rendezvous=[&](std::size_t index,const char* phase) {
            if(coordinates[index]<0)return;
            // Release this lifetime scope before the body reborrows its exact
            // native binding. Recursive shared_mutex acquisition is unnecessary.
            const auto use=native->acquire();
            if(!use.context().graphCaptureBoundaryRendezvous(label+phase,coordinates[index],collective_timeout_policy::kDefaultCollectiveTimeoutMs))
                throw std::runtime_error("Original native capture phase failed: "+label+phase);
        };
        std::vector<std::future<void>> pending;pending.reserve(jobs.size());
        try {
            for(std::size_t index=0;index<jobs.size();++index)
                pending.push_back(jobs[index].worker->submitAsync([&,index] {
                    try {
                        const auto& job=jobs[index];
                        ScopedBackendGraphCapture capture(*job.worker,*job.capture,label,job.dependencies);
                        try {
                            if(!capture.begin())throw std::runtime_error("Original native participant beginCapture failed: "+job.device.toString());
                            rendezvous(index,":phase=capture_begin:");
                            job.record();
                            rendezvous(index,":phase=body_recorded:");
                            capture.finish();
                            rendezvous(index,":phase=capture_end:");
                        }catch(...) {
                            // Abort while the capture scope still owns begin/end;
                            // only then may its destructor close a partial body.
                            fail(std::current_exception());throw;
                        }
                    }catch(...) {fail(std::current_exception());throw;}
                }));
        }catch(...) {fail(std::current_exception());}
        // Every job is scheduled before the first wait. Partial dispatch has
        // already signalled abort, waking existing native rendezvous/collectives.
        joinOriginalNativeWorkerSubmissions(pending,fail,std::chrono::steady_clock::now()+
            std::chrono::milliseconds(collective_timeout_policy::kDefaultCollectiveTimeoutMs));
        if(firstError)std::rethrow_exception(firstError);
    }
};
} // namespace llaminar2
