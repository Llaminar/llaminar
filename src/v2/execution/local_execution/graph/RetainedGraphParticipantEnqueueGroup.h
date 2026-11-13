/**
 * @file RetainedGraphParticipantEnqueueGroup.h
 * @brief Dispatch every original remote before the original source SDK enqueue.
 *
 * Native graph submission may apply host backpressure while a mapped receive
 * waits for its peer. Dispatch therefore cannot join a remote enqueue before
 * the source is submitted. The sole source owner joins only host enqueue jobs
 * afterwards; GPU completion belongs to the existing whole-request terminal.
 */
#pragma once
#include "backends/IWorkerGPUContext.h"
#include <future>
#include <functional>
#include <memory>
#include <vector>
#include <stdexcept>
namespace llaminar2 {
namespace qwen4exp {class PooledImmutableRetainedParticipants;}
namespace test {struct RetainedGraphParticipantsReplayAccess;}
/** @brief Cold-declared bounded remote job set, reused under its original source FIFO lock. */
class RetainedGraphParticipantEnqueueGroup final {
public:
    /** @brief A job owns no model state; it submits its exact original native definition. */
    struct Job {std::shared_ptr<IWorkerGPUContext> worker;std::function<void()> enqueue;};
    /** @brief Fail before any shared frame or worker is released with outstanding enqueue jobs. */
    ~RetainedGraphParticipantEnqueueGroup(){if(dispatched_)std::terminate();}
    /** @brief Queue all remote jobs without waiting for any native enqueue to return. */
    void dispatch(){
        if(dispatched_)throw std::logic_error("Original remote enqueue group is already outstanding");
        for(const auto& job:jobs_)if(job.worker->ownsCurrentThread())
            throw std::logic_error("Source submission cannot occupy an original remote worker");
        pending_.clear();
        try{
            for(const auto& job:jobs_){
                const Job* const original=&job;
                pending_.push_back(job.worker->submitAsync([original]{
                    // A peer SDK error cannot wait in a future while source
                    // submission is backpressured on that missing peer.
                    try{original->enqueue();}catch(...){std::terminate();}
                }));
            }
            dispatched_=true;
        }catch(...){if(!pending_.empty())std::terminate();throw;}
    }
    /** @brief Join CPU enqueue completion after the source call; never inspect GPU progress. */
    void completeOriginalEnqueue()noexcept{
        if(!dispatched_)std::terminate();
        try{for(auto& job:pending_)job.get();}catch(...){std::terminate();}
        pending_.clear();dispatched_=false;
    }
private:
    friend class qwen4exp::PooledImmutableRetainedParticipants;
    friend struct test::RetainedGraphParticipantsReplayAccess;
    /** @brief Only the actual source domain authors its original immutable job population. */
    explicit RetainedGraphParticipantEnqueueGroup(std::vector<Job> jobs):jobs_(std::move(jobs)){
        if(jobs_.empty())throw std::invalid_argument("Original remote enqueue group is empty");
        for(const auto& job:jobs_)if(!job.worker || !job.enqueue)
            throw std::invalid_argument("Original remote enqueue group lost a worker or definition");
        pending_.reserve(jobs_.size());
    }
    std::vector<Job> jobs_; ///< Cold immutable jobs, never replaced by scheduler/model state.
    std::vector<std::future<void>> pending_; ///< CPU submission receipts only; no GPU completion or token count.
    bool dispatched_=false;
};
/** @brief Typed source-owner completion called only after its native enqueue succeeds. */
class IRetainedGraphParticipantEnqueueCompletion {
public:
    /** @brief Keep every original worker/definition until CPU enqueue jobs have returned. */
    virtual ~IRetainedGraphParticipantEnqueueCompletion()=default;
    /** @brief Complete only the original host enqueue group, never its GPU readers. */
    virtual void completeOriginalEnqueue()const noexcept=0;
};
}
