/**
 * @file CPUWorkshareTiming.h
 * @brief One lexical authority for optional CPU retained-team timing contributions.
 *
 * OpenMP reductions over caller-local variables captured by an orphan workshare
 * are invalid and, for retained teams, are not a shared invocation ledger. This
 * scope elects one live stack publication before work, collects worker-local
 * counters once, then publishes one complete snapshot before releasing that
 * stack owner. Disabled profiling adds neither atomics nor collective edges.
 */
#pragma once
#include <atomic>
#include <cstdint>
#include <exception>
#include <omp.h>
#include <stdexcept>
#include <utility>

namespace llaminar2::cpu
{
/** @brief Optional two-phase timing authority shared by exactly one cooperative invocation. */
class CPUWorkshareTiming final
{
public:
    /** @brief Thread-local work-time counters, never a shared arithmetic reduction. */
    struct Counters
    {
        std::uint64_t qk_ns=0; ///< Original worker QK work time.
        std::uint64_t value_ns=0; ///< Original worker weighted-value work time.
    };
    /** @brief Elect one lexical stack authority before retained-team work begins.
     * @param enabled Immutable invocation-wide profiling policy.
     * @details All members of an already-active team encounter this boundary.
     * A standalone owner remains alive across its own parallel-region join.
     */
    explicit CPUWorkshareTiming(bool enabled):enabled_(enabled),retained_(omp_in_parallel()),authority_(&storage_)
    {
        if(enabled_&&retained_)
        {
            Totals*publication=&storage_;
#pragma omp single copyprivate(publication)
            {publication=&storage_;}
            authority_=publication;
        }
    }
    CPUWorkshareTiming(const CPUWorkshareTiming&)=delete;
    CPUWorkshareTiming&operator=(const CPUWorkshareTiming&)=delete;
    /** @brief Require completed publication before the elected owner can leave its lexical scope. */
    ~CPUWorkshareTiming(){if(enabled_&&state_!=State::Published)std::terminate();}
    /** @brief Execute one worker's lexical work and contribute after all its validation reads.
     * @tparam Work Immediate worker-local callback; @param work Consumer of private mutable counters.
     * @throws std::logic_error If work is attempted after publication.
     * @details The callback cannot return a counter handle. It executes directly,
     * without allocation, a worker-team transition or a separate arithmetic path.
     */
    template<class Work>void withThreadSample(Work&&work)
    {
        if(state_!=State::Collecting)throw std::logic_error("CPU workshare timing already published");
        Counters local;
        std::forward<Work>(work)(local);
        if(enabled_)
        {
            authority_->qk.fetch_add(local.qk_ns,std::memory_order_relaxed);
            authority_->value.fetch_add(local.value_ns,std::memory_order_relaxed);
            authority_->workers.store(omp_in_parallel()?omp_get_num_threads():1,std::memory_order_relaxed);
        }
    }
    /** @brief Bind immediate worker-local counter lifetime into an existing workshare callable.
     * @tparam Work Worker callback taking Counters&; @param work Original worksharing body.
     * @return Allocation-free callable suitable for the repository's retained/standalone workshare API.
     * @throws std::logic_error If work is bound after publication.
     */
    template<class Work>auto worker(Work&&work)
    {
        if(state_!=State::Collecting)throw std::logic_error("CPU workshare timing already published");
        return [this,callback=std::forward<Work>(work)]()mutable{withThreadSample(callback);};
    }
    /** @brief Publish exactly one complete immutable snapshot, then release the elected stack authority.
     * @tparam Publisher Immediate sink callable; @param publisher Sink(total_qk,total_value,actual_workers).
     * @throws std::logic_error For a repeated publication.
     * @details Retained workers finish contributions before the snapshot and
     * rendezvous again before any elected storage can be destroyed. After that
     * release edge, only each caller's private lifecycle state is touched.
     */
    template<class Publisher>void publish(Publisher&&publisher)
    {
        if(state_!=State::Collecting)throw std::logic_error("CPU workshare timing publication repeated");
        if(enabled_)
        {
            const auto emit=[&]{
                std::forward<Publisher>(publisher)(authority_->qk.load(std::memory_order_relaxed),
                    authority_->value.load(std::memory_order_relaxed),authority_->workers.load(std::memory_order_relaxed));
            };
            if(retained_)
            {
#pragma omp barrier
#pragma omp single
                {emit();}
            }
            else emit();
        }
        state_=State::Published;
    }
private:
    /** @brief Exactly one elected stack record receives contributions; other records remain dormant. */
    struct Totals
    {
        std::atomic<std::uint64_t> qk{0},value{0};
        std::atomic<int>workers{1};
    };
    /** @brief Private one-way lexical lifecycle, not a caller-written raw state flag. */
    enum class State:std::uint8_t{Collecting,Published};
    bool enabled_,retained_;
    Totals storage_;
    Totals*authority_;
    State state_=State::Collecting;
};
} // namespace llaminar2::cpu
