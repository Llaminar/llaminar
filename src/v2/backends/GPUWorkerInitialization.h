/**
 * @file GPUWorkerInitialization.h
 * @brief Exactly-once native initialization rollback and complete constructor readiness.
 *
 * Native handle creation remains in each backend's initializeOnWorker body.
 * This transaction surrounds that complete body: no false/exception branch can
 * publish failure or discharge the native-context lease before owning-thread
 * rollback has finished. Readiness publication then holds the constructor wait
 * mutex so notification cannot pass its predicate-to-wait gap. No second handle
 * inventory or memory ledger is kept.
 */
#pragma once
#include <concepts>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <exception>
#include <type_traits>

namespace llaminar2
{
    /** @brief One complete owning-worker initialization outcome for constructor publication. */
    enum class GPUWorkerReadiness { Ready, Failed };

    /**
     * @brief Publish constructor readiness under its actual wait mutex, then notify.
     * @param mutex Exact queue mutex held by the constructor's predicate wait.
     * @param condition Exact constructor/worker condition variable.
     * @param initialized Existing initialization publication flag.
     * @param running Existing complete worker-ready flag.
     * @param shutdown Existing terminal initialization-failure flag.
     * @param outcome Ready after complete initialization, or Failed after complete rollback.
     * @note Running precedes the release publication of initialized, so even
     * lock-free readers cannot observe initialized without complete readiness.
     * Mutex/condition/flag types also admit deterministic device-free observations.
     */
    template<class Mutex, class Condition, class Flag>
    void publishGPUWorkerReadiness(Mutex &mutex, Condition &condition,
        Flag &initialized, Flag &running, Flag &shutdown, GPUWorkerReadiness outcome)
    {
        {
            std::lock_guard lock(mutex);
            if (outcome == GPUWorkerReadiness::Ready)
            {
                running.store(true, std::memory_order_relaxed);
                initialized.store(true, std::memory_order_release);
            }
            else
                shutdown.store(true, std::memory_order_release);
        }
        condition.notify_all();
    }

    /**
     * @brief Initialize one unpublished worker or retire its partial obligations.
     * @param initialize Actual backend initializer, returning true only when ready.
     * @param rollback Exact owning-worker cleanup, returning only after complete retirement.
     * @return True after a committed initialization; false after completed rollback.
     * @throws Any initializer exception, after rollback has completed on this thread.
     *
     * Both callbacks run synchronously on the caller's owning worker thread.
     * Rollback must return void so a failed status cannot be accidentally ignored.
     * Native cleanup failures must throw or terminate; a rollback exception is
     * diagnosed and fatal, including during initializer exception unwinding.
     */
    template<class Initialize, class Rollback>
        requires std::same_as<std::invoke_result_t<Initialize &>, bool> &&
                 std::same_as<std::invoke_result_t<Rollback &>, void>
    [[nodiscard]] bool runGPUWorkerInitialization(Initialize &&initialize, Rollback &&rollback)
    {
        /** @brief Arm before any native creation and disarm only after full success. */
        struct Transaction
        {
            Rollback &cleanup;
            bool committed = false;
            /** @brief Retire each partial prefix before failure becomes observable. */
            ~Transaction() noexcept
            {
                if (committed) return;
                try { cleanup(); }
                catch (const std::exception &error)
                {
                    std::fprintf(stderr, "[ERROR] [GPUWorkerInitialization] Native initialization rollback failed: %s\n", error.what());
                    std::terminate();
                }
                catch (...)
                {
                    std::fputs("[ERROR] [GPUWorkerInitialization] Native initialization rollback failed with an unknown exception\n", stderr);
                    std::terminate();
                }
            }
        } transaction{rollback};
        if (!initialize()) return false;
        transaction.committed = true;
        return true;
    }
}
