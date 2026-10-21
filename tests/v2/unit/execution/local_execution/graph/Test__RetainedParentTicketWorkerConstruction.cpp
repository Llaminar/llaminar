/**
 * @file Test__RetainedParentTicketWorkerConstruction.cpp
 * @brief Force the actual persistent worker to run before its thread constructor returns.
 *
 * This executable alone interposes two public POSIX symbols. A creator-thread
 * TLS arm selects exactly one pthread_create call; its real entrypoint/argument
 * are forwarded unchanged. The constructor cannot resume until that actual
 * child enters its first condition wait or returns. Recycled aligned storage
 * retains bytes left by an actual previous worker destructor, never a copied
 * State definition or test-authored private-field mutation. All other threads
 * and condition waits forward directly to libc. No backend/model is used.
 */
#include "execution/local_execution/graph/RetainedParentTicketServiceWorker.h"

#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <new>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <unistd.h>

namespace
{
using Worker = llaminar2::RetainedParentTicketServiceWorker;
using NativeStart = void *(*)(void *);
using NativeCreate = int (*)(pthread_t *, const pthread_attr_t *, NativeStart, void *);
using NativeWait = int (*)(pthread_cond_t *, pthread_mutex_t *);

/** @brief A real child either reached the worker's wait boundary or exited prematurely. */
enum class FirstBoundary : unsigned char { None, ConditionWait, ThreadReturn };

/** @brief Per-construction test observation, retained until the actual worker joins. */
struct ConstructionProbe final
{
    NativeStart start = nullptr;
    void *argument = nullptr;
    sem_t reached{};
    std::atomic<FirstBoundary> first{FirstBoundary::None};
    std::atomic<bool> create_returned{false};
    std::atomic<bool> observed_before_create_return{false};

    /** @brief Create the test-only handshake before arming thread creation. */
    ConstructionProbe()
    {
        if (sem_init(&reached, 0, 0) != 0)
            std::abort();
    }
    /** @brief Every real child is joined before this handshake retires. */
    ~ConstructionProbe() { sem_destroy(&reached); }
    ConstructionProbe(const ConstructionProbe &) = delete;
    ConstructionProbe &operator=(const ConstructionProbe &) = delete;

    /** @brief Publish only the first actual boundary; later waits are irrelevant. */
    void observe(FirstBoundary boundary) noexcept
    {
        auto expected = FirstBoundary::None;
        if (first.compare_exchange_strong(expected, boundary))
        {
            observed_before_create_return.store(!create_returned.load(std::memory_order_acquire),
                                                std::memory_order_release);
            if (sem_post(&reached) != 0)
                _exit(80);
        }
    }
};

thread_local ConstructionProbe *armed_creator = nullptr;
thread_local ConstructionProbe *observed_child = nullptr;

/** @brief Resolve the real POSIX function or fail without recursion/exception handling. */
template<class Function>
Function nextSymbol(const char *name)
{
    auto *symbol = dlsym(RTLD_NEXT, name);
    if (!symbol)
        _exit(81);
    return reinterpret_cast<Function>(symbol);
}

/** @return Real libc thread creation, resolved before any armed construction. */
NativeCreate realCreate()
{
    static const auto function = nextSymbol<NativeCreate>("pthread_create");
    return function;
}

/** @return Real libc condition wait, resolved before any armed construction. */
NativeWait realWait()
{
    static const auto function = nextSymbol<NativeWait>("pthread_cond_wait");
    return function;
}

/** @brief Forward the actual libstdc++ thread entrypoint and observe only its return. */
void *observedStart(void *argument)
{
    auto &probe = *static_cast<ConstructionProbe *>(argument);
    observed_child = &probe;
    void *result = probe.start(probe.argument);
    probe.observe(FirstBoundary::ThreadReturn);
    observed_child = nullptr;
    return result;
}

/** @brief Bound a broken constructor in the already-isolated child; never fabricate progress. */
void awaitActualBoundary(ConstructionProbe &probe) noexcept
{
    timespec deadline{};
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
        _exit(82);
    deadline.tv_sec += 3;
    int result = 0;
    do
    {
        result = sem_timedwait(&probe.reached, &deadline);
    } while (result != 0 && errno == EINTR);
    if (result != 0)
    {
        constexpr char message[] = "Actual worker reached neither condition wait nor return before constructor deadline\n";
        const auto written = write(STDERR_FILENO, message, sizeof(message) - 1);
        (void)written; // The original failure still exits83 if diagnostic I/O fails.
        _exit(83);
    }
}

/** @brief Destroy a placement-constructed real worker without freeing caller storage. */
struct DestroyWorker final
{
    /** @param worker Actual object whose accepted tasks have already been awaited. */
    void operator()(Worker *worker) const noexcept { std::destroy_at(worker); }
};

/** @brief The same real borrowed task is submitted twice without inference dependencies. */
bool countInvocation(void *context)
{
    ++*static_cast<unsigned *>(context);
    return true;
}

/**
 * @brief Exercise twenty-one real lifetimes on identical aligned storage.
 * @return Zero only when every child was waiting before constructor return and
 *         every public dispatch/await completed on that same persistent thread.
 *
 * The first lifetime starts from ordinary zero-initialized raw storage. Each
 * later one consumes bytes left by the actual destructor (including Stopping
 * in the original implementation). No byte is inspected or rewritten between
 * lifetimes. An early-return observation deliberately skips dispatch so the
 * original broken implementation can retire/join instead of hanging await().
 */
int exerciseRecycledWorker()
{
    alarm(10); // A broken destructor/unexpected native stall cannot hang the gate.
    (void)realCreate();
    (void)realWait();
    alignas(Worker) std::array<std::byte, sizeof(Worker)> storage{};
    for (unsigned lifetime = 0; lifetime != 21; ++lifetime)
    {
        ConstructionProbe probe; // Outlives the worker and its trampoline.
        armed_creator = &probe;
        std::unique_ptr<Worker, DestroyWorker> worker(::new (storage.data()) Worker);
        if (armed_creator != nullptr)
            return 84; // The expected real pthread_create call was not interposed.
        if (!probe.observed_before_create_return.load(std::memory_order_acquire))
            return 85;
        if (probe.first.load(std::memory_order_acquire) == FirstBoundary::ThreadReturn)
        {
            std::fprintf(stderr, "Actual worker returned before constructor completion on recycled lifetime %u; no task dispatched\n", lifetime);
            return 86; // Idle teardown joins the already-returned child safely.
        }
        if (probe.first.load(std::memory_order_acquire) != FirstBoundary::ConditionWait || !worker->idle())
            return 87;
        unsigned calls = 0;
        const auto first = worker->dispatch(countInvocation, &calls);
        if (!worker->await(first) || calls != 1 || !worker->idle())
            return 88;
        const auto second = worker->dispatch(countInvocation, &calls);
        if (second.generation <= first.generation || !worker->await(second) || calls != 2 || !worker->idle())
            return 89;
        // unique_ptr teardown precedes probe destruction and leaves original
        // worker storage untouched for the next real constructor.
    }
    alarm(0);
    return 0;
}
} // namespace

/**
 * @brief Test-executable-only thread-create interposer; unarmed calls pass through unchanged.
 * @param thread Real pthread result.
 * @param attributes Original caller attributes.
 * @param start Original libstdc++ entrypoint.
 * @param argument Original libstdc++ thread state.
 * @return Original pthread_create result, after the selected actual child boundary.
 */
extern "C" int pthread_create(pthread_t *thread, const pthread_attr_t *attributes,
                              NativeStart start, void *argument) noexcept
{
    auto *probe = armed_creator;
    if (!probe)
        return realCreate()(thread, attributes, start, argument);
    armed_creator = nullptr; // One selected construction, never unrelated nested work.
    probe->start = start;
    probe->argument = argument;
    const int result = realCreate()(thread, attributes, observedStart, probe);
    if (result == 0)
        awaitActualBoundary(*probe);
    probe->create_returned.store(true, std::memory_order_release);
    return result;
}

/**
 * @brief Observe the selected actual child's wait before forwarding the exact condition/mutex.
 * @param condition Original condition variable.
 * @param mutex Original owned mutex, released/reacquired only by real libc.
 * @return Unmodified real pthread_cond_wait result.
 */
extern "C" int pthread_cond_wait(pthread_cond_t *condition, pthread_mutex_t *mutex)
{
    if (observed_child)
        observed_child->observe(FirstBoundary::ConditionWait);
    return realWait()(condition, mutex);
}

/** @test Construction cannot publish a worker before all lifecycle members are initialized. */
TEST(RetainedParentTicketWorkerConstruction, RecycledStorageStartsActualWorkerBeforeConstructorReturns)
{
    // Re-exec isolates the symbol interposition and any intentionally detected
    // early-exit constructor from other test lifetimes; no production hooks.
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(_exit(exerciseRecycledWorker()), ::testing::ExitedWithCode(0), "");
}
