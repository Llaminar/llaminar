/**
 * @file rccl_thread_lifetime_probe.cpp
 * @brief Literal native RCCL partial pthread creation and destructor ordering replay.
 * Native async launch, worker and cleanup bodies are included unchanged. Only
 * pthread operations are deterministic device-free seams; no thread race or
 * freed-memory dereference is required to expose the missing join obligation.
 */
#include "group.h"
#include "debug.h"
#include <array>
#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <csignal>
#include <thread>
int ncclDebugLevel = 0;
unsigned long ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char *, int, const char *, ...) {}
using Queue = ncclIntruQueue<ncclAsyncJob, &ncclAsyncJob::next>;
struct Launch
{
    pthread_t thread{};
    void *(*entry)(void *) = nullptr;
    void *argument = nullptr;
    bool entered = false, joined = false;
};
struct Facts
{
    bool eager = false, failJoin = false, real = false, failCreate = true, requireAbort = true;
    int creates = 0, joins = 0, joinAttempts = 0, executions = 0, destructors = 0, undos = 0,
        unsafe = 0;
    std::array<Launch, 2> launches{};
    std::array<uint32_t *, 8> abortFlags{};
} facts;
int fatalPipe = -1;
ncclAsyncJob *fatalOwner = nullptr;
extern "C" [[noreturn]] void __real_abort();
/** @brief Observe the actual fatal boundary, then invoke the real process abort. */
extern "C" [[noreturn]] void __wrap_abort()
{
    if (fatalPipe >= 0)
    {
        const bool retained =
            fatalOwner &&
            ncclLlaminarLoadThreadOwnership(fatalOwner) == ncclNativeThreadOwnership::JoinFailed &&
            facts.joinAttempts == 1 && facts.joins == 0 && facts.destructors == 0 &&
            facts.undos == 0 && facts.unsafe == 0;
        const char value = retained ? 'Y' : 'N';
        if (write(fatalPipe, &value, 1) != 1)
            _exit(3);
    }
    __real_abort();
}
/** @brief Actual native worker used by the real-pthread delayed/eager controls. */
void *ncclAsyncJobMain(void *);
void *replayRealEntry(void *opaque)
{
    auto &launch = *static_cast<Launch *>(opaque);
    auto *job = static_cast<ncclAsyncJob *>(launch.argument);
    if (!facts.eager)
        while (!__atomic_load_n(job->abortFlag, __ATOMIC_ACQUIRE))
            std::this_thread::yield();
    launch.entered = true;
    return launch.entry(launch.argument);
}
int replayCreate(pthread_t *output, const pthread_attr_t *, void *(*entry)(void *), void *argument)
{
    const int index = facts.creates++;
    if (index == 1 && facts.failCreate)
        return EAGAIN;
    *output = 100 + index;
    facts.launches[index] = {*output, entry, argument, false, false};
    if (facts.real)
    {
        const int error =
            ::pthread_create(output, nullptr, replayRealEntry, &facts.launches[index]);
        if (error)
            return error;
        facts.launches[index].thread = *output;
        if (facts.eager)
            while (__atomic_load_n(&static_cast<ncclAsyncJob *>(argument)->state,
                                   __ATOMIC_ACQUIRE) != ncclGroupJobDone)
                std::this_thread::yield();
        return 0;
    }
    if (facts.eager)
    {
        facts.launches[index].entered = true;
        entry(argument);
    }
    return 0;
}
int replayJoin(pthread_t thread, void **)
{
    ++facts.joinAttempts;
    if (facts.requireAbort)
        for (auto *flag : facts.abortFlags)
            if (!flag || *flag != 1)
                ++facts.unsafe;
    if (facts.failJoin)
        return EINVAL;
    for (auto &launch : facts.launches)
        if (launch.thread == thread && launch.entry)
        {
            if (launch.joined)
                std::abort();
            if (facts.real)
            {
                const int error = ::pthread_join(thread, nullptr);
                if (!error)
                {
                    launch.joined = true;
                    ++facts.joins;
                }
                return error;
            }
            if (!launch.entered)
            {
                launch.entered = true;
                launch.entry(launch.argument);
            }
            launch.joined = true;
            ++facts.joins;
            return 0;
        }
    return EINVAL;
}
#define pthread_create replayCreate
#define pthread_join replayJoin
#include "actual-job-destructor.inc"
#include "actual-worker.inc"
#include "actual-created-drain.inc"
#include "actual-async-launch.inc"
#undef pthread_create
#undef pthread_join
ncclResult_t ncclCommSetAsyncError(ncclComm_t, ncclResult_t)
{
    std::abort();
}
void cleanup(Queue *asyncJobsPtr, ncclResult_t error)
{
#include "actual-cleanup-guard.inc"
#include "actual-cleanup.inc"
}
ncclResult_t execute(ncclAsyncJob *)
{
    ++facts.executions;
    return ncclSuccess;
}
void destroy(void *source)
{
    ++facts.destructors;
    for (const auto &launch : facts.launches)
        if (launch.entry && !launch.joined)
            ++facts.unsafe;
}

/** @brief Existing native undo must also follow every created worker join. */
void undo(ncclAsyncJob *)
{
    ++facts.undos;
    for (const auto &launch : facts.launches)
        if (launch.entry && !launch.joined)
            ++facts.unsafe;
}
struct Fixture
{
    std::array<ncclAsyncJob, 2> jobs{};
    std::array<uint32_t, 2> host{}, device{}, childHost{}, childDevice{};
    Queue queue;
    Fixture()
    {
        ncclIntruQueueConstruct(&queue);
        for (int i = 0; i < 2; ++i)
        {
            auto &job = jobs[i];
            job.func = execute;
            job.undo = undo;
            job.destructor = destroy;
            job.abortFlag = &host[i];
            job.abortFlagDev = &device[i];
            job.childAbortFlag = &childHost[i];
            job.childAbortFlagDev = &childDevice[i];
            job.state = ncclGroupJobRunning;
            ncclIntruQueueEnqueue(&queue, &job);
            facts.abortFlags[4 * i] = &host[i];
            facts.abortFlags[4 * i + 1] = &device[i];
            facts.abortFlags[4 * i + 2] = &childHost[i];
            facts.abortFlags[4 * i + 3] = &childDevice[i];
        }
    }
    bool aborted() const
    {
        for (int i = 0; i < 2; ++i)
            if (host[i] != 1 || device[i] != 1 || childHost[i] != 1 || childDevice[i] != 1)
                return false;
        return true;
    }
};
int main()
{
    const rlimit noCore{0, 0};
    if (setrlimit(RLIMIT_CORE, &noCore))
        return 2;
    int violations = 0;
    for (int mode = 0; mode < 4; ++mode)
        for (int replay = 0; replay < 20; ++replay)
        {
            facts = {};
            facts.eager = mode % 2;
            facts.real = mode >= 2;
            Fixture fixture;
            bool abort = false;
            const auto status = asyncJobLaunch(&fixture.queue, &abort);
            if (status != ncclSystemError)
                throw std::runtime_error("Expected actual later create failure");
            if (!fixture.aborted() || facts.joins != 1 || facts.executions != 1 ||
                facts.destructors || facts.undos ||
                ncclLlaminarLoadThreadOwnership(&fixture.jobs[0]) !=
                    ncclNativeThreadOwnership::Joined ||
                ncclLlaminarLoadThreadOwnership(&fixture.jobs[1]) !=
                    ncclNativeThreadOwnership::Uncreated)
                ++violations;
            cleanup(&fixture.queue, status);
            if (facts.unsafe || facts.joins != 1 || facts.joinAttempts != 1 ||
                facts.executions != 1 || facts.destructors != 2 || facts.undos != 2)
                ++violations;
        }
    for (int mode = 0; mode < 2; ++mode)
        for (int replay = 0; replay < 20; ++replay)
        {
            int report[2];
            if (pipe(report))
                return 2;
            const auto child = fork();
            if (child < 0)
                return 2;
            if (!child)
            {
                close(report[0]);
                facts = {};
                facts.failJoin = true;
                facts.failCreate = mode == 0;
                facts.requireAbort = mode == 0;
                facts.eager = mode == 1;
                Fixture fixture;
                fatalOwner = &fixture.jobs[0];
                fatalPipe = report[1];
                bool abort = false;
                (void)asyncJobLaunch(&fixture.queue, &abort);
                _exit(4); // A real join failure must never reach any caller.
            }
            close(report[1]);
            char value = 0;
            const auto bytes = read(report[0], &value, 1);
            close(report[0]);
            int status = 0;
            if (waitpid(child, &status, 0) != child || bytes != 1 || value != 'Y' ||
                !WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT)
                ++violations;
        }
    std::cout << "{\"schema\":1,\"suite\":\"created-thread\",\"cases\":120,\"real_thread_cases\":40,\"join_failure_cases\":40,\"violations\":" << violations << "}\n";
    return violations ? 1 : 0;
}
