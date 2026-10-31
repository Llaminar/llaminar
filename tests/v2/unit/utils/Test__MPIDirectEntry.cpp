/**
 * @file Test__MPIDirectEntry.cpp
 * @brief Replay actual MPI/GTest executable mains without initializing runtimes.
 *
 * The target selects the original production E2E main or shared fixture main
 * with LLAMINAR_MPI_ENTRY_SOURCE. Only external APIs are inert. Exact terminal
 * statuses, argument identity, lexical GPU ownership, and entry-before-initialization
 * are checked over repeated runs. Original mode verifies the precise missing-owner observations
 * with ordinary success; an arbitrary failure never counts as a negative proof.
 */
#include "utils/MPISessionOwner.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <stdexcept>
#include <vector>

namespace mpi_entry_replay
{
    enum class Event { Owner, GPUEnter, MPIInit, TestsInit, TestsRun,
                       CollectiveShutdown, GPURetire, ContextShutdown, MPIFinalize, GPUExit };
    std::vector<Event> events;
    std::optional<int> terminal;
    int original_argc = 0;
    char **original_argv = nullptr;
    const llaminar2::GPUDeviceProcessLifetime *gpu_owner = nullptr;
    bool gpu_constructed = false, gpu_retired = false, throw_from_tests = false;
    int mpi_init_result = 0, mpi_provided = 3, mpi_finalize_result = 0;

    /** @brief Stop on any incomplete or changed observation. */
    void require(bool condition)
    {
        if (!condition)
        {
            std::fputs("MPI entrypoint replay invariant failed\n", stderr);
            std::abort();
        }
    }

    /**
     * @brief Begin one independent child-process observation after the previous frame retired.
     * @param status Ancestor terminal result, or absence to continue in the owned child.
     * @details The selected real main is invoked repeatedly without a process fork;
     * only this external-API witness resets. Production terminal state is not modified.
     */
    void beginReplay(std::optional<int> status = std::nullopt)
    {
        require(gpu_owner == nullptr);
        events.clear(); terminal = status;
        gpu_constructed = false; gpu_retired = false; throw_from_tests = false;
        mpi_init_result = 0; mpi_provided = 3; mpi_finalize_result = 0;
    }

    /** @brief Verify actual library admission cannot precede the lexical GPU owner. */
    void requireGPUAdmission()
    {
#ifndef LLAMINAR_MPI_ENTRY_ORIGINAL
        llaminar2::GPUDeviceProcessLifetime::requireActive();
#endif
    }

    /** @brief Verify original arguments before a library could parse them. */
    void arguments(int argc, char **argv)
    {
        require(argc == original_argc && argv == original_argv);
        require(std::string(argv[0]) == "direct-mpi-test");
        require(std::string(argv[1]) == "--authored=value with spaces");
        require(std::string(argv[2]) == "--gtest_filter=actual:*" && argv[3] == nullptr);
    }

    void initializeTests(int *argc, char **argv)
    {
        arguments(*argc, argv);
        requireGPUAdmission();
        events.push_back(Event::TestsInit);
    }
    int runTests()
    {
        requireGPUAdmission();
        events.push_back(Event::TestsRun);
        if (throw_from_tests) throw std::runtime_error("recorded test-run unwind");
        return 19;
    }
    void shutdownContexts() { events.push_back(Event::ContextShutdown); }
    void shutdownCollectives()
    {
        requireGPUAdmission();
        events.push_back(Event::CollectiveShutdown);
    }
}

/** @brief Witness lexical construction only after the ancestor's child decision. */
llaminar2::GPUDeviceProcessLifetime::GPUDeviceProcessLifetime()
{
    using namespace mpi_entry_replay;
    require(!terminal && !gpu_constructed && !gpu_owner && !gpu_retired);
    require(events == std::vector<Event>{Event::Owner});
    gpu_constructed = true; gpu_owner = this;
    events.push_back(Event::GPUEnter);
}
/** @brief Scope unwind closes admission once, including after explicit pre-MPI retirement. */
llaminar2::GPUDeviceProcessLifetime::~GPUDeviceProcessLifetime() noexcept
{
    using namespace mpi_entry_replay;
    require(gpu_owner == this);
    retire();
    require(gpu_retired);
    events.push_back(Event::GPUExit);
    gpu_owner = nullptr;
}
/** @brief Record this exact boundary once; never model native teardown internals. */
void llaminar2::GPUDeviceProcessLifetime::retire() noexcept
{
    mpi_entry_replay::require(mpi_entry_replay::gpu_owner == this);
    retireActive();
}
/** @brief Reject every initialization attempted outside the live lexical frame. */
void llaminar2::GPUDeviceProcessLifetime::requireActive()
{
    mpi_entry_replay::require(mpi_entry_replay::gpu_owner && !mpi_entry_replay::gpu_retired);
}
/** @brief Observe terminal retirement while retaining lexical object identity until destruction. */
void llaminar2::GPUDeviceProcessLifetime::retireActive() noexcept
{
    using namespace mpi_entry_replay;
    if (!gpu_owner || gpu_retired) return;
    gpu_retired = true;
    events.push_back(Event::GPURetire);
}

std::optional<int> llaminar2::MPISessionOwner::enterProcess(int argc, char **argv)
{
    mpi_entry_replay::arguments(argc, argv);
    mpi_entry_replay::require(mpi_entry_replay::events.empty());
    mpi_entry_replay::events.push_back(mpi_entry_replay::Event::Owner);
    return mpi_entry_replay::terminal;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
    mpi_entry_replay::arguments(*argc, *argv);
    mpi_entry_replay::require(required == 3);
    mpi_entry_replay::requireGPUAdmission();
    mpi_entry_replay::events.push_back(mpi_entry_replay::Event::MPIInit);
    *provided = mpi_entry_replay::mpi_provided;
    return mpi_entry_replay::mpi_init_result;
}
int MPI_Finalize()
{
#ifndef LLAMINAR_MPI_ENTRY_ORIGINAL
    mpi_entry_replay::require(mpi_entry_replay::gpu_owner && mpi_entry_replay::gpu_retired);
#endif
    mpi_entry_replay::events.push_back(mpi_entry_replay::Event::MPIFinalize);
    return mpi_entry_replay::mpi_finalize_result;
}

// Compile the complete selected production main, rather than copying its body.
#define main llaminarActualMPIEntry
#include LLAMINAR_MPI_ENTRY_SOURCE
#undef main

/** @brief Replay both terminal short-circuit and locally continuing entry paths. */
int main()
{
    using namespace mpi_entry_replay;
    char executable[] = "direct-mpi-test";
    char authored[] = "--authored=value with spaces";
    char filter[] = "--gtest_filter=actual:*";
    char *argv[] = {executable, authored, filter, nullptr};
    original_argc = 3;
    original_argv = argv;
    const std::vector<Event> continuation = {
#ifndef LLAMINAR_MPI_ENTRY_ORIGINAL
        Event::Owner, Event::GPUEnter,
#endif
#ifndef LLAMINAR_MPI_ENTRY_SHARED_GTEST
        Event::MPIInit,
#endif
        Event::TestsInit, Event::TestsRun,
#ifdef LLAMINAR_MPI_ENTRY_INTEGRATION
        Event::CollectiveShutdown,
#endif
#ifdef LLAMINAR_MPI_ENTRY_ORIGINAL
        Event::ContextShutdown, Event::MPIFinalize
#else
        Event::GPURetire,
#ifndef LLAMINAR_MPI_ENTRY_SHARED_GTEST
        Event::MPIFinalize,
#endif
        Event::GPUExit
#endif
    };
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        for (const int status : {0, 17, 125})
        {
            beginReplay(status);
            const int result = llaminarActualMPIEntry(original_argc, argv);
#ifdef LLAMINAR_MPI_ENTRY_ORIGINAL
            // The original main ignores the terminal capability and initializes
            // every library. Require that exact defect on every negative run.
            require(result == 19 && events == continuation);
#else
            require(result == status && events == std::vector<Event>{Event::Owner});
#endif
        }
        beginReplay();
        require(llaminarActualMPIEntry(original_argc, argv) == 19);
        require(events == continuation);
        require(gpu_owner == nullptr);

        // An exception must traverse the real main's lexical owner. The old
        // main control has no lifetime events at all; an arbitrary throw is
        // never accepted in place of that exact missing-owner observation.
        beginReplay();
        throw_from_tests = true;
        try { (void)llaminarActualMPIEntry(original_argc, argv); require(false); }
        catch (const std::runtime_error &error)
        { require(std::string(error.what()) == "recorded test-run unwind"); }
        const std::vector<Event> unwind = {
#ifndef LLAMINAR_MPI_ENTRY_ORIGINAL
            Event::Owner, Event::GPUEnter,
#endif
#ifndef LLAMINAR_MPI_ENTRY_SHARED_GTEST
            Event::MPIInit,
#endif
            Event::TestsInit, Event::TestsRun,
#ifndef LLAMINAR_MPI_ENTRY_ORIGINAL
            Event::GPURetire, Event::GPUExit
#endif
        };
        require(events == unwind && gpu_owner == nullptr);

#ifdef LLAMINAR_MPI_ENTRY_INTEGRATION
        // Early MPI failures return through the actual scope destructor;
        // finalization failure retires before preserving status3 unchanged.
        for (const bool bad_status : {false, true})
        {
            beginReplay();
            mpi_init_result = bad_status ? 7 : 0;
            mpi_provided = bad_status ? 3 : 2;
            require(llaminarActualMPIEntry(original_argc, argv) == 2);
            require(events == std::vector<Event>{Event::Owner, Event::GPUEnter,
                Event::MPIInit, Event::GPURetire, Event::GPUExit});
            require(gpu_owner == nullptr);
        }
        beginReplay();
        mpi_finalize_result = 9;
        require(llaminarActualMPIEntry(original_argc, argv) == 3);
        require(events == continuation && gpu_owner == nullptr);
#endif
    }
    std::puts("{\"iterations\":20,\"terminal_cases\":60,\"continuations\":20,\"unwinds\":20,\"complete\":true}");
    return 0;
}
