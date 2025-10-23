// SPDX-License-Identifier: MIT
// Shared MPI test utilities for Llaminar test suite.
// Provides a consistent initialization/finalization pattern, rank helpers,
// root-only execution helpers, and a canonical GTest main macro.

#pragma once

#include <mpi.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <functional>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstdio>

namespace llaminar::test_util
{

    // RAII + static style environment controller. We deliberately separate init and finalize
    // so that tests which need a custom main (e.g. they install signal handlers) can still
    // control the ordering. The majority of test TU's should just invoke the convenience macro
    // LLAMINAR_DEFINE_GTEST_MPI_MAIN() at the bottom of the file instead of writing a custom main.
    class MPIEnvironment
    {
    public:
        // Initialize MPI (idempotent). Requests MPI_THREAD_FUNNELED which is sufficient for current tests.
        static void init(int *argc = nullptr, char ***argv = nullptr)
        {
            if (initialized_)
            {
                return; // Already initialized by earlier test TU or custom main.
            }
            int was = 0;
            MPI_Initialized(&was);
            if (!was)
            {
                int provided = 0;
                MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
                int tmp_rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
                int tmp_world = 1;
                MPI_Comm_size(MPI_COMM_WORLD, &tmp_world);
                fprintf(stderr, "[MPIEnvironment] rank %d init complete (provided thread level FUNNELED) world=%d\n", tmp_rank, tmp_world);
            }
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_);
            initialized_ = true;
        }

        // Finalize MPI (idempotent). A barrier is used to reduce noisy WARNING messages from OpenMPI.
        static void finalize()
        {
            if (!initialized_ || finalized_)
            {
                if (!initialized_)
                {
                    fprintf(stderr, "[MPIEnvironment] finalize early return: not initialized (rank %d)\n", rank_);
                }
                else if (finalized_)
                {
                    fprintf(stderr, "[MPIEnvironment] finalize early return: already finalized (rank %d)\n", rank_);
                }
                return;
            }
            int fin = 0;
            MPI_Finalized(&fin);
            if (!fin)
            {
                // Early debug snapshot of environment-driven finalize control on each rank.
                const char *raw_skip = std::getenv("LLAMINAR_TEST_SKIP_FINALIZE");
                fprintf(stderr, "[MPIEnvironment] rank %d entering finalize path (skip_var=%s, internal_flag=%d)\n", rank_, raw_skip ? raw_skip : "<unset>", (int)skip_finalize_flag_);
                const char *skip_finalize = std::getenv("LLAMINAR_TEST_SKIP_FINALIZE");
                if (skip_finalize_flag_ || (skip_finalize && skip_finalize[0] != '\0' && !(skip_finalize[0] == '0' && skip_finalize[1] == '\0')))
                {
                    // Intentionally skip finalize (debug mode) to avoid hang; MPI resources may leak.
                    fprintf(stderr, "[MPIEnvironment] rank %d skipping MPI_Finalize due to %s\n", rank_, skip_finalize_flag_ ? "internal skip flag" : "LLAMINAR_TEST_SKIP_FINALIZE env");
                    finalized_ = true;
                    return;
                }

                // Optional pre-finalize barrier only if explicitly requested.
                const char *force_barrier = std::getenv("LLAMINAR_TEST_FINALIZE_BARRIER");
                if (force_barrier && force_barrier[0] != '\0' && !(force_barrier[0] == '0' && force_barrier[1] == '\0'))
                {
                    // Best-effort; if this hangs user can set SKIP_FINALIZE or omit FINALIZE_BARRIER.
                    fprintf(stderr, "[MPIEnvironment] rank %d entering pre-finalize barrier\n", rank_);
                    MPI_Barrier(MPI_COMM_WORLD);
                    fprintf(stderr, "[MPIEnvironment] rank %d exited pre-finalize barrier\n", rank_);
                }

                // Watchdog: if finalize appears to hang, force terminate process.
                int timeout_ms = 0;
                if (const char *wd = std::getenv("LLAMINAR_TEST_FINALIZE_TIMEOUT_MS"))
                {
                    timeout_ms = std::atoi(wd);
                }
                std::atomic<bool> done{false};
                if (timeout_ms > 0)
                {
                    std::thread([timeout_ms, &done]()
                                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
                    if (!done.load(std::memory_order_relaxed)) {
                        fprintf(stderr, "[MPIEnvironment] finalize watchdog (%d ms) firing; forcing _exit. Set LLAMINAR_TEST_SKIP_FINALIZE=1 to bypass.\n", timeout_ms);
                        fflush(stderr);
                        _exit(91); // hard exit
                    } })
                        .detach();
                }
                fprintf(stderr, "[MPIEnvironment] rank %d calling MPI_Finalize...\n", rank_);
                MPI_Finalize();
                fprintf(stderr, "[MPIEnvironment] rank %d returned from MPI_Finalize\n", rank_);
                done.store(true, std::memory_order_relaxed);
            }
            finalized_ = true;
        }

        // Automatically call finalize at process end if user forgot the macro main.
        struct AtExitHook
        {
            ~AtExitHook()
            {
                int fin = 0;
                MPI_Finalized(&fin);
                int rk = -1;
                if (!fin)
                    MPI_Comm_rank(MPI_COMM_WORLD, &rk);
                fprintf(stderr, "[MPIEnvironment] AtExitHook dtor rank %d fin_state=%d calling finalize()\n", rk, fin);
                MPIEnvironment::finalize();
            }
        };

        static int rank()
        {
            ensure_init();
            return rank_;
        }
        static int world()
        {
            ensure_init();
            return world_;
        }
        static bool is_root() { return rank() == 0; }
        static void request_skip_finalize() { skip_finalize_flag_ = true; }

        // Convenience wrappers.
        static void barrier()
        {
            ensure_init();
            MPI_Barrier(MPI_COMM_WORLD);
        }

        template <class Fn>
        static void root_only(Fn &&fn)
        {
            if (is_root())
            {
                fn();
            }
        }

        // Skip a test unless the world size matches an expected count.
        static void skip_unless_world(int expected)
        {
            ensure_init();
            if (world_ != expected)
            {
                GTEST_SKIP() << "Requires world size " << expected << ", got " << world_;
            }
        }

        // Skip a test unless world size >= min_world.
        static void skip_unless_world_at_least(int min_world)
        {
            ensure_init();
            if (world_ < min_world)
            {
                GTEST_SKIP() << "Requires world size >= " << min_world << ", got " << world_;
            }
        }

        // Scoped environment variable override. Restores prior state (value or absence)
        // when destroyed. Useful for temporarily forcing backend selection or logging
        // verbosity without leaking state across tests.
        //
        // Usage:
        //   {
        //       MPIEnvironment::ScopedEnvVar force_cosma("LLAMINAR_COSMA_FORCE", "1");
        //       // ... test body with variable set ...
        //   } // variable restored/unset automatically here
        //
        // To temporarily unset an existing variable:
        //   MPIEnvironment::ScopedEnvVar unset_debug("LLAMINAR_COSMA_LOG_LEVEL");
        class ScopedEnvVar
        {
        public:
            // Set to a new value (value may be nullptr meaning "unset now").
            ScopedEnvVar(const char *name, const char *value = nullptr)
                : name_(name ? name : "")
            {
                if (name_.empty())
                    return;
                const char *prev = std::getenv(name_.c_str());
                if (prev)
                {
                    had_old_ = true;
                    old_value_ = prev;
                }
                if (value)
                {
                    setenv(name_.c_str(), value, 1);
                }
                else
                {
                    unsetenv(name_.c_str());
                }
            }
            // Non-copyable, movable
            ScopedEnvVar(const ScopedEnvVar &) = delete;
            ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;
            ScopedEnvVar(ScopedEnvVar &&other) noexcept { *this = std::move(other); }
            ScopedEnvVar &operator=(ScopedEnvVar &&other) noexcept
            {
                if (this != &other)
                {
                    restore();
                    name_ = std::move(other.name_);
                    old_value_ = std::move(other.old_value_);
                    had_old_ = other.had_old_;
                    other.had_old_ = false; // prevent double restore
                    other.name_.clear();
                }
                return *this;
            }
            ~ScopedEnvVar() { restore(); }

            void restore()
            {
                if (name_.empty())
                    return;
                if (had_old_)
                {
                    setenv(name_.c_str(), old_value_.c_str(), 1);
                }
                else
                {
                    unsetenv(name_.c_str());
                }
                // ensure idempotence
                had_old_ = true; // mark as restored; old_value_ retained
            }

        private:
            std::string name_;
            std::string old_value_;
            bool had_old_ = false;
        };

    private:
        static void ensure_init()
        {
            if (!initialized_)
                init();
        }

        inline static bool initialized_ = false;
        inline static bool finalized_ = false;
        inline static int rank_ = 0;
        inline static int world_ = 1;
        inline static bool skip_finalize_flag_ = false;
        inline static AtExitHook at_exit_{}; // ensures cleanup if user forgets explicit finalize.
    };

} // namespace llaminar::test_util

// Define a canonical GTest main that all MPI-enabled test binaries can share.
// Usage: place at the bottom of a test .cpp file (only once per binary):
//   LLAMINAR_DEFINE_GTEST_MPI_MAIN();
// If a test needs a custom main (additional setup) it may provide one instead; in that
// case it should explicitly call MPIEnvironment::init / finalize.
#define LLAMINAR_DEFINE_GTEST_MPI_MAIN()                                                                                                                                                \
    int main(int argc, char **argv)                                                                                                                                                     \
    {                                                                                                                                                                                   \
        fprintf(stderr, "[MPIEnvironment] main entry (pre-init)\n");                                                                                                                    \
        fflush(stderr);                                                                                                                                                                 \
        ::testing::InitGoogleTest(&argc, argv);                                                                                                                                         \
        ::llaminar::test_util::MPIEnvironment::init(&argc, &argv);                                                                                                                      \
        fprintf(stderr, "[MPIEnvironment] main after init rank=%d world=%d\n", ::llaminar::test_util::MPIEnvironment::rank(), ::llaminar::test_util::MPIEnvironment::world());          \
        fflush(stderr);                                                                                                                                                                 \
        int result = RUN_ALL_TESTS();                                                                                                                                                   \
        fprintf(stderr, "[MPIEnvironment] main after RUN_ALL_TESTS rank=%d world=%d\n", ::llaminar::test_util::MPIEnvironment::rank(), ::llaminar::test_util::MPIEnvironment::world()); \
        fflush(stderr);                                                                                                                                                                 \
        ::llaminar::test_util::MPIEnvironment::finalize();                                                                                                                              \
        fprintf(stderr, "[MPIEnvironment] main exiting result=%d\n", result);                                                                                                           \
        fflush(stderr);                                                                                                                                                                 \
        return result;                                                                                                                                                                  \
    }
