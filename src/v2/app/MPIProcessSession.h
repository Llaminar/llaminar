/**
 * @file MPIProcessSession.h
 * @brief Scope-bound ownership of the process MPI runtime, not a communicator.
 *
 * Declare this owner before runner/context owners so it is destroyed after
 * them. Moving transfers finalization responsibility; a default or moved-from
 * object performs no MPI calls. Application modes terminate requests but must
 * never finalize the process while their adapters or callers still own state.
 */
#pragma once

namespace llaminar2
{
    /** @brief Single move-only owner of a process's initialized MPI runtime. */
    class MPIProcessSession final
    {
    public:
        /** @brief Empty scope for pure unit fixtures or explicit local discovery. */
        MPIProcessSession() noexcept = default;
        /**
         * @brief Initialize and collectively authenticate MPI_THREAD_MULTIPLE.
         * @param argc Argument count MPI may modify.
         * @param argv Argument vector MPI may modify.
         * @return The only scope responsible for this process's MPI shutdown.
         * @throws std::runtime_error if initialization/thread support fails.
         * @throws std::logic_error if MPI already has a process owner or ended.
         */
        [[nodiscard]] static MPIProcessSession initialize(int &argc, char **&argv);
        /** @brief Transfer finalization responsibility without touching MPI. */
        MPIProcessSession(MPIProcessSession &&other) noexcept;
        /** @brief Retire global MPI-dependent caches and finalize an owned session. */
        ~MPIProcessSession();

        MPIProcessSession(const MPIProcessSession &) = delete;
        MPIProcessSession &operator=(const MPIProcessSession &) = delete;
        // Replacing a live owner could finalize underneath its dependent
        // members. Construction/move construction are the only legal handoffs.
        MPIProcessSession &operator=(MPIProcessSession &&) = delete;

        /** @return Whether this scope, rather than its moved-from source, owns MPI. */
        [[nodiscard]] bool ownsMPI() const noexcept { return ownership_ == Ownership::Owned; }

    private:
        /** @brief Ownership, not MPI execution state or a shadow of rank data. */
        enum class Ownership { Unowned, Owned };
        /** @brief Only successful initialization can arm the finalizer. */
        explicit MPIProcessSession(Ownership ownership) noexcept : ownership_(ownership) {}
        Ownership ownership_{Ownership::Unowned};
    };
}
