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
#include <memory>
#include <utility>


namespace llaminar2
{
    class PhysicalMemoryAuthority;
    class NativeCPUCollectiveProgram;
    namespace detail { struct MPICollectiveUse; }
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
         * @throws std::logic_error if startup ownership is missing, MPI already has an owner, or ended.
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
        friend class NativeCPUCollectiveProgram;
        friend struct detail::MPICollectiveUse;
        friend void mpiShutdown();
        struct State;
        /** @brief Attach one admitted domain to the original process-pool lifetime. */
        [[nodiscard]] static std::shared_ptr<detail::MPICollectiveUse> retainCPUCollectivePool(const std::shared_ptr<PhysicalMemoryAuthority>&);
        /** @brief Require the same original session still open immediately before native split entry. */
        static void requireCPUCollectiveNativeEntry(const std::shared_ptr<detail::MPICollectiveUse>&);
        /** @brief Close new domain admission before cache/context teardown begins. */
        static void closeCPUCollectiveAdmission();
        /** @brief Require every native domain retired before the original MPI_Finalize. */
        static void requireCPUCollectiveRetirement();
        /** @brief Release the sole process commitment only after successful original MPI_Finalize. */
        static void publishFinalizedCPUCollectivePool();
        /** @brief Reject external finalization that bypassed a live admitted pool. */
        static void requireNoUnretiredCPUCollectivePool();
        static std::weak_ptr<State> active_;
        std::shared_ptr<State> state_;
        /** @brief Ownership, not MPI execution state or a shadow of rank data. */
        enum class Ownership { Unowned, Owned };
        /** @brief Only successful initialization can arm the finalizer. */
        explicit MPIProcessSession(Ownership ownership, std::shared_ptr<State> state) noexcept : state_(std::move(state)), ownership_(ownership) {}
        Ownership ownership_{Ownership::Unowned};
    };
}
