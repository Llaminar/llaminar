/**
 * @file MPIProcessSession.cpp
 * @brief One process initialization/finalization scope shared by every frontend.
 *
 * Initial thread support is agreed before creating any communicator owners.
 * Thereafter normal C++ scope order retires request handlers, runners and MPI
 * contexts before the existing infrastructure shutdown boundary is entered.
 * The executable entrypoint must already own an ancestor session lifetime,
 * including any singleton daemon MPI creates. Initialization never restarts
 * its caller; missing ownership fails before native MPI state is created.
 */
#include "app/MPIProcessSession.h"
#include "app/MPIShutdown.h"
#include "utils/MPISessionOwner.h"
#include <mpi.h>
#include <stdexcept>
#include <utility>
#include <mutex>
#include <optional>
#include "collective/NativeCPUCollectiveProgram.h"

namespace llaminar2
{
    /** @brief Original session ownership, not a second physical-byte ledger. */
    struct MPIProcessSession::State {
        std::mutex mutex;
        std::optional<PhysicalMemoryOwnerReservation> pool;
        std::size_t participations = 0;
        enum class Phase { Open, Closing, Finalizing, Finalized };
        Phase phase = Phase::Open;
    };
    std::weak_ptr<MPIProcessSession::State> MPIProcessSession::active_;
    namespace detail {
        /** @brief Split borrower retains only the original session lifecycle and one use count. */
        struct MPICollectiveUse {
            std::shared_ptr<MPIProcessSession::State> original_session;
            ~MPICollectiveUse() noexcept {
                if (!original_session) return;
                std::lock_guard lock(original_session->mutex);
                if (!original_session->participations || original_session->phase == MPIProcessSession::State::Phase::Finalized) std::terminate();
                --original_session->participations;
            }
        };
    }

    std::shared_ptr<detail::MPICollectiveUse> MPIProcessSession::retainCPUCollectivePool(const std::shared_ptr<PhysicalMemoryAuthority>& memory)
    {
        auto session = active_.lock();
        if (!session || !memory) throw std::logic_error("CPU collective requires its live original MPIProcessSession and admitted PMA");
        auto use = std::make_shared<detail::MPICollectiveUse>();
        std::lock_guard lock(session->mutex);
        if (session->phase != State::Phase::Open) throw std::logic_error("CPU collective process session is closing");
        // Reserve once. Later model admissions budget the same potential growth
        // and borrow the original canonical root, never concrete resident credit.
        if (!session->pool) session->pool.emplace(memory->reserveNewAllocations(DeviceId::cpu(), PhysicalMemoryOwner::NativeCPUCollectiveProcessPool, NativeCPUCollectiveProgram::processPoolBytes()));
        else memory->borrowCPUCollectiveProcessPool(*session->pool);
        use->original_session = session;
        ++session->participations;
        return use;
    }
    void MPIProcessSession::requireCPUCollectiveNativeEntry(const std::shared_ptr<detail::MPICollectiveUse>& use) {
        const auto session=use ? use->original_session : nullptr;
        if(!session || active_.lock()!=session) throw std::logic_error("CPU split requires its original live MPI session");
        std::lock_guard lock(session->mutex);
        if(session->phase!=State::Phase::Open || !session->participations)
            throw std::logic_error("CPU split native entry cannot follow original session admission closure");
    }
    void MPIProcessSession::closeCPUCollectiveAdmission() {
        if (auto session=active_.lock()) { std::lock_guard lock(session->mutex); if (session->phase == State::Phase::Open) session->phase=State::Phase::Closing; }
    }
    void MPIProcessSession::requireCPUCollectiveRetirement() {
        if (auto session=active_.lock()) { std::lock_guard lock(session->mutex); if(session->participations) throw std::logic_error("MPI_Finalize requires every admitted CPU domain's original native retirement");
            if(session->phase == State::Phase::Finalizing || session->phase == State::Phase::Finalized)
                throw std::logic_error("Previous MPI_Finalize has no original successful retirement; never retry native finalization");
            session->phase=State::Phase::Finalizing; }
    }
    void MPIProcessSession::publishFinalizedCPUCollectivePool() {
        if (auto session=active_.lock()) { std::lock_guard lock(session->mutex); if(session->participations || session->phase != State::Phase::Finalizing) std::terminate(); session->pool.reset(); session->phase=State::Phase::Finalized; }
    }
    void MPIProcessSession::requireNoUnretiredCPUCollectivePool() {
        if (auto session=active_.lock()) { std::lock_guard lock(session->mutex); if(session->pool || session->participations || session->phase == State::Phase::Finalizing) throw std::logic_error("MPI was finalized outside its original CPU collective pool owner"); }
    }

    MPIProcessSession MPIProcessSession::initialize(int &argc, char **&argv)
    {
        MPISessionOwner::requireProcessOwner();
        int initialized = 0, finalized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS || MPI_Finalized(&finalized) != MPI_SUCCESS)
            throw std::runtime_error("Cannot inspect process MPI lifecycle");
        if (initialized || finalized)
            throw std::logic_error("MPI process initialization requires an unowned, never-finalized runtime");

        auto original_state = std::make_shared<State>();
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
            throw std::runtime_error("Failed to initialize process MPI runtime");
        MPIProcessSession session(Ownership::Owned, std::move(original_state));
        active_ = session.state_;
        int compatible = provided >= MPI_THREAD_MULTIPLE;
        // All ranks must agree before any can enter discovery. On rejection
        // every armed scope unwinds together, rather than stranding a peer.
        if (MPI_Allreduce(MPI_IN_PLACE, &compatible, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS)
            throw std::runtime_error("Cannot authenticate process MPI thread support");
        if (!compatible)
            throw std::runtime_error("Llaminar requires MPI_THREAD_MULTIPLE on every process");
        return session;
    }

    MPIProcessSession::MPIProcessSession(MPIProcessSession &&other) noexcept
        : state_(std::move(other.state_)), ownership_(std::exchange(other.ownership_, Ownership::Unowned)) {}

    MPIProcessSession::~MPIProcessSession()
    {
        if (ownsMPI()) mpiShutdown();
    }
}
