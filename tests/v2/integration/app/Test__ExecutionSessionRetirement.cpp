/**
 * @file Test__ExecutionSessionRetirement.cpp
 * @brief Prove inactive MPI finalization beside live selected execution ranks.
 *
 * A SELF deletion callback authenticates entry into MPI_Finalize before active
 * ranks run their selected-context collectives. MPI permits communication from
 * this callback before other world-model state is retired. The callback's one
 * matched notification is the excluded rank's last communication obligation.
 *
 * This executable owns the process lifecycle: a GTest main that finalizes only
 * after all tests cannot exercise the early-finalization edge. It loads no
 * model and initializes no accelerator. CTest supplies the ordinary 30-second
 * protocol bound; no idle-rank command loop or production callback is added.
 */
#include "utils/MPIContext.h"
#include "app/AppContext.h"
#include <cstdio>
#include <stdexcept>

namespace
{
    constexpr int finalization_entry_tag = 9184;

    /** @brief Outlive the process session to authenticate finalization order. */
    struct RetirementObserver
    {
        std::weak_ptr<llaminar2::IMPIContext> dependent;
        int discovery_rank = -1;
    };

    /**
     * @brief Notify an active rank from MPI's finalization-entry callback.
     * @return MPI status directly; exceptions cannot cross the C callback ABI.
     *
     * Rank 1 posts this exact receive before it begins any selected execution.
     * Returning from the callback completes rank 0's final communication; all
     * subsequent work excludes rank 0 and uses the selected communicator.
     */
    int enteredFinalize(MPI_Comm, int, void *attribute, void *)
    {
        const auto &observer = *static_cast<RetirementObserver *>(attribute);
        if (!observer.dependent.expired())
        {
            std::fputs("FAIL: MPI finalized before its context owner retired\n", stderr);
            return MPI_ERR_OTHER;
        }
        if (observer.discovery_rank != 0) return MPI_SUCCESS;
        int entered = 1;
        return MPI_Send(&entered, 1, MPI_INT, 1, finalization_entry_tag, MPI_COMM_WORLD);
    }

    /** @brief Convert an unexpected MPI status into a fail-fast test error. */
    void requireMPI(int status, const char *operation)
    {
        if (status != MPI_SUCCESS) throw std::runtime_error(operation);
    }
}

/**
 * @brief Retire discovery rank 0 while selected ranks 2/1 continue communicating.
 * @return Zero only after selected execution and every process's MPI shutdown.
 */
int main(int argc, char **argv)
{
    RetirementObserver observer;
    // The real application aggregate, not a hand-written reset sequence, owns
    // the finalizer. Its observer stays alive until after that scope retires.
    llaminar2::AppContext context{
        .mpi_session = llaminar2::MPIProcessSession::initialize(argc, argv)};
    try
    {
        context.mpi_ctx = llaminar2::MPIContextFactory::global();
        auto discovery = context.mpi_ctx;
        if (discovery->world_size() != 3)
            throw std::runtime_error("Execution retirement proof requires three ranks and MPI_THREAD_MULTIPLE");
        bool second_owner_rejected = false;
        try { auto duplicate = llaminar2::MPIProcessSession::initialize(argc, argv); }
        catch (const std::logic_error &) { second_owner_rejected = true; }
        if (!second_owner_rejected) throw std::runtime_error("A second process MPI owner was admitted");

        observer.dependent = context.mpi_ctx;
        observer.discovery_rank = discovery->rank();
        int key = MPI_KEYVAL_INVALID;
        requireMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, enteredFinalize, &key, nullptr),
                   "Cannot create finalization observer");
        requireMPI(MPI_Comm_set_attr(MPI_COMM_SELF, key, &observer), "Cannot bind finalization observer");
        requireMPI(MPI_Comm_free_keyval(&key), "Cannot release observer key");
        auto selection = llaminar2::MPIContextFactory::selectRanks(discovery, {2, 1});
        if (auto *inactive = std::get_if<llaminar2::InactiveMPIRank>(&selection))
        {
            if (inactive->discovery_rank != 0) throw std::runtime_error("Unexpected inactive rank");
            // Returning retires every dependent before the AppContext's
            // process owner enters finalization. No idle service loop is needed.
            return 0;
        }
        auto active = std::get<std::shared_ptr<llaminar2::MPIContext>>(selection);
        context.mpi_ctx = active;
        observer.dependent = active;
        int entered = 0;
        if (discovery->rank() == 1)
            requireMPI(MPI_Recv(&entered, 1, MPI_INT, 0, finalization_entry_tag,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE), "Finalization entry was not delivered");
        // Original rank 1 is selected rank 1. From here onward, even topology
        // construction must exclude the process already in MPI_Finalize.
        requireMPI(MPI_Bcast(&entered, 1, MPI_INT, 1, active->communicator()), "Cannot publish finalization entry");
        if (entered != 1) throw std::runtime_error("Finalization callback was not observed");
        if (!active->topology()) throw std::runtime_error("Selected topology was not constructed");
        for (int iteration = 0; iteration < 1000; ++iteration)
        {
            float value = static_cast<float>(active->rank() + 1);
            active->allreduce_sum_inplace(&value, 1);
            if (value != 3.0f) throw std::runtime_error("Selected allreduce produced the wrong result");
            if (active->clusterInventory()->world_size != 2)
                throw std::runtime_error("Selected inventory lost membership");
        }
        if (active->is_root())
            std::puts("PASS: 1000 selected allreduces after excluded rank entered MPI_Finalize");
        // Scope order alone releases selected owners and derived topology.
        // The finalization observer rejects any remaining context owner.
        return 0;
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
