/**
 * @file MoEOverlayAsyncFailureProbe.cpp
 * @brief Native-process replay of async service failure retaining its original owner.
 *
 * Real original MPI ranks exchange production transaction tickets and receipts.
 * A device-free completion fence supplies the exact failure observation. The
 * failed process must terminate inside production follower teardown before its
 * retained owner can be released. An independent launcher validates durable
 * markers; no initialized MPI process forks and no failed work is acknowledged.
 */
#include "app/MPIProcessSession.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "execution/moe/MoEOverlayInferenceTransactionService.h"
#include "utils/MPIContext.h"
#include "utils/MPISessionOwner.h"
#include <mpi.h>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace {
std::filesystem::path report;
/** @brief Write one flushed observation owned exclusively by this child job. */
void mark(const char *name) { std::ofstream(report / name) << "original-command=11 abi=6\n"; }
/** @brief Track the storage authority retained by the actual async service fence. */
struct Owner {
    ~Owner() { mark("owner-released"); }
};
/** @brief Emit the expected production fatal boundary without unwinding live ownership. */
[[noreturn]] void fatal() noexcept { mark("fatal-before-owner-release"); std::_Exit(86); }
/** @brief Replay a concrete failed native observation, or its successful control. */
class Fence final : public llaminar2::IMoEOverlayInferenceCompletionFence {
    std::shared_ptr<Owner> owner_;
    bool fail_;
    int polls_ = 0;
public:
    Fence(std::shared_ptr<Owner> owner, bool fail) : owner_(std::move(owner)), fail_(fail) {}
    llaminar2::MoEOverlayInferenceCompletionFenceProgress poll(std::string *error) noexcept override {
        using P = llaminar2::MoEOverlayInferenceCompletionFenceProgress;
        if (polls_++ == 0) return P::Pending;
        if (fail_) { if (error) *error = "recorded-native-fence-failed"; return P::Failed; }
        return P::Ready;
    }
};
/** @brief Submit through the real service seam; the synchronous API cannot certify work. */
class Executor final : public llaminar2::IMoEOverlayInferenceTransactionExecutor {
    std::shared_ptr<Fence> fence_;
public:
    Executor(std::shared_ptr<Owner> owner, bool fail) : fence_(std::make_shared<Fence>(std::move(owner), fail)) {}
    /** @brief Withdraw the producer's setup reference so the service owns the sole native lease. */
    void releaseSubmissionOwner() noexcept { fence_.reset(); }
    bool executeMoEOverlayInferenceTransaction(const llaminar2::MoEOverlayInferenceTransactionTicket &, std::string *) override { return false; }
    llaminar2::MoEOverlayInferenceAsyncSubmission submitMoEOverlayInferenceTransaction(
        const llaminar2::MoEOverlayInferenceTransactionTicket &ticket, std::string *error) override {
        using P = llaminar2::MoEOverlayInferenceCompletionFenceProgress;
        if (ticket.command_id != 11 || ticket.request_generation != 9 || ticket.abi_version != llaminar2::MoEOverlayInferenceTransactionTicket::kABIVersion) {
            if (error) *error = "original-ticket-identity-mismatch";
            return {P::Failed, {}};
        }
        mark("submitted-original-ticket");
        return {P::Pending, fence_};
    }
};
}

/** @brief Run positive retirement or fail-closed teardown under the canonical original MPI owner. */
int main(int argc, char **argv) {
    if (auto status = llaminar2::MPISessionOwner::enterProcess(argc, argv)) return *status;
    if (argc != 3) return 125;
    report = argv[1];
    const std::string mode = argv[2];
    if (mode != "ready" && mode != "failed") return 125;
    llaminar2::GPUDeviceProcessLifetime device_lifetime;
    auto process = llaminar2::MPIProcessSession::initialize(argc, argv);
    auto context = llaminar2::MPIContextFactory::global();
    if (context->world_size() != 2) return 125;
    const llaminar2::MoEOverlayInferenceTopologyIdentity topology{
        .workspace_generation = 7, .topology_fingerprint_low = 123,
        .topology_fingerprint_high = 456, .source_world_rank = 0, .target_world_rank = 1};
    const llaminar2::MoEOverlayInferenceTransactionProtocol::Config protocol{
        .topology = topology, .slot_count = 2, .max_request_count = 1,
        .max_rows_per_request = 1, .max_mtp_draft_depth = 0};
    auto channel = std::make_shared<llaminar2::MoEOverlayMPIInferenceTransactionChannel>(
        llaminar2::MoEOverlayMPIInferenceTransactionChannel::Config{
            .mpi_ctx = context, .source_world_rank = 0, .target_world_rank = 1, .send_slot_count = 2});
    if (context->rank() == 0) {
        llaminar2::MoEOverlayInferenceTransactionPublisher publisher({channel, protocol});
        std::string error;
        if (!publisher.beginCommand({9, 11, 17}, &error)) return 125;
        const auto transaction = publisher.publish({
            .graph_role = llaminar2::MoEOverlayInferenceGraphRole::MainDecode,
            .logical_step_id = 1, .placement_epoch = 17, .request_count = 1,
            .logical_rows_per_request = 1, .physical_rows_per_request = 1});
        if (!transaction.ok) return 125;
        if (mode == "ready") {
            if (!publisher.retire(transaction, &error) || !publisher.complete(17, 0, &error)) return 125;
            mark("completion-retired");
        } else {
            int observed = 0;
            MPI_Recv(&observed, 1, MPI_INT, 1, 19009, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (observed != 1 || publisher.tryRetire(transaction, &error) !=
                llaminar2::MoEOverlayInferenceCompletionFenceProgress::Pending) return 125;
            mark("no-false-completion");
            MPI_Send(&observed, 1, MPI_INT, 1, 19010, MPI_COMM_WORLD);
            // The peer's expected native fatal must retire this original MPI job.
            // Returning here would destroy a live receive without terminal proof.
            MPI_Barrier(MPI_COMM_WORLD);
            return 125;
        }
    } else {
        auto owner = std::make_shared<Owner>();
        Executor executor(owner, mode == "failed");
        {
            llaminar2::MoEOverlayInferenceTransactionFollower follower({
                .channel = channel, .executor = &executor, .protocol = protocol});
            const auto result = follower.runOneCommand();
            // After submission, only the production service may retain the
            // native owner. A premature pending-queue release becomes visible.
            executor.releaseSubmissionOwner();
            owner.reset();
            if (mode == "ready") {
                if (!result.ok || result.executed_transactions != 1) return 125;
            } else {
                if (result.ok || result.error != "recorded-native-fence-failed") return 125;
                mark("service-retained-failed-fence");
                int observed = 1;
                MPI_Send(&observed, 1, MPI_INT, 0, 19009, MPI_COMM_WORLD);
                MPI_Recv(&observed, 1, MPI_INT, 0, 19010, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::set_terminate(fatal);
                // The production destructor rechecks the retained failed native
                // fence and terminates before dropping either fence or owner.
            }
        }
        if (mode == "failed") { mark("unexpected-owner-unwind"); return 125; }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
