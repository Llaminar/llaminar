/** @brief OpenMP-linked child of the isolated helper launch chain.
 *
 * Executed BY the placement helper, so the libgomp initialize_env
 * constructor has ALREADY run with the helper's exported OMP_PLACES /
 * OMP_PROC_BIND=close / OMP_NUM_THREADS. Verifies the ACTUAL native
 * contract, not just environment strings:
 *  - the receipt transport parses (strict module authority);
 *  - the native place list (omp_get_num_places/omp_get_place_proc_ids)
 *    equals the receipt's place sequence exactly;
 *  - omp_get_proc_bind() is not omp_proc_bind_false (managed intent is
 *    close; the ICV reports the runtime's actual decision);
 *  - in a parallel region, every thread's sched_getaffinity mask equals
 *    the proc-id set of ITS native place (omp_get_place_num), the master
 *    thread's place is the FIRST receipt place, and thread masks are
 *    pairwise disjoint for the full-budget case.
 * Exits 0 on full agreement, 1 with a stderr diagnostic otherwise.
 */
#include "planning/PlacementTopology.h"
#include "utils/DebugEnv.h"
#include "utils/MPISessionOwner.h"

#include <omp.h>
#include <sched.h>

#include <signal.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>

#include <algorithm>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace
{
    /** @brief Validate only the isolated helper's exec, not production admission. */
    void validateFixturePlacement()
    {
        const char *marker = llaminar2::DebugEnv::envValue("LLAMINAR_MANAGED_PLACEMENT");
        const char *transport = llaminar2::DebugEnv::envValue("LLAMINAR_PLACEMENT_RECEIPT_V1");
        const bool managed_context = marker && std::string_view(marker) == "1";

        if (managed_context)
        {
            if (!transport || !*transport)
                throw std::runtime_error(
                    "Managed placement context requires the placement receipt "
                    "(LLAMINAR_PLACEMENT_RECEIPT_V1 missing); refusing to run "
                    "without the agreed placement");
            llaminar2::placement::PlacementTopologyReject reject(
                llaminar2::placement::RejectReason::MalformedInput, "unset");
            const auto view = llaminar2::placement::parseReceiptTransportView(
                transport, reject);
            if (!view.has_value())
                throw std::runtime_error(std::string("Managed placement receipt rejected: ") +
                                         reject.what());
            // Owner-state loss in an owned context is FATAL before any
            // classification (never downgraded to an unowned mode).
            try
            {
                (void)llaminar2::MPISessionOwner::sharedDirectory();
            }
            catch (const std::exception &error)
            {
                throw std::runtime_error(std::string(
                    "Managed placement owner state is broken: ") + error.what());
            }
            // Explicit intent requires FULL placement: the native place list
            // must equal the receipt's sequence exactly (a runtime trim of a
            // receipt place is fatal, never silently accepted).
            const auto &places = view->receipt.ordered_places();
            if (omp_get_proc_bind() == omp_proc_bind_false)
                throw std::runtime_error(
                    "Managed placement requires bound places; the runtime "
                    "reports omp_proc_bind=false");
            if (omp_get_num_places() != static_cast<int>(places.size()))
                throw std::runtime_error("Managed placement receipt place count " +
                    std::to_string(places.size()) + " != native place count " +
                    std::to_string(omp_get_num_places()) +
                    " — explicit intent requires full placement");
            for (int i = 0; i < omp_get_num_places(); ++i)
            {
                std::vector<int> ids(CPU_SETSIZE, -1);
                omp_get_place_proc_ids(i, ids.data());
                ids.erase(std::remove(ids.begin(), ids.end(), -1), ids.end());
                std::sort(ids.begin(), ids.end());
                if (ids != places[static_cast<std::size_t>(i)].sibling_cpus)
                    throw std::runtime_error("Managed placement native place " +
                        std::to_string(i) + " does not match the receipt");
            }
            // Identity correlation (independent env, never self-declaration).
            const char *mpi_rank = llaminar2::DebugEnv::envValue("OMPI_COMM_WORLD_RANK");
            if (!mpi_rank || !*mpi_rank) mpi_rank = llaminar2::DebugEnv::envValue("PMI_RANK");
            if (!mpi_rank || !*mpi_rank || view->identity.rank != mpi_rank)
                throw std::runtime_error("Managed placement receipt rank '" +
                    view->identity.rank + "' does not match the MPI rank identity");
            char host[256];
            if (gethostname(host, sizeof(host)) != 0 ||
                view->identity.host != host)
                throw std::runtime_error("Managed placement receipt host '" +
                    view->identity.host + "' does not match this host");
            if (view->identity.budget < 1 ||
                view->identity.budget > static_cast<long>(places.size()))
                throw std::runtime_error("Managed placement receipt budget " +
                    std::to_string(view->identity.budget) + " outside [1, " +
                    std::to_string(places.size()) + "]");
            if (const char *threads = llaminar2::DebugEnv::envValue("OMP_NUM_THREADS"))
                if (view->identity.budget_origin_explicit &&
                    view->identity.budget != std::strtol(threads, nullptr, 10))
                    throw std::runtime_error("Managed placement receipt budget " +
                        std::to_string(view->identity.budget) +
                        " conflicts with authored OMP_NUM_THREADS=" + threads);
            // Deterministic team sizes; libgomp binds from the receipt env.
            omp_set_dynamic(0);
            return;
        }

        if (transport && *transport)
            throw std::runtime_error(
                "Unexpected placement receipt outside a managed placement "
                "context (stale environment?); refusing to run");

        // Native policies: MPI rank context or a plain direct run. The
        // authored OMP grammar is validated (malformed = fatal; the runtime's
        // warn-and-default is never silently certified); the runtime ICVs
        // decide bound vs unbound.
        const char *authored = llaminar2::DebugEnv::envValue("OMP_PLACES");
        llaminar2::placement::AuthoredPlacesInterpretation authored_view;
        bool authored_present = authored && *authored;
        if (authored_present)
            authored_view = llaminar2::placement::validateAuthoredOmpPlaces(authored);
        const bool bound = omp_get_proc_bind() != omp_proc_bind_false;
        if (bound)
        {
            // In an expected-bound mode a team thread outside any place is
            // fatal (checked for the master here; workers are checked by the
            // in-region occupancy regressions).
            if (omp_get_place_num() < 0)
                throw std::runtime_error(
                    "OpenMP binding is active but the master thread is not in "
                    "any place (omp_get_place_num() == -1)");
            if (authored_present &&
                authored_view.kind ==
                    llaminar2::placement::AuthoredPlacesInterpretation::Kind::Explicit)
            {
                // Explicit authored intent: the native list must equal the
                // validated interpretation (runtime trim of explicit intent
                // is fatal here too).
                if (omp_get_num_places() !=
                    static_cast<int>(authored_view.deterministic_places.size()))
                    throw std::runtime_error("Authored OMP_PLACES place count " +
                        std::to_string(authored_view.deterministic_places.size()) +
                        " != native place count " +
                        std::to_string(omp_get_num_places()) +
                        " — explicit intent requires full placement");
                for (int i = 0; i < omp_get_num_places(); ++i)
                {
                    std::vector<int> ids(CPU_SETSIZE, -1);
                    omp_get_place_proc_ids(i, ids.data());
                    ids.erase(std::remove(ids.begin(), ids.end(), -1), ids.end());
                    std::sort(ids.begin(), ids.end());
                    if (ids != authored_view.deterministic_places[
                            static_cast<std::size_t>(i)])
                        throw std::runtime_error("Authored OMP_PLACES native place " +
                            std::to_string(i) + " does not match the authored list");
                }
            }
            else if (authored_present && omp_get_num_places() == 0)
            {
                throw std::runtime_error(
                    "Authored OMP_PLACES present but the runtime built no "
                    "places (rejected value must not be silently defaulted)");
            }
        }
        // Unbound native policy is first-class: no occupancy checks, authored
        // budget (OMP_NUM_THREADS) remains the team authority.
        if (const char *threads = llaminar2::DebugEnv::envValue("OMP_NUM_THREADS"))
        {
            char *end = nullptr;
            const long value = std::strtol(threads, &end, 10);
            if (end == threads || *end != '\0' || value < 1)
                throw std::runtime_error("Authored OMP_NUM_THREADS='" +
                    std::string(threads) + "' is not a positive integer");
        }
        omp_set_dynamic(0);
    }
}

int main()
{
    // Observe the explicitly composed helper chain before the occupancy test.
    // This process is not the production rank placement authority.
    try
    {
        validateFixturePlacement();
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "[native-child] admission rejected: %s\n", error.what());
        return 1;
    }
    // Signal-cancellation mode: the admitted parked rank. The test provides
    // a DEDICATED ack channel (LLAMINAR_NATIVE_CHILD_ACK_SOCK): a unix
    // datagram socket addressed BY PATH (the env carries the path; a numeric
    // fd cannot survive the mpirun/helper/rank exec chain). Each record is
    // ONE datagram (atomic framing, no stream parsing):
    //  "RDY <attempt> <rank>\n"  — the admitted parked rank is ready;
    //  "GC <attempt> <pid>\n"    — the lineage sleeper's pid, WITH the
    //    creator-retained pidfd transferred via SCM_RIGHTS (the handle —
    //    opened immediately at fork, while the child is unreaped and its
    //    identity is pinned — is the authoritative identity; the numeric pid
    //    is correlation metadata only);
    //  "ADOPTED <attempt> <pid>\n" — sent BY the sleeper after it observes
    //    its own reparenting (the rank's death reparents it to the session
    //    guardian subreaper): actual surviving-parent adoption evidence.
    // The rank then parks until the guardian's SIGTERM.
    if (std::getenv("LLAMINAR_NATIVE_CHILD_PARK") != nullptr)
    {
        const char *ack_sock = std::getenv("LLAMINAR_NATIVE_CHILD_ACK_SOCK");
        const char *attempt = std::getenv("LLAMINAR_NATIVE_CHILD_ATTEMPT");
        const char *rank_env = std::getenv("OMPI_COMM_WORLD_RANK");

        const int sock = ack_sock && *ack_sock
            ? ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0) : -1;
        if (sock >= 0)
        {
            sockaddr_un address {};
            address.sun_family = AF_UNIX;
            std::strncpy(address.sun_path, ack_sock, sizeof(address.sun_path) - 1);
            if (::connect(sock, reinterpret_cast<sockaddr *>(&address),
                    sizeof(address)) == 0)
            {
                const std::string ack = std::string("RDY ") +
                    (attempt && *attempt ? attempt : "?") + " " +
                    (rank_env && *rank_env ? rank_env : "?") + "\n";
                ::send(sock, ack.data(), ack.size(), MSG_NOSIGNAL);
            }
        }

        if (std::getenv("LLAMINAR_NATIVE_CHILD_GRANDCHILD") != nullptr)
        {
            const pid_t grandchild = ::fork();
            if (grandchild == 0)
            {
                // The sleeper ESCAPES the MPI job's process group first
                // (setsid: a new session). Open MPI's job termination
                // signals the whole launched process group -- without this,
                // the sleeper dies alongside its parent and the
                // surviving-parent adoption is never exercised (strace
                // evidence: mpirun SIGTERMs the rank AND the in-group
                // sleeper). Detached, it outlives the rank, is reparented
                // to the session guardian (PR_SET_CHILD_SUBREAPER), reports
                // the adoption, and is terminated by the guardian. A setsid
                // failure degrades to group death and the test's missing
                // adoption record fails it loudly.
                ::setsid();
                // Block SIGTERM DURING the reparent watch: the guardian
                // signals an adopted child as soon as it observes it (the
                // race proven by strace -- the guardian's SIGTERM can land
                // between this watcher's polls). With SIGTERM blocked, the
                // guardian's termination stays PENDING while this sleeper
                // observes its own reparenting, sends the adoption record,
                // and only then restores the mask: the pending SIGTERM
                // completes the guardian's termination (SIGKILL remains the
                // guardian's unblockable backstop within its grace).
                sigset_t block, previous;
                sigemptyset(&block);
                sigaddset(&block, SIGTERM);
                ::sigprocmask(SIG_BLOCK, &block, &previous);
                if (sock >= 0) ::close(sock);
                const int own = ack_sock && *ack_sock
                    ? ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0) : -1;
                bool connected = false;
                if (own >= 0)
                {
                    sockaddr_un address {};
                    address.sun_family = AF_UNIX;
                    std::strncpy(address.sun_path, ack_sock,
                        sizeof(address.sun_path) - 1);
                    connected = ::connect(own,
                        reinterpret_cast<sockaddr *>(&address),
                        sizeof(address)) == 0;
                }
                const pid_t parent = ::getppid();
                for (;;)
                {
                    if (::getppid() != parent) break;
                    ::usleep(10000);
                }
                if (connected)
                {
                    const std::string report = std::string("ADOPTED ") +
                        (attempt && *attempt ? attempt : "?") + " " +
                        std::to_string(::getpid()) + "\n";
                    // The datagram queues atomically: even if the guardian's
                    // termination lands immediately after, the test receives
                    // the adoption record.
                    ::send(own, report.data(), report.size(), MSG_NOSIGNAL);
                }
                if (own >= 0) ::close(own);
                // Restore the mask: a pending guardian SIGTERM (if already
                // delivered while blocked) completes the termination now,
                // AFTER the adoption record was sent; otherwise the sleeper
                // parks and a later guardian signal terminates it normally.
                ::sigprocmask(SIG_SETMASK, &previous, nullptr);
                for (;;) ::pause();
            }
            // The creator-retained handle, opened IMMEDIATELY at fork: the
            // child is this process's unreaped direct child, so its identity
            // is pinned (a pidfd on an unreaped child references exactly that
            // fork — no exit/reap/reuse window exists).
            const int grandchild_pidfd = static_cast<int>(
                ::syscall(SYS_pidfd_open, grandchild, 0));
            if (grandchild_pidfd >= 0 && sock >= 0)
            {
                const std::string line = std::string("GC ") +
                    (attempt && *attempt ? attempt : "?") + " " +
                    std::to_string(grandchild) + "\n";
                // Transfer the retained handle with the record: SCM_RIGHTS
                // duplicates the pidfd into the receiving test process.
                char control[CMSG_SPACE(sizeof(int))] = {};
                msghdr message {};
                iovec vector { const_cast<char *>(line.data()), line.size() };
                message.msg_iov = &vector;
                message.msg_iovlen = 1;
                message.msg_control = control;
                message.msg_controllen = sizeof(control);
                cmsghdr *header = CMSG_FIRSTHDR(&message);
                header->cmsg_level = SOL_SOCKET;
                header->cmsg_type = SCM_RIGHTS;
                header->cmsg_len = CMSG_LEN(sizeof(int));
                std::memcpy(CMSG_DATA(header), &grandchild_pidfd, sizeof(int));
                message.msg_controllen = CMSG_LEN(sizeof(int));
                ::sendmsg(sock, &message, MSG_NOSIGNAL);
                ::close(grandchild_pidfd);
            }
            else if (grandchild_pidfd >= 0)
            {
                ::close(grandchild_pidfd);
            }
        }
        if (sock >= 0) ::close(sock);
        // Failure-injection variant: a rank that IGNORES SIGTERM (used by
        // the teardown escalation test; the guardian's own SIGKILL grace
        // still retires it).
        if (std::getenv("LLAMINAR_NATIVE_CHILD_IGNORE_SIGTERM") != nullptr)
            ::signal(SIGTERM, SIG_IGN);
        for (;;)
        {
            ::pause();
        }
    }
    const char *transport = std::getenv("LLAMINAR_PLACEMENT_RECEIPT_V1");
    // Admission (above) routes the mode; in native modes there is no receipt
    // and the occupancy checks below use the runtime's own place list.
    std::vector<std::vector<int>> receipt_places;
    if (transport && *transport)
    {
        llaminar2::placement::PlacementTopologyReject reject(
            llaminar2::placement::RejectReason::MalformedInput, "unset");
        const auto receipt = llaminar2::placement::parseReceiptTransport(transport, reject);
        if (!receipt.has_value())
        {
            std::fprintf(stderr, "[native-child] receipt rejected: %s\n", reject.what());
            return 1;
        }
        for (const auto &place : receipt->ordered_places())
            receipt_places.push_back(place.sibling_cpus);
    }
    const auto &places = receipt_places;
    // Managed intent is close binding; the runtime ICV must agree. Native
    // unbound policy (no receipt) is first-class and explicitly legal.
    if (!receipt_places.empty() && omp_get_proc_bind() == omp_proc_bind_false)
    {
        std::fprintf(stderr, "[native-child] runtime reports unbound (proc_bind=false)\n");
        return 1;
    }
    // The native place list must equal the receipt's sequence exactly
    // (managed mode only; native modes have no receipt to compare).
    if (!places.empty() && omp_get_num_places() != static_cast<int>(places.size()))
    {
        std::fprintf(stderr, "[native-child] native place count %d != receipt %zu\n",
            omp_get_num_places(), places.size());
        return 1;
    }
    // The native list is sized by the RUNTIME's place count (the receipt is
    // optional in native modes; comparing only when a receipt exists).
    std::vector<std::vector<int>> native(
        static_cast<std::size_t>(omp_get_num_places()));
    for (int i = 0; i < omp_get_num_places(); ++i)
    {
        // libgomp provides omp_get_place_proc_ids(place, ids); the id count
        // is bounded by CPU_SETSIZE, so size the probe buffer accordingly.
        std::vector<int> ids(CPU_SETSIZE, -1);
        omp_get_place_proc_ids(i, ids.data());
        ids.erase(std::remove(ids.begin(), ids.end(), -1), ids.end());
        native[static_cast<std::size_t>(i)] = ids;
    }
    for (std::size_t i = 0; !places.empty() && i < places.size(); ++i)
        if (native[i] != places[i])
        {
            std::fprintf(stderr, "[native-child] native place %zu != receipt place\n", i);
            return 1;
        }
    // In-region occupancy: every thread's mask == its native place's ids;
    // budget-aware (full budget => exact partition; partial budget => the
    // first-k places occupied, idle places legitimate); repeated regions +
    // a nested region re-verified at every entry (retained pool threads).
    const int team = omp_get_max_threads();
    int failures = 0;
    int master_place = -1;
    // Occupancy checks apply only in BOUND states; unbound native policy
    // legitimately runs threads anywhere (no place, no mask contract).
    const bool bound = omp_get_proc_bind() != omp_proc_bind_false;
    for (int repetition = bound ? 0 : 0; bound && repetition < 3; ++repetition)
    {
#pragma omp parallel num_threads(team) reduction(| : failures)
        {
            const int place_num = omp_get_place_num();
            const int thread_num = omp_get_thread_num();
            if (thread_num == 0) master_place = place_num;
            if (place_num < 0 || place_num >= omp_get_num_places())
            {
                std::fprintf(stderr, "[native-child] rep %d thread %d place %d invalid\n",
                    repetition, thread_num, place_num);
                failures |= 1;
            }
            else
            {
                std::vector<int> ids(CPU_SETSIZE, -1);
                omp_get_place_proc_ids(place_num, ids.data());
                ids.erase(std::remove(ids.begin(), ids.end(), -1), ids.end());
                cpu_set_t mask;
                CPU_ZERO(&mask);
                if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
                {
                    std::fprintf(stderr, "[native-child] thread %d cannot read mask\n",
                        thread_num);
                    failures |= 1;
                }
                else
                {
                    for (const int cpu : ids)
                        if (!CPU_ISSET(cpu, &mask))
                        {
                            std::fprintf(stderr,
                                "[native-child] rep %d thread %d mask misses place cpu %d\n",
                                repetition, thread_num, cpu);
                            failures |= 1;
                        }
                    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                        if (CPU_ISSET(cpu, &mask) &&
                            std::find(ids.begin(), ids.end(), cpu) == ids.end())
                        {
                            std::fprintf(stderr,
                                "[native-child] rep %d thread %d mask has extra cpu %d\n",
                                repetition, thread_num, cpu);
                            failures |= 1;
                        }
#pragma omp parallel num_threads(1) reduction(| : failures)
                    {
                        cpu_set_t nested_mask;
                        CPU_ZERO(&nested_mask);
                        if (sched_getaffinity(0, sizeof(nested_mask), &nested_mask) != 0 ||
                            std::memcmp(&nested_mask, &mask, sizeof(mask)) != 0)
                        {
                            std::fprintf(stderr,
                                "[native-child] rep %d nested mask differs from parent place\n",
                                repetition);
                            failures |= 1;
                        }
                    }
                }
            }
        }
    }
    // The master-place check applies only in BOUND states; unbound native
    // policy legitimately has no place (omp_get_place_num() == -1).
    if (omp_get_proc_bind() != omp_proc_bind_false && master_place != 0)
    {
        std::fprintf(stderr, "[native-child] master place %d != first receipt place\n",
            master_place);
        failures |= 1;
    }
    // Full-budget partition: team == place count => every place occupied.
    if (team == omp_get_num_places())
    {
        std::vector<char> covered(CPU_SETSIZE, 0);
#pragma omp parallel num_threads(team)
        {
            const int place_num = omp_get_place_num();
            if (place_num >= 0)
            {
                std::vector<int> ids(CPU_SETSIZE, -1);
                omp_get_place_proc_ids(place_num, ids.data());
#pragma omp critical
                {
                    for (const int cpu : ids)
                        if (cpu >= 0) covered[static_cast<std::size_t>(cpu)] = 1;
                }
            }
        }
        for (int place = 0; place < omp_get_num_places(); ++place)
        {
            std::vector<int> ids(CPU_SETSIZE, -1);
            omp_get_place_proc_ids(place, ids.data());
            for (const int cpu : ids)
                if (cpu >= 0 && !covered[static_cast<std::size_t>(cpu)])
                {
                    std::fprintf(stderr,
                        "[native-child] full-budget partition misses place %d cpu %d\n",
                        place, cpu);
                    failures |= 1;
                }
        }
    }
    return failures == 0 ? 0 : 1;
}