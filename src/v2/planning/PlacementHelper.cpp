/**
 * @file PlacementHelper.cpp
 * @brief The managed-placement helper executable (APPROVED bounded scope:
 *        isolated test launch chain only — NO production launcher wiring).
 *
 * Runs in the rank's process slot between mpirun's exec and the application
 * exec (the agreed per-rank APPLICATION argv boundary). OpenMP-free: links
 * ONLY llaminar_placement_topology (+OpenSSL via the module), never
 * llaminar2_core, never libgomp — its ELF closure must not contain libgomp
 * because the receipt must exist before the application's libgomp
 * initialize_env constructor runs.
 *
 * Behavior: read the inherited rank affinity mask (the full admitted set —
 * no OpenMP constructor has run in this image), derive the placement receipt
 * with the strict module authority, validate the authored budget against the
 * derived place count, serialize the transport with session identity read
 * from the INDEPENDENT MPI rank environment (never invented), export the
 * receipt + OMP placement variables, then execvp the application argv after
 * "--" verbatim. Failure branches never fall back to anything.
 *
 * Exit codes (typed): 0 = unreachable in practice (execvp replaces the
 * image); 2 = usage/argv error; 3 = typed topology/capacity rejection; 4 =
 * budget invalid or conflicts with an authored OMP_NUM_THREADS; 5 = execvp
 * of the application failed; 6 = independent rank identity unavailable.
 * All diagnostics are single-line, "[placement-helper]"-prefixed, on stderr.
 * The helper forks nothing, creates no session, writes no files, and keeps
 * the parent environment (adding only the placement variables).
 */
#include "planning/PlacementTopology.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include <sched.h>

namespace
{
    constexpr int kExitUsage = 2;
    constexpr int kExitTopology = 3;
    constexpr int kExitBudget = 4;
    constexpr int kExitExec = 5;
    constexpr int kExitIdentity = 6;

    [[noreturn]] void fail(const int code, const std::string &message)
    {
        std::fprintf(stderr, "[placement-helper] %s\n", message.c_str());
        std::_Exit(code);
    }

    std::vector<int> ownAdmittedCpus()
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
            fail(kExitTopology, "cannot read the inherited rank affinity mask");
        std::vector<int> admitted;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask))
                admitted.push_back(cpu);
        return admitted;
    }

    /// Independent MPI rank identity from the EXISTING environment bindings.
    std::string independentRankIdentity()
    {
        for (const char *key : {"OMPI_COMM_WORLD_RANK", "PMI_RANK"})
            if (const char *value = std::getenv(key))
                if (*value != '\0')
                    return value;
        return {};
    }
} // namespace

int main(int argc, char **argv)
{
    long authored_budget = 0;      // 0 = AutoLocal (resolve from own places)
    std::string session;
    std::size_t separator = argc;  // position of "--"
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--")
        {
            separator = static_cast<std::size_t>(i);
            break;
        }
        if (arg.rfind("--budget=", 0) == 0)
        {
            const std::string value = arg.substr(9);
            char *end = nullptr;
            const long parsed = std::strtol(value.c_str(), &end, 10);
            if (end == value.c_str() || *end != '\0' || parsed <= 0)
                fail(kExitUsage, "invalid --budget value '" + value + "'");
            authored_budget = parsed;
            continue;
        }
        if (arg.rfind("--session=", 0) == 0)
        {
            session = arg.substr(10);
            continue;
        }
        fail(kExitUsage, "unknown option '" + arg + "' (expected --budget=, --session=, then --)");
    }
    if (session.empty())
        fail(kExitUsage, "missing --session=<token>");
    if (separator >= static_cast<std::size_t>(argc) || separator + 1 >= static_cast<std::size_t>(argc))
        fail(kExitUsage, "missing '-- <application argv...>'");

    // Strict derivation from the inherited mask (fail-closed authority).
    std::vector<char *> app_argv;
    for (std::size_t i = separator + 1; i < static_cast<std::size_t>(argc); ++i)
        app_argv.push_back(argv[i]);
    app_argv.push_back(nullptr);

    const auto admitted = ownAdmittedCpus();
    // The receipt type has NO default constructor (construction barrier);
    // the only producer is the validating factory below.
    std::optional<llaminar2::placement::PlacementReceipt> derived;
    try
    {
        derived = llaminar2::placement::deriveLocalReceipt(admitted,
            "/sys/devices/system/cpu");
    }
    catch (const llaminar2::placement::PlacementTopologyReject &reject)
    {
        std::fprintf(stderr, "[placement-helper] reason=%d %s\n",
            static_cast<int>(reject.reason), reject.what());
        fail(kExitTopology, "strict receipt derivation failed");
    }
    catch (const std::exception &error)
    {
        fail(kExitTopology, std::string("strict receipt derivation failed: ") + error.what());
    }
    const llaminar2::placement::PlacementReceipt &receipt = *derived;

    const std::size_t place_count = receipt.ordered_places().size();
    long budget = authored_budget;
    bool budget_origin_explicit = authored_budget > 0;
    if (!budget_origin_explicit)
        budget = static_cast<long>(place_count); // AutoLocal: resolve ONCE from own places
    if (budget < 1 || budget > static_cast<long>(place_count))
        fail(kExitBudget, "budget " + std::to_string(budget) +
            " outside [1, place_count " + std::to_string(place_count) + "]");

    // Authored OMP_NUM_THREADS is preserved; a conflicting value is fatal —
    // never silently overwritten, never substituted with the place count.
    if (const char *authored = std::getenv("OMP_NUM_THREADS"))
    {
        const std::string value = authored;
        if (value != std::to_string(budget))
            fail(kExitBudget, "authored OMP_NUM_THREADS=" + value +
                " conflicts with the resolved budget " + std::to_string(budget));
    }
    else
    {
        setenv("OMP_NUM_THREADS", std::to_string(budget).c_str(), 1);
    }

    const std::string rank = independentRankIdentity();
    if (rank.empty())
        fail(kExitIdentity, "no independent MPI rank identity in the environment "
             "(OMPI_COMM_WORLD_RANK / PMI_RANK)");
    char host[256];
    if (gethostname(host, sizeof(host)) != 0)
        fail(kExitIdentity, "gethostname failed");

    llaminar2::placement::ReceiptSessionIdentity identity;
    identity.session = session;
    identity.host = host;
    identity.rank = rank;
    identity.budget = budget;
    identity.budget_origin_explicit = budget_origin_explicit;

    std::string transport;
    try
    {
        transport = llaminar2::placement::serializeReceiptTransport(receipt, identity);
    }
    catch (const llaminar2::placement::PlacementTopologyReject &reject)
    {
        std::fprintf(stderr, "[placement-helper] reason=%d %s\n",
            static_cast<int>(reject.reason), reject.what());
        fail(kExitIdentity, "transport serialization failed");
    }
    // Authored placement policy conflicts are FATAL — the helper never
    // silently runs a wrong layout (preserve-or-fatal, per review).
    if (const char *authored_places = std::getenv("OMP_PLACES"))
        fail(kExitBudget, "authored OMP_PLACES='" + std::string(authored_places) +
            "' conflicts with the managed receipt authority");
    if (const char *authored_bind = std::getenv("OMP_PROC_BIND"))
        if (std::strcmp(authored_bind, "close") != 0)
            fail(kExitBudget, "authored OMP_PROC_BIND='" + std::string(authored_bind) +
                "' conflicts with the managed close-binding intent");
    setenv("LLAMINAR_PLACEMENT_RECEIPT_V1", transport.c_str(), 1);
    // The receipt's place list is the managed authority for OMP_PLACES.
    setenv("OMP_PLACES", receipt.omp_places_string().c_str(), 1);
    // Binding policy: set close when the operator has not authored one.
    if (std::getenv("OMP_PROC_BIND") == nullptr)
        setenv("OMP_PROC_BIND", "close", 1);

    // Child semantics are execvp's DOCUMENTED behavior, verbatim: a
    // argv[0] containing a slash is executed directly; a bare name is
    // searched along PATH exactly as mpirun/the existing CLI already
    // behaves. The NO-PATH-FALLBACK rule binds the HELPER's own resolution
    // (this binary is only ever launched by absolute path), not the
    // application's existing CLI contract.
    ::execvp(app_argv[0], app_argv.data());
    fail(kExitExec, std::string("execvp of the application failed: ") +
        std::strerror(errno));
}
