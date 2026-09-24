/**
 * @file Test__FrontendStartupFailure.cpp
 * @brief Real frontend asymmetric preparation and failed dry-run retirement.
 *
 * This process deliberately invokes RuntimeInitPhase before MPI initialization,
 * exactly as a command does. A malformed rank-local argument must be published
 * through phase consensus; a failed dry-run must preserve requested intent and
 * return an explicit failure. A saved configuration must preserve explicit
 * startup policy through that same lifecycle. All paths retire MPI on return.
 * Saved subsets and reversed namespaces additionally exercise the public
 * --config parser: excluded peers finalize while selected ranks reach real
 * runner metadata admission. Asymmetric or absent selections fail before split.
 * No real model or accelerator is used, and the ordinary MPI timeout applies.
 */
#include "app/RuntimeInitPhase.h"
#include "config/OrchestrationConfigDocument.h"
#include "utils/MPIBootstrap.h"
#include "utils/DebugEnv.h"
#include <mpi.h>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include <unistd.h>

namespace
{
    /** @brief Mutually exclusive frontend entry contracts, not independent flags. */
    enum class Scenario
    {
        RankParseFailure, DryRunFailure, SavedConfiguration,
        SelectedFollower, ReversedSelection, AsymmetricSelection, AbsentSelection
    };

    /** @brief Own the exact per-rank input document until the frontend retires. */
    class SavedConfig final
    {
    public:
        /** @brief Persist a document with no real model, before entering MPI. */
        explicit SavedConfig(const llaminar2::OrchestrationConfig &config)
        {
            char pattern[] = "/tmp/llaminar-frontend-config-XXXXXX";
            const int fd = mkstemp(pattern);
            if (fd < 0) throw std::runtime_error("Cannot create saved frontend config");
            close(fd);
            path = pattern;
            std::ofstream output(path);
            output << llaminar2::serializeOrchestrationConfig(config);
            if (!output) throw std::runtime_error("Cannot write saved frontend config");
        }
        /** @brief Remove only this rank's private, uniquely created fixture. */
        ~SavedConfig() { std::remove(path.c_str()); }
        SavedConfig(const SavedConfig &) = delete;
        SavedConfig &operator=(const SavedConfig &) = delete;
        std::string path;
    };
}

/** @brief Run one complete process-owned frontend failure lifecycle. */
int main(int argc, char **argv)
{
    Scenario scenario = Scenario::RankParseFailure;
    if (argc == 2 && std::string(argv[1]) == "--dry-run-proof") scenario = Scenario::DryRunFailure;
    else if (argc == 2 && std::string(argv[1]) == "--saved-config-proof") scenario = Scenario::SavedConfiguration;
    else if (argc == 2 && std::string(argv[1]) == "--selected-follower-proof") scenario = Scenario::SelectedFollower;
    else if (argc == 2 && std::string(argv[1]) == "--reversed-selection-proof") scenario = Scenario::ReversedSelection;
    else if (argc == 2 && std::string(argv[1]) == "--asymmetric-selection-proof") scenario = Scenario::AsymmetricSelection;
    else if (argc == 2 && std::string(argv[1]) == "--absent-selection-proof") scenario = Scenario::AbsentSelection;
    else if (argc != 1) return 1;
    const auto launch = llaminar2::MPIBootstrap::detectMPIEnvironment();
    if (!launch.is_mpi_process || launch.detected_world_size != 2) return 1;
    std::vector<std::string> args{
        argv[0], "-m", "/__llaminar_frontend_failure_proof__/missing.gguf", "-d", "cpu", "--threads",
        (scenario == Scenario::RankParseFailure && launch.detected_rank == 1) ? "invalid-thread-count" : "1"};
    if (scenario == Scenario::DryRunFailure) args.push_back("--dry-run");
    std::optional<SavedConfig> saved;
    const bool saved_scenario = scenario != Scenario::RankParseFailure && scenario != Scenario::DryRunFailure;
    if (saved_scenario)
    {
        llaminar2::OrchestrationConfig input;
        input.model_path = args[2];
        input.device_for_this_rank = llaminar2::GlobalDeviceAddress::cpu(0);
        input.cpu_global_tp_all_local = true;
        input.n_threads = 1;
        input.dry_run = true;
        input.prefill_max_bucket_size = 1024;
        input.deterministic = true;
        input.temperature = 0.0f;
        if (scenario == Scenario::SelectedFollower)
            input.execution_rank_selection = llaminar2::ExecutionRankSelection({1});
        else if (scenario == Scenario::ReversedSelection)
            input.execution_rank_selection = llaminar2::ExecutionRankSelection({1, 0});
        else if (scenario == Scenario::AsymmetricSelection)
            input.execution_rank_selection = llaminar2::ExecutionRankSelection(
                launch.detected_rank == 0 ? std::vector<int>{1} : std::vector<int>{0, 1});
        else if (scenario == Scenario::AbsentSelection)
            input.execution_rank_selection = llaminar2::ExecutionRankSelection({2});
        saved.emplace(input);
        args = {argv[0], "--config", saved->path};
    }
    std::vector<char *> pointers;
    for (auto &argument : args) pointers.push_back(argument.data());
    pointers.push_back(nullptr);
    int count = static_cast<int>(args.size());
    char **arguments = pointers.data();
    llaminar2::OrchestrationConfig config;
    const auto result = llaminar2::RuntimeInitPhase{}.execute(config, count, arguments);
    const auto *terminal = std::get_if<llaminar2::RuntimeInitExit>(&result);
    int finalized = 0;
    MPI_Finalized(&finalized);
    const bool excluded = scenario == Scenario::SelectedFollower && launch.detected_rank == 0;
    const auto expected = excluded ? llaminar2::RuntimeInitExit::Completed : llaminar2::RuntimeInitExit::Failed;
    if (!terminal || *terminal != expected || !finalized ||
        (scenario != Scenario::RankParseFailure && !config.dry_run)) return 1;
    if (saved_scenario &&
        (config.prefill_max_bucket_size != 1024 || !config.deterministic ||
         !llaminar2::debugEnv().gemm.deterministic ||
         llaminar2::debugEnv().execution.prefill_graph_bucket_sizes != llaminar2::prefillGraphBucketSizes(1024)))
        return 1;
    if (scenario == Scenario::SelectedFollower && !excluded &&
        (config.device_map.size() != 1 || config.device_map.front().first != 0)) return 1;
    if (scenario == Scenario::ReversedSelection &&
        (config.device_map.size() != 1 || config.device_map.front().first != 1 - launch.detected_rank)) return 1;
    std::puts("PASS: frontend failure preserved intent and retired its MPI session");
    return 0;
}
