/**
 * @file Test__FrontendAutomaticSelection.cpp
 * @brief Real frontend auto-selection publication before execution-rank admission.
 *
 * Only discovery root owns a tiny metadata-only GGUF. A follower deliberately
 * starts with an unreadable model path: it must receive root's complete apply
 * document rather than running another search. Before selection, both ranks
 * measure one small CPU projection distributed from root's fixture through the
 * real PMA/sample catalog. The real RuntimeInitPhase, compiler, admission, MPI
 * split and runner dry-run execute. Synthetic decision costs do not claim model
 * performance. No accelerator/model is loaded, and MPI belongs to the frontend.
 */
#include "app/RuntimeInitPhase.h"
#include "planning/PlanningKernelServiceCatalog.h"
#include "utils/MPIBootstrap.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    /** @brief Independent success/failure obligations through one frontend entry. */
    enum class Scenario
    {
        DefaultSelection, SelectedFollower, CostFailure, MixedIntent,
        PreparationFailure, SampleFailure, FollowerEvaluator,
    };

    /** @brief Surface an exact failed invariant after the frontend has retired MPI. */
    void require(bool condition, const char *detail)
    {
        if (!condition) throw std::runtime_error(detail);
    }
}

/** @brief Exercise automatic subset selection through the actual command initialization. */
int main(int argc, char **argv)
{
    try
    {
        Scenario scenario = Scenario::DefaultSelection;
        if (argc == 2 && std::string(argv[1]) == "--selected-follower") scenario = Scenario::SelectedFollower;
        else if (argc == 2 && std::string(argv[1]) == "--cost-failure") scenario = Scenario::CostFailure;
        else if (argc == 2 && std::string(argv[1]) == "--mixed-intent") scenario = Scenario::MixedIntent;
        else if (argc == 2 && std::string(argv[1]) == "--preparation-failure") scenario = Scenario::PreparationFailure;
        else if (argc == 2 && std::string(argv[1]) == "--sample-failure") scenario = Scenario::SampleFailure;
        else if (argc == 2 && std::string(argv[1]) == "--follower-evaluator") scenario = Scenario::FollowerEvaluator;
        else require(argc == 1, "Unexpected automatic-selection proof arguments");
        const auto launch = llaminar2::MPIBootstrap::detectMPIEnvironment();
        require(launch.is_mpi_process && launch.detected_world_size == 2, "Proof requires two discovery ranks");
        // Root's fixture outlives RuntimeInitPhase, including MPI_Finalize's
        // join with the selected process. Only one bounded matrix is sampled;
        // the rest of this metadata-only model is never loaded for inference.
        std::optional<llaminar2::test::PlanningGGUFFixture> fixture;
        if (launch.detected_rank == 0) fixture.emplace();
        std::vector<std::string> args{argv[0], "-m",
            fixture ? fixture->path() : "/__unreadable_follower_planning_source__/missing.gguf",
            "--auto", "--only-backends", "cpu", "--only-strategies", "single",
            "--context-length", "512", "--kv-cache-precision", "fp32",
            "--plan-workload", "128,384",
            "--prefill-max-bucket-size", "64", "--threads", "1", "--dry-run"};
        if (scenario == Scenario::MixedIntent && launch.detected_rank != 0)
        {
            // Valid apply intent on one rank must disagree at the common
            // initialization consensus, not strand root in a payload broadcast.
            args = {argv[0], "-m", args[2], "-d", "cpu", "--threads", "1", "--dry-run"};
        }
        std::vector<char *> pointers;
        for (auto &argument : args) pointers.push_back(argument.data());
        pointers.push_back(nullptr);
        int count = static_cast<int>(args.size());
        char **arguments = pointers.data();
        llaminar2::OrchestrationConfig config;
        int evaluations = 0;
        int preparations = 0, source_loads = 0;
        bool collected = false;
        const auto evaluate = [&](const auto &candidate, const auto &workload)
        {
            ++evaluations;
            require(collected, "Candidate priced before complete discovery evidence");
            require(launch.detected_rank == 0, "Follower performed a second cost search");
            require(workload == llaminar2::OrchestrationPlanningWorkload(128, 384),
                "Runtime frontend discarded the shared workload hint");
            if (scenario == Scenario::CostFailure) throw std::runtime_error("Injected root cost failure");
            if (scenario != Scenario::SelectedFollower)
                return llaminar2::OrchestrationCostEstimate(workload, 1.0, .01,
                    "synthetic frontend failure/consensus oracle, not production ranking");
            // Only pricing is a decision oracle. Real hardware discovery,
            // model compilation, PMA and the complete frontend remain intact.
            const bool follower = candidate.membership().discoveryRanks() == std::vector<int>{1};
            return llaminar2::OrchestrationCostEstimate(workload, follower ? 1.0 : 2.0,
                .01, "synthetic frontend non-root selection witness");
        };
        const llaminar2::AutomaticPlanningStartup::Prepare prepare =
            [&](const llaminar2::AutomaticPlanningPreparation &context)
                -> std::optional<llaminar2::AutomaticOrchestrationPlanner::Evaluate>
        {
            ++preparations;
            llaminar2::acceptPlanningCostPreparation(context.mpi(), [&] {
                const auto &local = context.inventory().ranks.at(launch.detected_rank);
                require(local.cpuWorkerThreads() == 1 && omp_get_max_threads() == 1,
                    "Frontend did not preserve the requested one-thread CPU team");
            });
            // Use the same failure-atomic source/publication/catalog boundaries
            // as measured startup. No rank-local assertion precedes an unmatched
            // exchange; checks below run after the final collective has returned.
            const auto matrix = llaminar2::PlanningMatrixSamplePublication::describe(context.mpi(), [&] {
                return llaminar2::PlanningMatrixSamplePlan::resolve(context.rootSource(),
                    {"blk.0.ffn_gate.weight", llaminar2::PlanningMatrixRows{0, 32}});
            });
            const auto selection = std::get<llaminar2::AutomaticOrchestrationRequest>(
                llaminar2::resolveOrchestrationIntent(context.request()));
            const auto catalog = llaminar2::PlanningKernelServiceCatalog::collect(context.mpi(),
                context.inventory(), selection, llaminar2::PlanningProjectionServicePlan(matrix, 7),
                [&](const auto &memory) {
                    ++source_loads;
                    if (scenario == Scenario::SampleFailure) throw std::runtime_error("Injected root sample failure");
                    return matrix.load(context.rootSource(), memory, llaminar2::DeviceId::cpu());
                });
            require(evaluations == 0, "Search started before preparation completed");
            require(context.isRoot() == (launch.detected_rank == 0), "Wrong source authority");
            require(context.request().model_path == matrix.modelPath(), "Follower kept its unreadable model path");
            require(context.workload() == llaminar2::OrchestrationPlanningWorkload(128, 384),
                "Evidence used a different workload than selection");
            require(context.metadata().mainLayerCount() == 2, "Follower did not receive main-layer metadata");
            require(catalog.has_value() == context.isRoot(), "Measurement catalog has more than one authority");
            if (catalog)
            {
                require(catalog->records().size() == 2, "Missing discovery rank's CPU measurement");
                for (const auto &record : catalog->records())
                {
                    require(record.observer.device().is_cpu(), "CPU-only planning created a GPU observation");
                    const auto &observed = std::get<llaminar2::PlanningProjectionObservations>(record.observation);
                    require(observed.cpu && observed.cpu->workers == 1,
                        "Sample used physical core count instead of the requested CPU team");
                }
            }
            collected = true;
            if (!context.isRoot())
            {
                bool rejected = false;
                try { (void)context.rootSource(); }
                catch (const std::logic_error &) { rejected = true; }
                require(rejected, "Follower could borrow root's model source");
                if (scenario == Scenario::PreparationFailure)
                    throw std::runtime_error("Injected follower evaluator preparation failure");
                if (scenario != Scenario::FollowerEvaluator) return std::nullopt;
            }
            return evaluate;
        };
        const auto result = llaminar2::RuntimeInitPhase(prepare).execute(config, count, arguments);
        const auto *terminal = std::get_if<llaminar2::RuntimeInitExit>(&result);
        int finalized = 0;
        MPI_Finalized(&finalized);
        require(finalized != 0, "Frontend did not retire its MPI session");
        require(launch.detected_rank == 0 || evaluations == 0, "Follower priced an automatic candidate");
        require(preparations == (scenario == Scenario::MixedIntent ? 0 : 1),
            "Evidence preparation did not execute exactly once on every discovery rank");
        require(source_loads == (launch.detected_rank == 0 && scenario != Scenario::MixedIntent ? 1 : 0),
            "Sample payload was reread or opened by a follower");
        if (scenario == Scenario::CostFailure || scenario == Scenario::MixedIntent ||
            scenario == Scenario::PreparationFailure || scenario == Scenario::SampleFailure ||
            scenario == Scenario::FollowerEvaluator)
        {
            require(terminal && *terminal == llaminar2::RuntimeInitExit::Failed,
                "Asymmetric planning failure was not published to every discovery rank");
            if (scenario == Scenario::CostFailure && launch.detected_rank == 0)
                require(evaluations == 1, "A failed cost observation triggered another candidate");
            else require(evaluations == 0, "Search ran with failed or unauthenticated evidence");
            std::cout << "PASS: frontend planning failure reached every rank and retired MPI\n";
            return 0;
        }
        require(terminal && *terminal == llaminar2::RuntimeInitExit::Completed,
            "Automatic selection did not complete the real frontend dry-run");
        require(config.execution_rank_selection && config.execution_rank_selection->size() == 1,
            "Automatic selection was not published before rank admission");
        if (launch.detected_rank == 0) require(evaluations > 0, "Root never performed candidate selection");
        if (scenario == Scenario::SelectedFollower)
            require(config.execution_rank_selection->discoveryRanks() == std::vector<int>{1},
                "Frontend did not admit the selected non-root discovery rank");
        require(config.mpi_procs == 2, "Published plan lost its discovery process count");
        require(std::holds_alternative<llaminar2::ApplyOrchestrationRequest>(
            llaminar2::resolveOrchestrationIntent(config)), "Frontend retained automatic intent after selection");
        require(config.model_path != "/__unreadable_follower_planning_source__/missing.gguf",
            "Follower did not receive root's model identity");
        require(config.kv_cache_precision == "fp32" && config.prefill_max_bucket_size == 64,
            "Publication lost requested inference policy");
        std::cout << "PASS: frontend published automatic subset before admission and retired MPI\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
