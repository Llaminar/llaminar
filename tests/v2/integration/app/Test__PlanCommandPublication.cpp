/**
 * @file Test__PlanCommandPublication.cpp
 * @brief Public plan command produces one lossless apply document across MPI ranks.
 *
 * The real command parser, bootstrap, inventory, model-aware planner and selected
 * configuration publication run before MPI retirement. Only root owns readable
 * model metadata and a writable output path. Followers must neither reopen the
 * model nor write their private invalid path. The fixture uses an IQ4_XS source
 * whose real CPU projection ABI has non-empty workspace, proving that `plan`
 * initialized the same rank-local CPU backend required by serving. The same
 * executable also proves the explicit local-only command lifecycle, which must
 * not skip that ownership transition merely because it does not initialize MPI.
 */
#include "app/commands/PlanCommand.h"
#include "backends/BackendManager.h"
#include "config/OrchestrationConfigDocument.h"
#include "utils/MPIBootstrap.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace
{
    /** @brief Capture only the public command's summary, restoring stdout on exceptions. */
    class PlanSummaryCapture final
    {
    public:
        /** @brief Bind the owning string stream for this command invocation. */
        PlanSummaryCapture() : previous_(std::cout.rdbuf(output_.rdbuf())) {}
        /** @brief Restore stdout before releasing the captured buffer. */
        ~PlanSummaryCapture() { std::cout.rdbuf(previous_); }
        PlanSummaryCapture(const PlanSummaryCapture &) = delete;
        PlanSummaryCapture &operator=(const PlanSummaryCapture &) = delete;
        /** @return Exact public summary for objective/provenance assertions. */
        std::string text() const { return output_.str(); }
    private:
        std::ostringstream output_;
        std::streambuf *previous_;
    };

    /** @brief Exclusive root-owned output file, retained through command/MPI retirement. */
    class PlanOutputFile final
    {
    public:
        /** @brief Create a unique writable file without a shared fixed test filename. */
        PlanOutputFile()
        {
            char pattern[] = "/tmp/llaminar-plan-publication-XXXXXX";
            const int descriptor = mkstemp(pattern);
            if (descriptor < 0) throw std::runtime_error("Cannot create plan output fixture");
            close(descriptor);
            path = pattern;
        }
        /** @brief Remove only this fixture's exact generated document. */
        ~PlanOutputFile() { std::remove(path.c_str()); }
        PlanOutputFile(const PlanOutputFile &) = delete;
        PlanOutputFile &operator=(const PlanOutputFile &) = delete;
        std::string path;
    };
    /** @brief Report a missing public command invariant after MPI has retired. */
    void require(bool condition, const char *detail)
    {
        if (!condition) throw std::runtime_error(detail);
    }
}

/** @brief Invoke the real plan frontend, then authenticate its complete apply document. */
int main(int argc, char **argv)
{
    try
    {
        const bool local_only = argc == 2 && std::string{argv[1]} == "--local";
        require(local_only || argc == 1, "Unexpected plan-publication proof arguments");
        const auto launch = llaminar2::MPIBootstrap::detectMPIEnvironment();
        if (local_only)
            require(!launch.is_mpi_process, "Local-only proof unexpectedly entered MPI");
        else
            require(launch.is_mpi_process && launch.detected_world_size == 2,
                "MPI proof requires two discovery ranks");
        std::optional<llaminar2::test::PlanningGGUFFixture> model;
        std::optional<PlanOutputFile> output;
        if (local_only || launch.detected_rank == 0)
        {
            model.emplace(false, false, llaminar2::GGUFTensorType::IQ4_XS);
            output.emplace();
        }
        std::vector<std::string> args{argv[0], "-m",
            model ? model->path() : "/__plan_follower_must_not_read__/model.gguf",
            "--only-backends", "cpu", "--only-strategies", "single",
            "--context-length", "512", "--kv-cache-precision", "fp32",
            "--plan-workload", "128,384",
            "--prefill-max-bucket-size", "64", "--threads", "1", "--output",
            output ? output->path : "/__plan_follower_must_not_write__/plan.json"};
        if (local_only) args.emplace_back("--no-mpi-bootstrap");
        std::vector<char *> pointers;
        for (auto &argument : args) pointers.push_back(argument.data());
        pointers.push_back(nullptr);
        std::string summary;
        int result = 1;
        {
            PlanSummaryCapture capture;
            result = llaminar2::PlanCommand{}.execute(static_cast<int>(args.size()), pointers.data());
            summary = capture.text();
        }
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (local_only)
            require(result == 0 && mpi_initialized == 0,
                "Local plan command failed or unexpectedly initialized MPI");
        else
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            require(result == 0 && finalized != 0, "Plan command failed or did not retire MPI");
        }
        require(omp_get_max_threads() == 1 && !omp_get_dynamic(),
            "Plan command ignored the requested fixed CPU team");
        require(llaminar2::hasCPUBackend(),
            "Plan command did not initialize the rank-local CPU backend before sampling");
        if (local_only || launch.detected_rank == 0)
        {
            require(summary.find("Expected workload: 128 prefill + 384 generated tokens") != std::string::npos,
                "Plan command did not price the requested workload");
            require(summary.find("Cost evidence: ") != std::string::npos,
                "Plan summary omitted its estimate provenance");
            std::ifstream input(output->path);
            require(input.good(), "Root did not produce an apply document");
            const std::string document{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
            const auto applied = llaminar2::deserializeOrchestrationConfig(document);
            require(applied.model_path == model->path(), "Plan changed the model identity");
            require(applied.execution_rank_selection && applied.execution_rank_selection->size() == 1 &&
                applied.mpi_procs == (local_only ? 1 : 2),
                "Plan lost its exact subset or discovery process count");
            require(std::holds_alternative<llaminar2::ApplyOrchestrationRequest>(
                llaminar2::resolveOrchestrationIntent(applied)), "Saved plan is not apply-only");
            require(!applied.automatic_planning.specified(), "Saved plan retained automatic search hints");
            require(applied.kv_cache_precision == "fp32" && applied.prefill_max_bucket_size == 64 &&
                applied.max_seq_len == 512, "Saved plan lost the shared inference policy");
            require(llaminar2::serializeOrchestrationConfig(applied) == document,
                "Saved plan does not round-trip through the canonical config codec");
        }
        std::cout << "PASS: public plan published one complete apply document and preserved its command lifecycle\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
