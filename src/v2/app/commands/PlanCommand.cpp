/**
 * @file PlanCommand.cpp
 * @brief Public automatic planning using the same request policy as serving.
 *
 * The shared parser owns model, KV, MTP, memory and topology options. This
 * command adds only output presentation, gathers the canonical MPI inventory,
 * and asks the shared planner to emit a lossless apply document. Planning must
 * not price a smaller default policy than the one the server will execute.
 */
#include "app/commands/PlanCommand.h"
#include "app/commands/CommandMPI.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"
#include "config/OrchestrationPlanningPolicy.h"
#include "planning/AutomaticPlanningStartup.h"
#include "utils/Logger.h"

#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        /** @brief Presentation is command-owned, never duplicated runtime policy. */
        struct PlanOutput
        {
            enum class Display { Summary, Document };
            Display display = Display::Summary;
            std::string path;
        };

        /**
         * @brief Add only output flags to the shared runtime specification.
         * @param output Presentation owner that outlives parsing and help rendering.
         * @return Extensions whose names cannot shadow a shared option.
         */
        CliSpec<OrchestrationConfig> outputOptions(PlanOutput &output)
        {
            CliSpec<OrchestrationConfig> spec;
            spec.addCategory("Plan output");
            spec.add({
                .short_name = "-o",
                .long_name = "--output",
                .category = "Plan output",
                .value_label = "<file>",
                .description = "Write the lossless apply configuration to a file",
                .setter = [&output](OrchestrationConfig &, const std::string &value) {
                    output.path = value;
                },
            });
            spec.add({
                .long_name = "--format",
                .category = "Plan output",
                .value_label = "<table|json>",
                .description = "Print a planning summary or the lossless configuration (default: table)",
                .valid_values = {"table", "json"},
                .setter = [&output](OrchestrationConfig &, const std::string &value) {
                    output.display = value == "json" ? PlanOutput::Display::Document
                                                    : PlanOutput::Display::Summary;
                },
            });
            return spec;
        }
    }

    int PlanCommand::execute(int argc, char *argv[])
    {
        initializeLogging();
        PlanOutput output;
        const auto options = outputOptions(output);
        OrchestrationConfig request;
        try
        {
            request = OrchestrationConfigParser{}.parseCommandArgs(argc, argv, options);
            if (request.show_help)
            {
                std::cout << OrchestrationConfigParser::getCommandHelpText(options,
                    "Usage: llaminar2 plan -m <model> [inference options] [-o plan.json]\n\n"
                    "Automatically select placement using the same policy as serve.\n"
                    "Apply the result with: llaminar2 serve --config plan.json");
                return 0;
            }
            const auto errors = request.validate();
            if (!errors.empty())
            {
                for (const auto &error : errors) std::cerr << "Error: " << error << '\n';
                return 1;
            }
            if (!std::holds_alternative<AutomaticOrchestrationRequest>(
                    resolveOrchestrationIntent(request)))
                throw std::invalid_argument(
                    "plan requires automatic intent; use --only-backends/--only-strategies "
                    "to constrain search, or serve --config to apply an existing plan");
            if (request.validate_only) return 0;
        }
        catch (const std::exception &error)
        {
            std::cerr << "Error: " << error.what() << '\n';
            return 1;
        }

        // Command bootstrap borrows this exact parsed policy. A second default
        // config would lose backend filters, CPU intent and MPI process bounds.
        auto [session, early_exit] = CommandMPI::bootstrap({
            .subcommand = name(),
            .argc = argc,
            .argv = argv,
            .no_mpi_bootstrap = request.mpi_no_bootstrap,
            .hostfile = request.hostfile,
            .request = &request,
        });
        if (early_exit.has_value()) return *early_exit;

        try
        {
            // Both frontends keep discovery intact through evidence gathering;
            // only the common startup authority opens/prices the model on root.
            const auto startup = AutomaticPlanningStartup::run(request, session.inventory(), session.context());
            if (!session.is_output_rank) return 0;
            if (!startup.root_selection) throw std::logic_error("Planning output rank did not produce the selected admission");
            const auto &selected = *startup.root_selection;
            const auto document = serializeOrchestrationConfig(startup.applied);

            if (!output.path.empty())
            {
                std::ofstream file(output.path);
                file << document;
                file.close();
                if (!file) throw std::runtime_error("Cannot write complete plan to " + output.path);
            }
            if (output.display == PlanOutput::Display::Document)
                std::cout << document;
            else
            {
                std::cout << "=== Automatic Execution Plan ===\n"
                          << "Model: " << request.model_path << '\n'
                          << "Strategy: " << orchestrationStrategyName(selected.candidate().strategy()) << '\n'
                          << "Candidates: " << selected.counts().evaluated << " admitted ("
                          << selected.counts().capacity_rejected << " capacity-rejected)\n"
                          << "Expected workload: " << selected.cost().workload().promptTokens()
                          << " prefill + " << selected.cost().workload().generationTokens() << " generated tokens\n"
                          << "Estimated request seconds: " << selected.cost().requestSeconds() << '\n'
                          << "Cost evidence: " << selected.cost().evidence() << "\n\n"
                          << selected.candidate().physicalAdmission().plan().summary() << '\n';
                if (!output.path.empty()) std::cout << "Plan written to: " << output.path << '\n';
            }
        }
        catch (const std::exception &error)
        {
            std::cerr << "Error: automatic planning failed: " << error.what() << '\n';
            return 1;
        }
        return 0;
    }
} // namespace llaminar2
