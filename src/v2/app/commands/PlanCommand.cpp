/**
 * @file PlanCommand.cpp
 * @brief 'llaminar plan' — cluster analysis and execution plan generation.
 *
 * Lightweight command that does NOT do full MPI bootstrap or model loading.
 * It queries the local device inventory, reads the model GGUF header for
 * architecture metadata, and produces a YAML execution plan.
 */

#include "app/commands/PlanCommand.h"
#include "config/CliSpec.h"
#include "utils/Logger.h"
#include "utils/MPIBootstrap.h"
#include "app/MPIBootstrapPhase.h"
#include <iostream>
#include <fstream>

namespace llaminar2
{

    namespace
    {
        struct PlanConfig
        {
            bool show_help = false;
            std::string model_path;
            std::string strategy = "auto";
            std::string output_file;
            std::string format = "yaml";
            bool benchmark_devices = false;
        };

        CliSpec<PlanConfig> buildPlanSpec()
        {
            CliSpec<PlanConfig> spec;
            spec.addCategory("Model");
            spec.addCategory("Strategy");
            spec.addCategory("Output");

            spec.add({
                .short_name  = "-h",
                .long_name   = "--help",
                .category    = "Output",
                .description = "Show this help message",
                .setter      = setters::assignBoolTrue(&PlanConfig::show_help),
            });
            spec.add({
                .short_name  = "-m",
                .long_name   = "--model",
                .category    = "Model",
                .value_label = "<path>",
                .description = "Path to GGUF model file (required)",
                .setter      = setters::assignString(&PlanConfig::model_path),
            });
            spec.add({
                .short_name  = "-s",
                .long_name   = "--strategy",
                .category    = "Strategy",
                .value_label = "<type>",
                .description = "Placement strategy",
                .valid_values = {"auto", "single-gpu", "tp", "pp", "hybrid", "cpu-only"},
                .setter      = setters::assignString(&PlanConfig::strategy),
            });
            spec.add({
                .short_name  = "-o",
                .long_name   = "--output",
                .category    = "Output",
                .value_label = "<file>",
                .description = "Write plan to file (default: stdout)",
                .setter      = setters::assignString(&PlanConfig::output_file),
            });
            spec.add({
                .long_name    = "--format",
                .category     = "Output",
                .value_label  = "<fmt>",
                .description  = "Output format",
                .valid_values = {"yaml", "json"},
                .setter       = setters::assignString(&PlanConfig::format),
            });
            spec.add({
                .long_name   = "--benchmark-devices",
                .category    = "Strategy",
                .description = "Benchmark device throughput before planning",
                .setter      = setters::assignBoolTrue(&PlanConfig::benchmark_devices),
            });

            return spec;
        }
    } // anonymous namespace

    int PlanCommand::execute(int argc, char *argv[])
    {
        initializeLogging();

        auto spec = buildPlanSpec();
        PlanConfig cfg;

        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i)
            args.emplace_back(argv[i]);

        try
        {
            spec.parse(args, cfg);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << "\n\n";
            std::cout << spec.getHelpText("Usage: llaminar2 plan [options]") << std::endl;
            return 1;
        }

        if (cfg.show_help)
        {
            std::cout << spec.getHelpText(
                             "Usage: llaminar2 plan -m <model> [options]\n\n"
                             "Analyze the cluster and produce an execution plan YAML file.\n"
                             "The plan file can be consumed via: llaminar2 serve --config <plan.yaml>")
                      << std::endl;
            return 0;
        }

        if (cfg.model_path.empty())
        {
            std::cerr << "Error: --model/-m is required for 'plan'.\n";
            return 1;
        }

        // --- Gather cluster inventory ---
        auto topo = MPIBootstrap::detectCPUTopology();

        std::cout << "=== Execution Plan ===" << std::endl;
        std::cout << "Model: " << cfg.model_path << std::endl;
        std::cout << "Strategy: " << cfg.strategy << std::endl;
        std::cout << std::endl;

        std::cout << "CPU Topology:" << std::endl;
        std::cout << "  Sockets: " << topo.num_sockets << std::endl;
        std::cout << "  Physical cores: " << topo.physical_cores << std::endl;
        std::cout << "  NUMA nodes: " << topo.numa_nodes << std::endl;
        std::cout << std::endl;

        // Device inventory
        std::cout << "Devices:" << std::endl;
        MPIBootstrapPhase::listDevices();
        std::cout << std::endl;

        // --- Generate plan YAML ---
        std::ostringstream yaml;
        yaml << "# Llaminar execution plan\n"
             << "# Generated by: llaminar2 plan\n"
             << "# Model: " << cfg.model_path << "\n"
             << "# Strategy: " << cfg.strategy << "\n"
             << "\n"
             << "model_path: \"" << cfg.model_path << "\"\n";

        if (cfg.strategy == "cpu-only" || cfg.strategy == "auto")
        {
            yaml << "# device: cpu\n";
        }

        yaml << "# TODO: Full strategy heuristic will populate TP/PP/device fields\n";

        // Output
        if (!cfg.output_file.empty())
        {
            std::ofstream out(cfg.output_file);
            if (!out.is_open())
            {
                std::cerr << "Error: Cannot write to " << cfg.output_file << "\n";
                return 1;
            }
            out << yaml.str();
            std::cout << "Plan written to: " << cfg.output_file << std::endl;
        }
        else
        {
            std::cout << "--- Plan YAML ---\n"
                      << yaml.str() << std::endl;
        }

        return 0;
    }

} // namespace llaminar2
