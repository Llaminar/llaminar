/**
 * @file DescribeCommand.cpp
 * @brief 'llaminar describe' — cluster/device inventory.
 *
 * Consolidates the existing --show-topology, --show-numa, and --list-devices
 * functionality into a single subcommand that prints everything by default.
 */

#include "app/commands/DescribeCommand.h"
#include "app/MPIBootstrapPhase.h"
#include "config/CliSpec.h"
#include "utils/Logger.h"
#include "utils/MPIBootstrap.h"
#include <iostream>

namespace llaminar2
{

    namespace
    {
        struct DescribeConfig
        {
            bool show_help = false;
            bool show_topology = true;
            bool show_numa = true;
            bool show_devices = true;
            std::string format = "text";
            std::string output_file;
        };

        CliSpec<DescribeConfig> buildDescribeSpec()
        {
            CliSpec<DescribeConfig> spec;
            spec.addCategory("Output Control");

            spec.add({"-h", "--help", {}, "Output Control", "",
                      "Show this help message", {}, false,
                      setters::assignBoolTrue(&DescribeConfig::show_help)});
            spec.add({"", "--format", {}, "Output Control", "<fmt>",
                      "Output format",
                      {"text", "json", "yaml"}, false,
                      setters::assignString(&DescribeConfig::format)});
            spec.add({"-o", "--output", {}, "Output Control", "<file>",
                      "Write output to file instead of stdout", {}, false,
                      setters::assignString(&DescribeConfig::output_file)});
            spec.add({"", "--topology-only", {}, "Output Control", "",
                      "Show only CPU topology", {}, false,
                      [](DescribeConfig &c, const std::string &)
                      {
                          c.show_topology = true;
                          c.show_numa = false;
                          c.show_devices = false;
                      }});
            spec.add({"", "--numa-only", {}, "Output Control", "",
                      "Show only NUMA configuration", {}, false,
                      [](DescribeConfig &c, const std::string &)
                      {
                          c.show_topology = false;
                          c.show_numa = true;
                          c.show_devices = false;
                      }});
            spec.add({"", "--devices-only", {}, "Output Control", "",
                      "Show only device list", {}, false,
                      [](DescribeConfig &c, const std::string &)
                      {
                          c.show_topology = false;
                          c.show_numa = false;
                          c.show_devices = true;
                      }});

            return spec;
        }
    } // anonymous namespace

    int DescribeCommand::execute(int argc, char *argv[])
    {
        initializeLogging();

        auto spec = buildDescribeSpec();

        // Parse flags (argv[0] is binary name, skip it)
        DescribeConfig cfg;
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
            std::cout << spec.getHelpText("Usage: llaminar2 describe [options]") << std::endl;
            return 1;
        }

        if (cfg.show_help)
        {
            std::cout << spec.getHelpText("Usage: llaminar2 describe [options]\n\n"
                                          "Print cluster topology, NUMA configuration, and available devices.")
                      << std::endl;
            return 0;
        }

        if (cfg.show_topology)
        {
            auto topo = MPIBootstrap::detectCPUTopology();
            std::cout << "\n=== CPU Topology ===\n"
                      << "  Detection method  : " << topo.detection_method << "\n"
                      << "  Sockets           : " << topo.num_sockets << "\n"
                      << "  Physical cores    : " << topo.physical_cores << "\n"
                      << "  Logical cores     : " << topo.logical_cores << "\n"
                      << "  Cores/socket      : " << topo.cores_per_socket << "\n"
                      << "  Threads/core      : " << topo.threads_per_core << "\n"
                      << "  NUMA nodes        : " << topo.numa_nodes << "\n"
                      << "  Hyperthreading    : " << (topo.hyperthreading ? "yes" : "no") << "\n"
                      << std::endl;
        }

        if (cfg.show_numa)
        {
            auto topo = MPIBootstrap::detectCPUTopology();
            std::cout << "=== NUMA Configuration ===\n"
                      << "  NUMA nodes        : " << topo.numa_nodes << "\n"
                      << "  Sockets           : " << topo.num_sockets << "\n"
                      << "  Cores per node    : " << (topo.physical_cores / std::max(1, topo.numa_nodes)) << "\n"
                      << std::endl;
        }

        if (cfg.show_devices)
        {
            MPIBootstrapPhase::listDevices();
        }

        return 0;
    }

} // namespace llaminar2
