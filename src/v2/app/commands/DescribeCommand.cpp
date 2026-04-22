/**
 * @file DescribeCommand.cpp
 * @brief 'llaminar describe' — cluster/device inventory.
 *
 * Gathers the full cluster inventory (CPU sockets, GPUs, NUMA, P2P)
 * and outputs in text (libfort tables), JSON, or YAML.
 *
 * Uses MPI bootstrap to exercise the same code path as real inference:
 * self-launches via mpirun, calls MPI_Init, gathers inventory via
 * MPI_Allgatherv, and prints on rank 0 only.
 */

#include "app/commands/DescribeCommand.h"
#include "app/commands/CommandMPI.h"
#include "backends/ComputeBackend.h"
#include "backends/InventoryPrinter.h"
#include "config/CliSpec.h"
#include "planning/ClusterInventoryGatherer.h"
#include "utils/Logger.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

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
            // MPI bootstrap control
            bool no_mpi_bootstrap = false;
            std::string hostfile;
        };

        CliSpec<DescribeConfig> buildDescribeSpec()
        {
            CliSpec<DescribeConfig> spec;
            spec.addCategory("Output Control");
            spec.addCategory("MPI");

            spec.add({"-h", "--help", {}, "Output Control", "", "Show this help message", {}, false, setters::assignBoolTrue(&DescribeConfig::show_help)});
            spec.add({"", "--format", {}, "Output Control", "<fmt>", "Output format", {"text", "json", "yaml"}, false, setters::assignString(&DescribeConfig::format)});
            spec.add({"-o", "--output", {}, "Output Control", "<file>", "Write output to file instead of stdout", {}, false, setters::assignString(&DescribeConfig::output_file)});
            spec.add({"", "--topology-only", {}, "Output Control", "", "Show only CPU topology", {}, false, [](DescribeConfig &c, const std::string &)
                      {
                          c.show_topology = true;
                          c.show_numa = false;
                          c.show_devices = false;
                      }});
            spec.add({"", "--numa-only", {}, "Output Control", "", "Show only NUMA configuration", {}, false, [](DescribeConfig &c, const std::string &)
                      {
                          c.show_topology = false;
                          c.show_numa = true;
                          c.show_devices = false;
                      }});
            spec.add({"", "--devices-only", {}, "Output Control", "", "Show only device list", {}, false, [](DescribeConfig &c, const std::string &)
                      {
                          c.show_topology = false;
                          c.show_numa = false;
                          c.show_devices = true;
                      }});
            spec.add({"", "--no-mpi-bootstrap", {}, "MPI", "", "Skip MPI bootstrap (local-only inventory)", {}, false, setters::assignBoolTrue(&DescribeConfig::no_mpi_bootstrap)});
            spec.add({"", "--hostfile", {}, "MPI", "<path>", "MPI hostfile for multi-node inventory", {}, false, setters::assignString(&DescribeConfig::hostfile)});

            return spec;
        }

        // =================================================================
        // Helpers: format memory values
        // =================================================================

        double to_gb(size_t bytes)
        {
            return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        }

        std::string device_type_str(DeviceType t)
        {
            switch (t)
            {
            case DeviceType::CUDA:
                return "cuda";
            case DeviceType::ROCm:
                return "rocm";
            case DeviceType::Vulkan:
                return "vulkan";
            case DeviceType::Metal:
                return "metal";
            default:
                return "cpu";
            }
        }

        // =================================================================
        // JSON serialization
        // =================================================================

        nlohmann::ordered_json socketToJson(const CPUSocketInfo &s)
        {
            nlohmann::ordered_json j;
            j["socket_id"] = s.socket_id;
            j["numa_node"] = s.numa_node;
            j["model_name"] = s.model_name;
            j["physical_cores"] = s.num_physical_cores();
            j["threads"] = s.num_threads();
            j["hyperthreading"] = s.has_hyperthreading();
            j["memory_bytes"] = s.memory_bytes;
            j["memory_gb"] = to_gb(s.memory_bytes);
            j["core_ids"] = s.physical_cores;
            if (!s.ht_threads.empty())
                j["ht_thread_ids"] = s.ht_threads;
            return j;
        }

        nlohmann::ordered_json gpuToJson(const DeviceInfo &gpu)
        {
            nlohmann::ordered_json j;
            j["id"] = gpu.local_device_id;
            j["type"] = device_type_str(gpu.type);
            j["name"] = gpu.name;
            j["memory_bytes"] = gpu.memory_bytes;
            j["memory_gb"] = to_gb(gpu.memory_bytes);
            j["free_memory_bytes"] = gpu.free_memory_bytes;
            j["free_memory_gb"] = to_gb(gpu.free_memory_bytes);
            j["numa_node"] = gpu.numa_node;
            if (gpu.compute_capability_major > 0)
            {
                j["compute_capability"] = std::to_string(gpu.compute_capability_major) +
                                          "." + std::to_string(gpu.compute_capability_minor);
            }
            if (gpu.pcie_gen > 0)
            {
                nlohmann::ordered_json pcie;
                pcie["gen"] = gpu.pcie_gen;
                pcie["width"] = gpu.pcie_width;
                pcie["speed_gts"] = gpu.pcie_speed_gts;
                if (gpu.pcie_degraded)
                {
                    pcie["degraded"] = true;
                    pcie["max_width"] = gpu.pcie_max_width;
                    pcie["max_speed_gts"] = gpu.pcie_max_speed_gts;
                    if (!gpu.pcie_bottleneck_bdf.empty())
                        pcie["bottleneck_bdf"] = gpu.pcie_bottleneck_bdf;
                }
                j["pcie"] = pcie;
            }
            return j;
        }

        nlohmann::ordered_json p2pToJson(const std::vector<bool> &matrix, int count,
                                         const std::vector<DeviceInfo> &gpus)
        {
            nlohmann::ordered_json j = nlohmann::ordered_json::array();
            for (int i = 0; i < count; ++i)
            {
                for (int k = 0; k < count; ++k)
                {
                    if (i != k && matrix[i * count + k])
                    {
                        int from_id = (i < static_cast<int>(gpus.size())) ? gpus[i].local_device_id : i;
                        int to_id = (k < static_cast<int>(gpus.size())) ? gpus[k].local_device_id : k;
                        nlohmann::ordered_json pair;
                        pair["from"] = from_id;
                        pair["to"] = to_id;
                        j.push_back(pair);
                    }
                }
            }
            return j;
        }

        nlohmann::ordered_json inventoryToJson(const ClusterInventory &inv,
                                               const DescribeConfig &cfg)
        {
            nlohmann::ordered_json root;

            for (const auto &rank : inv.ranks)
            {
                nlohmann::ordered_json node;
                node["hostname"] = rank.hostname;
                node["rank"] = rank.rank;
                node["node_id"] = rank.node_id;

                if (cfg.show_topology || cfg.show_numa)
                {
                    nlohmann::ordered_json cpu;
                    cpu["cores"] = rank.cpu_cores;
                    cpu["sockets"] = rank.cpu_sockets;
                    cpu["numa_nodes"] = rank.numa_nodes;
                    cpu["memory_bytes"] = rank.cpu_memory_bytes;
                    cpu["memory_gb"] = to_gb(rank.cpu_memory_bytes);

                    if (!rank.cpu_socket_info.empty())
                    {
                        nlohmann::ordered_json sockets = nlohmann::ordered_json::array();
                        for (const auto &s : rank.cpu_socket_info)
                            sockets.push_back(socketToJson(s));
                        cpu["sockets_detail"] = sockets;
                    }
                    node["cpu"] = cpu;
                }

                if (cfg.show_devices && !rank.gpus.empty())
                {
                    nlohmann::ordered_json gpus = nlohmann::ordered_json::array();
                    for (const auto &gpu : rank.gpus)
                        gpus.push_back(gpuToJson(gpu));
                    node["gpus"] = gpus;

                    // P2P matrices
                    if (rank.p2p_cuda_count >= 2)
                    {
                        auto cuda_gpus_filtered = rank.gpus;
                        std::erase_if(cuda_gpus_filtered,
                                      [](const DeviceInfo &g)
                                      { return g.type != DeviceType::CUDA; });
                        node["p2p_cuda"] = p2pToJson(rank.p2p_cuda, rank.p2p_cuda_count,
                                                     cuda_gpus_filtered);
                    }
                    if (rank.p2p_rocm_count >= 2)
                    {
                        auto rocm_gpus_filtered = rank.gpus;
                        std::erase_if(rocm_gpus_filtered,
                                      [](const DeviceInfo &g)
                                      { return g.type != DeviceType::ROCm; });
                        node["p2p_rocm"] = p2pToJson(rank.p2p_rocm, rank.p2p_rocm_count,
                                                     rocm_gpus_filtered);
                    }
                }

                root["nodes"].push_back(node);
            }

            // Summary
            nlohmann::ordered_json summary;
            summary["world_size"] = inv.world_size;
            summary["node_count"] = inv.node_count;
            summary["total_gpus"] = inv.total_gpus;
            if (inv.total_gpu_memory > 0)
                summary["total_gpu_memory_gb"] = to_gb(inv.total_gpu_memory);
            if (inv.total_cpu_memory > 0)
                summary["total_cpu_memory_gb"] = to_gb(inv.total_cpu_memory);
            root["summary"] = summary;

            return root;
        }

        // =================================================================
        // YAML serialization (hand-rolled, no library dependency)
        // =================================================================

        std::string inventoryToYaml(const ClusterInventory &inv,
                                    const DescribeConfig &cfg)
        {
            std::ostringstream y;
            y << std::fixed;

            y << "# Llaminar cluster inventory\n\n";

            y << "summary:\n"
              << "  world_size: " << inv.world_size << "\n"
              << "  node_count: " << inv.node_count << "\n"
              << "  total_gpus: " << inv.total_gpus << "\n";
            if (inv.total_gpu_memory > 0)
                y << "  total_gpu_memory_gb: " << std::setprecision(1) << to_gb(inv.total_gpu_memory) << "\n";
            if (inv.total_cpu_memory > 0)
                y << "  total_cpu_memory_gb: " << std::setprecision(1) << to_gb(inv.total_cpu_memory) << "\n";
            y << "\n";

            y << "nodes:\n";
            for (const auto &rank : inv.ranks)
            {
                y << "  - hostname: " << rank.hostname << "\n"
                  << "    rank: " << rank.rank << "\n"
                  << "    node_id: " << rank.node_id << "\n";

                if (cfg.show_topology || cfg.show_numa)
                {
                    y << "    cpu:\n"
                      << "      cores: " << rank.cpu_cores << "\n"
                      << "      sockets: " << rank.cpu_sockets << "\n"
                      << "      numa_nodes: " << rank.numa_nodes << "\n"
                      << "      memory_gb: " << std::setprecision(1) << to_gb(rank.cpu_memory_bytes) << "\n";

                    if (!rank.cpu_socket_info.empty())
                    {
                        y << "      sockets_detail:\n";
                        for (const auto &s : rank.cpu_socket_info)
                        {
                            y << "        - socket_id: " << s.socket_id << "\n"
                              << "          numa_node: " << s.numa_node << "\n"
                              << "          model_name: \"" << s.model_name << "\"\n"
                              << "          physical_cores: " << s.num_physical_cores() << "\n"
                              << "          threads: " << s.num_threads() << "\n"
                              << "          memory_gb: " << std::setprecision(1) << to_gb(s.memory_bytes) << "\n";
                        }
                    }
                }

                if (cfg.show_devices && !rank.gpus.empty())
                {
                    y << "    gpus:\n";
                    for (const auto &gpu : rank.gpus)
                    {
                        y << "      - id: " << gpu.local_device_id << "\n"
                          << "        type: " << device_type_str(gpu.type) << "\n"
                          << "        name: \"" << gpu.name << "\"\n"
                          << "        memory_gb: " << std::setprecision(1) << to_gb(gpu.memory_bytes) << "\n"
                          << "        free_memory_gb: " << std::setprecision(1) << to_gb(gpu.free_memory_bytes) << "\n"
                          << "        numa_node: " << gpu.numa_node << "\n";
                        if (gpu.compute_capability_major > 0)
                            y << "        compute_capability: \"" << gpu.compute_capability_major
                              << "." << gpu.compute_capability_minor << "\"\n";
                        if (gpu.pcie_gen > 0)
                        {
                            y << "        pcie:\n"
                              << "          gen: " << gpu.pcie_gen << "\n"
                              << "          width: " << gpu.pcie_width << "\n";
                            if (gpu.pcie_degraded)
                                y << "          degraded: true\n";
                        }
                    }
                }
            }

            return y.str();
        }

        // =================================================================
        // Text output: uses InventoryPrinter (renders via LOG_INFO)
        // =================================================================

        void printInventoryText(const ClusterInventory &inv)
        {
            InventoryPrinter::printClusterInventory(inv);
        }

        int renderOutput(const DescribeConfig &cfg, const ClusterInventory &inventory)
        {
            std::string output;
            if (cfg.format == "json")
            {
                auto j = inventoryToJson(inventory, cfg);
                output = j.dump(2) + "\n";
            }
            else if (cfg.format == "yaml")
            {
                output = inventoryToYaml(inventory, cfg);
            }

            if (!cfg.output_file.empty())
            {
                std::ofstream out(cfg.output_file);
                if (!out.is_open())
                {
                    std::cerr << "Error: Cannot write to " << cfg.output_file << "\n";
                    return 1;
                }

                if (cfg.format == "text")
                {
                    printInventoryText(inventory);
                    auto j = inventoryToJson(inventory, cfg);
                    out << j.dump(2) << "\n";
                    std::cout << "Inventory written to: " << cfg.output_file << " (json)\n";
                }
                else
                {
                    out << output;
                    std::cout << "Inventory written to: " << cfg.output_file << "\n";
                }
            }
            else
            {
                if (cfg.format == "text")
                {
                    printInventoryText(inventory);
                }
                else
                {
                    std::cout << output;
                }
            }

            return 0;
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
            std::cout << spec.getHelpText(
                             "Usage: llaminar2 describe [options]\n\n"
                             "Print cluster topology, NUMA configuration, and available devices.\n"
                             "Bootstraps MPI to gather inventory across all ranks (use\n"
                             "--no-mpi-bootstrap for local-only mode).")
                      << std::endl;
            return 0;
        }

        // Initialize devices and gather cluster inventory (local or MPI)
        auto [session, early_exit] = CommandMPI::bootstrap({
            .subcommand = name(),
            .argc = argc,
            .argv = argv,
            .no_mpi_bootstrap = cfg.no_mpi_bootstrap,
            .hostfile = cfg.hostfile,
        });
        if (early_exit.has_value())
            return *early_exit;

        // Only the output rank (rank 0 or local-only) renders
        if (!session.is_output_rank)
            return 0;

        return renderOutput(cfg, session.inventory);
    }

} // namespace llaminar2
