/**
 * @file StartupBanner.cpp
 * @brief Consolidated startup banner renderer using libfort with ANSI teal theme.
 *
 * Replaces scattered LOG_INFO output from CUDAEnumeration, ROCmEnumeration,
 * MPIBootstrapPhase, InventoryPrinter, OrchestrationRunner, etc. with a single
 * cohesive display.
 *
 * @author David Sanftenberg
 * @date May 2026
 */
#include "app/StartupBanner.h"
#include "backends/CPUSocketInfo.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "fort.hpp"

#include <cstdlib>
#include <map>
#include <sstream>
#include <unistd.h>

namespace llaminar2
{
    namespace
    {
        // ANSI color codes matching Llaminar's blue/teal brand palette.
        // Same palette as Splash.cpp but simplified for table output.
        struct Colors
        {
            const char *teal;   // borders, section headers
            const char *bright; // emphasis values (device names, OK)
            const char *dim;    // secondary info
            const char *red;    // FAIL only
            const char *bold;
            const char *reset;
        };

        constexpr Colors kColor = {
            "\033[36m",       // teal (cyan)
            "\033[96m",       // bright cyan
            "\033[38;5;245m", // dim gray
            "\033[91m",       // bright red
            "\033[1m",        // bold
            "\033[0m"         // reset
        };

        constexpr Colors kNoColor = {"", "", "", "", "", ""};

        // =====================================================================
        // Helpers
        // =====================================================================

        std::string format_memory_gb(size_t bytes)
        {
            double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
            char buf[32];
            if (gb >= 100.0)
                snprintf(buf, sizeof(buf), "%.0f GB", gb);
            else if (gb >= 10.0)
                snprintf(buf, sizeof(buf), "%.0f GB", gb);
            else
                snprintf(buf, sizeof(buf), "%.1f GB", gb);
            return buf;
        }

        std::string format_rank_range(const std::vector<int> &ranks)
        {
            if (ranks.empty())
                return "-";
            if (ranks.size() == 1)
                return std::to_string(ranks[0]);

            // Check if contiguous
            bool contiguous = true;
            for (size_t i = 1; i < ranks.size(); ++i)
            {
                if (ranks[i] != ranks[i - 1] + 1)
                {
                    contiguous = false;
                    break;
                }
            }
            if (contiguous)
                return std::to_string(ranks.front()) + "-" + std::to_string(ranks.back());

            // Comma-separated
            std::ostringstream oss;
            for (size_t i = 0; i < ranks.size(); ++i)
            {
                if (i > 0)
                    oss << ",";
                oss << ranks[i];
            }
            return oss.str();
        }

        /// Group GPUs by name and return a summary string like "4x MI60 (32 GB)"
        std::string summarize_gpus(const std::vector<DeviceInfo> &gpus)
        {
            if (gpus.empty())
                return "";

            // Count by name
            std::map<std::string, std::pair<int, size_t>> gpu_groups; // name -> (count, memory)
            for (const auto &g : gpus)
            {
                auto &entry = gpu_groups[g.name];
                entry.first++;
                entry.second = g.memory_bytes;
            }

            std::ostringstream oss;
            bool first = true;
            for (const auto &[name, info] : gpu_groups)
            {
                if (!first)
                    oss << " + ";
                first = false;
                if (info.first > 1)
                    oss << info.first << "x ";
                oss << name << " (" << format_memory_gb(info.second) << ")";
            }
            return oss.str();
        }

        /// Summarize CPU for a node
        std::string summarize_cpus(const NodeInventory &node, const std::vector<RankInventory> &node_ranks)
        {
            if (node_ranks.empty())
                return "";

            // Use first rank's cpu_socket_info — authoritative per-socket topology
            const auto &rank = node_ranks[0];
            int sockets = rank.cpu_sockets;

            // Get per-socket stats from cpu_socket_info (most reliable source)
            int cores_per_socket = 0;
            int threads_per_socket = 0;
            bool ht = false;
            std::string model_name;

            if (!rank.cpu_socket_info.empty())
            {
                const auto &sock = rank.cpu_socket_info[0];
                cores_per_socket = sock.num_physical_cores();
                threads_per_socket = sock.num_threads();
                ht = sock.has_hyperthreading();
                model_name = sock.model_name;
            }
            else
            {
                // Fallback: derive from rank-level fields
                cores_per_socket = rank.cpu_cores;
                threads_per_socket = rank.cpu.compute_units > 0 ? rank.cpu.compute_units : cores_per_socket;
                ht = (threads_per_socket > cores_per_socket && cores_per_socket > 0);
            }

            std::ostringstream oss;
            if (sockets > 1)
                oss << sockets << "x ";
            if (!model_name.empty())
                oss << model_name;
            else if (!rank.cpu.name.empty())
                oss << rank.cpu.name;
            else
                oss << "CPU";
            oss << " (" << cores_per_socket << "c/" << threads_per_socket << "t";
            if (ht)
                oss << ", HT";
            oss << ")";
            oss << " -- " << format_memory_gb(node.total_cpu_memory);
            return oss.str();
        }

        /// Detect interconnect type for a node's GPUs
        std::string detect_interconnect(const std::vector<RankInventory> &node_ranks)
        {
            if (node_ranks.empty())
                return "N/A";

            // Check P2P matrices (flat vectors: p2p_X[i * count + j])
            bool has_p2p = false;
            for (const auto &rank : node_ranks)
            {
                for (bool v : rank.p2p_rocm)
                    if (v)
                        has_p2p = true;
                for (bool v : rank.p2p_cuda)
                    if (v)
                        has_p2p = true;
            }

            if (has_p2p)
                return "P2P";
            if (!node_ranks[0].gpus.empty())
                return "PCIe (no P2P)";
            return "N/A";
        }

        /// Build device lines grouped by NUMA node.
        /// Returns one string per line, e.g. "NUMA 0: 2x MI60 (32 GB) | NUMA 1: 2x MI60 (32 GB)"
        /// or for single-NUMA: "4x MI60 (32 GB)"
        std::string format_devices_by_numa(const std::vector<DeviceInfo> &gpus)
        {
            if (gpus.empty())
                return "";

            // Group by NUMA node
            std::map<int, std::vector<const DeviceInfo *>> by_numa;
            for (const auto &g : gpus)
            {
                int numa = g.numa_node >= 0 ? g.numa_node : -1;
                by_numa[numa].push_back(&g);
            }

            // If all on same NUMA (or all unknown), just show summary
            if (by_numa.size() == 1)
            {
                auto &devices = by_numa.begin()->second;
                // Group by name within the NUMA node
                std::map<std::string, std::pair<int, size_t>> groups;
                for (const auto *d : devices)
                {
                    auto &entry = groups[d->name];
                    entry.first++;
                    entry.second = d->memory_bytes;
                }
                std::ostringstream oss;
                bool first = true;
                for (const auto &[name, info] : groups)
                {
                    if (!first)
                        oss << " + ";
                    first = false;
                    if (info.first > 1)
                        oss << info.first << "x ";
                    oss << name << " (" << format_memory_gb(info.second) << ")";
                }
                return oss.str();
            }

            // Multiple NUMA nodes — format per-NUMA
            std::ostringstream oss;
            bool first_numa = true;
            for (const auto &[numa_id, devices] : by_numa)
            {
                if (!first_numa)
                    oss << "\n";
                first_numa = false;

                if (numa_id >= 0)
                    oss << "NUMA " << numa_id << ": ";
                else
                    oss << "Unknown: ";

                // Group by name within this NUMA
                std::map<std::string, std::pair<int, size_t>> groups;
                for (const auto *d : devices)
                {
                    auto &entry = groups[d->name];
                    entry.first++;
                    entry.second = d->memory_bytes;
                }
                bool first_dev = true;
                for (const auto &[name, info] : groups)
                {
                    if (!first_dev)
                        oss << " + ";
                    first_dev = false;
                    if (info.first > 1)
                        oss << info.first << "x ";
                    oss << name << " (" << format_memory_gb(info.second) << ")";
                }
            }
            return oss.str();
        }

        // =====================================================================
        // Table renderers
        // =====================================================================

        std::string render_topology_single_node(
            const ClusterInventory &cluster,
            int threads_per_rank,
            const std::string &bind_policy,
            const Colors &c)
        {
            std::ostringstream out;

            // Collect node info
            const auto &node = cluster.nodes[0];
            std::vector<RankInventory> node_ranks;
            std::vector<int> rank_ids;
            for (const auto &r : cluster.ranks)
            {
                if (r.node_id == node.node_id)
                {
                    node_ranks.push_back(r);
                    rank_ids.push_back(r.rank);
                }
            }

            std::string hostname = node.hostname;
            std::string cpu_line = summarize_cpus(node, node_ranks);
            std::string gpu_line = format_devices_by_numa(
                node_ranks.empty() ? std::vector<DeviceInfo>{} : node_ranks[0].gpus);
            std::string interconnect = detect_interconnect(node_ranks);

            // Build the table manually for the single-node compact format
            // (key-value layout, not multi-row)
            fort::char_table table;
            table.set_border_style(FT_BASIC_STYLE);

            // Title row (spans full width)
            table << fort::header
                  << "CLUSTER TOPOLOGY (1 node)" << "" << fort::endr;
            table[0][0].set_cell_span(2);
            table[0][0].set_cell_text_align(fort::text_align::center);
            if (c.teal[0])
                table[0][0].set_cell_content_fg_color(fort::color::cyan);

            table << "Node" << hostname << fort::endr;
            table << "CPU" << cpu_line << fort::endr;
            if (!gpu_line.empty())
            {
                table << "GPU" << gpu_line << fort::endr;
                table << "Interconnect" << interconnect << fort::endr;
            }

            // World summary row
            std::ostringstream world_oss;
            world_oss << cluster.world_size << " rank"
                      << (cluster.world_size > 1 ? "s" : "")
                      << " on 1 node";
            if (threads_per_rank > 0)
                world_oss << " | " << threads_per_rank << " threads/rank";
            if (!bind_policy.empty())
                world_oss << " | bind=" << bind_policy;

            table << fort::separator;
            table << "World" << world_oss.str() << fort::endr;
            if (c.teal[0])
            {
                // Color the key column
                table.column(0).set_cell_content_fg_color(fort::color::cyan);
            }

            out << c.teal << table.to_string() << c.reset;
            return out.str();
        }

        std::string render_topology_multi_node(
            const ClusterInventory &cluster,
            int threads_per_rank,
            const std::string &bind_policy,
            const Colors &c)
        {
            std::ostringstream out;

            fort::char_table table;
            table.set_border_style(FT_BASIC_STYLE);

            // Title
            std::string title = "CLUSTER TOPOLOGY (" + std::to_string(cluster.node_count) + " nodes)";
            table << fort::header << title << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
            if (c.teal[0])
                table[0][0].set_cell_content_fg_color(fort::color::cyan);

            // Column headers
            table << fort::header << "Node" << "Hostname" << "Ranks" << "Devices" << "Interconnect" << fort::endr;

            // Per-node rows
            for (const auto &node : cluster.nodes)
            {
                std::vector<RankInventory> node_ranks;
                std::vector<int> rank_ids;
                for (const auto &r : cluster.ranks)
                {
                    if (r.node_id == node.node_id)
                    {
                        node_ranks.push_back(r);
                        rank_ids.push_back(r.rank);
                    }
                }

                std::string devices;
                std::string gpu_summary = format_devices_by_numa(
                    node_ranks.empty() ? std::vector<DeviceInfo>{} : node_ranks[0].gpus);
                if (!gpu_summary.empty())
                    devices = gpu_summary;
                else
                    devices = summarize_cpus(node, node_ranks);

                table << std::to_string(node.node_id)
                      << node.hostname
                      << format_rank_range(rank_ids)
                      << devices
                      << detect_interconnect(node_ranks)
                      << fort::endr;
            }

            // World summary footer
            std::ostringstream world_oss;
            world_oss << cluster.world_size << " ranks on "
                      << cluster.node_count << " nodes";
            if (threads_per_rank > 0)
                world_oss << " | " << threads_per_rank << " threads/rank";
            if (!bind_policy.empty())
                world_oss << " | bind=" << bind_policy;

            table << fort::separator;
            table << "World" << world_oss.str() << "" << "" << "" << fort::endr;
            table[static_cast<int>(cluster.nodes.size()) + 2][1].set_cell_span(4);

            if (c.teal[0])
                table.column(0).set_cell_content_fg_color(fort::color::cyan);

            out << c.teal << table.to_string() << c.reset;
            return out.str();
        }

        std::string render_topology(const StartupBannerData &data, const Colors &c)
        {
            if (!data.cluster || data.cluster->nodes.empty())
                return "";

            if (data.cluster->node_count <= 1)
                return render_topology_single_node(*data.cluster, data.threads_per_rank, data.bind_policy, c);
            else
                return render_topology_multi_node(*data.cluster, data.threads_per_rank, data.bind_policy, c);
        }

        std::string render_config(const StartupBannerData &data, const Colors &c)
        {
            fort::char_table table;
            table.set_border_style(FT_BASIC_STYLE);

            table << fort::header << "INFERENCE CONFIGURATION" << "" << fort::endr;
            table[0][0].set_cell_span(2);
            table[0][0].set_cell_text_align(fort::text_align::center);
            if (c.teal[0])
                table[0][0].set_cell_content_fg_color(fort::color::cyan);

            table << "Device" << data.device_description << fort::endr;
            table << "Parallelism" << data.parallelism << fort::endr;
            table << "Precision" << data.precision << fort::endr;
            table << "Context Length" << data.context_length << fort::endr;
            table << "Backend" << data.backend << fort::endr;

            if (c.teal[0])
                table.column(0).set_cell_content_fg_color(fort::color::cyan);

            std::ostringstream out;
            out << c.teal << table.to_string() << c.reset;
            return out.str();
        }

        std::string render_model(const StartupBannerData &data, const Colors &c)
        {
            fort::char_table table;
            table.set_border_style(FT_BASIC_STYLE);

            std::string file_cell = data.model_filename;
            if (!data.model_size.empty())
                file_cell += " (" + data.model_size + ")";

            table << fort::header << "MODEL" << "" << fort::endr;
            table[0][0].set_cell_span(2);
            table[0][0].set_cell_text_align(fort::text_align::center);
            if (c.teal[0])
                table[0][0].set_cell_content_fg_color(fort::color::cyan);

            table << "File" << file_cell << fort::endr;
            table << "Architecture" << data.architecture << fort::endr;
            if (!data.quantization.empty())
                table << "Quantization" << data.quantization << fort::endr;
            table << "Vocab" << data.vocab << fort::endr;
            if (!data.thinking.empty())
                table << "Thinking" << data.thinking << fort::endr;

            if (c.teal[0])
                table.column(0).set_cell_content_fg_color(fort::color::cyan);

            std::ostringstream out;
            out << c.teal << table.to_string() << c.reset;
            return out.str();
        }

        std::string render_preflight(const StartupBannerData &data, const Colors &c)
        {
            if (data.preflight_checks.empty())
                return "";

            fort::char_table table;
            table.set_border_style(FT_BASIC_STYLE);

            table << fort::header << "PREFLIGHT CHECKS" << "" << "" << fort::endr;
            table[0][0].set_cell_span(3);
            table[0][0].set_cell_text_align(fort::text_align::center);
            if (c.teal[0])
                table[0][0].set_cell_content_fg_color(fort::color::cyan);

            table << fort::header << "Check" << "Status" << "Detail" << fort::endr;

            for (const auto &check : data.preflight_checks)
            {
                std::string status = check.passed ? "OK" : "FAIL";
                table << check.name << status << check.detail << fort::endr;
            }

            // Apply colors
            if (c.teal[0])
            {
                table.column(0).set_cell_content_fg_color(fort::color::cyan);
                // Color status column per-row
                for (size_t i = 0; i < data.preflight_checks.size(); ++i)
                {
                    int row = static_cast<int>(i) + 2; // skip title + header
                    if (data.preflight_checks[i].passed)
                        table[row][1].set_cell_content_fg_color(fort::color::green);
                    else
                        table[row][1].set_cell_content_fg_color(fort::color::light_red);
                }
            }

            std::ostringstream out;
            out << c.teal << table.to_string() << c.reset;

            // Append mitigation lines for failures (below the table)
            for (const auto &check : data.preflight_checks)
            {
                if (!check.passed && !check.mitigation.empty())
                {
                    out << c.red << "  -> " << check.mitigation << c.reset << "\n";
                }
            }

            return out.str();
        }

    } // namespace

    // =========================================================================
    // Public API
    // =========================================================================

    std::string StartupBanner::render(const StartupBannerData &data, bool use_color)
    {
        const Colors &c = use_color ? kColor : kNoColor;
        std::ostringstream out;

        std::string topology = render_topology(data, c);
        std::string config = render_config(data, c);
        std::string model = render_model(data, c);
        std::string preflight = render_preflight(data, c);

        if (!topology.empty())
            out << topology << "\n";
        if (!config.empty())
            out << config << "\n";
        if (!model.empty())
            out << model << "\n";
        if (!preflight.empty())
            out << preflight;

        return out.str();
    }

    bool StartupBanner::shouldUseColor()
    {
        if (std::getenv("NO_COLOR"))
            return false;
        if (std::getenv("LLAMINAR_NO_COLOR"))
            return false;
        // Check if stderr is a TTY (LOG_INFO goes to stderr)
        return isatty(fileno(stderr)) != 0;
    }

} // namespace llaminar2
