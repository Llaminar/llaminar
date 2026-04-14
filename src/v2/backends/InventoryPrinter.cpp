/**
 * @file InventoryPrinter.cpp
 * @brief Renders hardware inventory tables using libfort.
 *
 * Reads from ClusterInventory / RankInventory and prints CPU, GPU,
 * and P2P tables via LOG_INFO.
 *
 * @author David Sanftenberg
 * @date 2026-04-11
 */

#include "InventoryPrinter.h"
#include "CPUSocketInfo.h"
#include "../execution/mpi_orchestration/DeviceInventory.h"
#include "../utils/Logger.h"
#include "fort.hpp"

#include <sstream>
#include <map>
#include <set>

namespace llaminar2
{

    namespace
    {
        // =====================================================================
        // Format helpers
        // =====================================================================

        std::string format_memory_gb(size_t bytes)
        {
            return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
        }

        std::string format_pcie_link(const DeviceInfo &gpu)
        {
            if (gpu.pcie_speed_gts <= 0)
                return "N/A";
            double encoding_efficiency = (gpu.pcie_gen >= 3) ? (128.0 / 130.0) : 0.8;
            double bw = gpu.pcie_speed_gts * gpu.pcie_width * encoding_efficiency / 8.0;
            char buf[64];
            snprintf(buf, sizeof(buf), "Gen%d x%d %.0f GB/s",
                     gpu.pcie_gen, gpu.pcie_width, bw);
            return buf;
        }

        std::string format_numa(int numa_node)
        {
            if (numa_node < 0)
                return "-";
            return std::to_string(numa_node);
        }

        std::string strip_vendor_prefix(const std::string &name)
        {
            if (name.compare(0, 7, "NVIDIA ") == 0)
                return name.substr(7);
            if (name.compare(0, 4, "AMD ") == 0)
                return name.substr(4);
            return name;
        }

        std::string format_cpu_ranges(const std::vector<int> &cpus)
        {
            if (cpus.empty())
                return "-";
            std::ostringstream oss;
            int start = cpus[0], prev = cpus[0];
            for (size_t i = 1; i <= cpus.size(); ++i)
            {
                if (i < cpus.size() && cpus[i] == prev + 1)
                {
                    prev = cpus[i];
                }
                else
                {
                    if (oss.tellp() > 0)
                        oss << ", ";
                    if (start == prev)
                        oss << start;
                    else
                        oss << start << "-" << prev;
                    if (i < cpus.size())
                    {
                        start = cpus[i];
                        prev = cpus[i];
                    }
                }
            }
            return oss.str();
        }

        void log_table(const std::string &table_str)
        {
            std::istringstream stream(table_str);
            std::string line;
            while (std::getline(stream, line))
            {
                if (!line.empty())
                    LOG_INFO(line);
            }
        }

        // =====================================================================
        // Table renderers (from DeviceInventory types)
        // =====================================================================

        void render_cpu_table(const std::string &node_label,
                              const std::vector<CPUSocketInfo> &sockets)
        {
            if (sockets.empty())
                return;

            const bool has_ht = !sockets[0].ht_threads.empty();
            const int num_cols = has_ht ? 7 : 6;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            table << node_label;
            for (int c = 1; c < num_cols; ++c)
                table << "";
            table << fort::endr;
            table[0][0].set_cell_span(num_cols);
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.row(0).set_cell_row_type(fort::row_type::header);

            // Header
            if (has_ht)
                table << "Socket" << "Processor" << "NUMA" << "Physical Cores" << "HT Threads" << "Cores" << "Memory" << fort::endr;
            else
                table << "Socket" << "Processor" << "NUMA" << "Cores" << "Core Count" << "Memory" << fort::endr;
            table.row(1).set_cell_row_type(fort::row_type::header);

            table.column(0).set_cell_text_align(fort::text_align::center);
            table.column(1).set_cell_text_align(fort::text_align::left);
            table.column(2).set_cell_text_align(fort::text_align::center);
            if (has_ht)
            {
                table.column(3).set_cell_text_align(fort::text_align::left);
                table.column(4).set_cell_text_align(fort::text_align::left);
                table.column(5).set_cell_text_align(fort::text_align::center);
                table.column(6).set_cell_text_align(fort::text_align::right);
            }
            else
            {
                table.column(3).set_cell_text_align(fort::text_align::left);
                table.column(4).set_cell_text_align(fort::text_align::center);
                table.column(5).set_cell_text_align(fort::text_align::right);
            }

            // Data rows
            for (const auto &s : sockets)
            {
                std::string cores_str = std::to_string(s.num_physical_cores()) + "c/" + std::to_string(s.num_threads()) + "t";

                if (has_ht)
                {
                    table << std::to_string(s.socket_id)
                          << s.model_name
                          << std::to_string(s.numa_node)
                          << format_cpu_ranges(s.physical_cores)
                          << format_cpu_ranges(s.ht_threads)
                          << cores_str
                          << format_memory_gb(s.memory_bytes)
                          << fort::endr;
                }
                else
                {
                    table << std::to_string(s.socket_id)
                          << s.model_name
                          << std::to_string(s.numa_node)
                          << format_cpu_ranges(s.physical_cores)
                          << cores_str
                          << format_memory_gb(s.memory_bytes)
                          << fort::endr;
                }
            }

            // Total row (multi-socket)
            if (sockets.size() > 1)
            {
                int total_phys = 0, total_threads = 0;
                size_t total_mem = 0;
                for (const auto &s : sockets)
                {
                    total_phys += s.num_physical_cores();
                    total_threads += s.num_threads();
                    total_mem += s.memory_bytes;
                }
                std::string total_cores_str = std::to_string(total_phys) + "c/" + std::to_string(total_threads) + "t total";

                table << fort::separator;
                if (has_ht)
                    table << "" << "Total" << "" << "" << "" << total_cores_str << format_memory_gb(total_mem) << fort::endr;
                else
                    table << "" << "Total" << "" << "" << total_cores_str << format_memory_gb(total_mem) << fort::endr;
            }

            log_table(table.to_string());
        }

        void render_gpu_table(const char *title,
                              const std::vector<DeviceInfo> &gpus)
        {
            if (gpus.empty())
                return;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            table << title << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.row(0).set_cell_row_type(fort::row_type::header);

            // Header
            table << "ID" << "Name" << "VRAM" << "PCIe Link" << "NUMA" << fort::endr;
            table.row(1).set_cell_row_type(fort::row_type::header);

            table.column(0).set_cell_text_align(fort::text_align::center);
            table.column(1).set_cell_text_align(fort::text_align::left);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::left);
            table.column(4).set_cell_text_align(fort::text_align::center);

            for (const auto &gpu : gpus)
            {
                std::string display_name = strip_vendor_prefix(gpu.name);

                if (gpu.type == DeviceType::CUDA && gpu.compute_capability_major > 0)
                {
                    std::string sm = "SM " + std::to_string(gpu.compute_capability_major) + "." + std::to_string(gpu.compute_capability_minor);
                    if (display_name.find("SM") == std::string::npos)
                        display_name += " (" + sm + ")";
                }

                table << std::to_string(gpu.local_device_id)
                      << display_name
                      << format_memory_gb(gpu.memory_bytes)
                      << format_pcie_link(gpu)
                      << format_numa(gpu.numa_node)
                      << fort::endr;
            }

            log_table(table.to_string());

            // Degraded link warnings
            for (const auto &gpu : gpus)
            {
                if (gpu.pcie_degraded)
                {
                    int max_gen = (gpu.pcie_max_speed_gts >= 64.0)   ? 6
                                  : (gpu.pcie_max_speed_gts >= 32.0) ? 5
                                  : (gpu.pcie_max_speed_gts >= 16.0) ? 4
                                  : (gpu.pcie_max_speed_gts >= 8.0)  ? 3
                                  : (gpu.pcie_max_speed_gts >= 5.0)  ? 2
                                                                     : 1;
                    double eff = (max_gen >= 3) ? (128.0 / 130.0) : 0.8;
                    double max_bw = gpu.pcie_max_speed_gts * gpu.pcie_max_width * eff / 8.0;
                    char cap_buf[64];
                    snprintf(cap_buf, sizeof(cap_buf), "Gen%d x%d (%.1f GB/s)",
                             max_gen, gpu.pcie_max_width, max_bw);

                    const char *type_prefix = (gpu.type == DeviceType::CUDA) ? "cuda" : "rocm";
                    if (!gpu.pcie_bottleneck_bdf.empty())
                    {
                        LOG_WARN("  ⚠ " << type_prefix << ":" << gpu.local_device_id
                                        << " link degraded: " << format_pcie_link(gpu)
                                        << " — capable of " << cap_buf
                                        << " (bottleneck at upstream bridge " << gpu.pcie_bottleneck_bdf << ")");
                    }
                    else
                    {
                        LOG_WARN("  ⚠ " << type_prefix << ":" << gpu.local_device_id
                                        << " link degraded: " << format_pcie_link(gpu)
                                        << " — capable of " << cap_buf);
                    }
                }
            }
        }

        void render_p2p_table(const char *backend_name,
                              const std::vector<bool> &matrix,
                              int device_count,
                              const std::vector<DeviceInfo> &gpus)
        {
            if (device_count < 2)
                return;

            int p2p_pairs = 0;
            for (int i = 0; i < device_count; ++i)
                for (int j = 0; j < device_count; ++j)
                    if (i != j && matrix[i * device_count + j])
                        ++p2p_pairs;
            const int total_pairs = device_count * (device_count - 1);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title
            std::ostringstream title;
            title << backend_name << " P2P Access (" << p2p_pairs << "/" << total_pairs << " pairs)";
            table << title.str();
            for (int j = 0; j < device_count; ++j)
                table << "";
            table << fort::endr;
            table[0][0].set_cell_span(device_count + 1);
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.row(0).set_cell_row_type(fort::row_type::header);

            // Header: "" + GPU0 GPU1 ...
            table << "";
            for (int j = 0; j < device_count; ++j)
            {
                int dev_id = (j < static_cast<int>(gpus.size())) ? gpus[j].local_device_id : j;
                table << ("GPU" + std::to_string(dev_id));
            }
            table << fort::endr;
            table.row(1).set_cell_row_type(fort::row_type::header);

            for (int c = 0; c <= device_count; ++c)
                table.column(c).set_cell_text_align(fort::text_align::center);

            // Data rows
            for (int i = 0; i < device_count; ++i)
            {
                int dev_id = (i < static_cast<int>(gpus.size())) ? gpus[i].local_device_id : i;
                table << ("GPU" + std::to_string(dev_id));
                for (int j = 0; j < device_count; ++j)
                {
                    if (i == j)
                        table << "-";
                    else
                        table << (matrix[i * device_count + j] ? "✓" : "✗");
                }
                table << fort::endr;
            }

            log_table(table.to_string());
        }

        /// Filter GPUs from a RankInventory by type
        std::vector<DeviceInfo> filter_gpus(const RankInventory &rank, DeviceType type)
        {
            std::vector<DeviceInfo> result;
            for (const auto &gpu : rank.gpus)
            {
                if (gpu.type == type)
                    result.push_back(gpu);
            }
            return result;
        }

    } // anonymous namespace

    // =========================================================================
    // Public API
    // =========================================================================

    void InventoryPrinter::printRankInventory(const RankInventory &rank)
    {
        // CPU table
        if (!rank.cpu_socket_info.empty())
            render_cpu_table("CPU", rank.cpu_socket_info);

        // NVIDIA GPU table
        auto cuda_gpus = filter_gpus(rank, DeviceType::CUDA);
        if (!cuda_gpus.empty())
            render_gpu_table("NVIDIA GPU", cuda_gpus);

        // AMD GPU table
        auto rocm_gpus = filter_gpus(rank, DeviceType::ROCm);
        if (!rocm_gpus.empty())
            render_gpu_table("AMD GPU", rocm_gpus);

        // P2P tables
        if (rank.p2p_cuda_count >= 2)
            render_p2p_table("NVIDIA", rank.p2p_cuda, rank.p2p_cuda_count, cuda_gpus);
        if (rank.p2p_rocm_count >= 2)
            render_p2p_table("AMD", rank.p2p_rocm, rank.p2p_rocm_count, rocm_gpus);
    }

    void InventoryPrinter::printClusterInventory(const ClusterInventory &inventory)
    {
        if (inventory.ranks.empty())
            return;

        // Single-node cluster: print as a single RankInventory
        if (inventory.node_count <= 1)
        {
            // Merge all ranks' GPUs into one view (they share the same node)
            // Use the first rank's CPU info (same node, same CPUs)
            printRankInventory(inventory.ranks[0]);
            return;
        }

        // Multi-node: print per-node tables
        // Group ranks by node
        std::map<int, std::vector<int>> node_ranks;
        for (const auto &r : inventory.ranks)
            node_ranks[r.node_id].push_back(r.rank);

        for (const auto &[node_id, ranks] : node_ranks)
        {
            if (ranks.empty())
                continue;

            const auto &first_rank = inventory.ranks[ranks[0]];
            std::string node_label = "Node " + std::to_string(node_id) + " (" + first_rank.hostname + ")";

            // CPU
            if (!first_rank.cpu_socket_info.empty())
                render_cpu_table(node_label + " — CPU", first_rank.cpu_socket_info);

            // Merge GPUs from all ranks on this node
            std::vector<DeviceInfo> cuda_gpus, rocm_gpus;
            const RankInventory *best_rank = &first_rank; // Use first rank for P2P
            for (int r : ranks)
            {
                for (const auto &gpu : inventory.ranks[r].gpus)
                {
                    if (gpu.type == DeviceType::CUDA)
                        cuda_gpus.push_back(gpu);
                    else if (gpu.type == DeviceType::ROCm)
                        rocm_gpus.push_back(gpu);
                }
            }

            if (!cuda_gpus.empty())
                render_gpu_table((node_label + " — NVIDIA GPU").c_str(), cuda_gpus);
            if (!rocm_gpus.empty())
                render_gpu_table((node_label + " — AMD GPU").c_str(), rocm_gpus);

            // P2P from first rank (node-local)
            if (best_rank->p2p_cuda_count >= 2)
                render_p2p_table("NVIDIA", best_rank->p2p_cuda, best_rank->p2p_cuda_count, cuda_gpus);
            if (best_rank->p2p_rocm_count >= 2)
                render_p2p_table("AMD", best_rank->p2p_rocm, best_rank->p2p_rocm_count, rocm_gpus);
        }
    }

} // namespace llaminar2
