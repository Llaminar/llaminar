/**
 * @file ComputeBackend.cpp
 * @brief DeviceManager ownership of one observed hardware inventory and its views.
 *
 * Initialization publishes HardwareInventory before any cluster projection or
 * display. NUMA-filtered device lists and peer queries are views of that value,
 * never independent discovery/accounting paths.
 *
 * Supports:
 * - CPU (OpenBLAS/MKL)
 * - NVIDIA CUDA
 * - AMD ROCm
 *
 * Phase 6: Multi-GPU (heterogeneous)
 * GPU enumeration is now in separate compilation units to avoid header conflicts:
 *   - CUDAEnumeration.cu (CUDA only)
 *   - ROCmEnumeration.cpp (ROCm only, compiled with hipcc)
 * This allows CUDA and ROCm to coexist in the same binary.
 *
 * @author David Sanftenberg
 */

#include "ComputeBackend.h"
#include "HardwareInventory.h"
#include "../utils/Logger.h"
#include "../utils/CPUFeatures.h"
#include "../utils/NUMATopology.h"
#include "../kernels/cpu/ops/CPURoPEKernelT.h"
#include "../kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "../kernels/cpu/ops/CPUSoftmaxKernelT.h"
#include "fort.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <numa.h>

// ============================================================================
// GPU Header Includes REMOVED (Phase 6)
// ============================================================================
// GPU enumeration moved to separate compilation units to enable heterogeneous
// multi-GPU (CUDA + ROCm in same binary). See:
//   - backends/CUDAEnumeration.cu
//   - backends/ROCmEnumeration.cpp
//   - backends/GPUEnumeration.h (declarations)
// ============================================================================


namespace llaminar2
{

    // ============================================================================
    // Helper: Get backend type name
    // ============================================================================

    static const char *backend_type_name(ComputeBackendType type)
    {
        switch (type)
        {
        case ComputeBackendType::CPU:
            return "CPU";
        case ComputeBackendType::GPU_CUDA:
            return "NVIDIA CUDA";
        case ComputeBackendType::GPU_ROCM:
            return "AMD ROCm";
        case ComputeBackendType::GPU_VULKAN:
            return "Vulkan";
        default:
            return "Unknown";
        }
    }

    namespace
    {
        // Read CPU model name per socket from /proc/cpuinfo
        // Returns socket_id -> model name (different sockets may have different CPUs)
        // Format memory size as "XX GB"
        std::string format_memory_gb(size_t bytes)
        {
            return std::to_string(bytes / (1024ULL * 1024 * 1024)) + " GB";
        }

        // Format PCIe link info as "GenX xY ZZ GB/s"
        std::string format_pcie_link(const PCIeLinkInfo &pcie)
        {
            if (pcie.link_speed_gts <= 0)
                return "N/A";
            char buf[64];
            snprintf(buf, sizeof(buf), "Gen%d x%d %.0f GB/s",
                     pcie.pcie_gen, pcie.link_width, pcie.bandwidth_gbps());
            return buf;
        }

        // Format NUMA node(s) as a string
        std::string format_numa(int numa_node)
        {
            if (numa_node < 0)
                return "-";
            return std::to_string(numa_node);
        }

        // Strip vendor prefix from GPU name for cleaner display
        std::string strip_vendor_prefix(const std::string &name)
        {
            // Remove "NVIDIA " prefix
            if (name.compare(0, 7, "NVIDIA ") == 0)
                return name.substr(7);
            // Remove "AMD " prefix
            if (name.compare(0, 4, "AMD ") == 0)
                return name.substr(4);
            return name;
        }

        // Log a libfort table through the Logger (line by line)
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

        // Build and log a GPU table for a given backend
        void log_gpu_table(const char *title,
                           const std::vector<ComputeDevice> &devices)
        {
            if (devices.empty())
                return;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row spanning all columns
            table << title << "" << "" << "" << "" << fort::endr;
            table[0][0].set_cell_span(5);
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.row(0).set_cell_row_type(fort::row_type::header);

            // Header row
            table << "ID" << "Name" << "VRAM" << "PCIe Link" << "NUMA" << fort::endr;
            table.row(1).set_cell_row_type(fort::row_type::header);

            table.column(0).set_cell_text_align(fort::text_align::center);
            table.column(1).set_cell_text_align(fort::text_align::left);
            table.column(2).set_cell_text_align(fort::text_align::right);
            table.column(3).set_cell_text_align(fort::text_align::left);
            table.column(4).set_cell_text_align(fort::text_align::center);

            for (const auto &dev : devices)
            {
                // Build name with arch info embedded
                std::string display_name = strip_vendor_prefix(dev.name);

                // For CUDA devices, append SM version if not already in name
                if (dev.type == ComputeBackendType::GPU_CUDA)
                {
                    std::string sm = "SM " + std::to_string(dev.compute_capability / 10) + "." + std::to_string(dev.compute_capability % 10);
                    if (display_name.find("SM") == std::string::npos)
                        display_name += " (" + sm + ")";
                }

                table << std::to_string(dev.device_id)
                      << display_name
                      << format_memory_gb(dev.total_memory_bytes)
                      << format_pcie_link(dev.pcie)
                      << format_numa(dev.numa_node)
                      << fort::endr;
            }

            log_table(table.to_string());

            /*
             * Enumeration precedes model traffic, and autonomous PCIe power
             * management may temporarily reduce link speed or width. Keep the
             * observation available as a diagnostic without presenting this
             * pre-workload snapshot as an inference failure.
             */
            for (const auto &dev : devices)
            {
                if (dev.pcie.degraded)
                {
                    int max_gen = (dev.pcie.max_speed_gts >= 64.0)   ? 6
                                  : (dev.pcie.max_speed_gts >= 32.0) ? 5
                                  : (dev.pcie.max_speed_gts >= 16.0) ? 4
                                  : (dev.pcie.max_speed_gts >= 8.0)  ? 3
                                  : (dev.pcie.max_speed_gts >= 5.0)  ? 2
                                                                     : 1;
                    char cap_buf[64];
                    snprintf(cap_buf, sizeof(cap_buf), "Gen%d x%d (%.1f GB/s)",
                             max_gen, dev.pcie.max_width,
                             // Compute max bandwidth
                             dev.pcie.max_speed_gts * dev.pcie.max_width * ((max_gen >= 3) ? (128.0 / 130.0) : 0.8) / 8.0);

                    const char *type_prefix = (dev.type == ComputeBackendType::GPU_CUDA) ? "cuda" : "rocm";
                    if (!dev.pcie.bottleneck_bdf.empty())
                    {
                        LOG_DEBUG("  " << type_prefix << ":" << dev.device_id
                                       << " pre-workload link snapshot: " << format_pcie_link(dev.pcie)
                                       << "; capable of " << cap_buf
                                       << " (narrowest upstream bridge " << dev.pcie.bottleneck_bdf << ")");
                    }
                    else
                    {
                        LOG_DEBUG("  " << type_prefix << ":" << dev.device_id
                                       << " pre-workload link snapshot: " << format_pcie_link(dev.pcie)
                                       << "; capable of " << cap_buf);
                    }
                }
            }
        }

        // Build and log a P2P access matrix table
        void log_p2p_table(const char *backend_name, const P2PMatrix &matrix)
        {
            const int n = matrix.device_count();
            if (n < 2)
                return;

            // Count P2P-enabled pairs
            int p2p_pairs = 0;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    if (i != j && matrix.can_access[i][j])
                        ++p2p_pairs;
            const int total_pairs = n * (n - 1);

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            std::ostringstream title;
            title << backend_name << " P2P Access (" << p2p_pairs << "/" << total_pairs << " pairs)";
            // First cell + N GPU columns
            table << title.str();
            for (int j = 0; j < n; ++j)
                table << "";
            table << fort::endr;
            table[0][0].set_cell_span(n + 1);
            table[0][0].set_cell_text_align(fort::text_align::center);
            table.row(0).set_cell_row_type(fort::row_type::header);

            // Header row: empty + GPU0, GPU1, ...
            table << "";
            for (int j = 0; j < n; ++j)
                table << ("GPU" + std::to_string(matrix.device_ids[j]));
            table << fort::endr;
            table.row(1).set_cell_row_type(fort::row_type::header);

            // Set all columns to center
            for (int c = 0; c <= n; ++c)
                table.column(c).set_cell_text_align(fort::text_align::center);

            // Data rows
            for (int i = 0; i < n; ++i)
            {
                table << ("GPU" + std::to_string(matrix.device_ids[i]));
                for (int j = 0; j < n; ++j)
                {
                    if (i == j)
                        table << "-";
                    else
                        table << (matrix.can_access[i][j] ? "✓" : "✗");
                }
                table << fort::endr;
            }

            log_table(table.to_string());
        }
    } // anonymous namespace

    DeviceManager::DeviceManager() = default;
    DeviceManager::~DeviceManager() = default;

    void DeviceManager::initialize(int local_numa_node, bool log_inventory)
    {
        devices_.clear();
        contexts_.clear();
        local_numa_node_ = local_numa_node;

        // Log NUMA filtering mode
        if (local_numa_node >= 0)
        {
            LOG_DEBUG("[DeviceManager] Initializing with NUMA node " << local_numa_node << " filtering (MPI rank mode)");
        }
        else
        {
            LOG_DEBUG("[DeviceManager] Initializing without NUMA filtering (all devices visible)");
        }

        // Publish one complete observation independent of logging. Hardware
        // discovery already owns vendor startup policy, NUMA and P2P queries;
        // repeating those queries here used to leave hardware() unpopulated.
        hardware_ = std::make_unique<HardwareInventory>(HardwareInventory::detect());
        devices_.push_back(hardware_->cpuDevice(local_numa_node));

        auto cuda_devices = hardware_->cuda_devices;
        auto rocm_devices = hardware_->rocm_devices;
        const auto retainLocal = [local_numa_node](std::vector<ComputeDevice> &devices)
        {
            if (local_numa_node < 0) return;
            std::erase_if(devices, [local_numa_node](const ComputeDevice &device)
            {
                return !NUMATopology::isGPULocalToProcess(device.numa_node, local_numa_node);
            });
        };
        retainLocal(cuda_devices);
        retainLocal(rocm_devices);
        devices_.insert(devices_.end(), cuda_devices.begin(), cuda_devices.end());
        devices_.insert(devices_.end(), rocm_devices.begin(), rocm_devices.end());

        // Peer matrices retain their real backend ordinal maps. Exact-domain
        // queries select edges by ordinal, so filtered views cannot renumber
        // devices or turn asymmetric access into an all-peer assertion.
        p2p_matrices_.clear();
        if (hardware_->cuda_p2p) p2p_matrices_.push_back(*hardware_->cuda_p2p);
        if (hardware_->rocm_p2p) p2p_matrices_.push_back(*hardware_->rocm_p2p);

        // Resize contexts_ vector to match devices
        contexts_.resize(devices_.size(), nullptr);

        // ====================================================================
        // Device inventory tables (libfort) — print once on first init
        // ====================================================================
        if (log_inventory && !inventory_logged_)
        {
            inventory_logged_ = true;

            // --- CPU table (per-socket detail) ---
            {
                // Display consumes the exact same observation as planning;
                // quiet startup and verbose startup publish identical facts.
                const auto &sockets = hardware_->cpu_sockets;

                // Compute total memory for title
                size_t total_mem = 0;
                for (const auto &s : sockets)
                    total_mem += s.memory_bytes;

                // Build table
                const bool has_ht = !sockets.empty() && !sockets[0].ht_threads.empty();
                const int num_cols = has_ht ? 7 : 6;

                fort::utf8_table cpu_table;
                cpu_table.set_border_style(FT_DOUBLE2_STYLE);

                // Title row
                cpu_table << "CPU";
                for (int c = 1; c < num_cols; ++c)
                    cpu_table << "";
                cpu_table << fort::endr;
                cpu_table[0][0].set_cell_span(num_cols);
                cpu_table[0][0].set_cell_text_align(fort::text_align::center);
                cpu_table.row(0).set_cell_row_type(fort::row_type::header);

                // Header row
                if (has_ht)
                    cpu_table << "Socket" << "Processor" << "NUMA" << "Physical Cores" << "HT Threads" << "Cores" << "Memory" << fort::endr;
                else
                    cpu_table << "Socket" << "Processor" << "NUMA" << "Cores" << "Core Count" << "Memory" << fort::endr;
                cpu_table.row(1).set_cell_row_type(fort::row_type::header);

                // Column alignments
                cpu_table.column(0).set_cell_text_align(fort::text_align::center);
                cpu_table.column(1).set_cell_text_align(fort::text_align::left);
                cpu_table.column(2).set_cell_text_align(fort::text_align::center);
                if (has_ht)
                {
                    cpu_table.column(3).set_cell_text_align(fort::text_align::left);
                    cpu_table.column(4).set_cell_text_align(fort::text_align::left);
                    cpu_table.column(5).set_cell_text_align(fort::text_align::center);
                    cpu_table.column(6).set_cell_text_align(fort::text_align::right);
                }
                else
                {
                    cpu_table.column(3).set_cell_text_align(fort::text_align::left);
                    cpu_table.column(4).set_cell_text_align(fort::text_align::center);
                    cpu_table.column(5).set_cell_text_align(fort::text_align::right);
                }

                // Data rows
                for (const auto &s : sockets)
                {
                    std::string cores_str = std::to_string(s.physical_cores.size()) + "c/" + std::to_string(s.physical_cores.size() + s.ht_threads.size()) + "t";

                    if (has_ht)
                    {
                        cpu_table << std::to_string(s.socket_id)
                                  << s.model_name
                                  << std::to_string(s.numa_node)
                                  << HardwareInventory::formatCpuRanges(s.physical_cores)
                                  << HardwareInventory::formatCpuRanges(s.ht_threads)
                                  << cores_str
                                  << format_memory_gb(s.memory_bytes)
                                  << fort::endr;
                    }
                    else
                    {
                        cpu_table << std::to_string(s.socket_id)
                                  << s.model_name
                                  << std::to_string(s.numa_node)
                                  << HardwareInventory::formatCpuRanges(s.physical_cores)
                                  << cores_str
                                  << format_memory_gb(s.memory_bytes)
                                  << fort::endr;
                    }
                }

                // Total row
                if (sockets.size() > 1)
                {
                    int total_phys = 0, total_threads = 0;
                    for (const auto &s : sockets)
                    {
                        total_phys += static_cast<int>(s.physical_cores.size());
                        total_threads += static_cast<int>(s.physical_cores.size() + s.ht_threads.size());
                    }
                    std::string total_cores_str = std::to_string(total_phys) + "c/" + std::to_string(total_threads) + "t total";

                    cpu_table << fort::separator;
                    if (has_ht)
                        cpu_table << "" << "Total" << "" << "" << "" << total_cores_str << format_memory_gb(total_mem) << fort::endr;
                    else
                        cpu_table << "" << "Total" << "" << "" << total_cores_str << format_memory_gb(total_mem) << fort::endr;
                }

                log_table(cpu_table.to_string());
            }

            // --- NVIDIA CUDA GPU table ---
            if (!cuda_devices.empty())
            {
                log_gpu_table("NVIDIA CUDA GPUs", cuda_devices);
            }

            // --- AMD ROCm GPU table ---
            if (!rocm_devices.empty())
            {
                log_gpu_table("AMD ROCm GPUs", rocm_devices);
            }

            // --- P2P access matrices ---
            for (const auto &matrix : p2p_matrices_)
            {
                if (matrix.backend == ComputeBackendType::GPU_CUDA)
                    log_p2p_table("CUDA", matrix);
                else if (matrix.backend == ComputeBackendType::GPU_ROCM)
                    log_p2p_table("ROCm", matrix);
            }

        } // end if (!inventory_logged_)

        // Note: All devices are available for heterogeneous work distribution.
        // CPU may get 0% prefill but significant decode work due to memory bandwidth.
        // GPU kernels are under development; CPU backend is currently primary.
    }

    std::shared_ptr<ComputeContext> DeviceManager::create_context(size_t device_index)
    {
        if (device_index >= devices_.size())
        {
            LOG_ERROR("[DeviceManager] Invalid device index: " << device_index << "");
            return nullptr;
        }

        // Check if context already exists
        if (device_index < contexts_.size() && contexts_[device_index])
        {
            return contexts_[device_index]; // Reuse existing context
        }

        // Ensure contexts_ vector is large enough
        if (device_index >= contexts_.size())
        {
            contexts_.resize(device_index + 1, nullptr);
        }

        // Create concrete context based on backend type
        std::shared_ptr<ComputeContext> ctx;
        const auto &device = devices_[device_index];

        switch (device.type)
        {
        case ComputeBackendType::CPU:
            ctx = std::make_shared<CPUComputeContext>();
            break;

        // GPU context creation is owned by IBackend.
        case ComputeBackendType::GPU_CUDA:
        case ComputeBackendType::GPU_ROCM:
        case ComputeBackendType::GPU_VULKAN:
            LOG_ERROR("[DeviceManager] GPU context creation moved to IBackend (Phase 3)");
            return nullptr;

        default:
            LOG_ERROR("[DeviceManager] Unknown backend type");
            return nullptr;
        }

        // Cache context
        contexts_[device_index] = ctx;

        LOG_DEBUG("[DeviceManager] Created context for device " << device_index
                                                                << " (" << backend_type_name(device.type) << ")");

        return ctx;
    }

    int DeviceManager::find_device(ComputeBackendType type, int device_id) const
    {
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == type && devices_[i].device_id == device_id)
            {
                return static_cast<int>(i);
            }
        }
        return -1; // Not found
    }

    bool DeviceManager::deviceExists(const DeviceId &device) const
    {
        if (!device.is_valid())
        {
            return false;
        }

        if (device.is_cpu())
        {
            // CPU is always available
            return true;
        }

        // Map DeviceId type to ComputeBackendType
        ComputeBackendType backend_type;
        switch (device.type)
        {
        case DeviceType::CUDA:
            backend_type = ComputeBackendType::GPU_CUDA;
            break;
        case DeviceType::ROCm:
            backend_type = ComputeBackendType::GPU_ROCM;
            break;
        default:
            return false;
        }

        return find_device(backend_type, device.ordinal) >= 0;
    }

    std::optional<PeerAccessCoverage> DeviceManager::peerAccessCoverage(
        const std::vector<DeviceId> &devices) const
    {
        if (devices.size() < 2u || !devices.front().is_gpu())
            return std::nullopt;

        const DeviceType backend_type = devices.front().type;
        std::vector<int> ordinals;
        ordinals.reserve(devices.size());
        for (const DeviceId &device : devices)
        {
            if (!device.is_gpu() || device.type != backend_type)
                return std::nullopt;
            ordinals.push_back(device.ordinal);
        }

        const ComputeBackendType matrix_backend =
            backend_type == DeviceType::CUDA
                ? ComputeBackendType::GPU_CUDA
                : ComputeBackendType::GPU_ROCM;
        for (const auto &matrix : p2p_matrices_)
        {
            if (matrix.backend == matrix_backend)
                return matrix.coverageForDevices(ordinals);
        }
        return std::nullopt;
    }

    std::optional<bool> DeviceManager::peerAccessAvailable(
        DeviceId accessor,
        DeviceId peer) const
    {
        if (!accessor.is_gpu() || !peer.is_gpu() || accessor == peer ||
            accessor.type != peer.type)
        {
            return std::nullopt;
        }

        const ComputeBackendType matrix_backend =
            accessor.type == DeviceType::CUDA
                ? ComputeBackendType::GPU_CUDA
                : ComputeBackendType::GPU_ROCM;
        for (const auto &matrix : p2p_matrices_)
        {
            if (matrix.backend != matrix_backend)
                continue;
            if (!matrix.indexForDevice(accessor.ordinal).has_value() ||
                !matrix.indexForDevice(peer.ordinal).has_value())
            {
                return std::nullopt;
            }
            return matrix.canAccessDevice(accessor.ordinal, peer.ordinal);
        }
        return std::nullopt;
    }

    bool DeviceManager::deviceExists(const GlobalDeviceAddress &device, bool strict_numa) const
    {
        if (device.isCPU())
        {
            if (!strict_numa)
            {
                return true;
            }

            if (local_numa_node_ >= 0)
            {
                return device.numa_node == local_numa_node_;
            }

            const int total_numa = NUMATopology::getNumNUMANodes();
            return device.numa_node >= 0 && device.numa_node < std::max(1, total_numa);
        }

        ComputeBackendType backend_type;
        switch (device.device_type)
        {
        case DeviceType::CUDA:
            backend_type = ComputeBackendType::GPU_CUDA;
            break;
        case DeviceType::ROCm:
            backend_type = ComputeBackendType::GPU_ROCM;
            break;
        default:
            return false;
        }

        for (const auto &dev : devices_)
        {
            if (dev.type != backend_type || dev.device_id != device.device_ordinal)
            {
                continue;
            }

            if (!strict_numa)
            {
                return true;
            }

            if (dev.numa_node < 0 || dev.numa_node == device.numa_node)
            {
                return true;
            }
        }

        return false;
    }

    std::string DeviceManager::availableDevicesString() const
    {
        std::string result;
        for (const auto &dev : devices_)
        {
            if (!result.empty())
            {
                result += ", ";
            }
            switch (dev.type)
            {
            case ComputeBackendType::CPU:
                result += "CPU";
                break;
            case ComputeBackendType::GPU_CUDA:
                result += "CUDA:" + std::to_string(dev.device_id);
                if (!dev.name.empty())
                {
                    result += " (" + dev.name + ")";
                }
                break;
            case ComputeBackendType::GPU_ROCM:
                result += "ROCm:" + std::to_string(dev.device_id);
                if (!dev.name.empty())
                {
                    result += " (" + dev.name + ")";
                }
                break;
            default:
                result += "Unknown:" + std::to_string(dev.device_id);
                break;
            }
        }
        return result.empty() ? "(none)" : result;
    }

    size_t DeviceManager::select_device(size_t estimated_memory_bytes)
    {
        // NOTE: This method selects a PRIMARY device for legacy single-device code paths.
        // For heterogeneous tensor-parallel execution, use all devices via devices() and
        // let the work distributor allocate work based on device capabilities.
        //
        // Current behavior: Returns CPU (index 0) since GPU kernels are not yet implemented.
        // Future behavior: Will be deprecated in favor of multi-device orchestration.

        if (devices_.empty())
        {
            LOG_ERROR("[DeviceManager] No devices available");
            return 0;
        }

        // For now, always use CPU backend since GPU kernels are under development
        // All devices remain available for future heterogeneous work distribution
        LOG_DEBUG("[DeviceManager] Using CPU backend (GPU kernels under development)");
        return 0; // CPU is always device 0
    }

    bool DeviceManager::has_gpu() const
    {
        for (const auto &dev : devices_)
        {
            if (dev.type == ComputeBackendType::GPU_CUDA ||
                dev.type == ComputeBackendType::GPU_ROCM ||
                dev.type == ComputeBackendType::GPU_VULKAN)
            {
                return true;
            }
        }
        return false;
    }

    int DeviceManager::cuda_device_count() const
    {
        int count = 0;
        for (const auto &dev : devices_)
        {
            if (dev.type == ComputeBackendType::GPU_CUDA)
            {
                count++;
            }
        }
        return count;
    }

    int DeviceManager::rocm_device_count() const
    {
        int count = 0;
        for (const auto &dev : devices_)
        {
            if (dev.type == ComputeBackendType::GPU_ROCM)
            {
                count++;
            }
        }
        return count;
    }

    std::vector<size_t> DeviceManager::get_devices_by_type(ComputeBackendType type) const
    {
        std::vector<size_t> result;
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == type)
            {
                result.push_back(i);
            }
        }
        return result;
    }

    int DeviceManager::get_device_id_for_type(ComputeBackendType type, int local_index) const
    {
        int count = 0;
        for (const auto &dev : devices_)
        {
            if (dev.type == type)
            {
                if (count == local_index)
                {
                    return dev.device_id;
                }
                count++;
            }
        }
        return -1; // Not found
    }

    // ============================================================================
    // CPUComputeContext Implementation
    // ============================================================================

    struct CPUComputeContext::Impl
    {
        // Note: The typed kernels don't implement ITensorRoPE/ITensorSwiGLU interfaces,
        // so we use void* and cast when needed. This is a transitional pattern
        // that will be cleaned up when the compute context is refactored.
        std::unique_ptr<CPURoPEKernelT<ActivationPrecision::FP32>> rope_kernel;
        std::unique_ptr<CPUSoftmaxKernelT<ActivationPrecision::FP32>> softmax_kernel;
        std::unique_ptr<CPUSwiGLUKernelT<ActivationPrecision::FP32>> swiglu_kernel;
    };

    CPUComputeContext::CPUComputeContext()
        : pimpl_(std::make_unique<Impl>())
    {
    }

    CPUComputeContext::~CPUComputeContext() = default;

    void *CPUComputeContext::allocate(size_t bytes)
    {
        return std::malloc(bytes);
    }

    void CPUComputeContext::free(void *ptr)
    {
        std::free(ptr);
    }

    void CPUComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        std::memcpy(dst, src, bytes); // CPU-to-CPU copy
    }

    void CPUComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        std::memcpy(dst, src, bytes); // CPU-to-CPU copy
    }

    ITensorRoPE *CPUComputeContext::get_rope_kernel()
    {
        // Note: The typed kernels no longer implement ITensorRoPE interface.
        // This method is deprecated and returns nullptr.
        // Use KernelFactory::createRoPE() instead.
        return nullptr;
    }

    ITensorSoftmax *CPUComputeContext::get_softmax_kernel()
    {
        // Note: The typed kernels no longer implement ITensorSoftmax interface.
        // This method is deprecated and returns nullptr.
        // Use KernelFactory::createSoftmax() instead.
        return nullptr;
    }

    ITensorSwiGLU *CPUComputeContext::get_swiglu_kernel()
    {
        // Note: The typed kernels no longer implement ITensorSwiGLU interface.
        // This method is deprecated and returns nullptr.
        // Use KernelFactory::createSwiGLU() instead.
        return nullptr;
    }

} // namespace llaminar2
