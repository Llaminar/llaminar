/**
 * @file ComputeBackend.h
 * @brief Device manager and compute context interfaces (LEGACY)
 *
 * ⚠️ DEPRECATION NOTICE (Phase 3 - October 2025):
 * GPU contexts were removed and replaced with the IBackend architecture
 * (see backends/IBackend.h).
 *
 * CPU device enumeration (DeviceManager) is still functional and used by Main.cpp.
 * Full removal is deferred until V2 has a production-ready device manager.
 *
 * For GPU operations, use:
 * - backends/IBackend.h (abstract interface)
 * - backends/cuda/CUDABackend.h (CUDA implementation)
 * - backends/rocm/ROCmBackend.h (ROCm implementation)
 *
 * @author David Sanftenberg
 */

#pragma once

#include "DeviceId.h"
#include "GlobalDeviceAddress.h"
#include "PeerAccessCoverage.h"
#include <string>
#include <vector>
#include <memory>
#include <cstddef>
#include <optional>

namespace llaminar2
{
    // Forward declaration — full include only in ComputeBackend.cpp
    struct HardwareInventory;

    /**
     * @brief Compute backend types
     */
    enum class ComputeBackendType
    {
        CPU,        // CPU with custom kernels (AVX-512, VNNI, etc.)
        GPU_CUDA,   // NVIDIA GPU with CUDA
        GPU_ROCM,   // AMD GPU with ROCm
        GPU_VULKAN, // Cross-platform GPU with Vulkan
        GPU_METAL   // Apple GPU with Metal
    };

    /**
     * @brief PCIe link information for a device
     */
    struct PCIeLinkInfo
    {
        std::string pci_address;   // BDF address (e.g., "0000:8f:00.0")
        double link_speed_gts = 0; // Effective link speed in GT/s (bottleneck-aware)
        int link_width = 0;        // Effective link width (bottleneck-aware)
        double max_speed_gts = 0;  // Max (capable) link speed in GT/s
        int max_width = 0;         // Max (capable) link width
        int pcie_gen = 0;          // PCIe generation (3, 4, 5, ...)
        bool degraded = false;     // True if effective < max speed or width
        std::string bottleneck_bdf; // BDF of upstream bridge causing bottleneck (empty if none)

        double bandwidth_gbps() const // Effective bandwidth in GB/s (per direction)
        {
            // PCIe encoding: Gen1/2 = 8b/10b (80%), Gen3+ = 128b/130b (~98.46%)
            double encoding_efficiency = (pcie_gen >= 3) ? (128.0 / 130.0) : 0.8;
            return link_speed_gts * link_width * encoding_efficiency / 8.0;
        }
    };

    /**
     * @brief Driver-backed P2P access matrix for one GPU backend.
     *
     * Matrix indices are internal enumeration positions, while callers name
     * devices by backend ordinal.  The ordinal-aware helpers below prevent a
     * filtered or reordered inventory from being mistaken for a dense
     * zero-based device list.
     */
    struct P2PMatrix
    {
        ComputeBackendType backend = ComputeBackendType::CPU;
        std::vector<int> device_ids;               // Backend device IDs (ordered)
        std::vector<std::vector<bool>> can_access; // can_access[i][j] = device i can access device j's memory

        /** @return Number of backend ordinals represented by this matrix. */
        [[nodiscard]] int device_count() const noexcept
        {
            return static_cast<int>(device_ids.size());
        }

        /**
         * @brief Query one directed edge by matrix index.
         * @param from_idx Source matrix index.
         * @param to_idx Destination matrix index.
         * @return False for an invalid or structurally incomplete matrix.
         */
        [[nodiscard]] bool has_p2p(int from_idx, int to_idx) const noexcept
        {
            if (from_idx < 0 || from_idx >= device_count() || to_idx < 0 || to_idx >= device_count())
                return false;
            if (static_cast<std::size_t>(from_idx) >= can_access.size() ||
                static_cast<std::size_t>(to_idx) >=
                    can_access[static_cast<std::size_t>(from_idx)].size())
            {
                return false;
            }
            return can_access[static_cast<std::size_t>(from_idx)]
                             [static_cast<std::size_t>(to_idx)];
        }

        /**
         * @brief Resolve a backend ordinal to its matrix index.
         * @param device_id CUDA or ROCm device ordinal.
         * @return Matrix index, or empty when the ordinal was not queried.
         */
        [[nodiscard]] std::optional<int> indexForDevice(
            int device_id) const noexcept
        {
            for (std::size_t index = 0; index < device_ids.size(); ++index)
            {
                if (device_ids[index] == device_id)
                    return static_cast<int>(index);
            }
            return std::nullopt;
        }

        /**
         * @brief Query one directed peer edge by backend device ordinal.
         * @param from_device_id Source CUDA or ROCm ordinal.
         * @param to_device_id Destination CUDA or ROCm ordinal.
         * @return Driver-reported accessibility, or false for unknown devices.
         */
        [[nodiscard]] bool canAccessDevice(
            int from_device_id,
            int to_device_id) const noexcept
        {
            const auto from = indexForDevice(from_device_id);
            const auto to = indexForDevice(to_device_id);
            return from.has_value() && to.has_value() &&
                   has_p2p(*from, *to);
        }

        /**
         * @brief Classify direct peer access for an exact device domain.
         *
         * `Partial` deliberately includes an asymmetric edge.  Native
         * NCCL/RCCL remains authoritative whenever the driver exposes any P2P
         * opportunity because the collective library can select a topology
         * better than a forced host bounce.  A custom mapped transport is
         * eligible only for `None`.
         *
         * @param requested_device_ids Distinct backend ordinals in the domain.
         * @return Coverage, or empty for fewer than two devices, duplicates,
         *         missing ordinals, or a malformed matrix.
         */
        [[nodiscard]] std::optional<PeerAccessCoverage> coverageForDevices(
            const std::vector<int> &requested_device_ids) const
        {
            if (requested_device_ids.size() < 2u ||
                can_access.size() != device_ids.size())
            {
                return std::nullopt;
            }
            for (const auto &row : can_access)
            {
                if (row.size() != device_ids.size())
                    return std::nullopt;
            }

            std::vector<int> indices;
            indices.reserve(requested_device_ids.size());
            for (std::size_t requested = 0;
                 requested < requested_device_ids.size();
                 ++requested)
            {
                for (std::size_t prior = 0; prior < requested; ++prior)
                {
                    if (requested_device_ids[prior] ==
                        requested_device_ids[requested])
                    {
                        return std::nullopt;
                    }
                }
                const auto index =
                    indexForDevice(requested_device_ids[requested]);
                if (!index)
                    return std::nullopt;
                indices.push_back(*index);
            }

            bool any_directed_edge = false;
            bool every_pair_bidirectional = true;
            for (std::size_t from = 0; from < indices.size(); ++from)
            {
                for (std::size_t to = from + 1u; to < indices.size(); ++to)
                {
                    const bool forward = has_p2p(indices[from], indices[to]);
                    const bool reverse = has_p2p(indices[to], indices[from]);
                    any_directed_edge = any_directed_edge || forward || reverse;
                    every_pair_bidirectional =
                        every_pair_bidirectional && forward && reverse;
                }
            }

            if (every_pair_bidirectional)
                return PeerAccessCoverage::Complete;
            return any_directed_edge ? PeerAccessCoverage::Partial
                                     : PeerAccessCoverage::None;
        }
    };

    /**
     * @brief Device descriptor
     */
    struct ComputeDevice
    {
        ComputeBackendType type;
        int device_id;             // Backend-specific device ID (e.g., CUDA device 0, 1, ...)
        int numa_node;             // NUMA node/socket affinity (-1 if unknown)
        std::string name;          // Human-readable name
        size_t total_memory_bytes; // Total device memory
        size_t free_memory_bytes;  // Free device memory (approximate)
        int compute_units = 0;     // CUDA SMs, ROCm CUs, or backend-equivalent units
        int compute_capability;    // Backend-specific capability (e.g., CUDA compute 8.6 → 86)
        bool supports_fp16;        // Hardware FP16 support
        bool supports_bf16;        // Hardware BF16 support
        bool supports_int8;        // Hardware INT8 support
        PCIeLinkInfo pcie;         // PCIe link information (GPU devices only)
    };

    /**
     * @brief Abstract compute context interface
     *
     * The surviving implementation is CPUComputeContext. GPU execution uses IBackend.
     */
    class ComputeContext
    {
    public:
        virtual ~ComputeContext() = default;

        // Memory management
        virtual void *allocate(size_t bytes) = 0;
        virtual void free(void *ptr) = 0;

        // Data transfers
        virtual void copy_to_device(void *dst, const void *src, size_t bytes) = 0;
        virtual void copy_from_device(void *dst, const void *src, size_t bytes) = 0;

        // Synchronization
        virtual void synchronize() = 0;

        // Capabilities
        virtual ComputeBackendType backend_type() const = 0;
        virtual bool supports_bf16() const = 0;
        virtual bool supports_fp16() const = 0;
        virtual bool supports_int8() const = 0;
    };

    // Forward declarations for kernel types
    class ITensorGemm;
    class ITensorRoPE;
    class ITensorSoftmax;
    class ITensorRMSNorm;
    class ITensorSwiGLU;

    /**
     * @brief CPU compute context (OpenBLAS or Intel MKL)
     */
    class CPUComputeContext : public ComputeContext
    {
    public:
        CPUComputeContext();
        ~CPUComputeContext() override;

        void *allocate(size_t bytes) override;
        void free(void *ptr) override;
        void copy_to_device(void *dst, const void *src, size_t bytes) override;
        void copy_from_device(void *dst, const void *src, size_t bytes) override;
        void synchronize() override { /* no-op for CPU */ }

        ComputeBackendType backend_type() const override
        {
            return ComputeBackendType::CPU;
        }
        bool supports_bf16() const override { return true; } // Software emulation (or MKL native)
        bool supports_fp16() const override { return true; }
        bool supports_int8() const override { return true; }

        // Kernel access (lazily created on first access)
        ITensorRoPE *get_rope_kernel();
        ITensorSoftmax *get_softmax_kernel();
        ITensorSwiGLU *get_swiglu_kernel();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

    /**
     * @brief Device manager singleton
     *
     * Enumerates all available compute devices and manages context creation.
     *
     * NUMA-Aware Filtering (Phase 1):
     * When initialized with a specific NUMA node, only enumerates devices
     * affine to that socket. This is critical for MPI multi-socket execution
     * to avoid cross-socket performance penalties (40-60% slower).
     */
    class DeviceManager
    {
    public:
        static DeviceManager &instance()
        {
            static DeviceManager instance;
            return instance;
        }

        /**
         * @brief Initialize device manager with optional NUMA filtering
         *
         * Call once at startup. Scans for:
         * - CPU (OpenBLAS/MKL) - always on local_numa_node
         * - CUDA devices (cudaGetDeviceCount) - filtered by NUMA affinity
         * - ROCm devices (hipGetDeviceCount) - filtered by NUMA affinity
         * - Vulkan devices (vkEnumeratePhysicalDevices) - not filtered (unknown affinity)
         *
         * @param local_numa_node NUMA node for this process/rank (-1 = enumerate all devices)
         *
         * Usage:
         *   // MPI rank bound to socket 0
         *   dm.initialize(0);  // Only sees GPUs on socket 0
         *
         *   // Testing or single-socket system
         *   dm.initialize(-1);  // Sees all devices
         *
         * @param log_inventory If true, print device inventory tables (CPU, GPU, P2P)
         */
        void initialize(int local_numa_node = -1, bool log_inventory = true);

        /**
         * @brief Get all enumerated devices
         */
        const std::vector<ComputeDevice> &devices() const { return devices_; }

        /**
         * @brief Get local NUMA node (socket) this manager is filtering for
         *
         * @return NUMA node or -1 if no filtering active
         */
        int local_numa_node() const { return local_numa_node_; }

        /**
         * @brief Create context for specific device
         *
         * @param device_index Index into devices() vector
         * @return Cached context (created on first call, reused thereafter)
         */
        std::shared_ptr<ComputeContext> create_context(size_t device_index);

        /**
         * @brief Find device by backend type and device ID
         *
         * @param type Backend type (e.g., GPU_CUDA)
         * @param device_id Backend-specific device ID (default: 0)
         * @return Device index in devices() vector, or -1 if not found
         */
        int find_device(ComputeBackendType type, int device_id = 0) const;

        /**
         * @brief Get the default CPU device index
         *
         * CPU is always enumerated first (index 0) after initialization.
         * Use this instead of hardcoding 0 or using magic values like -1.
         *
         * @return Index of the CPU device (always 0 after initialization)
         */
        int cpuDeviceIndex() const { return 0; }

        /**
         * @brief Validate a device index
         *
         * @param device_idx Device index to validate
         * @return true if valid (>= 0 and < devices().size())
         */
        bool isValidDeviceIndex(int device_idx) const
        {
            return device_idx >= 0 && static_cast<size_t>(device_idx) < devices_.size();
        }

        /**
         * @brief Auto-select primary device (DEPRECATED - transitional)
         *
         * NOTE: This method is for legacy single-device code paths and will be
         * deprecated in favor of heterogeneous multi-device orchestration.
         *
         * Current behavior: Always returns CPU (index 0) since GPU kernels
         * are not yet implemented. All devices remain enumerated and available
         * for future heterogeneous tensor-parallel execution.
         *
         * Future direction: Work distribution will use all devices based on
         * their capabilities. CPUs may get 0% prefill but significant decode
         * work due to memory bandwidth characteristics.
         *
         * @param estimated_memory_bytes Ignored (kept for API compatibility)
         * @return Device index (currently always 0 = CPU)
         */
        size_t select_device(size_t estimated_memory_bytes = 0);

        /**
         * @brief Check if any GPU is available
         */
        bool has_gpu() const;

        /**
         * @brief Get total number of enumerated devices
         */
        size_t device_count() const { return devices_.size(); }

        /**
         * @brief Get count of CUDA devices
         */
        int cuda_device_count() const;

        /**
         * @brief Get count of ROCm devices
         */
        int rocm_device_count() const;

        /**
         * @brief Get all devices of specific type
         *
         * @param type Backend type
         * @return Vector of device indices
         */
        std::vector<size_t> get_devices_by_type(ComputeBackendType type) const;

        /**
         * @brief Get the backend-specific device ID for the N-th device of a given type
         *
         * This is useful for NUMA-aware device selection. When user specifies "rocm:0",
         * this method returns the actual HIP device ordinal of the first NUMA-local ROCm
         * device, which may differ from 0 if the rank is on NUMA node 1.
         *
         * @param type Backend type (e.g., GPU_ROCM)
         * @param local_index Index into the filtered list of devices of this type (0-based)
         * @return The backend-specific device_id, or -1 if index out of range
         */
        int get_device_id_for_type(ComputeBackendType type, int local_index) const;

        /**
         * @brief Check if a device with the given DeviceId actually exists
         *
         * Validates that the hardware device is enumerated and available.
         * Use this to fail early with a clear error message instead of
         * cascading failures deep in the stack.
         *
         * @param device DeviceId to validate
         * @return true if a matching device was found in the inventory
         */
        bool deviceExists(const DeviceId &device) const;

        /**
         * @brief Check if a global device address exists in current inventory
         *
         * @param device Global device address to validate
         * @param strict_numa When true, require matching NUMA node as well as type+ordinal
         * @return true if matching device exists
         */
        bool deviceExists(const GlobalDeviceAddress &device, bool strict_numa) const;

        /**
         * @brief Classify driver-reported P2P coverage for a GPU device cell.
         *
         * The matrix is populated during every `initialize()` call regardless
         * of whether inventory tables are printed.  Mixed CUDA/ROCm cells have
         * no native collective P2P domain and therefore return empty; callers
         * must handle heterogeneous transport explicitly rather than treating
         * missing topology as a no-P2P measurement.
         *
         * @param devices Distinct process-local devices in one homogeneous cell.
         * @return Exact coverage, or empty for invalid, mixed, unknown, or
         *         incompletely enumerated cells.
         */
        [[nodiscard]] std::optional<PeerAccessCoverage>
        peerAccessCoverage(const std::vector<DeviceId> &devices) const;

        /**
         * @brief Query one exact driver-reported directed GPU peer edge.
         *
         * Direction matters on asymmetric PCIe/IOMMU topologies.  `accessor`
         * names the device whose runtime stream will read memory owned by
         * `peer`; for a destination-owned peer copy this is therefore
         * `peerAccessAvailable(destination, source)`.  Returning an optional
         * keeps an invalid or mixed-backend query distinct from a measured
         * same-backend edge whose answer is false.
         *
         * @param accessor GPU that would directly address peer memory.
         * @param peer Distinct same-backend GPU owning that memory.
         * @return Driver capability, or empty for invalid/mixed/unknown input.
         */
        [[nodiscard]] std::optional<bool> peerAccessAvailable(
            DeviceId accessor,
            DeviceId peer) const;

        /**
         * @brief Get a formatted string listing all available devices
         *
         * Useful for error messages when a requested device doesn't exist.
         * Format: "CPU, CUDA:0 (NVIDIA A100), ROCm:0 (AMD MI300X)"
         *
         * @return Comma-separated list of available devices
         */
        std::string availableDevicesString() const;

        /**
         * @brief Get the hardware inventory detected at startup
         *
         * Contains the complete, unfiltered view of all hardware:
         * CPU sockets (model, cores, HT, memory), GPU devices, P2P matrices.
         * Detected once on first initialize() call and cached.
         *
         * @return Reference to the hardware inventory
         */
        const HardwareInventory *hardware() const { return hardware_.get(); }

    private:
        DeviceManager();
        ~DeviceManager();

        std::vector<ComputeDevice> devices_;
        std::vector<std::shared_ptr<ComputeContext>> contexts_; // Cached per device
        std::vector<P2PMatrix> p2p_matrices_;                   // P2P access matrices (one per GPU backend)
        std::unique_ptr<HardwareInventory> hardware_;           // Complete hardware inventory (detected once)
        bool hardware_detected_ = false;                        // Whether hardware_ has been populated        size_t last_selected_device_ = 0;                       // Round-robin state
        int local_numa_node_ = -1;                              // NUMA node filter (-1 = no filter)
        bool inventory_logged_ = false;                         // Only print device tables once
    };

} // namespace llaminar2
