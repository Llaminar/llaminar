/**
 * @file TPDomainConfig.h
 * @brief Configuration for a Tensor Parallel domain (Phase 1.1)
 *
 * A TP domain is a group of devices that work together on tensor-parallel
 * sharded operations. Within a domain, devices use a collective backend
 * (NCCL, RCCL, HOST, etc.) for allreduce operations.
 *
 * This is a configuration-only structure (no MPI handles). It serves as
 * the input for creating ILocalTPContext instances at runtime.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "backends/DeviceId.h"
#include "config/OrchestrationConfig.h" // For CollectiveBackendType
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    // Forward declaration
    class ILocalTPContext;

    /**
     * @brief Configuration for a Tensor Parallel domain
     *
     * A TP domain is a group of devices that work together on sharded operations.
     * Within a domain, devices use a collective backend (NCCL, RCCL, HOST, etc.)
     * for allreduce operations.
     *
     * This is a pure configuration structure - it holds the specification for a
     * TP domain but does not own any runtime resources (MPI communicators, etc.).
     * Use createTPContext() to instantiate a runtime context.
     *
     * Example configurations:
     * - 2-way CUDA TP: devices=[cuda:0, cuda:1], backend=NCCL
     * - Mixed GPU TP: devices=[cuda:0, rocm:0], backend=HOST
     * - Heterogeneous TP: devices=[cuda:0, rocm:0], weights=[0.73, 0.27]
     */
    struct TPDomainConfig
    {
        /// Domain name (e.g., "gpu_tp_nvidia", "cpu_tp", "mixed_gpu_tp")
        std::string name;

        /// Devices in this domain
        std::vector<DeviceId> devices;

        /// Collective backend for TP allreduce within this domain
        CollectiveBackendType tp_backend = CollectiveBackendType::AUTO;

        /// Optional proportional weights for heterogeneous TP (e.g., [0.6, 0.4] for 3090+MI50)
        /// Empty = equal distribution
        std::vector<float> device_weights;

        // =========================================================================
        // Accessors
        // =========================================================================

        /**
         * @brief Get the primary device (first device in the domain)
         * @return First device if not empty, DeviceId::cpu() otherwise
         */
        DeviceId primaryDevice() const;

        /**
         * @brief Get the TP degree (number of devices)
         * @return Number of devices in this domain
         */
        int degree() const;

        /**
         * @brief Check if all devices are the same type (homogeneous)
         * @return true if all devices have the same DeviceType
         */
        bool isHomogeneous() const;

        /**
         * @brief Check if this is a GPU domain (all devices are GPUs)
         * @return true if all devices are CUDA or ROCm GPUs
         */
        bool isGPUDomain() const;

        /**
         * @brief Check if this is a CPU domain (all devices are CPUs)
         * @return true if all devices are CPUs
         */
        bool isCPUDomain() const;

        /**
         * @brief Check if this is a cross-vendor domain (mixed CUDA/ROCm)
         * @return true if domain has both CUDA and ROCm devices
         */
        bool isCrossVendor() const;

        // =========================================================================
        // Validation
        // =========================================================================

        /**
         * @brief Validate the configuration
         * @param error_msg Output: error message if invalid (nullptr to skip)
         * @return true if valid
         *
         * Checks:
         * - Name is not empty
         * - At least one device
         * - Weights (if provided) match device count
         * - Weights (if provided) sum to approximately 1.0 (tolerance: 0.01)
         * - Backend is valid for device types (e.g., NCCL only for CUDA-only)
         */
        bool validate(std::string *error_msg = nullptr) const;

        // =========================================================================
        // Factory
        // =========================================================================

        /**
         * @brief Create an ILocalTPContext for this domain
         *
         * Creates a runtime TP context from this configuration. The context
         * owns MPI communicators, stream handles, and other runtime resources.
         *
         * @return Unique pointer to context, or nullptr if creation fails
         *
         * @note Currently returns nullptr (TODO: wire up to actual implementation)
         */
        std::unique_ptr<ILocalTPContext> createTPContext() const;
    };

} // namespace llaminar2
