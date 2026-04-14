/**
 * @file CollectiveBackendType.h
 * @brief Backend type enumeration for collective operations
 *
 * Extracted to separate header to avoid circular dependencies with
 * ParallelismTree.h and OrchestrationConfig.h.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include <optional>
#include <string>

namespace llaminar2
{

    /**
     * @brief Backend type for collective operations
     */
    enum class CollectiveBackendType
    {
        AUTO,          ///< Automatically select based on device types
        NCCL,          ///< NVIDIA NCCL (CUDA only)
        RCCL,          ///< AMD RCCL (ROCm only)
        HETEROGENEOUS, ///< Multi-GPU heterogeneous (orchestrates NCCL+RCCL+HOST for mixed vendor)
        UPI,           ///< Intel UPI interconnect (CPU cross-socket)
        MPI,           ///< Fallback MPI_Allreduce
        HOST           ///< Host-staged (copy to CPU, reduce, copy back)
    };

    /**
     * @brief Convert CollectiveBackendType to string
     */
    const char *collectiveBackendTypeToString(CollectiveBackendType type);

    /**
     * @brief Parse CollectiveBackendType from string
     * @param str String representation (case-insensitive)
     * @return Parsed type or nullopt if invalid
     */
    std::optional<CollectiveBackendType> parseCollectiveBackendType(const std::string &str);

} // namespace llaminar2
