/**
 * @file WorkspaceDescriptor.h
 * @brief Describes device workspace buffer requirements for centralized allocation
 *
 * This header defines the structures used to declare workspace buffer
 * requirements. Kernels return WorkspaceRequirements from their
 * getWorkspaceRequirements() method.
 *
 * Works with any device type: CPU, CUDA, ROCm.
 * (Formerly GpuWorkspaceDescriptor.h)
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace llaminar2
{

    /**
     * @brief Describes a single device workspace buffer requirement
     *
     * Each buffer has a unique name, size, alignment requirement, and
     * a flag indicating whether it's required or optional.
     *
     * (Formerly GpuWorkspaceDescriptor)
     */
    struct WorkspaceDescriptor
    {
        std::string name;       ///< Unique identifier (e.g., "gemm_quant_a")
        size_t size_bytes = 0;  ///< Required size in bytes
        size_t alignment = 256; ///< Alignment requirement (default 256 for device)
        bool required = true;   ///< If false, allocation failure is not fatal

        // Default constructor
        WorkspaceDescriptor() = default;

        // Convenience constructor
        WorkspaceDescriptor(
            const std::string &name_,
            size_t size_,
            size_t align_ = 256,
            bool req_ = true)
            : name(name_), size_bytes(size_), alignment(align_), required(req_)
        {
        }

        // Convenience constructor with const char*
        WorkspaceDescriptor(
            const char *name_,
            size_t size_,
            size_t align_ = 256,
            bool req_ = true)
            : name(name_), size_bytes(size_), alignment(align_), required(req_)
        {
        }
    };

    /**
     * @brief Describes complete workspace requirements for a kernel
     *
     * A kernel's getWorkspaceRequirements() returns this structure containing
     * all buffers needed for execution at specific dimensions.
     *
     * (Formerly GpuWorkspaceRequirements)
     */
    struct WorkspaceRequirements
    {
        std::vector<WorkspaceDescriptor> buffers;

        /**
         * @brief Calculate total bytes needed for all buffers
         *
         * Note: Does not account for alignment padding between buffers.
         * Actual allocation may require more memory due to alignment.
         *
         * @return Sum of all buffer size_bytes
         */
        size_t total_bytes() const
        {
            size_t total = 0;
            for (const auto &buf : buffers)
            {
                total += buf.size_bytes;
            }
            return total;
        }

        /**
         * @brief Calculate total bytes including worst-case alignment padding
         *
         * Assumes each buffer needs up to (alignment-1) padding bytes.
         *
         * @return Upper bound on memory needed including alignment
         */
        size_t total_bytes_with_alignment() const
        {
            size_t total = 0;
            for (const auto &buf : buffers)
            {
                // Round up to alignment boundary
                size_t aligned_size = (buf.size_bytes + buf.alignment - 1) & ~(buf.alignment - 1);
                total += aligned_size;
            }
            return total;
        }

        /**
         * @brief Check if any buffers are required
         */
        bool has_required_buffers() const
        {
            for (const auto &buf : buffers)
            {
                if (buf.required)
                    return true;
            }
            return false;
        }

        /**
         * @brief Get buffer descriptor by name
         *
         * @param name Buffer name to find
         * @return Pointer to descriptor, or nullptr if not found
         */
        const WorkspaceDescriptor *find(const std::string &name) const
        {
            for (const auto &buf : buffers)
            {
                if (buf.name == name)
                    return &buf;
            }
            return nullptr;
        }

        // =========================================================================
        // Factory methods for common requirement patterns
        // =========================================================================

        /**
         * @brief Create requirements for standard INT8 quantized GEMM
         *
         * Includes:
         * - quant_a: INT8 quantized activations [m × k]
         * - scales_a: FP32 per-row scales [m]
         * - acc_int32: INT32 accumulator [m × n]
         *
         * @param m Maximum sequence length
         * @param n Output features
         * @param k Input features
         * @return Requirements for INT8 GEMM
         */
        static WorkspaceRequirements forQuantizedGemm(int m, int n, int k)
        {
            WorkspaceRequirements req;
            req.buffers.push_back({"gemm_quant_a",
                                   static_cast<size_t>(m) * k * sizeof(int8_t), 256, true});
            req.buffers.push_back({"gemm_scales_a",
                                   static_cast<size_t>(m) * sizeof(float), 256, true});
            req.buffers.push_back({"gemm_acc_int32",
                                   static_cast<size_t>(m) * n * sizeof(int32_t), 256, true});
            return req;
        }

        /**
         * @brief Create requirements for FP16 GEMM with full-matrix buffers
         *
         * Includes A, B, C FP16 buffers for full-matrix conversion.
         * This is the legacy/simple path with high memory usage.
         *
         * @param m Maximum sequence length
         * @param n Output features
         * @param k Input features
         * @return Requirements for full-matrix FP16 GEMM
         */
        static WorkspaceRequirements forFP16Gemm(int m, int n, int k)
        {
            WorkspaceRequirements req;
            req.buffers.push_back({"gemm_full_a_fp16",
                                   static_cast<size_t>(m) * k * sizeof(uint16_t), 256, true});
            req.buffers.push_back({"gemm_full_b_fp16",
                                   static_cast<size_t>(k) * n * sizeof(uint16_t), 256, true});
            req.buffers.push_back({"gemm_full_c_fp16",
                                   static_cast<size_t>(m) * n * sizeof(uint16_t), 256, true});
            return req;
        }

        /**
         * @brief Create requirements for slab-based FP16 GEMM
         *
         * Uses fixed-size slab buffers instead of full matrices.
         * Much lower memory usage for large matrices.
         *
         * @param slab_m Slab rows
         * @param slab_n Slab columns
         * @param slab_k Slab inner dimension
         * @return Requirements for slab-based FP16 GEMM
         */
        static WorkspaceRequirements forSlabFP16Gemm(int slab_m, int slab_n, int slab_k)
        {
            WorkspaceRequirements req;
            req.buffers.push_back({"gemm_slab_a_fp16",
                                   static_cast<size_t>(slab_m) * slab_k * sizeof(uint16_t), 256, true});
            req.buffers.push_back({"gemm_slab_b_fp16",
                                   static_cast<size_t>(slab_k) * slab_n * sizeof(uint16_t), 256, true});
            req.buffers.push_back({"gemm_slab_c_fp16",
                                   static_cast<size_t>(slab_m) * slab_n * sizeof(uint16_t), 256, true});
            return req;
        }

        /**
         * @brief Merge requirements from another set
         *
         * Adds buffers from other that don't already exist in this.
         * If a buffer with the same name exists, keeps the larger size.
         *
         * @param other Requirements to merge
         */
        void merge(const WorkspaceRequirements &other)
        {
            for (const auto &buf : other.buffers)
            {
                bool found = false;
                for (auto &existing : buffers)
                {
                    if (existing.name == buf.name)
                    {
                        // Keep larger size
                        if (buf.size_bytes > existing.size_bytes)
                        {
                            existing.size_bytes = buf.size_bytes;
                        }
                        // Keep stricter alignment
                        if (buf.alignment > existing.alignment)
                        {
                            existing.alignment = buf.alignment;
                        }
                        // Keep required if either is required
                        existing.required = existing.required || buf.required;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    buffers.push_back(buf);
                }
            }
        }
    };

} // namespace llaminar2
