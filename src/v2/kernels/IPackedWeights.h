/**
 * @file IPackedWeights.h
 * @brief Abstract interface for device-packed weight data
 *
 * Decouples packed weight storage from GEMM compute engines, enabling:
 * - Cross-NUMA weight transfer (clone to destination socket)
 * - Cross-device weight conversion (CPU ↔ GPU, phase 2)
 * - Memory lifecycle management (release departed expert weights)
 * - MPI serialization for inter-rank weight transfer (phase 2)
 *
 * Each concrete implementation wraps a device-specific packed format:
 * - CPU: CPUNativeVNNIPackedWeights (VNNI-interleaved layout)
 * - CUDA: CUDAPackedWeights (INT8 column-major, phase 2)
 * - ROCm: ROCmPackedWeights (INT8 column-major, phase 2)
 */

#pragma once

#include "../backends/DeviceId.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Identifies the packed weight format for compatibility checking.
     *
     * Same-format transfers are a straight memcpy (e.g., CPU_NATIVE_VNNI → CPU_NATIVE_VNNI
     * across NUMA nodes). Cross-format transfers require conversion (phase 2).
     */
    enum class PackedWeightsFormat : uint8_t
    {
        CPU_NATIVE_VNNI,  ///< CPUNativeVNNIPackedWeights (VNNI-interleaved + native payload)
        CUDA_INT8,        ///< CUDAPackedWeights (INT8 column-major on device)
        ROCM_INT8,        ///< ROCmPackedWeights (INT8 on device)
        UNKNOWN
    };

    /**
     * @brief Abstract interface for transferable packed weight data.
     *
     * Packed weights are the device-optimized representation of model weights
     * (e.g., VNNI-interleaved for CPU AVX-512, INT8 column-major for CUDA).
     * This interface enables transferring packed weights between devices
     * without re-packing from the original quantized tensor.
     *
     * ## Transfer Matrix (phase 1 = implemented, phase 2 = stubbed)
     *
     * | Source → Dest     | Same Format? | Status  |
     * |-------------------|-------------|---------|
     * | CPU → CPU         | Yes         | Phase 1 |
     * | CUDA → CUDA       | Yes         | Phase 2 |
     * | ROCm → ROCm       | Yes         | Phase 2 |
     * | CPU → CUDA         | No          | Phase 2 |
     * | CUDA → CPU         | No          | Phase 2 |
     * | CPU → ROCm         | No          | Phase 2 |
     * | ROCm → CPU         | No          | Phase 2 |
     */
    class IPackedWeights
    {
    public:
        virtual ~IPackedWeights() = default;

        /// The packed weight format (for compatibility checking).
        virtual PackedWeightsFormat format() const = 0;

        /// The device type these weights are packed for.
        virtual DeviceType deviceType() const = 0;

        /// Output dimension (rows) of the weight matrix.
        virtual int N() const = 0;

        /// Input dimension (columns) of the weight matrix.
        virtual int K() const = 0;

        /// Total size of packed weight data in bytes.
        /// For memory accounting and transfer budgeting.
        virtual size_t sizeBytes() const = 0;

        /**
         * @brief Deep-copy packed weights with NUMA-local allocation.
         *
         * The clone allocates new buffers on the calling thread's NUMA node
         * (via first-touch policy). This is the primary mechanism for
         * cross-socket weight migration on multi-socket CPU systems.
         *
         * @return New IPackedWeights with identical data on local NUMA node,
         *         or nullptr on failure.
         */
        virtual std::unique_ptr<IPackedWeights> clone() const = 0;

        /**
         * @brief Convert packed weights to a different device format.
         *
         * For same-format transfers (e.g., CPU→CPU), use clone() instead.
         * This is for cross-device-type transfers that require format conversion.
         *
         * @param target Target device type
         * @return Converted weights, or nullptr if conversion not supported.
         *
         * @note Phase 2: All cross-device conversions return nullptr for now.
         */
        virtual std::unique_ptr<IPackedWeights> convertTo(DeviceType target) const
        {
            (void)target;
            return nullptr; // Phase 2: implement CPU↔CUDA, CPU↔ROCm conversions
        }

        /**
         * @brief Check if these weights are compatible with a target device type.
         *
         * Same device type → always compatible (straight copy via clone()).
         * Cross device type → requires conversion (phase 2).
         */
        bool isCompatibleWith(DeviceType target) const
        {
            return deviceType() == target;
        }

        /**
         * @brief Check if these weights can be transferred to a target format.
         *
         * Same format → clone() works.
         * Different format → convertTo() needed (may not be implemented yet).
         */
        bool canTransferTo(PackedWeightsFormat target_format) const
        {
            return format() == target_format;
        }

        /**
         * @brief Serialize packed weights to a transferable byte buffer.
         *
         * The buffer contains a self-describing header + data sections that
         * can be deserialized on any rank without the original tensor.
         * Default returns empty (not supported for this format).
         */
        virtual std::vector<uint8_t> serialize() const { return {}; }
    };

} // namespace llaminar2
