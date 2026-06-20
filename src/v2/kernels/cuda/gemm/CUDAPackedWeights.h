/**
 * @file CUDAPackedWeights.h
 * @brief Stub IPackedWeights implementation for CUDA INT8 format.
 *
 * Phase 4 stub: Compilable but non-functional. Ensures the IPackedWeights
 * interface generalizes beyond CPU before GPU MoE kernels are written.
 * convertTo() and serialize() return nullptr/empty with LOG_WARN.
 */

#pragma once

#include "kernels/IPackedWeights.h"
#include "utils/Logger.h"

#include <vector>

namespace llaminar2::cuda
{

    /**
     * @brief IPackedWeights stub for CUDA INT8 packed weights.
     *
     * Owns a host-side byte buffer as a placeholder for device memory.
     * Real implementation (phase 5) will use cudaMalloc/cudaFree and
     * proper device↔host transfers.
     */
    class CUDAPackedWeights : public IPackedWeights
    {
    public:
        CUDAPackedWeights(int N, int K, std::vector<uint8_t> device_data)
            : N_(N), K_(K), device_data_(std::move(device_data)) {}

        ~CUDAPackedWeights() override = default;

        // ── IPackedWeights interface ─────────────────────

        PackedWeightsFormat format() const override { return PackedWeightsFormat::CUDA_INT8; }
        DeviceType deviceType() const override { return DeviceType::CUDA; }
        int N() const override { return N_; }
        int K() const override { return K_; }
        size_t sizeBytes() const override { return device_data_.size(); }

        std::unique_ptr<IPackedWeights> clone() const override
        {
            return std::make_unique<CUDAPackedWeights>(N_, K_, device_data_);
        }

        std::unique_ptr<IPackedWeights> convertTo(DeviceType target) const override
        {
            if (target == DeviceType::CUDA)
                return clone();

            LOG_WARN("[CUDAPackedWeights] Cross-device conversion CUDA -> "
                     << static_cast<int>(target)
                     << " not implemented (phase 5)");
            return nullptr;
        }

        std::vector<uint8_t> serialize() const override
        {
            LOG_WARN("[CUDAPackedWeights] Serialization not implemented (phase 5)");
            return {};
        }

        // ── Accessors ────────────────────────────────────

        /// Access the placeholder device data (read-only).
        const std::vector<uint8_t> &deviceData() const { return device_data_; }

    private:
        int N_;
        int K_;
        std::vector<uint8_t> device_data_; ///< Placeholder — real impl uses device memory
    };

} // namespace llaminar2::cuda
