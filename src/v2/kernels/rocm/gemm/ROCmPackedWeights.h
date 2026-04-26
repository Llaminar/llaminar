/**
 * @file ROCmPackedWeights.h
 * @brief Stub IPackedWeights implementation for ROCm INT8 format.
 *
 * Phase 4 stub: Compilable but non-functional. Ensures the IPackedWeights
 * interface generalizes beyond CPU before GPU MoE kernels are written.
 * convertTo() and serialize() return nullptr/empty with LOG_WARN.
 */

#pragma once

#include "kernels/IPackedWeights.h"
#include "utils/Logger.h"

#include <vector>

namespace llaminar2::rocm
{

    /**
     * @brief IPackedWeights stub for ROCm INT8 packed weights.
     *
     * Owns a host-side byte buffer as a placeholder for device memory.
     * Real implementation (phase 5) will use hipMalloc/hipFree and
     * proper device↔host transfers.
     */
    class ROCmPackedWeights : public IPackedWeights
    {
    public:
        ROCmPackedWeights(int N, int K, std::vector<uint8_t> device_data)
            : N_(N), K_(K), device_data_(std::move(device_data)) {}

        ~ROCmPackedWeights() override = default;

        // ── IPackedWeights interface ─────────────────────

        PackedWeightsFormat format() const override { return PackedWeightsFormat::ROCM_INT8; }
        DeviceType deviceType() const override { return DeviceType::ROCm; }
        int N() const override { return N_; }
        int K() const override { return K_; }
        size_t sizeBytes() const override { return device_data_.size(); }

        std::unique_ptr<IPackedWeights> clone() const override
        {
            return std::make_unique<ROCmPackedWeights>(N_, K_, device_data_);
        }

        std::unique_ptr<IPackedWeights> convertTo(DeviceType target) const override
        {
            if (target == DeviceType::ROCm)
                return clone();

            LOG_WARN("[ROCmPackedWeights] Cross-device conversion ROCm -> "
                     << static_cast<int>(target)
                     << " not implemented (phase 5)");
            return nullptr;
        }

        std::vector<uint8_t> serialize() const override
        {
            LOG_WARN("[ROCmPackedWeights] Serialization not implemented (phase 5)");
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

} // namespace llaminar2::rocm
