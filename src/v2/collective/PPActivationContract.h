/**
 * @file PPActivationContract.h
 * @brief Formal specification for inter-stage activation transfers in PP
 *
 * PPActivationContract codifies the shape, dtype, and transfer method for
 * hidden state tensors passed between PP stages. This replaces informal
 * inline calculations like `seq_len * model_ctx_->embeddingLength() * sizeof(float)`.
 *
 * Built once during graph construction; validated before execution.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#pragma once

#include "../backends/DeviceId.h"
#include "../transfer/TransferMethod.h"
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Data type for activation tensors in PP transfers
     */
    enum class ActivationDType : uint8_t
    {
        FP32, ///< 32-bit float (default)
        BF16, ///< Brain float 16
        FP16, ///< IEEE half-precision
    };

    /**
     * @brief Size in bytes of a single element for an ActivationDType
     */
    inline constexpr size_t activationDTypeBytes(ActivationDType dtype)
    {
        switch (dtype)
        {
        case ActivationDType::FP32:
            return 4;
        case ActivationDType::BF16:
        case ActivationDType::FP16:
            return 2;
        }
        return 4; // default FP32
    }

    inline constexpr const char *activationDTypeName(ActivationDType dtype)
    {
        switch (dtype)
        {
        case ActivationDType::FP32:
            return "FP32";
        case ActivationDType::BF16:
            return "BF16";
        case ActivationDType::FP16:
            return "FP16";
        }
        return "FP32";
    }

    /**
     * @brief Contract for a single inter-stage activation transfer
     *
     * Describes what is transferred between two adjacent PP stages.
     * Established during graph construction, immutable during execution.
     */
    struct PPStageTransferContract
    {
        int source_stage = -1; ///< Source PP stage index
        int target_stage = -1; ///< Target PP stage index

        DeviceId source_device; ///< Device producing the activation
        DeviceId target_device; ///< Device consuming the activation

        size_t embedding_dim = 0;                      ///< Hidden dimension (d_model)
        ActivationDType dtype = ActivationDType::FP32; ///< Activation data type
        size_t max_seq_len = 0;                        ///< Maximum sequence length

        /**
         * @brief Compute active transfer size for a given sequence length
         */
        size_t activeBytes(int seq_len) const
        {
            return static_cast<size_t>(seq_len) * embedding_dim * activationDTypeBytes(dtype);
        }

        /**
         * @brief Compute maximum transfer size (full sequence)
         */
        size_t maxBytes() const
        {
            return max_seq_len * embedding_dim * activationDTypeBytes(dtype);
        }

        /**
         * @brief Whether this is a cross-vendor transfer (CUDA↔ROCm)
         */
        bool isCrossVendor() const
        {
            return (source_device.is_cuda() && target_device.is_rocm()) ||
                   (source_device.is_rocm() && target_device.is_cuda());
        }

        /**
         * @brief Whether source and target are the same device
         */
        bool isSameDevice() const
        {
            return source_device == target_device;
        }

        /**
         * @brief Whether either endpoint involves CPU
         */
        bool involvesCPU() const
        {
            return source_device.is_cpu() || target_device.is_cpu();
        }

        /**
         * @brief Select the appropriate transfer method
         */
        TransferMethod selectMethod() const
        {
            if (isSameDevice())
                return TransferMethod::NOOP;
            if (isCrossVendor() || involvesCPU())
                return TransferMethod::HOST_STAGED;
            if (source_device.type == target_device.type)
                return TransferMethod::DEVICE_TO_DEVICE_SAME_BACKEND;
            return TransferMethod::HOST_STAGED;
        }

        std::string describe() const
        {
            std::ostringstream oss;
            oss << "Stage " << source_stage << " → " << target_stage
                << " (" << source_device.to_string() << " → " << target_device.to_string() << ")"
                << " " << embedding_dim << "×" << activationDTypeName(dtype)
                << " max_seq=" << max_seq_len
                << " method=" << to_string(selectMethod());
            return oss.str();
        }
    };

    /**
     * @brief Complete activation contract for a PP pipeline
     *
     * Contains transfer contracts for all adjacent stage pairs.
     * Built once during graph construction, validated before execution.
     *
     * Usage:
     * @code
     * PPActivationContract contract;
     * contract.addTransfer({.source_stage=0, .target_stage=1,
     *                       .source_device=DeviceId::cuda(0),
     *                       .target_device=DeviceId::cuda(1),
     *                       .embedding_dim=896,
     *                       .dtype=ActivationDType::FP32,
     *                       .max_seq_len=2048});
     *
     * // During forward:
     * size_t bytes = contract.transfer(0).activeBytes(seq_len);
     * @endcode
     */
    struct PPActivationContract
    {
        /**
         * @brief Add a transfer contract between adjacent stages
         */
        void addTransfer(PPStageTransferContract transfer)
        {
            transfers_.push_back(std::move(transfer));
        }

        /**
         * @brief Get the transfer contract for a specific stage boundary
         *
         * @param source_stage Source stage index
         * @return Reference to the transfer contract
         */
        const PPStageTransferContract &transfer(int source_stage) const
        {
            for (const auto &t : transfers_)
            {
                if (t.source_stage == source_stage)
                    return t;
            }
            throw std::runtime_error(
                "PPActivationContract: no transfer from stage " + std::to_string(source_stage));
        }

        /**
         * @brief Number of inter-stage transfers
         */
        size_t numTransfers() const { return transfers_.size(); }

        /**
         * @brief Validate the contract for consistency
         * @return Empty string on success, error description on failure
         */
        std::string validate() const
        {
            for (size_t i = 0; i < transfers_.size(); ++i)
            {
                const auto &t = transfers_[i];
                if (t.source_stage < 0 || t.target_stage < 0)
                    return "Transfer " + std::to_string(i) + " has invalid stage indices";
                if (t.embedding_dim == 0)
                    return "Transfer " + std::to_string(i) + " has zero embedding_dim";
                if (t.max_seq_len == 0)
                    return "Transfer " + std::to_string(i) + " has zero max_seq_len";
                if (!t.source_device.is_valid() || !t.target_device.is_valid())
                    return "Transfer " + std::to_string(i) + " has invalid device(s)";
            }
            return {};
        }

        /**
         * @brief Whether any transfer is cross-vendor
         */
        bool hasCrossVendorTransfer() const
        {
            for (const auto &t : transfers_)
            {
                if (t.isCrossVendor())
                    return true;
            }
            return false;
        }

        /**
         * @brief Iterate all transfers
         */
        const std::vector<PPStageTransferContract> &transfers() const { return transfers_; }

    private:
        std::vector<PPStageTransferContract> transfers_;
    };

} // namespace llaminar2
