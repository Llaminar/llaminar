/**
 * @file PPStage.h
 * @brief Hierarchical pipeline parallel stage model
 *
 * A PP stage can be EITHER:
 * - A single device (leaf node)
 * - A nested parallelism context (TP domain or nested PP)
 *
 * This enables topologies like:
 *   PipelineParallel(TensorParallel(rocm:0, rocm:1), cuda:0, cpu)
 *
 * Where stage 0 is a TP domain with 2 devices, and LocalPPContext::transfer()
 * understands how to get data into/out of the TP domain properly.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "../backends/GlobalDeviceAddress.h"
#include "IGlobalTPContext.h"
#include <memory>
#include <variant>

namespace llaminar2
{

    // Forward declarations
    class ILocalTPContext;
    class ILocalPPContext;
    class IGlobalTPContext;

    /**
     * @brief Type of parallelism context at a PP stage
     */
    enum class PPStageType
    {
        SINGLE_DEVICE,    ///< Stage is a single device (leaf)
        TP_DOMAIN,        ///< Stage is a LOCAL TensorParallel domain
        GLOBAL_TP_DOMAIN, ///< Stage is a GLOBAL TensorParallel domain (cross-MPI-rank)
        NESTED_PP,        ///< Stage is a nested PipelineParallel context (future)
    };

    /**
     * @brief A hierarchical pipeline parallel stage
     *
     * Represents one stage in a pipeline parallel topology. The stage can be:
     * - A single device (traditional PP model)
     * - A TP domain with multiple devices (PP wrapping TP)
     * - A nested PP context (future: PP of PP)
     *
     * ## Usage
     *
     * ```cpp
     * // Single device stage
     * PPStage stage0 = PPStage::fromDevice(DeviceId::cuda(0));
     *
     * // TP domain stage (wraps existing TP context)
     * auto tp_ctx = createLocalTPContext({rocm:0, rocm:1}, {}, AUTO);
     * PPStage stage1 = PPStage::fromTPContext(tp_ctx);
     *
     * // Query stage properties
     * if (stage1.isTPDomain()) {
     *     auto* tp = stage1.asTPContext();
     *     // Access TP context methods
     * }
     *
     * // Get representative device for transfers
     * DeviceId src = stage1.representativeDevice();  // First device in TP domain
     * ```
     *
     * ## Transfer Semantics
     *
     * When transferring FROM a TP domain:
     * - After TP allreduce, all devices have identical data
     * - Use any device as source (typically device 0)
     * - The TP context may have registered output tensors for cross-vendor transfer
     *
     * When transferring TO a TP domain:
     * - For replicated data: broadcast to all devices
     * - For sharded data: scatter to appropriate devices
     * - (Usually PP transfers replicated activations, TP handles sharding internally)
     */
    class PPStage
    {
    public:
        /**
         * @brief Create a single-device stage
         */
        static PPStage fromDevice(GlobalDeviceAddress device)
        {
            PPStage stage;
            stage.type_ = PPStageType::SINGLE_DEVICE;
            stage.device_ = device;
            return stage;
        }

        /**
         * @brief Create a TP domain stage (wraps existing TP context)
         *
         * The PP context does NOT own the TP context - caller must ensure
         * the TP context outlives the PP context.
         */
        static PPStage fromTPContext(std::shared_ptr<ILocalTPContext> tp_ctx)
        {
            PPStage stage;
            stage.type_ = PPStageType::TP_DOMAIN;
            stage.tp_context_ = std::move(tp_ctx);
            return stage;
        }

        /**
         * @brief Create a global TP domain stage (wraps GlobalTPContext)
         *
         * For global TP, the "representative device" is the local rank's CPU device
         * in the global TP domain. PP transfers operate on this local device.
         */
        static PPStage fromGlobalTPContext(std::shared_ptr<IGlobalTPContext> global_tp_ctx)
        {
            PPStage stage;
            stage.type_ = PPStageType::GLOBAL_TP_DOMAIN;
            stage.global_tp_context_ = std::move(global_tp_ctx);
            return stage;
        }

        /**
         * @brief Create a nested PP stage (future)
         */
        static PPStage fromNestedPP(std::shared_ptr<ILocalPPContext> pp_ctx)
        {
            PPStage stage;
            stage.type_ = PPStageType::NESTED_PP;
            stage.pp_context_ = std::move(pp_ctx);
            return stage;
        }

        // =====================================================================
        // Type Queries
        // =====================================================================

        PPStageType type() const { return type_; }

        bool isSingleDevice() const { return type_ == PPStageType::SINGLE_DEVICE; }
        bool isTPDomain() const { return type_ == PPStageType::TP_DOMAIN; }
        bool isGlobalTPDomain() const { return type_ == PPStageType::GLOBAL_TP_DOMAIN; }
        bool isNestedPP() const { return type_ == PPStageType::NESTED_PP; }

        // =====================================================================
        // Access Underlying Context
        // =====================================================================

        /**
         * @brief Get the single device (only valid for SINGLE_DEVICE type)
         */
        const GlobalDeviceAddress &device() const
        {
            if (type_ != PPStageType::SINGLE_DEVICE)
            {
                throw std::logic_error("PPStage::device() called on non-single-device stage");
            }
            return device_;
        }

        /**
         * @brief Get the TP context (only valid for TP_DOMAIN type)
         */
        ILocalTPContext *asTPContext() const
        {
            if (type_ != PPStageType::TP_DOMAIN)
            {
                return nullptr;
            }
            return tp_context_.get();
        }

        /**
         * @brief Get shared pointer to TP context (for lifetime management)
         */
        std::shared_ptr<ILocalTPContext> tpContextPtr() const
        {
            return tp_context_;
        }

        /**
         * @brief Get the global TP context (only valid for GLOBAL_TP_DOMAIN type)
         */
        IGlobalTPContext *asGlobalTPContext() const
        {
            if (type_ != PPStageType::GLOBAL_TP_DOMAIN)
            {
                return nullptr;
            }
            return global_tp_context_.get();
        }

        /**
         * @brief Get shared pointer to global TP context (for lifetime management)
         */
        std::shared_ptr<IGlobalTPContext> globalTPContextPtr() const
        {
            return global_tp_context_;
        }

        /**
         * @brief Get the nested PP context (only valid for NESTED_PP type)
         */
        ILocalPPContext *asNestedPP() const
        {
            if (type_ != PPStageType::NESTED_PP)
            {
                return nullptr;
            }
            return pp_context_.get();
        }

        // =====================================================================
        // Unified Interface (works for all stage types)
        // =====================================================================

        /**
         * @brief Get representative device for this stage
         *
         * For single device: returns the device
         * For TP domain: returns first device (device 0) - after allreduce, all have same data
         * For nested PP: returns first stage's representative device
         *
         * This is the "canonical" device used for PP transfers.
         */
        GlobalDeviceAddress representativeDevice() const;

        /**
         * @brief Get all devices at this stage
         *
         * For single device: returns vector with one device
         * For TP domain: returns all devices in the TP context
         * For nested PP: returns all devices across all nested stages
         */
        std::vector<GlobalDeviceAddress> allDevices() const;

        /**
         * @brief Get total device count at this stage
         */
        int deviceCount() const;

        /**
         * @brief Check if this stage contains a specific device
         */
        bool containsDevice(const GlobalDeviceAddress &device) const;

        /**
         * @brief Human-readable description
         */
        std::string describe() const;

    private:
        PPStage() = default;

        PPStageType type_ = PPStageType::SINGLE_DEVICE;

        // Only one of these is valid based on type_
        GlobalDeviceAddress device_;
        std::shared_ptr<ILocalTPContext> tp_context_;
        std::shared_ptr<IGlobalTPContext> global_tp_context_;
        std::shared_ptr<ILocalPPContext> pp_context_;
    };

} // namespace llaminar2
