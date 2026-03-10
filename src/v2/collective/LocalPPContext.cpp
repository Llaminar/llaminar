/**
 * @file LocalPPContext.cpp
 * @brief Implementation of LOCAL pipeline parallelism context
 *
 * Provides activation transfer between PP stages within a single MPI rank.
 * Backend selection is automatic based on device types:
 * - CUDA→CUDA: NCCL
 * - ROCm→ROCm: RCCL
 * - CUDA↔ROCm: PCIeBAR
 * - GPU↔CPU or CPU→CPU: HOST
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ILocalPPContext.h"
#include "ILocalTPContext.h"
#include "PPStage.h"
#include "DeviceGroup.h"
#include "ICollectiveBackend.h"
#include "backends/HostBackend.h"
#include "../backends/BackendManager.h"
#include "../tensors/TensorClasses.h"
#include "../utils/Logger.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

// Conditionally include GPU-specific backends
#ifdef HAVE_CUDA
#include "backends/NCCLBackend.h"
#include <cuda_runtime.h>
#endif

#ifdef HAVE_ROCM
#include "backends/RCCLBackend.h"
#include "../backends/rocm/ROCmBackend.h"
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/PCIeBARBackend.h"
#include "../backends/p2p/DirectP2P.h"
#endif

namespace llaminar2
{

    // =========================================================================
    // Helper: Backend selection based on device types
    // =========================================================================

    /**
     * @brief Select optimal backend for transfer between two device types
     *
     * @param src Source device type
     * @param dst Destination device type
     * @return Optimal backend type for this transfer path
     */
    static CollectiveBackendType selectBackendForTransfer(DeviceType src, DeviceType dst)
    {
        // Same CUDA → NCCL
        if (src == DeviceType::CUDA && dst == DeviceType::CUDA)
        {
            return CollectiveBackendType::NCCL;
        }

        // Same ROCm → RCCL
        if (src == DeviceType::ROCm && dst == DeviceType::ROCm)
        {
            return CollectiveBackendType::RCCL;
        }

        // Cross-vendor GPU → PCIeBAR
        if ((src == DeviceType::CUDA && dst == DeviceType::ROCm) ||
            (src == DeviceType::ROCm && dst == DeviceType::CUDA))
        {
            return CollectiveBackendType::PCIE_BAR;
        }

        // Any path involving CPU → HOST
        return CollectiveBackendType::HOST;
    }

    /**
     * @brief Get size in bytes for a collective data type
     */
    static size_t collectiveDataTypeSize(CollectiveDataType dtype)
    {
        switch (dtype)
        {
        case CollectiveDataType::FLOAT32:
            return sizeof(float);
        case CollectiveDataType::FLOAT16:
        case CollectiveDataType::BFLOAT16:
            return 2;
        case CollectiveDataType::INT32:
            return sizeof(int32_t);
        case CollectiveDataType::INT8:
            return sizeof(int8_t);
        }
        return sizeof(float); // Default
    }

    /**
     * @brief Convert tensor dtype to collective dtype
     */
    static CollectiveDataType tensorToCollectiveDataType(const TensorBase *tensor)
    {
        if (!tensor)
            return CollectiveDataType::FLOAT32;

        // Check tensor type - for now assume FP32
        // TODO: Extend for BF16, FP16 tensors when needed
        return CollectiveDataType::FLOAT32;
    }

    // =========================================================================
    // LocalPPContext Implementation
    // =========================================================================

    /**
     * @brief Concrete implementation of ILocalPPContext
     *
     * Manages pipeline parallel activation transfers between devices owned
     * by a single MPI rank. Backends are created lazily on first use.
     */
    class LocalPPContext : public ILocalPPContext
    {
    public:
        explicit LocalPPContext(const LocalPPConfig &config);
        ~LocalPPContext() override;

        // =====================================================================
        // Configuration (ILocalPPContext)
        // =====================================================================

        int numStages() const override { return config_.numStages(); }

        const GlobalDeviceAddress &deviceForStage(int stage) const override
        {
            if (stage < 0 || stage >= numStages())
            {
                throw std::out_of_range("LocalPPContext::deviceForStage: invalid stage index " +
                                        std::to_string(stage));
            }
            return config_.stage_devices[stage];
        }

        CollectiveBackendType backendForTransfer(int stage_from, int stage_to) const override
        {
            if (stage_from < 0 || stage_from >= numStages() ||
                stage_to < 0 || stage_to >= numStages())
            {
                return CollectiveBackendType::HOST;
            }

            DeviceType src_type = config_.stage_devices[stage_from].device_type;
            DeviceType dst_type = config_.stage_devices[stage_to].device_type;

            return selectBackendForTransfer(src_type, dst_type);
        }

        std::pair<int, int> layerRangeForStage(int stage) const override
        {
            return config_.layerRangeForStage(stage);
        }

        int stageForLayer(int layer) const override
        {
            return config_.stageForLayer(layer);
        }

        const std::vector<GlobalDeviceAddress> &stageDevices() const override
        {
            return config_.stage_devices;
        }

        const std::vector<int> &layerBoundaries() const override
        {
            return config_.layer_boundaries;
        }

        // =====================================================================
        // Activation Transfer Operations
        // =====================================================================

        bool transfer(TensorBase *activations, int stage_from, int stage_to,
                      size_t active_bytes = 0) override;
        bool transferAsync(TensorBase *activations, int stage_from, int stage_to, void *stream) override;

        // =====================================================================
        // Synchronization
        // =====================================================================

        void synchronize() override;
        void synchronizeStream(void *stream) override;

        // =====================================================================
        // Utility
        // =====================================================================

        bool sameDevice(int stage_a, int stage_b) const override
        {
            if (stage_a < 0 || stage_a >= numStages() ||
                stage_b < 0 || stage_b >= numStages())
            {
                return false;
            }
            return config_.stage_devices[stage_a] == config_.stage_devices[stage_b];
        }

        int totalLayers() const override
        {
            if (config_.layer_boundaries.empty())
                return 0;
            return config_.layer_boundaries.back();
        }

        bool reserveStagingBufferBytes(size_t bytes) override;

    private:
        // =====================================================================
        // Member Variables
        // =====================================================================

        /// Configuration
        LocalPPConfig config_;
    };

    // =========================================================================
    // LocalPPContext Implementation
    // =========================================================================

    LocalPPContext::LocalPPContext(const LocalPPConfig &config)
        : config_(config)
    {
        if (!config_.isValid())
        {
            throw std::invalid_argument("LocalPPContext: invalid configuration");
        }

        LOG_INFO("LocalPPContext created: " << numStages() << " stages, "
                                            << totalLayers() << " layers");

        // Log stage configuration
        for (int s = 0; s < numStages(); ++s)
        {
            auto [first, last] = layerRangeForStage(s);
            LOG_DEBUG("  Stage " << s << ": layers [" << first << ", " << last << ") on "
                                 << config_.stage_devices[s].toShortString());
        }
    }

    LocalPPContext::~LocalPPContext()
    {
        LOG_DEBUG("LocalPPContext destroyed");
    }

    // =========================================================================
    // Transfer Operations
    // =========================================================================

    bool LocalPPContext::transfer(TensorBase *activations, int stage_from, int stage_to,
                                  size_t active_bytes)
    {
        if (!activations)
        {
            LOG_ERROR("LocalPPContext::transfer: null activations tensor");
            return false;
        }

        // Validate stage indices
        if (stage_from < 0 || stage_from >= numStages() ||
            stage_to < 0 || stage_to >= numStages())
        {
            LOG_ERROR("LocalPPContext::transfer: invalid stage indices "
                      << stage_from << " -> " << stage_to);
            return false;
        }

        // Same device - no-op
        if (sameDevice(stage_from, stage_to))
        {
            LOG_DEBUG("LocalPPContext::transfer: same device, no-op (stage "
                      << stage_from << " -> " << stage_to << ")");
            return true;
        }

        const DeviceId src_device = config_.stage_devices[stage_from].toLocalDeviceId();
        const DeviceId dst_device = config_.stage_devices[stage_to].toLocalDeviceId();

        // Handle GPU↔CPU transfers via host memory (transferTo only supports GPU-to-GPU)
        if (src_device.is_cpu() || dst_device.is_cpu())
        {
            // GPU → CPU: Use ensureOnHost() to download data
            if (!dst_device.is_cpu())
            {
                // CPU → GPU: First ensure on host (should already be), then upload to GPU
                if (!activations->ensureOnDevice(dst_device))
                {
                    LOG_ERROR("LocalPPContext::transfer: Failed to upload to "
                              << dst_device.toString());
                    return false;
                }
                activations->mark_device_dirty();
                LOG_DEBUG("LocalPPContext::transfer: CPU → GPU transfer to "
                          << dst_device.toString() << " (" << activations->numel() << " elements)");
            }
            else
            {
                // GPU → CPU or CPU → CPU: Ensure data is on host
                // The data() call will sync from GPU if device-dirty
                if (!activations->data())
                {
                    LOG_ERROR("LocalPPContext::transfer: Failed to sync to host from "
                              << src_device.toString());
                    return false;
                }
                // Mark GPU data as stale - host is now authoritative
                activations->invalidateGpuData();
                LOG_DEBUG("LocalPPContext::transfer: GPU → CPU transfer from "
                          << src_device.toString() << " (" << activations->numel() << " elements)");
            }
            return true;
        }

        // Ensure tensor is on source device first (if not already authoritative there)
        if (!activations->isDeviceAuthoritative(src_device))
        {
            if (!activations->ensureOnDevice(src_device))
            {
                LOG_ERROR("LocalPPContext::transfer: Failed to ensure on source device "
                          << src_device.toString());
                return false;
            }
            activations->mark_device_dirty();
        }

        // Cross-vendor GPU transfer (CUDA↔ROCm): use host bounce if tensor
        // is not BAR-backed. PCIeBAR requires allocateInBarRegion(); normal
        // device memory won't work for cross-vendor direct copies.
        const bool is_cross_vendor =
            (src_device.is_cuda() && dst_device.is_rocm()) ||
            (src_device.is_rocm() && dst_device.is_cuda());

        if (is_cross_vendor && !activations->isBARBacked())
        {
            // GPU → host: use ensureOnHost() for proper D2H coherence sync.
            // NOTE: Do NOT use data() here — for quantized tensors (Q4_0, Q8_0, etc.),
            // data() returns a dequantized float view without touching coherence flags,
            // leaving host_valid_ = false and causing a COHERENCE ERROR downstream.
            if (!activations->ensureOnHost())
            {
                LOG_ERROR("LocalPPContext::transfer: Failed to sync to host from "
                          << src_device.toString() << " for cross-vendor bounce");
                return false;
            }

            // Invalidate old GPU data — host is now authoritative
            activations->invalidateGpuData();

            // Host → destination GPU
            if (!activations->ensureOnDevice(dst_device))
            {
                LOG_ERROR("LocalPPContext::transfer: Failed to upload to "
                          << dst_device.toString() << " after cross-vendor host bounce");
                return false;
            }
            activations->mark_device_dirty();

            LOG_DEBUG("LocalPPContext::transfer: Cross-vendor host bounce "
                      << src_device.toString() << " → CPU → " << dst_device.toString()
                      << " (" << activations->numel() << " elements)");
            return true;
        }

        // Same-vendor GPU-to-GPU or BAR-backed cross-vendor: direct transfer
        if (!activations->transferTo(dst_device, active_bytes))
        {
            LOG_ERROR("LocalPPContext::transfer: transferTo() failed "
                      << src_device.toString() << " → " << dst_device.toString());
            return false;
        }

        LOG_DEBUG("LocalPPContext::transfer: Direct P2P/BAR transfer "
                  << src_device.toString() << " → " << dst_device.toString()
                  << " (" << activations->numel() << " elements)");
        return true;
    }

    bool LocalPPContext::transferAsync(TensorBase *activations, int stage_from, int stage_to, void *stream)
    {
        if (!activations)
        {
            LOG_ERROR("LocalPPContext::transferAsync: null activations tensor");
            return false;
        }

        // Validate stage indices
        if (stage_from < 0 || stage_from >= numStages() ||
            stage_to < 0 || stage_to >= numStages())
        {
            LOG_ERROR("LocalPPContext::transferAsync: invalid stage indices "
                      << stage_from << " -> " << stage_to);
            return false;
        }

        // Same device - no-op
        if (sameDevice(stage_from, stage_to))
        {
            LOG_DEBUG("LocalPPContext::transferAsync: same device, no-op");
            return true;
        }

        const DeviceId src_device = config_.stage_devices[stage_from].toLocalDeviceId();
        const DeviceId dst_device = config_.stage_devices[stage_to].toLocalDeviceId();

        LOG_DEBUG("LocalPPContext::transferAsync: " << src_device.toString() << " -> "
                                                    << dst_device.toString()
                                                    << " (stream=" << stream << ")");

        // For now, transferTo() is synchronous. Future: add stream parameter.
        // Delegate to synchronous transfer.
        (void)stream;
        return transfer(activations, stage_from, stage_to);
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    void LocalPPContext::synchronize()
    {
        // Synchronize all GPU devices - transfers go through TensorBase::transferTo()
        // which uses GlobalBackendRouter, so we just need device-level sync here
#ifdef HAVE_CUDA
        (void)cudaDeviceSynchronize();
#endif

#ifdef HAVE_ROCM
        auto *rocm_backend = getROCmBackend();
        if (rocm_backend)
        {
            rocm_backend->synchronize(0); // Sync default ROCm device
        }
#endif

        LOG_TRACE("LocalPPContext::synchronize: devices synchronized");
    }

    void LocalPPContext::synchronizeStream(void *stream)
    {
        if (!stream)
        {
            synchronize();
            return;
        }

#ifdef HAVE_CUDA
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
#endif

#ifdef HAVE_ROCM
        // hipStreamSynchronize(static_cast<hipStream_t>(stream));
#endif
    }

    // =========================================================================
    // Utility
    // =========================================================================

    bool LocalPPContext::reserveStagingBufferBytes(size_t bytes)
    {
        // Interface method - no-op since transfers go through TensorBase::transferTo()
        // which uses GlobalBackendRouter for actual data movement
        (void)bytes;
        return true;
    }

    // =========================================================================
    // HierarchicalPPContext Implementation
    // =========================================================================

    /**
     * @brief Pipeline parallel context supporting nested TP/PP domains
     *
     * This implementation extends LocalPPContext to support:
     * - Stages that are TP domains (multiple devices with shared state)
     * - Automatic BAR-backed tensor handling for cross-vendor transfers
     * - Proper coordination with nested TP context synchronization
     *
     * Key insight: When a stage is a TP domain, the transfer() method needs to:
     * 1. Get data from the TP domain's representative device
     * 2. If TP uses BAR-backed buffers, use those for cross-vendor transfer
     * 3. Distribute data to the destination stage (broadcast for TP domains)
     */
    class HierarchicalPPContext : public ILocalPPContext
    {
    public:
        explicit HierarchicalPPContext(const HierarchicalPPConfig &config);
        ~HierarchicalPPContext() override;

        // =====================================================================
        // Configuration (ILocalPPContext)
        // =====================================================================

        int numStages() const override { return config_.numStages(); }

        const GlobalDeviceAddress &deviceForStage(int stage) const override
        {
            if (stage < 0 || stage >= numStages())
            {
                throw std::out_of_range("HierarchicalPPContext::deviceForStage: invalid stage index");
            }
            // Return representative device (for TP domains, this is device 0)
            return flat_devices_[stage];
        }

        CollectiveBackendType backendForTransfer(int stage_from, int stage_to) const override
        {
            if (stage_from < 0 || stage_from >= numStages() ||
                stage_to < 0 || stage_to >= numStages())
            {
                return CollectiveBackendType::HOST;
            }

            DeviceType src_type = flat_devices_[stage_from].device_type;
            DeviceType dst_type = flat_devices_[stage_to].device_type;

            return selectBackendForTransfer(src_type, dst_type);
        }

        std::pair<int, int> layerRangeForStage(int stage) const override
        {
            return config_.layerRangeForStage(stage);
        }

        int stageForLayer(int layer) const override
        {
            return config_.stageForLayer(layer);
        }

        const std::vector<GlobalDeviceAddress> &stageDevices() const override
        {
            return flat_devices_;
        }

        const std::vector<int> &layerBoundaries() const override
        {
            return config_.layer_boundaries;
        }

        // =====================================================================
        // Activation Transfer Operations
        // =====================================================================

        bool transfer(TensorBase *activations, int stage_from, int stage_to,
                      size_t active_bytes = 0) override;
        bool transferAsync(TensorBase *activations, int stage_from, int stage_to, void *stream) override;

        // =====================================================================
        // Synchronization
        // =====================================================================

        void synchronize() override;
        void synchronizeStream(void *stream) override;

        // =====================================================================
        // Utility
        // =====================================================================

        bool sameDevice(int stage_a, int stage_b) const override
        {
            if (stage_a < 0 || stage_a >= numStages() ||
                stage_b < 0 || stage_b >= numStages())
            {
                return false;
            }
            return flat_devices_[stage_a] == flat_devices_[stage_b];
        }

        int totalLayers() const override
        {
            if (config_.layer_boundaries.empty())
                return 0;
            return config_.layer_boundaries.back();
        }

        bool reserveStagingBufferBytes(size_t bytes) override;

    private:
        // =====================================================================
        // Transfer Implementation
        // =====================================================================

        /**
         * @brief Find the best source device from a TP domain for transfer to dst_device
         *
         * For heterogeneous TP domains, prefer same-vendor device to avoid cross-vendor
         * PCIe BAR staging. Falls back to representative device (index 0) if no
         * same-vendor device exists.
         *
         * @param tp_ctx TP context containing the source devices
         * @param dst_device Destination device to transfer to
         * @return DeviceId of the best source device
         */
        DeviceId findBestSourceDevice(ILocalTPContext *tp_ctx, const DeviceId &dst_device);

        /**
         * @brief Transfer from TP domain to another stage
         *
         * For TP domains:
         * 1. After allreduce, all devices have identical data
         * 2. Pick best source device (prefer same vendor as destination)
         * 3. Use TP context's DirectP2PEngine if available (BAR-backed)
         */
        bool transferFromTPDomain(TensorBase *activations,
                                  const PPStage &src_stage,
                                  const PPStage &dst_stage,
                                  size_t active_bytes = 0);

        /**
         * @brief Transfer to TP domain from another stage
         *
         * For TP domains:
         * 1. Find best destination device in TP domain (prefer same vendor as source)
         * 2. Transfer to that device
         * 3. Broadcast to all other devices in the TP domain
         */
        bool transferToTPDomain(TensorBase *activations,
                                const PPStage &src_stage,
                                const PPStage &dst_stage,
                                size_t active_bytes = 0);

        /**
         * @brief Transfer from a global TP domain to another stage
         *
         * For global TP:
         * - After global TP allreduce (which happens during layer computation),
         *   each rank has the result on its local CPU
         * - PP transfer is a LOCAL operation: transfer from local CPU to destination
         * - The destination can be a single device or a local TP domain
         */
        bool transferFromGlobalTPDomain(TensorBase *activations,
                                        const PPStage &src_stage,
                                        const PPStage &dst_stage,
                                        size_t active_bytes = 0);

        /**
         * @brief Transfer to a global TP domain from another stage
         *
         * For global TP:
         * - PP transfer moves data TO the local CPU
         * - The global TP allreduce happens during LAYER COMPUTATION (via TPAllreduceStage)
         * - This is a LOCAL operation: transfer from source to local CPU
         */
        bool transferToGlobalTPDomain(TensorBase *activations,
                                      const PPStage &src_stage,
                                      const PPStage &dst_stage,
                                      size_t active_bytes = 0);

        /**
         * @brief Standard transfer between single devices
         */
        bool transferSingleToSingle(TensorBase *activations,
                                    const DeviceId &src_device,
                                    const DeviceId &dst_device,
                                    size_t active_bytes = 0);

        // =====================================================================
        // Member Variables
        // =====================================================================

        HierarchicalPPConfig config_;

        /// Flat device list (representative device per stage) for backward compatibility
        std::vector<GlobalDeviceAddress> flat_devices_;

        /// Staging buffer for HOST backend transfers
        std::vector<char> staging_buffer_;

        /// Mutex for thread-safe access
        mutable std::mutex mutex_;
    };

    // =========================================================================
    // HierarchicalPPContext Implementation
    // =========================================================================

    HierarchicalPPContext::HierarchicalPPContext(const HierarchicalPPConfig &config)
        : config_(config)
    {
        if (!config_.isValid())
        {
            throw std::invalid_argument("HierarchicalPPContext: invalid configuration");
        }

        // Build flat device list from representative devices
        for (const auto &stage : config_.stages)
        {
            flat_devices_.push_back(stage.representativeDevice());
        }

        LOG_INFO("HierarchicalPPContext created: " << numStages() << " stages, "
                                                   << totalLayers() << " layers");

        // Log stage configuration with hierarchy info
        for (int s = 0; s < numStages(); ++s)
        {
            auto [first, last] = layerRangeForStage(s);
            LOG_INFO("  Stage " << s << ": layers [" << first << ", " << last << ") on "
                                << config_.stages[s].describe());
        }
    }

    HierarchicalPPContext::~HierarchicalPPContext()
    {
        LOG_DEBUG("HierarchicalPPContext destroyed");
    }

    bool HierarchicalPPContext::transfer(TensorBase *activations, int stage_from, int stage_to,
                                         size_t active_bytes)
    {
        if (!activations)
        {
            LOG_ERROR("HierarchicalPPContext::transfer: null activations tensor");
            return false;
        }

        // Validate stage indices
        if (stage_from < 0 || stage_from >= numStages() ||
            stage_to < 0 || stage_to >= numStages())
        {
            LOG_ERROR("HierarchicalPPContext::transfer: invalid stage indices "
                      << stage_from << " -> " << stage_to);
            return false;
        }

        // Same stage - no-op
        if (stage_from == stage_to)
        {
            LOG_DEBUG("HierarchicalPPContext::transfer: same stage, no-op");
            return true;
        }

        const PPStage &src_stage = config_.stages[stage_from];
        const PPStage &dst_stage = config_.stages[stage_to];

        LOG_DEBUG("HierarchicalPPContext::transfer: " << src_stage.describe()
                                                      << " → " << dst_stage.describe());

        // Dispatch based on stage types
        // Priority: Global TP > Local TP > Single Device

        // Handle Global TP domains first (CPU-only, cross-MPI-rank)
        if (src_stage.isGlobalTPDomain())
        {
            return transferFromGlobalTPDomain(activations, src_stage, dst_stage, active_bytes);
        }
        else if (dst_stage.isGlobalTPDomain())
        {
            return transferToGlobalTPDomain(activations, src_stage, dst_stage, active_bytes);
        }
        // Handle Local TP domains
        else if (src_stage.isTPDomain())
        {
            return transferFromTPDomain(activations, src_stage, dst_stage, active_bytes);
        }
        else if (dst_stage.isTPDomain())
        {
            return transferToTPDomain(activations, src_stage, dst_stage, active_bytes);
        }
        else
        {
            // Both are single devices
            return transferSingleToSingle(
                activations,
                src_stage.device().toLocalDeviceId(),
                dst_stage.device().toLocalDeviceId(),
                active_bytes);
        }
    }

    DeviceId HierarchicalPPContext::findBestSourceDevice(ILocalTPContext *tp_ctx,
                                                         const DeviceId &dst_device)
    {
        // For heterogeneous TP domains (e.g., TP(rocm:0, cuda:0)), prefer a source device
        // that matches the destination's vendor to avoid cross-vendor PCIe BAR staging.
        //
        // After TP allreduce, all devices in the domain have identical data, so any device
        // can serve as the source. Choosing a same-vendor device enables fast D2D transfer
        // via NCCL/RCCL/NVLink instead of the slower 2-hop BAR path.

        const int degree = tp_ctx->degree();

        // Single device TP domain - no choice
        if (degree == 1)
        {
            return tp_ctx->deviceAt(0).toLocalDeviceId();
        }

        // Check destination type
        const bool dst_is_cuda = dst_device.is_cuda();
        const bool dst_is_rocm = dst_device.is_rocm();
        const bool dst_is_gpu = dst_is_cuda || dst_is_rocm;

        // For CPU destination, any source works equally well (all paths go through host)
        if (!dst_is_gpu)
        {
            return tp_ctx->deviceAt(0).toLocalDeviceId();
        }

        // Search for a same-vendor device in the TP domain
        for (int i = 0; i < degree; ++i)
        {
            DeviceId candidate = tp_ctx->deviceAt(i).toLocalDeviceId();

            if ((dst_is_cuda && candidate.is_cuda()) ||
                (dst_is_rocm && candidate.is_rocm()))
            {
                LOG_DEBUG("HierarchicalPPContext::findBestSourceDevice: "
                          << "selected same-vendor device " << candidate.toString()
                          << " (index " << i << ") for destination " << dst_device.toString());
                return candidate;
            }
        }

        // No same-vendor device found - fall back to representative (index 0)
        LOG_DEBUG("HierarchicalPPContext::findBestSourceDevice: "
                  << "no same-vendor device found in TP domain, using representative "
                  << tp_ctx->deviceAt(0).toString() << " for destination " << dst_device.toString());
        return tp_ctx->deviceAt(0).toLocalDeviceId();
    }

    bool HierarchicalPPContext::transferFromTPDomain(TensorBase *activations,
                                                     const PPStage &src_stage,
                                                     const PPStage &dst_stage,
                                                     size_t active_bytes)
    {
        ILocalTPContext *tp_ctx = src_stage.asTPContext();
        if (!tp_ctx)
        {
            LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: src_stage is not a TP domain");
            return false;
        }

        // After TP allreduce, all devices in the TP domain have identical data.
        // For heterogeneous TP domains, prefer a same-vendor source device to avoid
        // cross-vendor PCIe BAR staging when possible.
        DeviceId dst_device = dst_stage.representativeDevice().toLocalDeviceId();

        // Smart source selection: find best source device from TP domain
        DeviceId src_device = findBestSourceDevice(tp_ctx, dst_device);

        LOG_DEBUG("HierarchicalPPContext::transferFromTPDomain: "
                  << "TP domain (" << tp_ctx->degree() << " devices) → "
                  << dst_device.toString()
                  << " using selected source " << src_device.toString());

        // Check if cross-vendor transfer (requires special handling)
        bool is_cross_vendor = (src_device.is_rocm() && dst_device.is_cuda()) ||
                               (src_device.is_cuda() && dst_device.is_rocm());

        if (!is_cross_vendor)
        {
            // Same-vendor transfer: use standard path
            return transferSingleToSingle(activations, src_device, dst_device, active_bytes);
        }

        // Cross-vendor transfer: check for BAR-backed fast path
        if (activations->isBARBacked())
        {
            LOG_DEBUG("HierarchicalPPContext::transferFromTPDomain: Tensor is BAR-backed, "
                      << "CUDA can read directly via PCIe BAR");

            // For BAR-backed tensors, we just need to switch the view
            // The physical memory (in AMD BAR region) is already accessible to both vendors
            if (!activations->ensureOnDevice(dst_device))
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "Failed to set up destination view of BAR-backed tensor");
                return false;
            }

            activations->mark_device_dirty();
            LOG_DEBUG("HierarchicalPPContext::transferFromTPDomain: BAR-backed zero-copy complete");
            return true;
        }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Cross-vendor transfer: use PCIe BAR staging (ROCm VRAM → BAR → CUDA or vice versa)

        // Check if we have DirectP2P with active BAR
        auto p2p = DirectP2PEngine::getSharedInstance();
        if (!p2p || !p2p->isPCIeBarActive())
        {
            LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                      "PCIe BAR not initialized for cross-vendor transfer "
                      << src_device.toString() << " → " << dst_device.toString());
            return false;
        }

        // PCIe BAR staging path: ROCm VRAM → BAR → CUDA (or reverse)
        LOG_DEBUG("HierarchicalPPContext::transferFromTPDomain: PCIe BAR staging "
                  << src_device.toString() << " → BAR → " << dst_device.toString());

        const size_t bytes = (active_bytes > 0 && active_bytes <= activations->size_bytes())
                                 ? active_bytes
                                 : activations->size_bytes();

        // Ensure tensor has data on the source device
        if (!activations->ensureOnDevice(src_device))
        {
            LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                      "Failed to ensure on source device for BAR staging");
            return false;
        }

        void *src_ptr = activations->gpu_data_ptr();
        if (!src_ptr)
        {
            LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                      "Source GPU pointer is null after ensureOnDevice");
            return false;
        }

        if (src_device.is_rocm() && dst_device.is_cuda())
        {
            // ROCm → CUDA via BAR
            // Step 1: ROCm D2D copy to BAR (hipMemcpy can use BAR mmap as destination)
            void *bar_host_ptr = p2p->getBarHostPtr();
            if (!bar_host_ptr)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "BAR host pointer is null");
                return false;
            }

            // Use ROCm backend to copy to BAR (D2D copy, very fast)
            // Need concrete ROCmBackend for deviceToDevice() method
            auto *rocm_backend = dynamic_cast<ROCmBackend *>(getROCmBackend());
            if (!rocm_backend)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "ROCm backend not available or wrong type");
                return false;
            }

            LOG_DEBUG("HierarchicalPPContext::transferFromTPDomain: "
                      "ROCm D2D to BAR: "
                      << bytes << " bytes");

            // hipMemcpy D2D from ROCm VRAM to BAR mmap address
            bool d2d_ok = rocm_backend->deviceToDevice(bar_host_ptr, src_ptr, bytes,
                                                       src_device.toKernelDeviceIndex());
            if (!d2d_ok)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "ROCm D2D to BAR failed");
                return false;
            }

            rocm_backend->synchronize(src_device.toKernelDeviceIndex());

            // Step 2: CUDA D2D copy from BAR to CUDA device
            void *cuda_bar_ptr = p2p->getCudaBarPointer();
            if (!cuda_bar_ptr)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "CUDA BAR pointer is null");
                return false;
            }

            // Ensure tensor is allocated on destination
            if (!activations->allocateOnDevice(dst_device))
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "Failed to allocate on CUDA device");
                return false;
            }

            void *dst_ptr = activations->gpu_data_ptr();

#ifdef HAVE_CUDA
            cudaSetDevice(dst_device.toKernelDeviceIndex());
            cudaError_t err = cudaMemcpy(dst_ptr, cuda_bar_ptr, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "CUDA D2D from BAR failed: "
                          << cudaGetErrorString(err));
                return false;
            }
            cudaDeviceSynchronize();
#endif

            // Clear any ROCm completion event before marking dirty
            activations->clearCompletionEvent();
            activations->mark_device_dirty();

            LOG_DEBUG("HierarchicalPPContext::transferFromTPDomain: "
                      "PCIe BAR staging complete: "
                      << bytes << " bytes");
            return true;
        }
        else if (src_device.is_cuda() && dst_device.is_rocm())
        {
            // CUDA → ROCm via BAR
            // Step 1: CUDA D2D copy to BAR
            void *cuda_bar_ptr = p2p->getCudaBarPointer();
            if (!cuda_bar_ptr)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "CUDA BAR pointer is null for CUDA→ROCm");
                return false;
            }

#ifdef HAVE_CUDA
            cudaSetDevice(src_device.toKernelDeviceIndex());
            cudaError_t err = cudaMemcpy(cuda_bar_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "CUDA D2D to BAR failed: "
                          << cudaGetErrorString(err));
                return false;
            }
            cudaDeviceSynchronize();

            // Step 2: ROCm D2D copy from BAR to ROCm device
            void *bar_host_ptr = p2p->getBarHostPtr();
            if (!bar_host_ptr)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "BAR host pointer is null for CUDA→ROCm");
                return false;
            }

            // Need concrete ROCmBackend for deviceToDevice() method
            auto *rocm_backend = dynamic_cast<ROCmBackend *>(getROCmBackend());
            if (!rocm_backend)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "ROCm backend not available or wrong type");
                return false;
            }

            // Ensure tensor is allocated on destination
            if (!activations->allocateOnDevice(dst_device))
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "Failed to allocate on ROCm device");
                return false;
            }

            void *dst_ptr = activations->gpu_data_ptr();

            // hipMemcpy D2D from BAR mmap to ROCm VRAM
            bool d2d_ok = rocm_backend->deviceToDevice(dst_ptr, bar_host_ptr, bytes,
                                                       dst_device.toKernelDeviceIndex());
            if (!d2d_ok)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                          "ROCm D2D from BAR failed");
                return false;
            }
            rocm_backend->synchronize(dst_device.toKernelDeviceIndex());

            // Clear any CUDA completion event before marking dirty
            activations->clearCompletionEvent();
            activations->mark_device_dirty();

            LOG_DEBUG("HierarchicalPPContext::transferFromTPDomain: "
                      "PCIe BAR staging complete (CUDA→ROCm): "
                      << bytes << " bytes");
            return true;
#else
            LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                      "CUDA not available for CUDA→ROCm transfer");
            return false;
#endif
        }

        LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                  "Unsupported cross-vendor transfer direction: "
                  << src_device.toString() << " → " << dst_device.toString());
        return false;
#else
        // Cross-vendor transfers not supported without both CUDA and ROCm
        LOG_ERROR("HierarchicalPPContext::transferFromTPDomain: "
                  "Cross-vendor transfer not supported (requires both CUDA and ROCm): "
                  << src_device.toString() << " → " << dst_device.toString());
        return false;
#endif // HAVE_CUDA && HAVE_ROCM
    }

    bool HierarchicalPPContext::transferFromGlobalTPDomain(TensorBase *activations,
                                                           const PPStage &src_stage,
                                                           const PPStage &dst_stage,
                                                           size_t active_bytes)
    {
        IGlobalTPContext *global_tp_ctx = src_stage.asGlobalTPContext();
        if (!global_tp_ctx)
        {
            LOG_ERROR("HierarchicalPPContext::transferFromGlobalTPDomain: src_stage is not a global TP domain");
            return false;
        }

        // Global TP: after allreduce (in layer computation), each rank has result on local CPU.
        // PP transfer is LOCAL: from this rank's CPU to the destination.
        DeviceId src_device = global_tp_ctx->localDevice().toLocalDeviceId();

        LOG_DEBUG("HierarchicalPPContext::transferFromGlobalTPDomain: "
                  << "Global TP domain (" << global_tp_ctx->degree() << " ranks) → "
                  << dst_stage.describe()
                  << " using local CPU " << src_device.toString());

        // Dispatch based on destination type
        if (dst_stage.isTPDomain())
        {
            // Global TP → Local TP: transfer to TP domain's representative device,
            // then TP handles broadcast
            ILocalTPContext *tp_ctx = dst_stage.asTPContext();
            if (!tp_ctx)
            {
                LOG_ERROR("HierarchicalPPContext::transferFromGlobalTPDomain: "
                          "dst_stage claims to be TP domain but asTPContext() returned null");
                return false;
            }

            // Transfer to TP domain's first device (or best match)
            DeviceId dst_device = tp_ctx->deviceAt(0).toLocalDeviceId();

            if (!transferSingleToSingle(activations, src_device, dst_device, active_bytes))
            {
                LOG_ERROR("HierarchicalPPContext::transferFromGlobalTPDomain: "
                          "Failed to transfer to TP domain representative");
                return false;
            }

            // Broadcast to all devices in the local TP domain
            if (tp_ctx->degree() > 1)
            {
                LOG_DEBUG("HierarchicalPPContext::transferFromGlobalTPDomain: "
                          << "Broadcasting from device 0 to all " << tp_ctx->degree() << " TP devices");
                if (!tp_ctx->broadcast(activations, 0))
                {
                    LOG_ERROR("HierarchicalPPContext::transferFromGlobalTPDomain: "
                              "Failed to broadcast to TP domain");
                    return false;
                }
            }

            LOG_DEBUG("HierarchicalPPContext::transferFromGlobalTPDomain: "
                      << "Global TP → Local TP transfer complete");
            return true;
        }
        else if (dst_stage.isGlobalTPDomain())
        {
            // Global TP → Global TP: should not happen in normal PP topology
            // (consecutive global TP stages would be merged)
            LOG_WARN("HierarchicalPPContext::transferFromGlobalTPDomain: "
                     "Transferring between two global TP domains - this is unusual");
            DeviceId dst_device = dst_stage.asGlobalTPContext()->localDevice().toLocalDeviceId();
            return transferSingleToSingle(activations, src_device, dst_device, active_bytes);
        }
        else
        {
            // Global TP → Single device
            DeviceId dst_device = dst_stage.device().toLocalDeviceId();
            return transferSingleToSingle(activations, src_device, dst_device, active_bytes);
        }
    }

    bool HierarchicalPPContext::transferToGlobalTPDomain(TensorBase *activations,
                                                         const PPStage &src_stage,
                                                         const PPStage &dst_stage,
                                                         size_t active_bytes)
    {
        IGlobalTPContext *global_tp_ctx = dst_stage.asGlobalTPContext();
        if (!global_tp_ctx)
        {
            LOG_ERROR("HierarchicalPPContext::transferToGlobalTPDomain: dst_stage is not a global TP domain");
            return false;
        }

        // Global TP: PP transfer moves data TO this rank's local CPU.
        // The global TP allreduce happens during LAYER COMPUTATION (TPAllreduceStage), NOT here.
        DeviceId dst_device = global_tp_ctx->localDevice().toLocalDeviceId();

        LOG_DEBUG("HierarchicalPPContext::transferToGlobalTPDomain: "
                  << src_stage.describe() << " → Global TP domain ("
                  << global_tp_ctx->degree() << " ranks)"
                  << " via local CPU " << dst_device.toString());

        // Dispatch based on source type
        if (src_stage.isTPDomain())
        {
            // Local TP → Global TP: after local TP allreduce, transfer from
            // TP domain's representative device to local CPU
            ILocalTPContext *tp_ctx = src_stage.asTPContext();
            if (!tp_ctx)
            {
                LOG_ERROR("HierarchicalPPContext::transferToGlobalTPDomain: "
                          "src_stage claims to be TP domain but asTPContext() returned null");
                return false;
            }

            // Use the first device (after TP allreduce, all have same data)
            DeviceId src_device = tp_ctx->deviceAt(0).toLocalDeviceId();

            LOG_DEBUG("HierarchicalPPContext::transferToGlobalTPDomain: "
                      << "Local TP (" << tp_ctx->degree() << " devices) → Global TP CPU "
                      << "using TP device 0: " << src_device.toString());

            return transferSingleToSingle(activations, src_device, dst_device, active_bytes);
        }
        else if (src_stage.isGlobalTPDomain())
        {
            // Global TP → Global TP: should not happen in normal PP topology
            LOG_WARN("HierarchicalPPContext::transferToGlobalTPDomain: "
                     "Transferring between two global TP domains - this is unusual");
            DeviceId src_device = src_stage.asGlobalTPContext()->localDevice().toLocalDeviceId();
            return transferSingleToSingle(activations, src_device, dst_device, active_bytes);
        }
        else
        {
            // Single device → Global TP
            DeviceId src_device = src_stage.device().toLocalDeviceId();
            return transferSingleToSingle(activations, src_device, dst_device, active_bytes);
        }
    }

    bool HierarchicalPPContext::transferToTPDomain(TensorBase *activations,
                                                   const PPStage &src_stage,
                                                   const PPStage &dst_stage,
                                                   size_t active_bytes)
    {
        ILocalTPContext *tp_ctx = dst_stage.asTPContext();
        if (!tp_ctx)
        {
            LOG_ERROR("HierarchicalPPContext::transferToTPDomain: dst_stage is not a TP domain");
            return false;
        }

        DeviceId src_device = src_stage.representativeDevice().toLocalDeviceId();
        const int tp_degree = tp_ctx->degree();

        // Single-device TP domain: simple transfer
        if (tp_degree == 1)
        {
            DeviceId dst_device = tp_ctx->deviceAt(0).toLocalDeviceId();
            LOG_DEBUG("HierarchicalPPContext::transferToTPDomain: "
                      << src_device.toString() << " → single-device TP domain ("
                      << dst_device.toString() << ")");
            return transferSingleToSingle(activations, src_device, dst_device, active_bytes);
        }

        // Multi-device TP domain: find best initial destination (same vendor as source)
        // then broadcast to all other devices.

        // Find same-vendor device in TP domain if possible
        const bool src_is_cuda = src_device.is_cuda();
        const bool src_is_rocm = src_device.is_rocm();
        const bool src_is_gpu = src_is_cuda || src_is_rocm;

        int initial_dst_index = 0; // Default to representative

        if (src_is_gpu)
        {
            // Search for a same-vendor device in the TP domain
            for (int i = 0; i < tp_degree; ++i)
            {
                DeviceId candidate = tp_ctx->deviceAt(i).toLocalDeviceId();
                if ((src_is_cuda && candidate.is_cuda()) ||
                    (src_is_rocm && candidate.is_rocm()))
                {
                    initial_dst_index = i;
                    LOG_DEBUG("HierarchicalPPContext::transferToTPDomain: "
                              << "selected same-vendor device " << candidate.toString()
                              << " (index " << i << ") as initial destination");
                    break;
                }
            }
        }

        DeviceId initial_dst_device = tp_ctx->deviceAt(initial_dst_index).toLocalDeviceId();

        LOG_DEBUG("HierarchicalPPContext::transferToTPDomain: "
                  << src_device.toString() << " → TP domain (" << tp_degree << " devices), "
                  << "initial transfer to " << initial_dst_device.toString()
                  << " (index " << initial_dst_index << ")");

        // Step 1: Transfer to initial destination device
        if (!transferSingleToSingle(activations, src_device, initial_dst_device, active_bytes))
        {
            LOG_ERROR("HierarchicalPPContext::transferToTPDomain: "
                      "Failed to transfer to initial destination "
                      << initial_dst_device.toString());
            return false;
        }

        // Step 2: Broadcast from initial destination to all other devices in TP domain
        LOG_DEBUG("HierarchicalPPContext::transferToTPDomain: "
                  << "Broadcasting from device " << initial_dst_index << " to all "
                  << tp_degree << " TP devices");

        if (!tp_ctx->broadcast(activations, initial_dst_index))
        {
            LOG_ERROR("HierarchicalPPContext::transferToTPDomain: "
                      "Failed to broadcast to TP domain");
            return false;
        }

        LOG_DEBUG("HierarchicalPPContext::transferToTPDomain: "
                  << "Transfer + broadcast to TP domain complete");
        return true;
    }

    bool HierarchicalPPContext::transferSingleToSingle(TensorBase *activations,
                                                       const DeviceId &src_device,
                                                       const DeviceId &dst_device,
                                                       size_t active_bytes)
    {
        // Same device - no-op
        if (src_device == dst_device)
        {
            LOG_DEBUG("HierarchicalPPContext::transferSingleToSingle: same device, no-op");
            return true;
        }

        // Check if CPU is involved - use coherence model instead of transferTo()
        if (src_device.is_cpu() || dst_device.is_cpu())
        {
            LOG_DEBUG("HierarchicalPPContext::transferSingleToSingle: CPU involved, using coherence model");

            if (dst_device.is_cpu())
            {
                // GPU → CPU: ensure data on source, then sync to host
                if (!activations->ensureOnDevice(src_device))
                {
                    LOG_ERROR("HierarchicalPPContext::transferSingleToSingle: "
                              "Failed to ensure on source device "
                              << src_device.toString());
                    return false;
                }
                activations->mark_device_dirty();

                // Force sync to host via data() - this sets authoritative_device_ = nullopt
                (void)activations->data();

                LOG_DEBUG("HierarchicalPPContext::transferSingleToSingle: "
                          << src_device.toString() << " → CPU complete");
                return true;
            }
            else
            {
                // CPU → GPU: ensure we have host data, then upload
                (void)activations->data(); // Ensure host has data

                if (!activations->ensureOnDevice(dst_device))
                {
                    LOG_ERROR("HierarchicalPPContext::transferSingleToSingle: "
                              "Failed to upload to "
                              << dst_device.toString());
                    return false;
                }
                activations->mark_device_dirty();

                LOG_DEBUG("HierarchicalPPContext::transferSingleToSingle: "
                          << "CPU → " << dst_device.toString() << " complete");
                return true;
            }
        }

        // GPU → GPU: Ensure tensor is on source device
        if (!activations->isDeviceAuthoritative(src_device))
        {
            if (!activations->ensureOnDevice(src_device))
            {
                LOG_ERROR("HierarchicalPPContext::transferSingleToSingle: "
                          "Failed to ensure on source device "
                          << src_device.toString());
                return false;
            }
            activations->mark_device_dirty();
        }

        // Cross-vendor GPU transfer (CUDA↔ROCm): use host bounce if tensor
        // is not BAR-backed. PCIeBAR requires allocateInBarRegion(); normal
        // device memory won't work for cross-vendor direct copies.
        const bool is_cross_vendor =
            (src_device.is_cuda() && dst_device.is_rocm()) ||
            (src_device.is_rocm() && dst_device.is_cuda());

        if (is_cross_vendor && !activations->isBARBacked())
        {
            // GPU → host (data() triggers D2H sync for device-dirty tensors)
            const void *host_ptr = activations->data();
            if (!host_ptr)
            {
                LOG_ERROR("HierarchicalPPContext::transferSingleToSingle: "
                          "Failed to sync to host from "
                          << src_device.toString()
                          << " for cross-vendor bounce");
                return false;
            }

            // Invalidate old GPU data — host is now authoritative
            activations->invalidateGpuData();

            // Host → destination GPU
            if (!activations->ensureOnDevice(dst_device))
            {
                LOG_ERROR("HierarchicalPPContext::transferSingleToSingle: "
                          "Failed to upload to "
                          << dst_device.toString()
                          << " after cross-vendor host bounce");
                return false;
            }
            activations->mark_device_dirty();

            LOG_DEBUG("HierarchicalPPContext::transferSingleToSingle: "
                      "Cross-vendor host bounce "
                      << src_device.toString() << " → CPU → " << dst_device.toString()
                      << " (" << activations->numel() << " elements)");
            return true;
        }

        // Same-vendor GPU-to-GPU or BAR-backed cross-vendor: use direct transfer
        if (!activations->transferTo(dst_device, active_bytes))
        {
            LOG_ERROR("HierarchicalPPContext::transferSingleToSingle: transferTo() failed "
                      << src_device.toString() << " → " << dst_device.toString());
            return false;
        }

        LOG_DEBUG("HierarchicalPPContext::transferSingleToSingle: "
                  << src_device.toString() << " → " << dst_device.toString()
                  << " (" << activations->numel() << " elements)");
        return true;
    }

    bool HierarchicalPPContext::transferAsync(TensorBase *activations, int stage_from, int stage_to, void *stream)
    {
        // For now, delegate to synchronous transfer
        (void)stream;
        return transfer(activations, stage_from, stage_to, 0);
    }

    void HierarchicalPPContext::synchronize()
    {
        // Synchronize all TP domains (local and global)
        for (const auto &stage : config_.stages)
        {
            if (stage.isTPDomain())
            {
                ILocalTPContext *tp_ctx = stage.asTPContext();
                if (tp_ctx)
                {
                    tp_ctx->synchronize();
                }
            }
            else if (stage.isGlobalTPDomain())
            {
                IGlobalTPContext *global_tp_ctx = stage.asGlobalTPContext();
                if (global_tp_ctx)
                {
                    // Global TP barrier ensures all ranks have completed
                    global_tp_ctx->barrier();
                }
            }
            else if (stage.isSingleDevice())
            {
                // Non-TP stages: synchronize the device directly via backend
                DeviceId device = stage.representativeDevice().toLocalDeviceId();
                if (device.is_cuda())
                {
#ifdef HAVE_CUDA
                    (void)cudaSetDevice(device.ordinal);
                    (void)cudaDeviceSynchronize();
#endif
                }
                else if (device.is_rocm())
                {
#ifdef HAVE_ROCM
                    auto *rocm_backend = getROCmBackend();
                    if (rocm_backend)
                    {
                        rocm_backend->synchronize(device.ordinal);
                    }
#endif
                }
                // CPU stages don't need synchronization
            }
            else if (stage.isNestedPP())
            {
                // Nested PP: recursively synchronize
                ILocalPPContext *nested_pp = stage.asNestedPP();
                if (nested_pp)
                {
                    nested_pp->synchronize();
                }
            }
        }
    }

    void HierarchicalPPContext::synchronizeStream(void *stream)
    {
        (void)stream;
        synchronize();
    }

    bool HierarchicalPPContext::reserveStagingBufferBytes(size_t bytes)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (staging_buffer_.size() < bytes)
        {
            staging_buffer_.resize(bytes);
            LOG_DEBUG("HierarchicalPPContext: Reserved " << bytes << " bytes for staging");
        }
        return true;
    }

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<ILocalPPContext> createLocalPPContext(const LocalPPConfig &config)
    {
        return std::make_unique<LocalPPContext>(config);
    }

    std::unique_ptr<ILocalPPContext> createLocalPPContext(const HierarchicalPPConfig &config)
    {
        return std::make_unique<HierarchicalPPContext>(config);
    }

} // namespace llaminar2
