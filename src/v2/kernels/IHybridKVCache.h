/**
 * @file IHybridKVCache.h
 * @brief Interface for hybrid KV caches that combine FA and GDN layer state
 *
 * Extends IKVCache with GDN state access methods. CPU caches expose their
 * host-owned live vectors; GPU caches expose kernel resources while keeping
 * all live recurrent state private to device-resident kernel banks.
 *
 *   auto* hybrid = dynamic_cast<IHybridKVCache*>(kv_cache);
 *   if (hybrid && hybrid->isGDNLayer(layer_idx)) {
 *       float* conv_state = hybrid->getConvState(layer_idx); // CPU only
 *       ITensorShortConvolution* kernel = hybrid->getConvKernel(layer_idx);
 *       ...
 *   }
 */

#pragma once

#include "kernels/common/GDNRecurrenceStateLayout.h"
#include "kernels/CPUGDNStateLayout.h"
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    class ITensorShortConvolution;
    class ITensorGatedDeltaNet;
    struct HybridGDNLayerState;

    /** @brief Durable wire records and process-local rollback are distinct explicit programs. */
    enum class HybridPrefixStatePurpose { DurablePrefix, LocalCheckpoint };

    /** @brief Immutable cache-owned physical section identity, independent of payload contents. */
    struct HybridPrefixStateMetadata
    {
        int total_layers = 0;
        int gdn_layers = 0;
        size_t host_bytes = 0;
        size_t device_bytes = 0;
        bool has_device_kernel_state = false;
        GDNRecurrenceStateLayout host_recurrence_layout =
            GDNRecurrenceStateLayout::KeyMajorValueContiguous; ///< Host section physical format.
        GDNRecurrenceStateLayout device_recurrence_layout =
            GDNRecurrenceStateLayout::KeyMajorValueContiguous; ///< Device section physical format.
        HybridPrefixHostEncoding host_encoding = HybridPrefixHostEncoding::None;
            ///< CPU host recurrence requires explicit PackedV2 durable encoding.
        size_t host_serialized_capacity_bytes = 0;
            ///< Stable maximum wire storage, separate from physical host_bytes.
    };

    /** @brief Authenticated source/destination formats and ordered prefix operation extent. */
    struct HybridPrefixStateDescriptor
    {
        int seq_idx = 0;
        int logical_token_count = 0;
        void *stream = nullptr;
        // Prefix-cache persistence needs completed payloads on return; live rollback
        // checkpoints can use stream ordering and avoid a host-side sync.
        bool synchronize = true;
        bool include_host_state = true;
        bool include_device_state = true;
        GDNRecurrenceStateLayout host_recurrence_layout =
            GDNRecurrenceStateLayout::KeyMajorValueContiguous; ///< Authenticated source/target host
                                                               ///< format.
        GDNRecurrenceStateLayout device_recurrence_layout =
            GDNRecurrenceStateLayout::KeyMajorValueContiguous; ///< Authenticated source/target
                                                               ///< device format.
        HybridPrefixStatePurpose purpose = HybridPrefixStatePurpose::DurablePrefix;
        HybridPrefixHostEncoding host_encoding = HybridPrefixHostEncoding::None;
        size_t host_payload_bytes = 0; ///< Exact bounded host source/destination extent, never capacity.
    };

    /**
     * @brief Authenticate selected physical sections before any prefix payload access.
     * @param metadata Original cache-owned geometry/layout metadata.
     * @param request Immutable declared source/target prefix operation.
     * @return False for unknown or incompatible formats even when byte counts match.
     * @details Export names the requested destination format; import names the
     * authenticated payload format. No conversion or live state is inferred.
     */
    [[nodiscard]] inline bool
    hybridPrefixStateLayoutsMatch(const HybridPrefixStateMetadata &metadata,
                                  const HybridPrefixStateDescriptor &request) noexcept
    {
        const auto known_encoding = [](HybridPrefixHostEncoding encoding) {
            return encoding == HybridPrefixHostEncoding::None ||
                   encoding == HybridPrefixHostEncoding::CPUFixedEpochLocalImage ||
                   encoding == HybridPrefixHostEncoding::CPUFixedEpochPackedV2;
        };
        const bool known_purpose = request.purpose == HybridPrefixStatePurpose::DurablePrefix ||
                                   request.purpose == HybridPrefixStatePurpose::LocalCheckpoint;
        const bool host_format = !request.include_host_state || metadata.host_bytes == 0 ||
            (metadata.host_encoding == HybridPrefixHostEncoding::CPUFixedEpochPackedV2 &&
             ((request.purpose == HybridPrefixStatePurpose::DurablePrefix &&
               request.host_encoding == HybridPrefixHostEncoding::CPUFixedEpochPackedV2) ||
              (request.purpose == HybridPrefixStatePurpose::LocalCheckpoint &&
               request.host_encoding == HybridPrefixHostEncoding::CPUFixedEpochLocalImage)));
        return known_purpose && known_encoding(metadata.host_encoding) &&
               known_encoding(request.host_encoding) && host_format &&
               validGDNRecurrenceStateLayout(metadata.host_recurrence_layout) &&
               validGDNRecurrenceStateLayout(metadata.device_recurrence_layout) &&
               validGDNRecurrenceStateLayout(request.host_recurrence_layout) &&
               validGDNRecurrenceStateLayout(request.device_recurrence_layout) &&
               (!request.include_host_state || !metadata.host_bytes ||
                request.host_recurrence_layout == metadata.host_recurrence_layout) &&
               (!request.include_device_state || !metadata.device_bytes ||
                request.device_recurrence_layout == metadata.device_recurrence_layout);
    }

    /**
     * @brief Extended KV cache interface with GDN state management
     *
     * Implemented by CPUHybridRingKVCache, CUDAHybridRingKVCache,
     * ROCmHybridRingKVCache. Provides per-layer GDN state access
     * for hybrid models (e.g., Qwen 3.5) that mix full-attention
     * and GDN layers.
     */
    class IHybridKVCache
    {
    public:
        virtual ~IHybridKVCache() = default;

        /**
         * @brief Retain the canonical claim for CPU-owned recurrent payloads.
         *
         * KernelFactory acquires this claim before constructing a CPU hybrid
         * cache. Derived recurrent vectors are destroyed before this interface
         * base, so the lease returns bytes only after their storage is gone.
         * GPU live state is owned and claimed by HybridGDNDeviceStateArena and
         * therefore never binds this host-payload lease.
         *
         * @param lease Non-null type-erased PhysicalMemoryAllocationLease.
         * @throws std::invalid_argument for a null lease.
         * @throws std::logic_error if the cache was already bound.
         */
        void bindRecurrentLiveMemoryLease(std::shared_ptr<void> lease)
        {
            if (!lease)
            {
                throw std::invalid_argument(
                    "Hybrid KV cache recurrent-state lease cannot be null");
            }
            if (recurrent_live_memory_lease_)
            {
                throw std::logic_error(
                    "Hybrid KV cache recurrent-state lease cannot be rebound");
            }
            recurrent_live_memory_lease_ = std::move(lease);
        }

        /** @return Whether CPU recurrent storage has an admitted live claim. */
        [[nodiscard]] bool hasRecurrentLiveMemoryLease() const noexcept
        {
            return recurrent_live_memory_lease_ != nullptr;
        }

        // =====================================================================
        // Layer Type Queries
        // =====================================================================

        /// Check if a layer is a GDN (linear attention) layer
        virtual bool isGDNLayer(int layer) const = 0;

        /// Check if a layer is a full-attention layer
        virtual bool isFullAttentionLayer(int layer) const = 0;

        /// Number of KV cache (full-attention) layers
        virtual int kvLayerCount() const = 0;

        /// Number of GDN layers
        virtual int gdnLayerCount() const = 0;

        // =====================================================================
        // GDN State Access
        // =====================================================================

        /// Get full GDN state for a layer (nullptr if FA layer)
        virtual HybridGDNLayerState *getGDNState(int layer) = 0;
        virtual const HybridGDNLayerState *getGDNState(int layer) const = 0;

        /// Get CPU-owned complete phase/checkpoint/history state (CPUGDNStateLayout).
        /// Returns nullptr for FA layers and every GPU cache.
        virtual float *getRecurrenceState(int layer) = 0;

        /// Get CPU-owned conv state [qkv_dim, conv_kernel-1].
        /// Returns nullptr for FA layers and every GPU cache.
        virtual float *getConvState(int layer) = 0;

        // =====================================================================
        // GDN Kernel Access
        // =====================================================================

        /// Get short convolution kernel for a GDN layer (nullptr if FA)
        virtual ITensorShortConvolution *getConvKernel(int layer) = 0;

        /// Get gated delta net recurrence kernel for a GDN layer (nullptr if FA)
        virtual ITensorGatedDeltaNet *getRecurrenceKernel(int layer) = 0;

        // =====================================================================
        // GDN Lifecycle
        // =====================================================================

        /// Reset all GDN states to zero (for new sequence)
        virtual void resetGDNStates() = 0;

        /// Total GDN state memory in bytes
        virtual size_t gdnMemoryBytes() const = 0;

        // =====================================================================
        // Prefix/Rollback State
        // =====================================================================

        virtual HybridPrefixStateMetadata hybridPrefixStateMetadata() const = 0;

        /**
         * @brief Query one exact ordered host payload extent without allocating storage.
         * @param desc Explicit durable/local purpose, format and request slot; byte bound is not yet authored.
         * @return Exact bytes, or no value when the declared program/source is invalid.
         * @details Dense device owners retain their existing zero-host format. CPU
         * derives live durable bytes from its original phases; immutable metadata
         * remains a phase-independent capacity/identity authority.
         */
        virtual std::optional<size_t> hybridPrefixHostPayloadBytes(
            const HybridPrefixStateDescriptor &desc) const
        {
            const auto metadata = hybridPrefixStateMetadata();
            if (!hybridPrefixStateLayoutsMatch(metadata, desc)) return std::nullopt;
            return desc.include_host_state ? metadata.host_bytes : 0;
        }

        /**
         * @brief Export the explicitly selected bounded durable/local state representation.
         * @param desc Authenticated purpose, encoding, exact host byte extent and stream/order.
         * @param dst_host Host destination with the declared extent, absent for device-only state.
         * @param dst_device Original native device destination, unchanged by CPU host encoding.
         * @return False before host mutation for incompatible format, extent, phase or overlap.
         */
        virtual bool exportHybridPrefixState(
            const HybridPrefixStateDescriptor &desc,
            void *dst_host,
            void *dst_device) const = 0;

        /**
         * @brief Restore an exact declared state representation through its original owner.
         * @param desc Authenticated source purpose/encoding and bounded host extent.
         * @param src_host Immutable source retained until this ordered operation completes.
         * @param src_device Original native device source on the declared stream.
         * @return False before any CPU host mutation when complete durable validation fails.
         */
        virtual bool importHybridPrefixState(
            const HybridPrefixStateDescriptor &desc,
            const void *src_host,
            const void *src_device) = 0;

    private:
        std::shared_ptr<void> recurrent_live_memory_lease_;
    };

} // namespace llaminar2
