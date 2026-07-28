/**
 * @file IKVCache.h
 * @brief Unified KV cache interface for CPU and GPU implementations
 *
 * This interface abstracts the common operations between CPU (CPUKVCache)
 * and GPU (CUDARingKVCache) implementations, allowing stages to work with
 * either cache type through a single pointer.
 */

#pragma once

#include "../execution/config/RuntimeConfig.h" // ActivationPrecision
#include "../tensors/ITensor.h"                // Lightweight interface (no MPI)
#include "../tensors/TensorLayout.h"
#include "../utils/Logger.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    class TurboQuantContext; // Forward declaration for KVReadParams

    /**
     * @brief Unified interface for KV cache implementations
     *
     * Both CPU (ICPUKVCache) and GPU (ICUDARingKVCache) caches inherit
     * from this interface. Stages that only need to query cache state or
     * append data can use IKVCache* without knowing the underlying implementation.
     */
    class IKVCache
    {
    public:
        virtual ~IKVCache() = default;

        /**
         * @brief Semantic owner of one cache instance's mutable sequence state.
         *
         * A main-model cache and an MTP sidecar can have identical tensor
         * geometry while living at different sequence offsets and crossing
         * different lifecycle boundaries.  Recording this role on the cache
         * makes that distinction inspectable and prevents orchestration code
         * from treating every IKVCache as interchangeable request storage.
         */
        enum class StateRole : uint8_t
        {
            Standalone,       ///< Directly constructed cache used outside an orchestrator.
            CommittedMain,    ///< Canonical committed KV/GDN state for the live request.
            PipelineShard,    ///< Committed state owned by one pipeline stage/device.
            MTPShiftedSidecar ///< Speculative cache shifted by one MTP depth.
        };

        /**
         * @brief Immutable lifecycle identity assigned after cache construction.
         */
        struct StateOwnership
        {
            StateRole role = StateRole::Standalone;
            int mtp_depth = -1;

            bool valid() const
            {
                return role == StateRole::MTPShiftedSidecar
                           ? mtp_depth >= 0
                           : mtp_depth == -1;
            }

            bool operator==(const StateOwnership &other) const
            {
                return role == other.role && mtp_depth == other.mtp_depth;
            }
        };

        /**
         * @brief Semantic boundary that makes previously cached rows unreachable.
         *
         * This is deliberately more precise than the historical word "clear".
         * Payload allocations remain alive at every boundary.  Only logical
         * visibility, graph append bindings, and cache-owned recurrent state
         * cross the boundary.
         */
        enum class StateResetBoundary : uint8_t
        {
            RequestBoundary,   ///< Start a new independent prompt/session.
            PrefixReplacement, ///< Replace live state with a promoted prefix snapshot.
            SequenceRetirement, ///< Retire one request slot while siblings remain live.
            TestReinitialization ///< Explicit test-fixture reuse; never a serving fallback.
        };

        /**
         * @brief Caller-owned ordering context for a cache-state reset.
         *
         * GPU implementations require @ref execution_stream and enqueue every
         * metadata/GDN/short-conv mutation on exactly that stream.  They must
         * never select a default stream or synchronize internally.  The caller
         * publishes one event after all participating state owners have reset.
         *
         * CPU implementations require a null stream because their mutations
         * complete synchronously on the calling thread.
         */
        struct StateResetContext
        {
            StateResetBoundary boundary = StateResetBoundary::RequestBoundary;
            void *execution_stream = nullptr;
            const char *reason = nullptr;

            bool hasReason() const
            {
                return reason != nullptr && reason[0] != '\0';
            }

            /**
             * @brief Whether this boundary may retire every live cache row.
             */
            bool permitsRequestReset() const
            {
                return boundary == StateResetBoundary::RequestBoundary ||
                       boundary == StateResetBoundary::PrefixReplacement ||
                       boundary == StateResetBoundary::TestReinitialization;
            }

            /**
             * @brief Whether this boundary may retire one request slot.
             */
            bool permitsSequenceReset() const
            {
                return boundary == StateResetBoundary::SequenceRetirement ||
                       boundary == StateResetBoundary::TestReinitialization;
            }

            /**
             * @brief Whether this boundary may reset one layer/request pair.
             */
            bool permitsLayerSequenceReset() const
            {
                return boundary == StateResetBoundary::PrefixReplacement ||
                       boundary == StateResetBoundary::SequenceRetirement ||
                       boundary == StateResetBoundary::TestReinitialization;
            }

            /**
             * @brief Whether this boundary may reset one complete layer.
             */
            bool permitsLayerReset() const
            {
                return boundary == StateResetBoundary::PrefixReplacement ||
                       boundary == StateResetBoundary::TestReinitialization;
            }

            /**
             * @brief Build the explicit fixture-reuse boundary used by tests.
             *
             * GPU integration tests pass their exact test stream. CPU unit
             * tests pass nullptr. This factory keeps test intent readable
             * without weakening production stream requirements.
             */
            static StateResetContext testReinitialization(
                void *execution_stream,
                const char *reason = "test-reinitialization")
            {
                return {
                    .boundary = StateResetBoundary::TestReinitialization,
                    .execution_stream = execution_stream,
                    .reason = reason,
                };
            }
        };

        /**
         * @brief Bind this cache to one immutable orchestration lifetime.
         *
         * Binding is idempotent only for the exact same identity.  Conflicting
         * rebinding is a construction bug and throws rather than silently
         * changing whether a cache represents committed or speculative state.
         *
         * @param ownership Immutable role and optional MTP depth.
         * @throws std::invalid_argument for an internally inconsistent identity.
         * @throws std::logic_error when an already-bound cache is rebound.
         */
        void bindStateOwnership(const StateOwnership &ownership)
        {
            if (!ownership.valid())
            {
                throw std::invalid_argument(
                    "IKVCache state ownership has inconsistent role/depth");
            }
            if (state_ownership_bound_ && !(state_ownership_ == ownership))
            {
                throw std::logic_error(
                    "IKVCache state ownership cannot change after binding");
            }
            state_ownership_ = ownership;
            state_ownership_bound_ = true;
        }

        /**
         * @brief Return the cache's lifecycle identity.
         *
         * Direct factory users remain Standalone until an orchestrator binds a
         * stronger role.  Production orchestrators bind every owned cache
         * immediately after successful construction.
         */
        const StateOwnership &stateOwnership() const
        {
            return state_ownership_;
        }

        /**
         * @brief Descriptor for copying a logical KV block in oldest-to-newest order.
         *
         * The token range is logical within the sequence, not a physical ring row
         * range. Implementations may remap global_layer through first_layer_index().
         *
         * GPU implementations require @ref stream to be a non-null explicit
         * stream for any import or non-empty export. The caller, not the cache,
         * owns ordering that stream after graph replay, prefix restore/truncate,
         * accepted MTP publication, and other live inference-state producers.
         * Falling back to a default stream would hide lifetime bugs at the
         * request boundary, so GPU caches must fail clearly when a streamful
         * access is required but missing.
         */
        /**
         * @brief Memory domain that owns a logical KV block payload.
         *
         * The domain is explicit because a GPU pointer must never be guessed
         * from the cache implementation or from the current tensor coherence
         * flags.  Prefix-cache execution uses @ref Device so harvest and
         * restore remain device-to-device and stream ordered.  @ref Host is
         * reserved for CPU caches and explicit diagnostic publication.
         */
        enum class KVCacheLogicalBlockPayloadDomain : uint8_t
        {
            Host,
            Device,
        };

        /**
         * @brief Describe one logical KV block transfer.
         *
         * GPU device-domain transfers are asynchronous.  The caller owns
         * readiness through @p stream and must record or consume an event
         * before reusing either payload.  A GPU implementation must reject a
         * null stream rather than silently entering the CUDA/HIP default
         * stream.
         */
        struct KVCacheLogicalBlockDescriptor
        {
            int layer = 0;
            int seq_idx = 0;
            int logical_token_start = 0;
            int token_count = 0;
            void *stream = nullptr;
            KVCacheLogicalBlockPayloadDomain payload_domain =
                KVCacheLogicalBlockPayloadDomain::Host;
        };

        /**
         * @brief Packed byte layout for a logical KV block payload.
         *
         * Payloads preserve native K/V precision and KV cache layout. For
         * KV_POS_HEAD_DIM, rows are packed by token. For KV_HEAD_POS_DIM, rows
         * are packed by head, then logical token, then head dimension.
         */
        struct KVCacheLogicalBlockLayout
        {
            ActivationPrecision k_precision = ActivationPrecision::FP32;
            ActivationPrecision v_precision = ActivationPrecision::FP32;
            TensorLayout layout = TensorLayout::UNKNOWN;
            int local_kv_heads = 0;
            int kv_head_start = 0;
            int head_dim = 0;
            size_t k_bytes = 0;
            size_t v_bytes = 0;
            bool device_resident = false;
        };

        /**
         * @brief Logical sequence state for cache import/export planning.
         */
        struct KVCacheSequenceState
        {
            int cached_tokens = 0;
            int implementation_head = 0;
            bool wrapped = false;
        };

        /**
         * @brief Device-resident sequence-state publication request.
         *
         * vLLM-style MTP publication derives accepted verifier rows and target
         * cache lengths in GPU memory.  A cache implementation may consume this
         * request to update its device-side sequence metadata without first
         * copying accepted counts back to the CPU.
         *
         * Long-context ring caches must treat @p target_cached_tokens_device as
         * the target valid-token count.  Publication must preserve the live
         * ring tail and move the head to `tail + target_cached_tokens`, matching
         * truncateSequence() semantics after a verifier graph has written
         * temporary rows.  @p accepted_state_counts_device records how many
         * verifier rows were accepted for validation/accounting; it is not a
         * reliable source of ring-head position because the live device head may
         * already include rejected verifier rows.
         *
         * The stream is mandatory for GPU implementations.  Work enqueued by
         * this call must be ordered after the verifier outcome producer and
         * before any graph replay that consumes the updated KV state.
         */
        struct DeviceSequenceStatePublicationRequest
        {
            int request_count = 0;
            int first_seq_idx = 0;
            const int32_t *target_cached_tokens_device = nullptr;
            const int32_t *accepted_state_counts_device = nullptr;
            const int32_t *publication_ok_flags_device = nullptr;
            void *stream = nullptr;

            bool valid() const
            {
                return request_count > 0 &&
                       first_seq_idx >= 0 &&
                       target_cached_tokens_device != nullptr &&
                       accepted_state_counts_device != nullptr &&
                       publication_ok_flags_device != nullptr &&
                       stream != nullptr;
            }
        };

        // =================================================================
        // Query Operations
        // =================================================================

        /**
         * @brief Get the precision/data type of cached K tensors
         * @return ActivationPrecision (FP32, BF16, FP16, Q8_1, TQ8, etc.)
         */
        virtual ActivationPrecision k_precision() const = 0;

        /**
         * @brief Get the precision/data type of cached V tensors
         *
         * Defaults to k_precision() for symmetric caches. Override for
         * asymmetric K/V storage (e.g., TQ8 for K, TQ4 for V).
         *
         * @return ActivationPrecision (FP32, BF16, FP16, Q8_1, TQ4, etc.)
         */
        virtual ActivationPrecision v_precision() const { return k_precision(); }

        /**
         * @brief Get the precision of cached K/V tensors (deprecated)
         *
         * @deprecated Use k_precision() and v_precision() instead.
         *             This returns k_precision() and is incorrect for asymmetric caches.
         * @return ActivationPrecision of K cache
         */
        [[deprecated("Use k_precision() and v_precision() instead")]]
        ActivationPrecision precision() const
        {
            return k_precision();
        }

        /**
         * @brief Get number of cached tokens for a layer/sequence
         * @param layer Layer index (global, will be remapped if first_layer_index > 0)
         * @param seq_idx Sequence index (default 0 for single-sequence)
         * @return Number of tokens currently cached
         */
        virtual int get_cached_tokens(int layer, int seq_idx = 0) const = 0;

        /**
         * @brief Get maximum sequence length
         */
        virtual int max_seq_len() const = 0;

        /**
         * @brief Get number of layers
         */
        virtual int n_layers() const = 0;

        /**
         * @brief Get the first layer index for this cache (Pipeline Parallelism)
         *
         * When > 0, this cache handles layers [first_layer_index, first_layer_index + n_layers()).
         * Incoming global layer indices are remapped: local_idx = global_idx - first_layer_index.
         *
         * @return First layer index (0 for non-PP caches)
         */
        virtual int first_layer_index() const { return 0; }

        /**
         * @brief Remap a global layer index to a local cache index
         *
         * Handles Pipeline Parallelism where each device's cache covers a subset of layers.
         * For example, PP stage 1 with layers [12, 23] uses cache with first_layer_index=12.
         * Global layer 15 maps to local cache index 3.
         *
         * @param global_layer The global layer index (0-based across all layers)
         * @return Local cache index (0-based within this cache), or -1 if out of range
         */
        int remapLayerIndex(int global_layer) const
        {
            int local_idx = global_layer - first_layer_index();
            if (local_idx < 0 || local_idx >= n_layers())
            {
                return -1; // Out of range for this cache
            }
            return local_idx;
        }

        /**
         * @brief Get KV cache tensor layout
         *
         * @return TensorLayout indicating memory ordering
         *         Default is KV_POS_HEAD_DIM (position-major)
         */
        virtual TensorLayout kv_layout() const { return TensorLayout::KV_POS_HEAD_DIM; }

        /**
         * @brief Inspect the current physical ring head for diagnostics.
         *
         * For ring caches this is the physical row that represents the oldest
         * valid token. Non-ring implementations may return 0.
         */
        virtual int ring_head(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return 0;
        }

        /**
         * @brief Describe the byte layout for a packed logical KV block.
         */
        virtual KVCacheLogicalBlockLayout logicalBlockLayout(int global_layer, int token_count) const
        {
            (void)global_layer;
            (void)token_count;
            return {};
        }

        /**
         * @brief Inspect logical sequence state for a global layer/sequence.
         */
        virtual KVCacheSequenceState sequenceState(int global_layer, int seq_idx) const
        {
            (void)global_layer;
            (void)seq_idx;
            return {};
        }

        /**
         * @brief Export a logical KV block into packed native-precision buffers.
         *
         * Exported floating payloads use the canonical logical-block
         * serialization defined by the KV cache codec: positive and negative
         * zero are normalized to positive zero while every non-zero payload bit
         * is preserved.  Prefix caches and diagnostics must compare this
         * canonical byte representation, not incidental backend scratch bytes.
         *
         * CPU caches accept host payloads and copy synchronously. GPU caches
         * support two explicit modes:
         *
         * - `Host`: diagnostic D2H publication followed by an explicit-stream
         *   wait before returning.
         * - `Device`: asynchronous D2D publication whose gather kernel reads
         *   the canonical ring head/count rows on device. No host sequence
         *   mirror, D2H copy, or host synchronization is permitted.
         *
         * Callers must first order the stream through the live inference-state
         * observation boundary so either export cannot race graph-captured KV
         * writers.
         */
        virtual bool exportLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, void *dst_k, void *dst_v) const
        {
            (void)desc;
            (void)dst_k;
            (void)dst_v;
            return false;
        }

        /**
         * @brief Import a packed logical KV block into this cache.
         *
         * CPU caches accept host payloads. GPU host-domain imports are explicit
         * diagnostic/compatibility publication. GPU device-domain imports are
         * asynchronous D2D scatter operations that validate and advance the
         * canonical ring metadata entirely on device. Prefix restore and
         * related mutation paths must pass a stream that has already waited for
         * any live-state producers they are about to overwrite.
         */
        virtual bool importLogicalBlock(const KVCacheLogicalBlockDescriptor &desc, const void *src_k, const void *src_v)
        {
            (void)desc;
            (void)src_k;
            (void)src_v;
            return false;
        }

        /**
         * @brief Truncate every layer for one sequence to cached_tokens.
         */
        virtual bool truncateSequence(int seq_idx, int cached_tokens, void *stream = nullptr)
        {
            (void)seq_idx;
            (void)cached_tokens;
            (void)stream;
            return false;
        }

        /**
         * @brief Return the bytes required to checkpoint one sequence's live metadata.
         *
         * GPU ring caches use this opaque payload to preserve every canonical
         * head/count row without publishing those values to the host. The
         * payload is backend-private: callers may retain and return the bytes
         * to the same cache implementation, but must never inspect or modify
         * them. A zero result means device-resident checkpointing is not
         * implemented and is a hard capability failure for GPU MTP rollback.
         */
        virtual size_t deviceSequenceStateCheckpointBytes() const
        {
            return 0;
        }

        /**
         * @brief Capture one sequence's canonical metadata into device memory.
         *
         * Implementations enqueue device-to-device work on @p stream. They
         * must not allocate, synchronize, copy through host memory, or retain
         * a host mirror. The destination must contain at least
         * deviceSequenceStateCheckpointBytes() bytes and remain alive until an
         * event recorded after this call has completed.
         *
         * @param seq_idx Sequence whose canonical state is captured.
         * @param checkpoint_device Opaque device allocation owned by the caller.
         * @param checkpoint_bytes Size of @p checkpoint_device.
         * @param stream Explicit backend stream ordered after all state writers.
         * @param error Optional diagnostic populated on failure.
         */
        virtual bool captureDeviceSequenceStateCheckpoint(
            int seq_idx,
            void *checkpoint_device,
            size_t checkpoint_bytes,
            void *stream,
            std::string *error = nullptr) const
        {
            (void)seq_idx;
            (void)checkpoint_device;
            (void)checkpoint_bytes;
            (void)stream;
            if (error)
            {
                *error =
                    "KV cache does not implement device-resident sequence-state checkpointing";
            }
            return false;
        }

        /**
         * @brief Restore canonical metadata from an opaque device checkpoint.
         *
         * Implementations enqueue only device work on @p stream and restore
         * the exact ring heads/counts captured earlier. This is deliberately
         * different from truncateSequence(): speculative rows remain in
         * payload storage but become unreachable when the canonical metadata
         * is restored, preserving serial-decode state without a host-observed
         * count or a row replay.
         *
         * @param seq_idx Sequence whose canonical state is replaced.
         * @param checkpoint_device Opaque device checkpoint from this cache.
         * @param checkpoint_bytes Size of @p checkpoint_device.
         * @param stream Explicit backend stream ordered after checkpoint readiness.
         * @param error Optional diagnostic populated on failure.
         */
        virtual bool restoreDeviceSequenceStateCheckpoint(
            int seq_idx,
            const void *checkpoint_device,
            size_t checkpoint_bytes,
            void *stream,
            std::string *error = nullptr)
        {
            (void)seq_idx;
            (void)checkpoint_device;
            (void)checkpoint_bytes;
            (void)stream;
            if (error)
            {
                *error =
                    "KV cache does not implement device-resident sequence-state restoration";
            }
            return false;
        }

        /**
         * @brief Device pointer to the canonical live cached-token count.
         *
         * GPU graph-captured stages can use this pointer once they are taught
         * to compute dynamic attention/KV parameters on device.  The pointer is
         * GPU implementations expose their authoritative device allocation;
         * callers must not copy it into persistent host bookkeeping.
         */
        virtual const int *deviceCachedTokenCountPtr(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        /**
         * @brief Device pointer to the canonical live cached-token count for a request.
         *
         * Verifier publication needs the cache length before the verifier appends
         * target rows.  GPU caches expose that value through per-layer device
         * rows; hybrid models may not have attention state at model layer zero.
         * This helper returns the first available per-layer count so callers can
         * consume a device-owned sequence length without knowing the model's
         * attention/GDN layout.
         */
        virtual const int *deviceSequenceCachedTokenCountPtr(int seq_idx = 0) const
        {
            const int first_layer = first_layer_index();
            const int layer_count = n_layers();
            for (int layer = first_layer; layer < first_layer + layer_count; ++layer)
            {
                if (const int *ptr = deviceCachedTokenCountPtr(layer, seq_idx))
                    return ptr;
            }
            return nullptr;
        }

        /**
         * @brief Device pointer to the canonical live ring head.
         *
         * Wrapped ring-cache publication requires both the target cached-token
         * count and the current ring head.  This accessor gives future Phase 10
         * kernels a backend-neutral way to consume that head without reaching
         * into CUDA/HIP cache internals.
         */
        virtual const int *deviceRingHeadPtr(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        /**
         * @brief Whether this cache can publish sequence metadata from device buffers.
         *
         * Returning true means the cache can consume
         * DeviceSequenceStatePublicationRequest without a D2H sync and future
         * graph-dynamic KV consumers will observe the published state from
         * device-owned metadata.  Implementations must not return true if they
         * only update an auxiliary device buffer while get_cached_tokens(),
         * attention dynamic params, or graph signatures still depend on stale
         * host counters.
         */
        virtual bool supportsDeviceResidentSequenceStatePublication() const
        {
            return false;
        }

        /**
         * @brief Publish accepted verifier cache counts from device metadata.
         *
         * The default hard-fails so callers cannot accidentally hide the host
         * synchronization that Phase 10 is trying to remove.  GPU backends
         * should implement this only together with device-readable count/head
         * metadata for subsequent attention and append stages.
         */
        virtual bool publishSequenceStateFromDeviceMetadata(
            const DeviceSequenceStatePublicationRequest &request,
            std::string *error = nullptr)
        {
            (void)request;
            if (error)
            {
                *error =
                    "KV cache does not support device-resident sequence-state publication";
            }
            return false;
        }

        // =================================================================
        // ITensor Access (unified CPU/GPU interface)
        // =================================================================

        /**
         * @brief Get both K and V cache tensors for attention computation
         *
         * This is the preferred interface for accessing KV cache - fetches both
         * K and V in a single call, which aligns with GPU batch operations and
         * enables potential optimizations.
         *
         * For CPU caches: returns TensorBase* pointers (which inherit ITensor)
         * For GPU caches: returns CUDATensorBase* pointers (which inherit ITensor)
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @param out_k Output: pointer to K cache tensor
         * @param out_v Output: pointer to V cache tensor
         * @param out_kv_len Output: number of cached tokens (optional, can be nullptr)
         * @return true on success, false if layer/seq_idx invalid
         */
        virtual bool get_kv(int layer, int seq_idx,
                            ITensor **out_k, ITensor **out_v,
                            int *out_kv_len = nullptr) = 0;

        virtual bool get_kv(int layer, int seq_idx,
                            const ITensor **out_k, const ITensor **out_v,
                            int *out_kv_len = nullptr) const = 0;

        /**
         * @brief Return a graph-snapshot-only direct physical KV view.
         *
         * Normal readers should use get_kv(), which may linearize wrapped ring
         * buffers. This hook is for graph-captured diagnostics that need a
         * stable device pointer before and after an append stage. Implementations
         * must return false when the requested logical token span is not a
         * direct contiguous physical range.
         */
        virtual bool get_kv_snapshot_view(int layer, int seq_idx,
                                          int token_count,
                                          ITensor **out_k, ITensor **out_v,
                                          int *out_kv_len = nullptr)
        {
            (void)layer;
            (void)seq_idx;
            (void)token_count;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            return false;
        }

        virtual bool get_kv_snapshot_view(int layer, int seq_idx,
                                          int token_count,
                                          const ITensor **out_k, const ITensor **out_v,
                                          int *out_kv_len = nullptr) const
        {
            ITensor *k = nullptr;
            ITensor *v = nullptr;
            const bool ok = const_cast<IKVCache *>(this)->get_kv_snapshot_view(
                layer, seq_idx, token_count, &k, &v, out_kv_len);
            if (ok)
            {
                if (out_k)
                    *out_k = k;
                if (out_v)
                    *out_v = v;
            }
            return ok;
        }

        // Convenience overloads for seq_idx=0
        bool get_kv(int layer, ITensor **out_k, ITensor **out_v, int *out_kv_len = nullptr)
        {
            return get_kv(layer, 0, out_k, out_v, out_kv_len);
        }

        bool get_kv(int layer, const ITensor **out_k, const ITensor **out_v, int *out_kv_len = nullptr) const
        {
            return get_kv(layer, 0, out_k, out_v, out_kv_len);
        }

        /**
         * @brief Get K cache tensor as ITensor for a layer/sequence
         *
         * @deprecated Use get_kv() instead for unified access to both K and V.
         *             This method will be removed in a future version.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @return ITensor* to K cache, or nullptr if not available
         */
        virtual ITensor *get_k(int layer, int seq_idx = 0)
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }
        virtual const ITensor *get_k(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        /**
         * @brief Get V cache tensor as ITensor for a layer/sequence
         *
         * @deprecated Use get_kv() instead for unified access to both K and V.
         *             This method will be removed in a future version.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index (default 0)
         * @return ITensor* to V cache, or nullptr if not available
         */
        virtual ITensor *get_v(int layer, int seq_idx = 0)
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }
        virtual const ITensor *get_v(int layer, int seq_idx = 0) const
        {
            (void)layer;
            (void)seq_idx;
            return nullptr;
        }

        // =================================================================
        // Append Operations
        // =================================================================

        /**
         * @brief Append K/V tensors to cache
         *
         * For CPU caches: reads from tensor host data
         * For GPU caches: reads from tensor GPU data (must be on device)
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param K Key tensor to append
         * @param V Value tensor to append
         * @param num_tokens Number of tokens to append
         * @return true on success
         */
        virtual bool append(int layer, int seq_idx, const ITensor *K, const ITensor *V, int num_tokens) = 0;

        // Convenience overload for seq_idx=0
        bool append(int layer, const ITensor *K, const ITensor *V, int num_tokens)
        {
            return append(layer, 0, K, V, num_tokens);
        }

        /**
         * @brief Append MTP verifier rows with decode-equivalent ring semantics.
         *
         * The verifier path computes several future rows at once, but those rows
         * must become KV-cache state exactly as if normal decode had appended
         * them one token at a time.  Backends that implement this method should
         * write all rows in one grouped/concurrent operation and update ring
         * metadata once, preserving the same oldest-to-newest order and
         * wraparound behavior as serial decode.
         *
         * Source tensors may be ordinary position-major rows
         * `[verifier_rows][local_kv_heads * head_dim]` or head-major rows
         * `[local_kv_heads * verifier_rows][head_dim]`.  Implementations must
         * fail closed for unsupported layouts rather than silently falling back
         * to stage-level row replay.
         */
        virtual bool appendVerifierRowsDecodeEquivalent(int layer,
                                                        int seq_idx,
                                                        const ITensor *K,
                                                        const ITensor *V,
                                                        int verifier_rows,
                                                        void *gpu_stream = nullptr)
        {
            (void)layer;
            (void)seq_idx;
            (void)K;
            (void)V;
            (void)verifier_rows;
            (void)gpu_stream;
            return false;
        }

        /**
         * @brief Stream-aware append for GPU graph capture compatibility
         *
         * GPU KV cache implementations override this to dispatch the append
         * kernel on the specified stream instead of the default stream (0).
         * This is critical for GPU graph capture where all operations must
         * execute on the same stream.
         *
         * Implementations that support stream-aware append must override this
         * method. The base implementation fails so GPU callers cannot silently
         * drop to append(), which may use the default stream or skip required
         * format conversion.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param K Key tensor to append
         * @param V Value tensor to append
         * @param num_tokens Number of tokens to append
         * @param gpu_stream Opaque GPU stream pointer (hipStream_t or cudaStream_t)
         * @return true on success
         */
        virtual bool appendWithStream(int layer, int seq_idx, const ITensor *K, const ITensor *V,
                                      int num_tokens, void *gpu_stream)
        {
            (void)layer;
            (void)seq_idx;
            (void)K;
            (void)V;
            (void)num_tokens;
            (void)gpu_stream;
            LOG_ERROR("[IKVCache::appendWithStream] Stream-aware append is not implemented by this cache");
            return false;
        }

        // =================================================================
        // Graph Capture Support
        // =================================================================

        /**
         * @brief Check whether append kernels consume canonical device state.
         *
         * A true result promises that captured append kernels read and advance
         * device-owned head/count allocations directly. It does not advertise a
         * host upload or post-replay adoption mechanism.
         */
        virtual bool isGraphCaptureReady() const { return false; }

        /**
         * @brief Bind a captured append to its device-resident row-count source.
         *
         * Request-batched prefill records one fixed-width append kernel per
         * request so a captured graph can be reused for every prompt that fits
         * the bucket. Each request can nevertheless have a different logical
         * length. GPU caches bind the persistent request-length row directly;
         * exact-shape graphs pass nullptr and use @p captured_max_tokens.
         *
         * This contract deliberately has no host count argument. The source
         * row has already been admitted into persistent device-owned request
         * metadata before graph planning. Implementations must not read, write,
         * or otherwise adopt host sequence metadata here. Canonical ring head
         * and cached-token count remain untouched and are consumed directly by
         * the captured kernels.
         *
         * @param layer Global or cache-local layer index accepted by the cache.
         * @param seq_idx Request index inside the cache batch.
         * @param append_tokens_device Optional device pointer to one positive
         *        int32 count; nullptr selects the exact captured width.
         * @param captured_max_tokens Maximum rows represented by the captured
         *        append kernel for this request.
         * @param gpu_stream Explicit graph stream. No copy is implied.
         * @return true when the backend accepted the immutable graph binding.
         */
        virtual bool bindGraphAppendCountSource(
            int layer,
            int seq_idx,
            const int32_t *append_tokens_device,
            int captured_max_tokens,
            void *gpu_stream)
        {
            (void)layer;
            (void)seq_idx;
            (void)append_tokens_device;
            (void)captured_max_tokens;
            (void)gpu_stream;
            return false;
        }

        // =================================================================
        // Explicit State-Lifetime Reset Operations
        // =================================================================

        /**
         * @brief Reset all layers and request slots at a named lifecycle boundary.
         *
         * Implementations make every previous row unreachable while retaining
         * model-lifetime payload allocations, graph bindings whose addresses
         * remain stable, and pre-bound workspace.  GPU implementations enqueue
         * work on @p context.execution_stream and return without waiting.
         *
         * @return true after every required mutation has been accepted.
         */
        virtual bool resetRequestState(const StateResetContext &context) = 0;

        /**
         * @brief Retire one request slot across every cache layer.
         *
         * @param seq_idx Request slot whose rows become unreachable.
         * @param context Explicit semantic and stream-ordering context.
         * @return true after the reset has been accepted.
         */
        virtual bool resetSequenceState(
            int seq_idx,
            const StateResetContext &context) = 0;

        /**
         * @brief Reset one request slot in one model/cache layer.
         *
         * Hybrid caches interpret @p layer as a global model layer and reset
         * the corresponding KV or recurrent owner.  Invalid indices are
         * contract violations and must fail rather than become silent no-ops.
         */
        virtual bool resetLayerSequenceState(
            int layer,
            int seq_idx,
            const StateResetContext &context) = 0;

        /**
         * @brief Reset every request slot, or recurrent state, in one layer.
         *
         * @param layer Global or cache-local layer accepted by the implementation.
         * @param context Explicit semantic and stream-ordering context.
         * @return true after the reset has been accepted.
         */
        virtual bool resetLayerState(
            int layer,
            const StateResetContext &context) = 0;

        // =================================================================
        // Batched Operations
        // =================================================================

        /**
         * @brief Gather K/V from multiple sequences for batched attention
         *
         * Copies K/V from sequences [0..num_seqs-1] into contiguous output
         * tensors with padding to max_kv_len.
         *
         * Output layout: [num_seqs * max_kv_len, kv_dim]
         *
         * @param layer Layer index
         * @param num_sequences Number of sequences to gather
         * @param out_k Output K tensor
         * @param out_v Output V tensor
         * @param out_kv_lens Output: per-sequence kv_lens (size = num_seqs)
         * @return Maximum kv_len found, or -1 on error
         */
        virtual int gather_kv_batched(
            int layer,
            int num_sequences,
            ITensor *out_k,
            ITensor *out_v,
            std::vector<int> &out_kv_lens)
        {
            // Default: not supported (for caches that don't implement batched gather)
            (void)layer;
            (void)num_sequences;
            (void)out_k;
            (void)out_v;
            (void)out_kv_lens;
            return -1;
        }

        /**
         * @brief Publish a graph-capturable device view of multiple KV slots.
         *
         * GPU ring caches generally allocate each logical request in a separate
         * device allocation.  Batched attention therefore cannot obtain a valid
         * `[request, token, head, dim]` tensor by taking sequence zero's pointer
         * and applying a larger batch stride.  This contract asks the cache to
         * materialize the selected slots into persistent device workspace while
         * reading ring heads and counts from device-owned metadata.
         *
         * The returned tensors use the cache's native K and V formats and a
         * fixed request stride of @ref max_seq_len. Their pointers and shapes
         * are stable for the lifetime of the bound workspace, making the gather
         * and its consumers safe to capture in one CUDA/HIP graph while the
         * canonical device count grows. Implementations must fail when they do
         * not provide a native batched path; production callers must not replay
         * rows or silently substitute sequence zero.
         *
         * Only rows below each request's canonical device count are initialized
         * by an invocation. Consumers must use that same device count as their
         * logical bound. Inactive capacity is deliberately unspecified so a
         * short decode does not clear an entire maximum-context allocation.
         *
         * @param layer Model/cache layer whose entries will be gathered.
         * @param first_seq_idx First logical cache sequence to gather.
         * @param request_count Number of consecutive request slots.
         * @param out_k Receives cache-owned device K view.
         * @param out_v Receives cache-owned device V view.
         * @param gpu_stream Explicit backend stream ordering append, gather, and attention.
         * @return true when a native device gather was enqueued successfully.
         */
        virtual bool get_kv_batched_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ITensor **out_k,
            ITensor **out_v,
            void *gpu_stream)
        {
            (void)layer;
            (void)first_seq_idx;
            (void)request_count;
            (void)gpu_stream;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            LOG_ERROR("[IKVCache::get_kv_batched_device_view] Native device batched KV read is not implemented by this cache");
            return false;
        }

        // =================================================================
        // Sharding Info (for tensor parallelism)
        // =================================================================

        /**
         * @brief Check if cache is sharded across ranks
         * @return true if sharded, false otherwise (default: not sharded)
         */
        virtual bool is_sharded() const { return false; }

        /**
         * @brief Get local number of KV heads (for sharding)
         * @return Number of KV heads on this rank (default: 0)
         */
        virtual int local_n_kv_heads() const { return 0; }

        /**
         * @brief Get total number of KV heads across all ranks.
         *
         * Most concrete caches already expose this. The default preserves
         * non-sharded semantics for older implementations.
         */
        virtual int n_kv_heads() const { return local_n_kv_heads(); }

        /**
         * @brief Get local KV dimension (local_n_kv_heads * head_dim)
         * @return Local KV dimension (default: 0)
         */
        virtual int local_kv_dim() const { return 0; }

        /**
         * @brief Get starting KV head index for this rank
         * @return Starting KV head index (default: 0)
         */
        virtual int kv_head_start() const { return 0; }

        // =================================================================
        // Converted KV Access (dequant-on-read with optional fused RoPE)
        // =================================================================

        /**
         * @brief Parameters for fused RoPE during KV cache read.
         *
         * When rope_theta > 0, RoPE is applied to K inline during dequantization.
         * This is "RoPE-on-read": K is stored WITHOUT position embeddings in the cache,
         * and the embeddings are applied lazily when K is consumed by attention.
         *
         * Benefits:
         * - Fused dequant+RoPE eliminates a separate RoPE pass over K
         * - Position-free cache enables future speculative decoding
         * - For quantized caches, avoids dequant→RoPE→requant roundtrip
         */
        struct KVReadParams
        {
            float rope_theta = 0.0f;                           ///< RoPE frequency base (<=0 disables RoPE)
            int position_start = 0;                            ///< RoPE position of the first cached token
            int n_kv_heads = 0;                                ///< Number of KV heads
            int head_dim = 0;                                  ///< Dimension per attention head
            int rope_dim = 0;                                  ///< Number of dimensions to rotate per head (0 = full head_dim)
            int requested_token_count = 0;                     ///< Optional graph-captured logical span to expose after a device-side append.
            const TurboQuantContext *turboquant_ctx = nullptr; ///< Required for TQ cache dequant (optional otherwise)
            void *gpu_stream = nullptr;                        ///< Optional GPU stream for fused conversion/read kernels
        };

        /**
         * @brief Materialize a converted request-batched view from device-owned state.
         *
         * GPU RoPE-on-read cannot use an incrementally validated host shadow:
         * captured append and publication kernels are the sole owners of live
         * ring metadata. This contract therefore launches a fixed-shape grouped
         * gather/dequant operation which reads each request's canonical device
         * head/count, optionally applies RoPE to K, and writes stable cache-owned
         * output storage on the supplied stream.
         *
         * The converted view has a fixed request stride of @ref max_seq_len.
         * Rows beyond a request's device count are unspecified and must not be
         * consumed. Implementations must be graph-capturable, must not perform
         * D2H observation, and must not replay requests or rows through a scalar
         * production path.
         *
         * @param layer Model/cache layer whose entries will be materialized.
         * @param first_seq_idx First request slot in the contiguous batch.
         * @param request_count Number of request-local rings to process.
         * @param target Requested output precision.
         * @param out_k Receives the stable converted K view.
         * @param out_v Receives the stable converted V view.
         * @param read Device-read and RoPE policy; @c gpu_stream is mandatory.
         * @return true when one grouped device-owned materialization was enqueued.
         */
        virtual bool get_kv_batched_converted_device_view(
            int layer,
            int first_seq_idx,
            int request_count,
            ActivationPrecision target,
            ITensor **out_k,
            ITensor **out_v,
            const KVReadParams &read)
        {
            (void)layer;
            (void)first_seq_idx;
            (void)request_count;
            (void)target;
            (void)read;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            LOG_ERROR("[IKVCache::get_kv_batched_converted_device_view] Native grouped converted KV read is not implemented by this cache");
            return false;
        }

        /**
         * @brief Get K/V converted to a target precision with optional fused RoPE.
         *
         * Returns FP32 (or other target precision) views of the cached K/V data.
         * The cache manages internal shadow buffers and performs incremental
         * conversion — only newly appended rows are processed each call.
         *
         * When rope.rope_theta > 0, RoPE is fused into K dequantization,
         * eliminating a separate RoPE computation pass.
         *
         * @param layer Layer index
         * @param seq_idx Sequence index
         * @param target Target output precision (e.g., FP32)
         * @param out_k Output: pointer to converted K tensor
         * @param out_v Output: pointer to converted V tensor
         * @param out_kv_len Output: number of cached tokens (optional)
         * @param rope Optional RoPE parameters for fused application on K
         * @return true on success
         */
        virtual bool get_kv_converted(int layer, int seq_idx,
                                      ActivationPrecision target,
                                      ITensor **out_k, ITensor **out_v,
                                      int *out_kv_len = nullptr,
                                      const KVReadParams *rope = nullptr)
        {
            // No base fallback: callers that request converted/RoPE-applied KV
            // require an implementation that owns the necessary scratch buffers.
            (void)target;
            (void)rope;
            if (out_k)
                *out_k = nullptr;
            if (out_v)
                *out_v = nullptr;
            if (out_kv_len)
                *out_kv_len = 0;
            LOG_ERROR("[IKVCache::get_kv_converted] Converted KV read is not implemented by this cache");
            return false;
        }

        /**
         * @brief Templated convenience for get_kv_converted().
         *
         * Usage:
         *   cache->get_kv<ActivationPrecision::FP32>(layer, seq, &k, &v, &len, &rope);
         *
         * @tparam Target Output precision (compile-time ActivationPrecision)
         */
        template <ActivationPrecision Target>
        bool get_kv(int layer, int seq_idx,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr,
                    const KVReadParams *rope = nullptr)
        {
            return get_kv_converted(layer, seq_idx, Target, out_k, out_v, out_kv_len, rope);
        }

        /// Convenience: templated get_kv with seq_idx=0
        template <ActivationPrecision Target>
        bool get_kv(int layer,
                    ITensor **out_k, ITensor **out_v,
                    int *out_kv_len = nullptr,
                    const KVReadParams *rope = nullptr)
        {
            return get_kv_converted(layer, 0, Target, out_k, out_v, out_kv_len, rope);
        }

    private:
        StateOwnership state_ownership_{};
        bool state_ownership_bound_ = false;
    };

} // namespace llaminar2
